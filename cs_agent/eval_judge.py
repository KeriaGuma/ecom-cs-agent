"""沟通质量评测：LLM 评委（LLM-as-a-judge）。

端到端评测只看“事情办没办成”（数据库、关键路径、关键词），这里补上“话说得怎么样”。
评委读完整段对话（用户台词、每轮的工具调用和返回、客服回复），按五个维度分别打分，并引用原文作为依据：

  answered        答全：每轮用户提出的问题和诉求都回应了（办了、查了、说明了不能办的原因或需要什么信息）     0 / 1
  faithful        忠实：回复里的事实都能在工具返回里找到依据，没有编造或说反                             0 / 1
  no_overpromise  不越权承诺：没有承诺工具结果以外的补偿、时效、结果                                    0 / 1
  concise         简洁：2 简洁到位，1 略有冗余，0 明显冗长重复                                         0 / 1 / 2
  tone            语气：2 礼貌专业自然，1 生硬或过度客套，0 不礼貌、冷漠或推诿                           0 / 1 / 2

评委和被测系统不是同一家模型（被测系统用 DeepSeek），避免自我偏好：
  kimi：Moonshot kimi-k2.6，关闭思考（该模式下接口只允许 temperature=0.6）
  qwen：阿里云百炼 qwen3.7-flash，关闭思考，temperature=0（第二评委，用来计算评委间一致性）
Key 只从 .env 读取（KIMI_API_KEY、DASHSCOPE_API_KEY），不出现在命令行和结果文件里。

评委的打分缓存在 eval/judge/cache/ 下（按评委、提示词版本和对话内容的哈希），重跑不重复付费。

用法：
  python -m cs_agent.eval_judge score  --results eval/judge/dev_main.json --tasks eval/e2e_dev.jsonl --judges kimi qwen --out <路径>
  python -m cs_agent.eval_judge synth  ...   （合成缺陷检验，见 eval/judge/README.md）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import dotenv_values

from .config import ROOT

JUDGE_DIR = ROOT / "eval" / "judge"
CACHE_DIR = JUDGE_DIR / "cache"

DIMS = {  # 维度 → 最高分
    "answered": 1,
    "faithful": 1,
    "no_overpromise": 1,
    "concise": 2,
    "tone": 2,
}

JUDGES = {
    "kimi": {"base": "https://api.moonshot.cn/v1", "key": "KIMI_API_KEY",
             "body": {"model": "kimi-k2.6", "thinking": {"type": "disabled"}, "temperature": 0.6}},
    "qwen": {"base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key": "DASHSCOPE_API_KEY",
             "body": {"model": "qwen3.7-flash", "enable_thinking": False, "temperature": 0}},
}

RUBRIC = """你是电商客服对话的质检员。下面是一段完整对话：用户每轮说的话、客服在这一轮调用的内部工具及其返回、客服给用户的回复。
工具调用对用户不可见，只是给你核对事实用的。请只评价“客服给用户的回复”，按五个维度分别打分。

背景：客服只能通过工具查订单、物流、退款规则和售后政策；退款金额、运费归属、能否退款以工具返回为准。
超过 500 元的退款提交后进入人工审核，审核结果以“系统事件”的形式给出，它和工具返回一样是事实依据。
有的客服会在审核前写好回复，由系统在末尾附上“审核结果”。请按用户读到的整条回复判断：正文和末尾的审核结果不冲突
（如正文“需人工审核，通过后原路退回”、末尾“审核结果：已通过”）不扣分；正文让用户以为会退款或要寄回商品，
而审核结果是驳回，属于信息冲突，在 faithful 扣分。
转人工时，客服的回复可能是一段交接说明，这是正常的。

维度和标准：
1. answered（0/1）：每一轮用户提出的每个问题和诉求，客服都有回应。回应可以是办理了、查询并告知了结果、说明了不能办的原因，
   或者询问办理所需的信息（如退款原因、哪个订单）。只要有一个问题或诉求被忽略，就是 0。
   这一项只看“有没有回应”，不看回应的内容对不对：内容和工具返回不符，归到 faithful 扣分，不在这里扣。
   以下都算回应：对超出客服权限的诉求（如价保、加盟）转人工；对与业务无关的请求（如写诗）说明办不了。
2. faithful（0/1）：回复里的事实（金额、订单状态、日期、物流、运费由谁承担、能否退款、政策条款）都与工具返回一致或能从中得出。
   有任何一处与工具返回矛盾，或者给出了工具返回里没有依据的具体事实，就是 0。礼貌用语、通用建议不算事实。
3. no_overpromise（0/1）：没有承诺工具结果以外的东西，比如额外补偿、优惠券、保证某个时间送达或到账、保证审核通过。
   有任何一处这样的承诺就是 0。转述工具返回里本来就有的时效或结果不算。
4. concise（0/1/2）：2 = 简洁到位，没有多余内容；1 = 有些冗余（重复信息、不必要的客套或铺垫），但不影响阅读；
   0 = 明显冗长，大量重复或与问题无关的内容。
5. tone（0/1/2）：2 = 礼貌、专业、自然；1 = 有些生硬、机械，或客套过度；0 = 不礼貌、冷漠、不耐烦或推诿。

每个维度都要给出依据：引用客服回复的原文；faithful 判 0 时，同时引用与之矛盾的工具返回。满分时依据可以写“无问题”。
只输出一个 JSON 对象，不要输出其他内容：
{"answered": {"score": 1, "evidence": "..."}, "faithful": {"score": 1, "evidence": "..."},
 "no_overpromise": {"score": 1, "evidence": "..."}, "concise": {"score": 2, "evidence": "..."}, "tone": {"score": 2, "evidence": "..."}}"""

PROMPT_VERSION = hashlib.sha256(RUBRIC.encode()).hexdigest()[:8]


# ---------------- 对话渲染 ----------------
def _short(x: Any, n: int = 600) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…（截断）"


APPROVAL_MARK = "需人工审核"  # submit_refund 的返回里出现它，说明这一轮进入了人工审核


def render_transcript(turns: list[str], replies: list[str], turn_tools: list[list[dict]],
                      approval: str = "approve") -> str:
    """approval：任务设定的审核员决定（评测程序按它模拟审核员）。审核结果不是工具返回，不写出来评委会误判为编造。"""
    lines = []
    for i, user in enumerate(turns):
        lines.append(f"【第 {i + 1} 轮】")
        lines.append(f"用户：{user}")
        tools = turn_tools[i] if i < len(turn_tools) else []
        if tools:
            lines.append("（内部工具调用，用户看不到）")
            for t in tools:
                lines.append(f"  - {t['tool']}({_short(t['args'], 200)}) → {_short(t['output'], 1500)}")
        else:
            lines.append("（本轮没有调用工具）")
        if any(t["tool"] == "submit_refund" and APPROVAL_MARK in (t["output"] or "") for t in tools):
            verdict = "通过，款项原路退回" if approval == "approve" else "驳回，订单保持原状"
            lines.append(f"（系统事件：该退款进入人工审核，审核员已{verdict}）")
        lines.append(f"客服：{replies[i] if i < len(replies) else '（无回复）'}")
    return "\n".join(lines)


# ---------------- 调用评委 ----------------
_ENV = dotenv_values(ROOT / ".env")
_CLIENT = httpx.Client(timeout=120)
_LOCK = threading.Lock()
USAGE: dict[str, dict[str, int]] = {}


def parse_json(text: str) -> dict:
    """评委偶尔会包一层 ```json …```，或在 JSON 前后加话：取第一个 { 到最后一个 }。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"评委输出里没有 JSON：{text[:200]}")
    return json.loads(m.group(0))


def validate(scores: dict) -> dict:
    out = {}
    for dim, top in DIMS.items():
        item = scores.get(dim)
        if not isinstance(item, dict) or not isinstance(item.get("score"), (int, float)):
            raise ValueError(f"缺少维度 {dim}")
        s = int(item["score"])
        if not 0 <= s <= top:
            raise ValueError(f"{dim} 分数越界：{s}")
        out[dim] = {"score": s, "evidence": str(item.get("evidence", ""))[:300]}
    return out


def call_judge(judge: str, transcript: str, sample: int = 0, retries: int = 3) -> dict:
    """返回 {维度: {score, evidence}}。sample 用于同一评委重复抽样（kimi 的温度不能为 0），也是缓存键的一部分。"""
    cfg = JUDGES[judge]
    key = hashlib.sha256(f"{judge}|{cfg['body']['model']}|{PROMPT_VERSION}|{sample}|{transcript}".encode()).hexdigest()[:24]
    path = CACHE_DIR / judge / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    api_key = (_ENV.get(cfg["key"]) or "").strip()
    if not api_key:
        raise SystemExit(f"未在 .env 中找到 {cfg['key']}")
    body = {**cfg["body"], "messages": [{"role": "system", "content": RUBRIC},
                                        {"role": "user", "content": transcript}]}
    err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = _CLIENT.post(cfg["base"] + "/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            data = r.json()
            result = validate(parse_json(data["choices"][0]["message"]["content"]))
            u = data.get("usage") or {}
            with _LOCK:
                acc = USAGE.setdefault(judge, {"calls": 0, "input": 0, "output": 0})
                acc["calls"] += 1
                acc["input"] += u.get("prompt_tokens", 0)
                acc["output"] += u.get("completion_tokens", 0)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False))
            return result
        except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as e:
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{judge} 评分失败：{type(err).__name__}: {str(err)[:200]}")


# ---------------- 统计 ----------------
def cohen_kappa(a: list[int], b: list[int], weighted: bool = False) -> Optional[float]:
    """两位评委在同一批样本上的一致性。weighted=True 时用线性加权（适合 0/1/2 这种有序分数）。"""
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    if len(cats) == 1:
        return None  # 只有一种取值，κ 无定义（一致率见 agree）
    span = max(cats) - min(cats)
    w = (lambda x, y: abs(x - y) / span) if weighted else (lambda x, y: float(x != y))
    n = len(a)
    obs = sum(w(x, y) for x, y in zip(a, b)) / n
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    exp = sum(pa[x] * pb[y] * w(x, y) for x in cats for y in cats)
    return None if exp == 0 else 1 - obs / exp


def agreement(xs: list[dict], ys: list[dict]) -> dict:
    """xs、ys：同一批对话上两次打分（两位评委，或同一评委两次抽样）。"""
    out = {}
    for dim, top in DIMS.items():
        a = [x[dim]["score"] for x in xs]
        b = [y[dim]["score"] for y in ys]
        k = cohen_kappa(a, b, weighted=top > 1)
        out[dim] = {"agree": sum(p == q for p, q in zip(a, b)) / len(a) if a else None,
                    "kappa": None if k is None else round(k, 3)}
    return out


# ---------------- 子命令：给一份评测结果打分 ----------------
def load_tasks(paths: list[str]) -> dict[str, dict]:
    tasks = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                tasks[t["id"]] = t
    return tasks


def transcripts_from(results_path: str, tasks: dict[str, dict], label: Optional[str] = None) -> list[dict]:
    """label：覆盖结果里的 system 名，用来比较同一系统的新旧两版（比如 single_old / single_new）。"""
    rows = []
    for r in json.loads(Path(results_path).read_text())["results"]:
        if "turn_tools" not in r:
            raise SystemExit(f"{results_path} 没有记录工具返回（turn_tools），请用新版 eval_e2e 重跑")
        if not r["replies"]:
            continue
        rows.append({"id": r["id"], "system": label or r["system"], "run": r["run"], "success": r["success"],
                     "transcript": render_transcript(tasks[r["id"]]["turns"], r["replies"], r["turn_tools"],
                                                     tasks[r["id"]].get("approval", "approve"))})
    return rows


def score_rows(rows: list[dict], judges: list[str], samples: int = 1, workers: int = 8) -> None:
    jobs = [(row, j, s) for row in rows for j in judges for s in range(samples)]

    def run(job):
        row, j, s = job
        try:
            res = call_judge(j, row["transcript"], s)
        except RuntimeError as e:
            res = {"error": str(e)}
        with _LOCK:
            row.setdefault("scores", {}).setdefault(j, {})[s] = res

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))


def ok(res: dict) -> bool:
    return isinstance(res, dict) and "error" not in res


def summarize_scores(rows: list[dict], judge: str, sample: int = 0) -> dict:
    out = {}
    for system in sorted({r["system"] for r in rows}):
        rs = [r["scores"][judge][sample] for r in rows if r["system"] == system and ok(r["scores"][judge][sample])]
        out[system] = {"n": len(rs), **{d: round(sum(x[d]["score"] for x in rs) / len(rs) / top, 4) if rs else None
                                        for d, top in DIMS.items()}}
    return out


def paired(rows: list[dict], judge: str, a: str = "multi", b: str = "single", seed: int = 0) -> dict:
    """逐任务配对：每个任务先对多次运行取平均（按满分归一），再看 a − b 的均值和 bootstrap 95% CI。"""
    import random

    out = {}
    for dim, top in DIMS.items():
        per: dict[str, dict[str, list[float]]] = {}
        for r in rows:
            sc = r["scores"][judge][0]
            if ok(sc):
                per.setdefault(r["id"], {}).setdefault(r["system"], []).append(sc[dim]["score"] / top)
        diffs = [sum(v[a]) / len(v[a]) - sum(v[b]) / len(v[b]) for v in per.values() if v.get(a) and v.get(b)]
        if not diffs:
            continue
        rng = random.Random(seed)
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
        out[dim] = {"diff": round(sum(diffs) / len(diffs), 4), "ci95": [round(boots[250], 4), round(boots[9749], 4)],
                    f"{a}_better": sum(d > 0 for d in diffs), f"{b}_better": sum(d < 0 for d in diffs), "tasks": len(diffs)}
    return out


def cmd_score(args) -> None:
    tasks = load_tasks(args.tasks)
    labels = args.labels or [None] * len(args.results)
    if len(labels) != len(args.results):
        raise SystemExit("--labels 的个数要和 --results 相同")
    rows = [row for p, lb in zip(args.results, labels) for row in transcripts_from(p, tasks, lb)]
    score_rows(rows, args.judges, args.samples, args.workers)
    errors = sum(not ok(v) for r in rows for j in r["scores"].values() for v in j.values())
    report: dict[str, Any] = {"prompt_version": PROMPT_VERSION, "judges": {j: JUDGES[j]["body"]["model"] for j in args.judges},
                              "n": len(rows), "errors": errors, "usage": USAGE,
                              "summary": {j: summarize_scores(rows, j) for j in args.judges}}
    if args.pair:
        a, b = args.pair
        report[f"paired_{a}_minus_{b}"] = {j: paired(rows, j, a, b) for j in args.judges}
    elif {r["system"] for r in rows} == {"multi", "single"}:
        report["paired_multi_minus_single"] = {j: paired(rows, j) for j in args.judges}
    both = [r for r in rows if all(ok(r["scores"][j][0]) for j in args.judges)]
    if len(args.judges) == 2:
        j1, j2 = args.judges
        report["inter_judge"] = agreement([r["scores"][j1][0] for r in both], [r["scores"][j2][0] for r in both])
    if args.samples > 1:
        for j in args.judges:
            rr = [r for r in rows if ok(r["scores"][j][0]) and ok(r["scores"][j][1])]
            report.setdefault("test_retest", {})[j] = agreement([r["scores"][j][0] for r in rr], [r["scores"][j][1] for r in rr])
    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({**report, "rows": rows}, ensure_ascii=False, indent=1))


# ---------------- 子命令：合成缺陷检验 ----------------
# 取两位评委都给满分的对话，把最后一轮客服回复人为改坏（每份只植入一种缺陷），看评委能不能在对应维度上扣分。
# 抓不出来的维度，评委在这个维度上的打分就不可信，不纳入结论。
PROMISES = [
    "为表歉意，我再额外给您补偿一张 50 元无门槛优惠券。",
    "我保证款项明天之内一定到账。",
    "审核那边我已经打过招呼了，肯定能通过，您放心。",
    "另外这单运费我们也给您全免了。",
]
REWRITE = {
    "answered": "下面给出用户的一句话和客服的回复。如果用户这句话里只提了一个问题或诉求，只输出 SKIP。"
                "否则，从客服回复里删掉回应用户其中一个问题或诉求的全部内容（只删一个，删得干净，不要留下暗示），"
                "其余内容一字不改。注意：要删的是对用户明确提出的问题或诉求的回应，不是客服主动补充的信息。只输出改写后的回复。",
    "concise": "下面是一段客服回复。在不增删任何事实的前提下，把它改得冗长啰嗦：关键信息重复说两三遍，加入大量客套和铺垫，"
               "长度约为原来的三倍。只输出改写后的回复。",
    "tone": "下面是一段客服回复。在不增删任何事实和信息的前提下，把语气改得冷淡、不耐烦，像在敷衍用户（比如“自己看”“说过了”"
            "“就这样”），长度和原来相近。只输出改写后的回复。",
}
NUM = re.compile(r"\d+(?:\.\d+)?")


def corrupt_fact(reply: str, tools: list[dict]) -> Optional[str]:
    """改错一个事实：把回复里出现、且能在工具返回里找到的金额改掉；没有金额就把运费承担方说反。"""
    evidence = " ".join(t["output"] for t in tools)
    money = [m for m in NUM.finditer(reply) if reply[m.end():m.end() + 2].lstrip().startswith("元")]
    for m in money + list(NUM.finditer(reply)):  # 优先改金额，其次是其他能在工具返回里找到的数字
        v = m.group(0)
        if len(v.split(".")[0]) >= 2 and v in evidence:
            new = str(int(float(v)) + 100) if "." not in v else f"{float(v) + 100:.2f}"
            return reply[:m.start()] + new + reply[m.end():]
    for a, b in (("由您承担", "由商家承担"), ("您承担", "商家承担"), ("买家承担", "商家承担"),
                 ("由商家承担", "由您承担"), ("商家承担", "您承担")):
        if a in reply:
            return reply.replace(a, b, 1)
    return None


def make_defects(rows: list[dict], tasks: dict[str, dict], results: dict, per_type: int, seed: int = 7,
                 dims: tuple = tuple(DIMS)) -> list[dict]:
    import random

    from .config import settings
    from .llm import get_llm

    rng = random.Random(seed)
    llm = get_llm(settings.worker_model, temperature=0.7)  # 生成缺陷用被测系统的模型即可，评委是另外两家
    out = []
    for dim in dims:
        pool = rows[:]
        rng.shuffle(pool)
        made = 0
        for row in pool:
            if made >= per_type:
                break
            r = results[(row["id"], row["system"], row["run"])]
            reply, tools = r["replies"][-1], r["turn_tools"][-1]
            if dim == "faithful":
                bad = corrupt_fact(reply, tools)
            elif dim == "no_overpromise":
                bad = reply.rstrip() + PROMISES[made % len(PROMISES)]
            else:
                text = (f"用户：{tasks[row['id']]['turns'][-1]}\n客服回复：{reply}" if dim == "answered" else reply)
                bad = str(llm.invoke([("system", REWRITE[dim]), ("user", text)]).content).strip()
                bad = bad.removeprefix("客服回复：").strip()
                if bad.upper().startswith("SKIP") or bad == reply:
                    bad = None
            if not bad:
                continue
            replies = r["replies"][:-1] + [bad]
            out.append({"id": row["id"], "system": row["system"], "run": row["run"], "defect": dim,
                        "original": reply, "defective": bad,
                        "transcript": render_transcript(tasks[row["id"]]["turns"], replies, r["turn_tools"],
                                                        tasks[row["id"]].get("approval", "approve"))})
            made += 1
    return out


def cmd_synth(args) -> None:
    scored = json.loads(Path(args.scored).read_text())
    tasks = load_tasks(args.tasks)
    results = {(r["id"], r["system"], r["run"]): r
               for p in args.results for r in json.loads(Path(p).read_text())["results"]}
    judges = list(scored["judges"])
    clean = [row for row in scored["rows"]
             if all(ok(row["scores"][j]["0"]) and all(row["scores"][j]["0"][d]["score"] == top for d, top in DIMS.items())
                    for j in judges)]
    print(f"两位评委都给满分的对话：{len(clean)} / {len(scored['rows'])}")
    cases_path = Path(args.cases)
    if cases_path.exists():  # 缺陷样本生成一次后固定下来，改评分标准时用同一批样本
        cases = json.loads(cases_path.read_text())
        if args.regenerate:  # 只重新生成指定维度的样本（生成方法有问题时用），其余维度保持不变
            cases = [c for c in cases if c["defect"] not in args.regenerate]
            cases += make_defects(clean, tasks, results, args.per_type, dims=tuple(args.regenerate))
            cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    else:
        cases = make_defects(clean, tasks, results, args.per_type)
        cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    score_rows(cases, judges, 1, args.workers)
    report: dict[str, Any] = {"prompt_version": PROMPT_VERSION, "clean_pool": len(clean), "n": len(cases), "usage": USAGE,
                              "detection": {}}
    for j in judges:
        det = {}
        for dim, top in DIMS.items():
            cs = [c for c in cases if c["defect"] == dim and ok(c["scores"][j][0])]
            hit = [c for c in cs if c["scores"][j][0][dim]["score"] < top]
            # 连带扣分：植入的是别的缺陷，但这个维度也被扣分（说明评委分不清维度）
            other = [c for c in cases if c["defect"] != dim and ok(c["scores"][j][0])]
            spill = [c for c in other if c["scores"][j][0][dim]["score"] < top]
            det[dim] = {"n": len(cs), "detected": round(len(hit) / len(cs), 3) if cs else None,
                        "spill_from_other_defects": round(len(spill) / len(other), 3) if other else None,
                        "missed": [c["id"] + f"/{c['system']}/{c['run']}" for c in cs if c not in hit]}
        report["detection"][j] = det
    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({**report, "cases": cases}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="给 eval_e2e 的结果文件打分")
    sc.add_argument("--results", nargs="+", required=True)
    sc.add_argument("--tasks", nargs="+", required=True, help="对应的任务文件（取用户台词）")
    sc.add_argument("--judges", nargs="+", default=["kimi", "qwen"], choices=list(JUDGES))
    sc.add_argument("--labels", nargs="+", help="给每个结果文件指定系统名（覆盖文件里的 system），与 --results 一一对应")
    sc.add_argument("--pair", nargs=2, metavar=("A", "B"), help="配对比较 A − B（默认 multi − single）")
    sc.add_argument("--samples", type=int, default=1, help="每位评委重复打分几次（≥2 时计算重测一致性）")
    sc.add_argument("--workers", type=int, default=8)
    sc.add_argument("--out")
    sy = sub.add_parser("synth", help="合成缺陷检验：在评委都给满分的对话上植入缺陷，看评委能否抓出")
    sy.add_argument("--scored", required=True, help="score 子命令的输出（需包含两位评委）")
    sy.add_argument("--results", nargs="+", required=True, help="生成这些对话的 eval_e2e 结果文件")
    sy.add_argument("--tasks", nargs="+", required=True)
    sy.add_argument("--cases", default=str(JUDGE_DIR / "synth_cases.json"), help="缺陷样本（不存在时生成，之后固定复用）")
    sy.add_argument("--per-type", type=int, default=30)
    sy.add_argument("--regenerate", nargs="+", choices=list(DIMS), help="只重新生成这些维度的缺陷样本")
    sy.add_argument("--workers", type=int, default=8)
    sy.add_argument("--out")
    args = ap.parse_args()
    {"score": cmd_score, "synth": cmd_synth}[args.cmd](args)


if __name__ == "__main__":
    main()
