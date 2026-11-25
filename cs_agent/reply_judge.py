"""回复检查的 LLM 复核：关键词没命中时，再请评委判断回复是不是换了种说法表达了同一个要点。

端到端评测的回复检查是“关键词组”：每组列几个同义说法，命中一个就算回复说到了这个要点。
同义词表难免列不全，比如要点是“已经退款”，回复写“459 元已原路退回”就没命中“已退回”。
历史结果里，回复检查的失败全部是“缺少关键词”（“不应出现”一次也没有误判），所以只复核这一种：

- 关键词命中 → 直接通过，不调评委（绝大多数情况）；
- 没命中 → 评委先写出这组关键词代表的要点，再判断回复有没有**明确**说出这个要点，并引用回复原文；
  引用的原文必须真的出现在回复里（代码核对），否则按没说到处理；
- 两个评委（通义 + Kimi，都不是被测系统用的模型）**都**判“说到了”才改判为通过，有一个不同意就维持失败。
  宁可漏掉几个同义说法，也不放过真正答错的回复。

改判只影响“回复：缺少 …”这一类失败；原始的关键词结果保留在结果文件里（keyword_failures），可以对照。

用法：
  python -m cs_agent.reply_judge calib                 # 在开发集上检验评委（真实案例 + 合成正反例）
  python -m cs_agent.reply_judge regrade --results X.json --tasks T.jsonl --out Y.json   # 给已有结果重新判分
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import ROOT
from .eval_judge import _CLIENT, _ENV, _LOCK, CACHE_DIR, JUDGES, USAGE, load_tasks, parse_json

MISSING = "回复：缺少 "
DEFAULT_JUDGES = ("qwen", "kimi")
CALIB_DIR = ROOT / "eval" / "judge"

PROMPT = """你在复核电商客服评测的“回复检查”。评测规定：客服的最后一条回复必须说到某个要点，标注人用一组关键词表示这个要点
（列出了几种常见说法，命中任意一个就算说到）。这条回复一个关键词都没命中，可能有两种原因：
A. 回复换了种说法，其实明确说到了这个要点（这时应判通过）；
B. 回复确实没说到：没提、只是沾边、含糊其辞、说反了，或者只把它当成假设或条件（这时应判不通过）。

请按下面的步骤判断：
1. 根据关键词和用户说的话，用一句话写出这组关键词代表的要点（point）。组里的词是同一个要点的不同说法，
   要点要把它们都概括进去，不要只按第一个词理解（比如“已取消 / 已退款 / 已退回”的要点是“订单已经取消或退款已经办好”）。
2. 在回复里找明确表达这个要点的原文（evidence），原样摘抄回复中连续的一段文字，不要改写、不要拼接；找不到就留空。
3. conveyed：找到了明确表达要点的原文就是 true，否则是 false。

判断标准：
- 关键词之间是“或”的关系：回复明确表达了其中任意一个词的意思，就算说到。
- 同义说法、不同措辞、数字写法不同（如“1–3 个工作日”和“一到三个工作日”）、用“您”代替“买家”，都算说到。
- 只是沾边不算。要点是一个结论时，回复必须给出这个结论，不能只给出相关的状态或过程。
- 分清“还没有结果”和“结果是否定的”：“尚未”“还没”“暂未”“仍在处理中”说的是事情还没有结论，
  不等于“没通过”“被拒绝”这类已经定了的否定结论。
- 假设和条件不算：“审核通过后会退回”“如果需要可以帮您办理”都不等于“已经退回”“已经办好”。
- 说反了不算；同时出现互相矛盾的说法也不算。
- 只看这一条回复，不看之前的回复；不评价回复是否礼貌、是否还有别的问题。

只输出 JSON：{"point": "...", "evidence": "...", "conveyed": true 或 false}"""

PROMPT_VERSION = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]


def render(turns: list, reply: str, group: list[str]) -> str:
    lines = [f"用户第 {i + 1} 轮：{t if isinstance(t, str) else t.get('text', '')}" for i, t in enumerate(turns)]
    return ("\n".join(lines) + f"\n\n关键词（任意一个即可）：{' / '.join(group)}\n\n"
            f"客服的最后一条回复：\n<<<\n{reply}\n>>>")


def _norm(s: str) -> str:
    return re.sub(r"[\s*`#>\-|\"'“”‘’「」『』]+", "", s)  # 去掉空白、Markdown 符号和引号再比对，评委摘抄时常会改掉这些


def _verdict(raw: dict, reply: str) -> dict:
    if not isinstance(raw.get("conveyed"), bool):
        raise ValueError("缺少 conveyed")
    evidence = str(raw.get("evidence") or "")
    quoted = bool(evidence.strip()) and _norm(evidence) in _norm(reply)
    return {"point": str(raw.get("point", ""))[:200], "evidence": evidence[:300],
            "conveyed": raw["conveyed"] and quoted,  # 引用不在回复里：按没说到处理
            "said": raw["conveyed"], "quoted": quoted}


def ask(judge: str, turns: list, reply: str, group: list[str], retries: int = 3) -> dict:
    cfg = JUDGES[judge]
    content = render(turns, reply, group)
    key = hashlib.sha256(f"reply|{judge}|{cfg['body']['model']}|{PROMPT_VERSION}|{content}".encode()).hexdigest()[:24]
    path = CACHE_DIR / "reply" / judge / f"{key}.json"
    if path.exists():
        return _verdict(json.loads(path.read_text()), reply)
    api_key = (_ENV.get(cfg["key"]) or "").strip()
    if not api_key:
        raise SystemExit(f"未在 .env 中找到 {cfg['key']}")
    body = {**cfg["body"], "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]}
    err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = _CLIENT.post(cfg["base"] + "/chat/completions", json=body, headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            data = r.json()
            raw = parse_json(data["choices"][0]["message"]["content"])
            out = _verdict(raw, reply)
            u = data.get("usage") or {}
            with _LOCK:
                acc = USAGE.setdefault(f"reply-{judge}", {"calls": 0, "input": 0, "output": 0})
                acc["calls"] += 1
                acc["input"] += u.get("prompt_tokens", 0)
                acc["output"] += u.get("completion_tokens", 0)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, ensure_ascii=False))
            return out
        except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as e:
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{judge} 复核失败：{type(err).__name__}: {str(err)[:200]}")


def judge_group(turns: list, reply: str, group: list[str], judges=DEFAULT_JUDGES) -> dict:
    verdicts = {j: ask(j, turns, reply, group) for j in judges}
    return {"group": group, "verdicts": verdicts, "conveyed": all(v["conveyed"] for v in verdicts.values())}


def missing_groups(expect: dict, reply: str) -> list[list[str]]:
    return [g for g in expect.get("reply", []) if not any(s in reply for s in g)]


def recheck(task: dict, replies: list[str], failures: list[str], judges=DEFAULT_JUDGES) -> tuple[list[str], list[dict]]:
    """返回（复核后的失败列表，复核记录）。只改判“回复：缺少 …”，其他失败原样保留。"""
    reply = replies[-1] if replies else ""
    groups = missing_groups(task.get("expect", {}), reply)
    if not groups or not reply:
        return failures, []
    records = [judge_group(task["turns"], reply, g, judges) for g in groups]
    passed = {MISSING + "/".join(r["group"]) for r in records if r["conveyed"]}
    return [f for f in failures if f not in passed], records


# ---------------- 给已有结果重新判分 ----------------
def regrade(results: list[dict], tasks: dict[str, dict], judges=DEFAULT_JUDGES, workers: int = 8) -> list[dict]:
    def one(r: dict) -> dict:
        if r.get("reply_rechecks") is not None or not any(f.startswith(MISSING) for f in r["failures"]):
            return r
        fails, records = recheck(tasks[r["id"]], r["replies"], r["failures"], judges)
        # “缺少要点”是最轻的一级（沟通），去掉它们不会改变剩下失败的最高级别；全部改判通过时清空
        return {**r, "keyword_failures": r["failures"], "failures": fails, "success": not fails,
                "severity": r.get("severity") if fails else None, "reply_rechecks": records}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, results))


def cmd_regrade(args) -> None:
    data = json.loads(Path(args.results).read_text())
    tasks = load_tasks(args.tasks)
    data["results"] = regrade(data["results"], tasks, tuple(args.judges))
    data["reply_judge"] = {"prompt_version": PROMPT_VERSION, "judges": list(args.judges),
                           "models": {j: JUDGES[j]["body"]["model"] for j in args.judges}}
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=1))
    flips = [r for r in data["results"] if r.get("reply_rechecks") and r["success"]]
    rechecked = [r for r in data["results"] if r.get("reply_rechecks")]
    print(f"复核 {len(rechecked)} 次运行，改判通过 {len(flips)} 次 → {args.out}")


# ---------------- 开发集检验 ----------------
def cmd_calib(args) -> None:
    """三类样本：真实案例（开发集结果里关键词没命中的回复，人工标注）、合成反例、合成正例。"""
    rows = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]

    def run(row: dict) -> dict:
        return {**row, **judge_group(row["turns"], row["reply"], row["group"], tuple(args.judges))}

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(run, rows))
    by_kind: dict[str, list] = {}
    for r in out:
        by_kind.setdefault(r["kind"], []).append(r)
    print(f"提示词版本 {PROMPT_VERSION}，评委 {args.judges}")
    for kind, rs in by_kind.items():
        rs = [r for r in rs if r["label"] is not None]  # 存疑样本不计入
        pos = [r for r in rs if r["label"] == 1]
        neg = [r for r in rs if r["label"] == 0]
        line = f"{kind:8s} n={len(rs):3d}"
        if pos:
            line += f"  正例改判率 {sum(r['conveyed'] for r in pos)}/{len(pos)}"
        if neg:
            line += f"  反例错放 {sum(r['conveyed'] for r in neg)}/{len(neg)}"
        for j in args.judges:
            line += f"  | {j} 与标注一致 {sum(r['verdicts'][j]['conveyed'] == r['label'] for r in rs)}/{len(rs)}"
        print(line)
    both = [r for r in out if r["label"] is not None]
    agree = sum(len({r["verdicts"][j]["conveyed"] for j in args.judges}) == 1 for r in both)
    print(f"评委间一致 {agree}/{len(both)}")
    for r in out:
        if (r["label"] is not None and r["conveyed"] != r["label"]) or args.verbose:
            vs = " ".join(f"{j}={'Y' if v['conveyed'] else 'N'}{'' if v['quoted'] or not v['said'] else '(引用不符)'}"
                          for j, v in r["verdicts"].items())
            print(f"  [{'错' if r['conveyed'] != r['label'] else '对'}] {r['kind']} {r['id']} 标注={r['label']} {vs} "
                  f"| {'/'.join(r['group'])[:40]} | {r['reply'][:120]!r}")
    if args.out:
        Path(args.out).write_text(json.dumps({"prompt_version": PROMPT_VERSION, "rows": out}, ensure_ascii=False, indent=1))
    print("用量：", USAGE)


def main() -> None:
    ap = argparse.ArgumentParser(description="回复检查的 LLM 复核")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib")
    c.add_argument("--cases", default=str(CALIB_DIR / "reply_dev_cases.jsonl"))
    c.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES))
    c.add_argument("--out")
    c.add_argument("--verbose", action="store_true")
    g = sub.add_parser("regrade")
    g.add_argument("--results", required=True)
    g.add_argument("--tasks", nargs="+", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES))
    args = ap.parse_args()
    {"calib": cmd_calib, "regrade": cmd_regrade}[args.cmd](args)


if __name__ == "__main__":
    main()
