"""多 Agent 对照基线的 Supervisor 路由离线评测：python -m cs_agent.eval_router [--split dev|test|all]

统计准确率（含 Wilson 95% 置信区间）、各标签召回、按 tag 分组准确率、混淆矩阵及延迟分位数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from bench.multi_agent.router import ACTIONS, SUPERVISOR_PROMPT, LLMRouter, Router

from .config import ROOT, settings
from .llm import get_llm

EVAL_DIR = ROOT / "eval"
TEST_LOG = EVAL_DIR / "reports" / "test_runs.log"
PROMPT_VERSION = hashlib.sha256(SUPERVISOR_PROMPT.encode()).hexdigest()[:8]
SPLITS = {"dev": ["route_dev.jsonl"], "test": ["route_test.jsonl"], "all": ["route_dev.jsonl", "route_test.jsonl"]}


def load_cases(paths: list[Path]) -> list[dict]:
    cases = []
    for p in paths:
        cases += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases


def to_state(case: dict) -> dict:
    msgs = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in case["messages"]]
    return {"messages": msgs, "facts": case.get("facts", {}), "turn_results": case.get("turn_results", [])}


def accepted(case: dict) -> set[str]:
    return {case["expected"], *case.get("also_ok", [])}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _percentile(sorted_vals: list[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))]


def _predict(router: Router, case: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        pred = router.decide(to_state(case)).next
    except Exception as e:
        pred = f"ERROR:{type(e).__name__}"
    return pred, (time.perf_counter() - t0) * 1000


def evaluate(router: Router, cases: list[dict], workers: int = 1) -> dict:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outputs = list(pool.map(lambda c: _predict(router, c), cases))

    correct, errors = 0, []
    total_by, hit_by = Counter(), Counter()
    tag_total, tag_hit = Counter(), Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    for c, (pred, _) in zip(cases, outputs):
        ok = pred in accepted(c)
        exp = c["expected"]
        total_by[exp] += 1
        confusion[exp][exp if ok else pred] += 1
        for t in c.get("tags", []):
            tag_total[t] += 1
            tag_hit[t] += ok
        if ok:
            correct += 1
            hit_by[exp] += 1
        else:
            errors.append({"id": c["id"], "expected": exp, "pred": pred, "text": c["messages"][-1]["content"]})

    lat = sorted(ms for _, ms in outputs)
    lo, hi = wilson(correct, len(cases))
    return {
        "n": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases),
        "ci95": [lo, hi],
        "recall_by_label": {k: {"n": v, "recall": hit_by[k] / v} for k, v in total_by.items()},
        "acc_by_tag": {k: {"n": v, "acc": tag_hit[k] / v} for k, v in tag_total.items()},
        "p50_ms": _percentile(lat, 0.5),
        "p95_ms": _percentile(lat, 0.95),
        "errors": errors,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def print_report(report: dict, title: str) -> None:
    lo, hi = report["ci95"]
    print(f"{title}  用例 {report['n']} 条")
    print(f"准确率 {report['accuracy']:.1%}（{report['correct']}/{report['n']}，95% CI {lo:.1%}～{hi:.1%}）"
          f"   p50 {report['p50_ms']:.0f}ms   p95 {report['p50_ms']:.0f}ms")

    print("\n按标签召回：")
    for label in ACTIONS:
        if label in report["recall_by_label"]:
            r = report["recall_by_label"][label]
            print(f"  {label:<14} {r['recall']:>6.1%}  (n={r['n']})")

    print("\n按 tag 准确率：")
    for tag, r in sorted(report["acc_by_tag"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {tag:<12} {r['acc']:>6.1%}  (n={r['n']})")

    labels = [a for a in ACTIONS if a in report["confusion"]]
    preds = labels + sorted({p for row in report["confusion"].values() for p in row} - set(labels))
    short = {"order_agent": "order", "refund_agent": "refund", "kb_agent": "kb", "handoff_human": "handoff"}
    print("\n混淆矩阵（行=期望，列=预测）：")
    print(" " * 10 + "".join(f"{short.get(p, p)[:8]:>9}" for p in preds))
    for e in labels:
        row = report["confusion"][e]
        print(f"{short.get(e, e)[:9]:<10}" + "".join(f"{row.get(p, 0) or '.':>9}" for p in preds))

    if report["errors"]:
        print("\n错例：")
        for e in report["errors"]:
            print(f"  {e['id']:<16} 期望 {e['expected']:<14} 实际 {e['pred']:<14} {e['text']}")


def _data_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _guard_test_run(model: str, data: str) -> None:
    if not TEST_LOG.exists():
        return
    tag = f"{model}\tprompt={PROMPT_VERSION}\tdata={data}\t"
    if any(tag in line for line in TEST_LOG.read_text(encoding="utf-8").splitlines()):
        raise SystemExit(f"模型 {model} + 提示词 {PROMPT_VERSION} 已经在测试集 {data} 上跑过（见 {TEST_LOG}）。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="dev")
    ap.add_argument("--cases", help="指定用例文件，覆盖 --split")
    ap.add_argument("--model", default=settings.router_model)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", help="把完整报告写成 JSON")
    args = ap.parse_args()

    paths = [Path(args.cases)] if args.cases else [EVAL_DIR / f for f in SPLITS[args.split]]
    is_test = args.split in ("test", "all") and not args.cases
    data = _data_version(EVAL_DIR / "route_test.jsonl") if is_test else "-"
    if is_test:
        _guard_test_run(args.model, data)

    router = LLMRouter(get_llm(args.model, temperature=0))
    cases = load_cases(paths)
    runs = [evaluate(router, cases, workers=args.workers) for _ in range(max(1, args.repeat))]
    report = runs[0]
    accs = [r["accuracy"] for r in runs]
    report.update(model=args.model, split=args.cases or args.split, prompt_version=PROMPT_VERSION,
                  data_version=data, runs_accuracy=accs, runs_errors=[r["errors"] for r in runs[1:]])
    print_report(report, f"模型 {args.model}  提示词 {PROMPT_VERSION}  数据集 {report['split']}")
    if len(runs) > 1:
        print(f"\n{len(runs)} 次运行准确率：" + " / ".join(f"{a:.1%}" for a in accs)
              + f"   均值 {statistics.mean(accs):.1%}  最低 {min(accs):.1%}  最高 {max(accs):.1%}")

    if is_test:
        TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TEST_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{args.model}\tprompt={PROMPT_VERSION}\tdata={data}"
                    f"\tsplit={args.split}\tacc={statistics.mean(accs):.4f}\truns={len(runs)}\tn={report['n']}\n")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
