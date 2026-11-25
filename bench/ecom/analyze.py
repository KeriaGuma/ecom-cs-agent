"""汇总 ECom-Bench 运行结果：pass^k、失败出在哪一项、Token、每轮延迟，以及两种实现的逐任务配对比较。

用法：python bench/ecom/analyze.py <结果目录A> [<结果目录B> ...]
每个目录里是 ECom-Bench 的结果 JSON（一份）和 run.py 写的 usage_<impl>.json。第一个目录作为配对比较的基线。
"""
import glob
import json
import random
import statistics as st
import sys
from math import comb


def load(d: str):
    res = [f for f in glob.glob(f"{d}/*.json") if "/usage_" not in f]
    assert len(res) == 1, f"{d} 里应只有一份结果文件：{res}"
    rows = json.load(open(res[0]))
    usage = json.load(open(glob.glob(f"{d}/usage_*.json")[0]))
    per: dict[int, list[int]] = {}
    parts = {"action": 0, "search": 0, "output": 0}
    for r in rows:
        per.setdefault(r["task_id"], []).append(1 if r["reward"] >= 1 else 0)
        for k in parts:
            parts[k] += 1 - int(r["detail_reward"][k])
    return per, parts, usage, len(rows)


def passk(per, k):
    return st.mean(comb(sum(v), k) / comb(len(v), k) for v in per.values() if len(v) >= k)


def pct(x):
    return f"{100 * x:.1f}%"


def impl_label(usage):
    return f"{usage['impl']}/{usage.get('runtime', 'langgraph')}"


runs = {d: load(d) for d in sys.argv[1:]}
for d, (per, parts, usage, n) in runs.items():
    secs = sorted(usage["turn_seconds"])
    p95 = secs[min(len(secs) - 1, int(0.95 * len(secs)))] if secs else 0
    tok = (usage["input_tokens"] + usage["output_tokens"]) / n
    print(f"{impl_label(usage):18} 任务 {len(per)} × {n // len(per)} 次 | pass^1/2/3 = "
          + " / ".join(pct(passk(per, k)) for k in (1, 2, 3))
          + f" | 失败项（{n} 次中）数据库 {parts['action']} 查询 {parts['search']} 关键词 {parts['output']}"
          + f" | 客服 Token/次 {tok:.0f} 模型调用/次 {usage['calls'] / n:.1f} | 每轮延迟 p50 {st.median(secs):.1f}s p95 {p95:.1f}s")

names = list(runs)
base = runs[names[0]][0]
for d in names[1:]:
    per = runs[d][0]
    ids = sorted(set(base) & set(per))
    diffs = [st.mean(per[t]) - st.mean(base[t]) for t in ids]
    random.seed(0)
    bs = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(10000))
    print(f"{impl_label(runs[d][2])} − {impl_label(runs[names[0]][2])}：{100 * st.mean(diffs):+.1f}%  95% CI "
          f"[{100 * bs[250]:+.1f}%, {100 * bs[9749]:+.1f}%]  更好 {sum(x > 0 for x in diffs)} 更差 {sum(x < 0 for x in diffs)} "
          f"持平 {sum(x == 0 for x in diffs)}")
    print("  逐任务（基线, 对比）:", {t: (sum(base[t]), sum(per[t])) for t in ids if base[t] != per[t]})
