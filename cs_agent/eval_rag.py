"""知识库检索评测：python -m cs_agent.eval_rag [--queries eval/kb_dev.jsonl]

同一批问题对比几种检索方式：
  keyword        原来的字符二元组匹配（FAQ 降级仍在用）
  bm25           jieba 分词 + BM25
  vector         向量检索（需要 DASHSCOPE_API_KEY）
  hybrid(rrf)    BM25 + 向量，等权 RRF 融合（对照）
  *+rr:model     加 DashScope 专用重排模型
  *+rr:llm       加对话模型批量打分重排
指标：Recall@1、Recall@3（前 k 条里有任一标准答案）、MRR@10，以及单次检索延迟。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from .config import ROOT, settings

TEST_LOG = ROOT / "eval" / "reports" / "rag_test_runs.log"  # 测试集每跑一次追加一行：同一配置 + 同一份测试集只允许跑一次
from .kb import PolicyKB
from .rag import DashScopeReranker, LLMReranker, build_retriever


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score(ranked: list[str], expected: list[str]) -> tuple[float, float, float]:
    exp = set(expected)
    r1 = float(bool(ranked[:1]) and ranked[0] in exp)
    r3 = float(any(x in exp for x in ranked[:3]))
    rr = next((1.0 / i for i, x in enumerate(ranked[:10], 1) if x in exp), 0.0)
    return r1, r3, rr


def run(name: str, search, queries: list[dict]) -> dict:
    rows, lat, misses = [], [], []
    by_tag: dict[str, list[float]] = {}
    for q in queries:
        t0 = time.perf_counter()
        ranked = search(q["query"])
        lat.append((time.perf_counter() - t0) * 1000)
        s = score(ranked, q["expected"])
        rows.append(s)
        by_tag.setdefault(q.get("tag", "all"), []).append(s[0])
        if s[1] == 0:
            misses.append((q["id"], q["query"], ranked[:3]))
    n = len(rows)
    return {"mode": name, "n": n, "recall@1": sum(r[0] for r in rows) / n, "recall@3": sum(r[1] for r in rows) / n,
            "mrr@10": sum(r[2] for r in rows) / n, "p50_ms": statistics.median(lat), "misses": misses,
            "recall@1_by_tag": {t: sum(v) / len(v) for t, v in by_tag.items()}}


def _rrf_order(retr, q: str) -> list[str]:
    retr.order_without_rerank = "rrf"
    try:
        return [f"{h.source} · {h.title}" for h in retr.search(q, k=10)]
    finally:
        retr.order_without_rerank = "vector"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), default="dev", help="dev 用来选型；test 只在报告最终数字时跑一次")
    ap.add_argument("--queries", help="指定问题文件，覆盖 --split")
    ap.add_argument("--out")
    args = ap.parse_args()
    path = Path(args.queries) if args.queries else ROOT / "eval" / f"kb_{args.split}.jsonl"
    queries = load(path)

    is_test = args.split == "test" and not args.queries
    kb_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(settings.kb_dir).glob("*.md")))).hexdigest()[:8]
    config = f"embed={settings.embedding_model}\trerank={settings.rerank_model}\tkb={kb_hash}"
    data = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    if is_test and TEST_LOG.exists() and f"{config}\tdata={data}\t" in TEST_LOG.read_text(encoding="utf-8"):
        raise SystemExit(f"当前检索配置已经在这份测试集上跑过（见 {TEST_LOG}），不允许重复跑。")

    kw = PolicyKB(settings.kb_dir)
    retr = build_retriever(settings)
    retr.reranker = None  # 默认配置会带上重排；对照组必须先去掉，重排在下面按组单独挂
    ref = lambda hits: [f"{h.source} · {h.title}" for h in hits]  # noqa: E731

    reports = [
        run("keyword", lambda q: ref(kw.search(q, k=10)), queries),
        run("bm25", lambda q: ref(retr.search(q, k=10, use_vector=False)), queries),
    ]
    if retr.vector is not None:
        from .llm import get_llm

        model_rr = DashScopeReranker(settings.embedding_api_key, settings.rerank_model)
        llm_rr = LLMReranker(get_llm(settings.worker_model, temperature=0))
        reports += [
            run("vector", lambda q: ref(retr.search(q, k=10, use_bm25=False)), queries),
            run("hybrid(rrf)", lambda q: _rrf_order(retr, q), queries),
        ]
        for name, rr in (("model", model_rr), ("llm", llm_rr)):
            retr.reranker = rr
            reports.append(run(f"vector+rr:{name}", lambda q: ref(retr.search(q, k=10, use_bm25=False)), queries))
            reports.append(run(f"hybrid+rr:{name}", lambda q: ref(retr.search(q, k=10)), queries))
        retr.reranker = None
    else:
        print("（未配置 DASHSCOPE_API_KEY，跳过向量检索和重排）")

    print(f"\n检索评测  {len(queries)} 个问题  知识库 {len(retr.chunks)} 条\n")
    tags = sorted({q.get("tag", "all") for q in queries})
    print(f"{'方式':<17}{'Recall@1':>10}{'Recall@3':>10}{'MRR@10':>9}{'延迟 p50':>11}" + "".join(f"{'R@1·' + t:>16}" for t in tags))
    for r in reports:
        print(f"{r['mode']:<17}{r['recall@1']:>10.1%}{r['recall@3']:>10.1%}{r['mrr@10']:>9.3f}{r['p50_ms']:>9.0f}ms"
              + "".join(f"{r['recall@1_by_tag'].get(t, 0):>16.1%}" for t in tags))
    for r in reports:
        if r["misses"]:
            print(f"\n[{r['mode']}] 前 3 条都没命中：")
            for qid, q, top in r["misses"]:
                print(f"  {qid} {q}  →  {top}")
    if is_test:
        TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TEST_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{config}\tdata={data}\t"
                    + "\t".join(f"{r['mode']}={r['recall@1']:.4f}" for r in reports) + "\n")
    if args.out:
        Path(args.out).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
