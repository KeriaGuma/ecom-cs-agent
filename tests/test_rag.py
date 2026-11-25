"""混合检索：BM25、向量、加权/等权 RRF、去重并集重排、相关性阈值过滤与多级故障退化。"""
import numpy as np

from cs_agent.config import settings
from cs_agent.rag import Chunk, HybridRetriever, LLMReranker, load_chunks, rrf
from cs_agent.tools import make_kb_tools
from tests.test_workers import ScriptedLLM, call


class CharEmbedder:
    VOCAB = "退换修运费款发票积分会员快递物流港澳台扣钱优惠券价保账号安装生鲜定制激活"

    def __init__(self, fail_query=False):
        self.doc_calls = 0
        self.fail_query = fail_query

    def _vec(self, t):
        return [float(t.count(c)) + 1e-3 for c in self.VOCAB]

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        if self.fail_query:
            raise TimeoutError("embedding down")
        return self._vec(text)


CHUNKS = load_chunks(settings.kb_dir)


def test_kb_loaded_with_all_documents_and_categories():
    assert len(CHUNKS) >= 40
    assert {c.source for c in CHUNKS} >= {"after_sales_policy.md", "shipping_logistics.md", "membership.md"}
    assert all(c.category for c in CHUNKS)


def test_bm25_only_when_no_embedder():
    r = HybridRetriever(CHUNKS)
    assert r.mode == "bm25"
    top = r.search("七天无理由退货运费谁出", k=3)
    assert top[0].title == "七天无理由退货" and top[0].detail == {"bm25": 1}


def test_hybrid_fuses_both_lists(tmp_path):
    r = HybridRetriever(CHUNKS, CharEmbedder(), cache_dir=str(tmp_path), model="fake")
    assert r.mode == "hybrid"
    hits = r.search("能寄到港澳台吗", k=5)
    assert any("vector" in h.detail and h.detail["vector"] for h in hits)
    assert "配送范围" in [h.title for h in hits]


def test_vector_cache_reused(tmp_path):
    e1 = CharEmbedder()
    HybridRetriever(CHUNKS, e1, cache_dir=str(tmp_path), model="fake")
    e2 = CharEmbedder()
    HybridRetriever(CHUNKS, e2, cache_dir=str(tmp_path), model="fake")
    assert e1.doc_calls == 1 and e2.doc_calls == 0
    assert len(list(tmp_path.glob("emb_*.npy"))) == 1


def test_vector_failure_at_query_time_falls_back_to_bm25(tmp_path):
    r = HybridRetriever(CHUNKS, CharEmbedder(fail_query=True), cache_dir=str(tmp_path), model="fake")
    hits = r.search("七天无理由退货运费谁出", k=3)
    assert hits and all("vector" not in h.detail for h in hits)


def test_rerank_reorders_filters_threshold_and_falls_back_on_error():
    chunks = [Chunk("a.md", "甲", "退货运费"), Chunk("b.md", "乙", "退货运费说明"), Chunk("c.md", "丙", "退货")]
    r = HybridRetriever(chunks, min_rerank_score=3.0)
    base = [h.title for h in r.search("退货运费", k=3)]
    r.reranker = LLMReranker(ScriptedLLM(replies=[call("Scores", {"scores": [1, 9, 5][: len(base)]})]))
    ranked = r.search("退货运费", k=3)
    assert [h.title for h in ranked] == [base[1]]  # 分数 1 < 阈值 3.0 被过滤
    assert ranked[0].detail["rerank"] == 9

    r.reranker = LLMReranker(ScriptedLLM(replies=[]))
    assert [h.title for h in r.search("退货运费", k=3)] == base


def test_rrf_supports_equal_and_weighted_fusion():
    fused = dict(rrf([[(1, 9.0), (2, 5.0)], [(2, 0.9), (3, 0.8)]]))
    assert fused[2] > fused[1] > fused[3]

    weighted = dict(rrf([[(0, 0.95)], [(1, 12.0)]], weights=[0.85, 0.15]))
    assert weighted[0] > weighted[1]


def test_search_policy_tool_cites_sources():
    tool = make_kb_tools(HybridRetriever(CHUNKS))[0]
    out = tool.invoke({"query": "重复扣款多久退回"})
    assert "[payment_and_invoice.md · 重复扣款]" in out


def test_embeddings_normalized(tmp_path):
    r = HybridRetriever(CHUNKS, CharEmbedder(), cache_dir=str(tmp_path), model="fake")
    assert np.allclose(np.linalg.norm(r.vector._mat, axis=1), 1.0, atol=1e-5)


def _ranked(chunks, vector_order, bm25_order):
    r = HybridRetriever(chunks)
    r.vector = type("V", (), {"search": lambda self, q, k: [(i, 1.0 - n * 0.1) for n, i in enumerate(vector_order)]})()
    r.bm25 = type("B", (), {"search": lambda self, q, k: [(i, 10.0 - n) for n, i in enumerate(bm25_order)]})()
    return r


def test_without_rerank_vector_order_wins_over_equal_rrf():
    chunks = [Chunk("x.md", t, t) for t in "甲乙丙丁"]
    r = _ranked(chunks, vector_order=[0, 1, 2], bm25_order=[3, 2, 1])
    assert [h.title for h in r.search("q", k=3)] == ["甲", "乙", "丙"]
    r.order_without_rerank = "rrf"
    assert r.search("q", k=1)[0].title != "甲"


def test_rerank_failure_falls_back_to_vector_order():
    chunks = [Chunk("x.md", t, t) for t in "甲乙丙丁"]
    r = _ranked(chunks, vector_order=[1, 0, 2], bm25_order=[3, 2, 0])
    r.reranker = LLMReranker(ScriptedLLM(replies=[]))
    assert [h.title for h in r.search("q", k=2)] == ["乙", "甲"]


def test_rerank_sees_union_of_both_recalls():
    chunks = [Chunk("x.md", t, t) for t in "甲乙丙丁"]
    seen = []

    class Spy:
        def rerank(self, q, cs):
            seen.extend(c.title for c in cs)
            return [1.0 if c.title == "丁" else 0.0 for c in cs]

    r = _ranked(chunks, vector_order=[0, 1], bm25_order=[3])
    r.reranker = Spy()
    assert r.search("q", k=1)[0].title == "丁"
    assert set(seen) == {"甲", "乙", "丁"}
