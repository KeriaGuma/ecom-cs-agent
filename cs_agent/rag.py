"""售后政策知识库的混合检索管线：BM25 + 向量两路召回 → 去重并集 → Cross-Encoder 重排（带相关性阈值过滤）。

线上检索与多级降级策略（依据 eval_rag 评测实验确立）：
1. **标准线上模式（BM25 + Vector + 专用 Reranker）**：
   - BM25（补充订单号、型号、专有名词等精确字面召回）与向量检索（捕获口语化同义表达）各召回 Top-K，
     直接做有序去重并集送入 DashScope Cross-Encoder 重排模型打分，并按 `min_rerank_score` 过滤低相关噪声。
2. **重排服务故障或未开启时**：
   - 支持向量优先或加权 RRF（Weighted RRF，默认向量权重高于 BM25，避免口语提问时 BM25 词频噪声挤压语义命中结果）。
3. **向量服务不可用或离线环境**：
   - 自动无感降级为纯 BM25 检索；主模型亦故障时，外层编排器进一步降级至零外部依赖的 `kb.PolicyKB`。

注：当前知识库共 8 份文档、48 条政策条款（原型规模），按二级标题切块以保留完整条款与引用出处；
向量按内容 SHA-256 哈希持久化缓存至本地磁盘，条款变更后自动重建。
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .kb import PolicyKB

log = logging.getLogger(__name__)

RRF_K = 60


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str
    category: str = ""

    @property
    def ref(self) -> str:
        return f"{self.source} · {self.title}"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    detail: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def title(self) -> str:
        return self.chunk.title

    @property
    def text(self) -> str:
        return self.chunk.text


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def load_chunks(kb_dir: str) -> list[Chunk]:
    return [Chunk(s.source, s.title, s.text, category=Path(s.source).stem) for s in PolicyKB(kb_dir).sections]


# ---------------- BM25 ----------------
_PUNCT = re.compile(r"[\s，。、？！：；,.?!:;（）()“”\"'【】\[\]·…—\-]+")
_STOP = {"的", "了", "吗", "呢", "吧", "啊", "我", "你", "是", "在", "有", "和", "就", "都", "也", "要", "能", "可以", "怎么", "什么"}


def tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut_for_search(_PUNCT.sub(" ", text)) if t.strip() and t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(c.title + " " + c.title + " " + c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


# ---------------- 向量索引 ----------------
class VectorIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache_dir: Optional[str] = None, model: str = ""):
        texts = [f"{c.title}\n{c.text}" for c in chunks]
        key = hashlib.sha256(("\n\x00".join([model] + texts)).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"emb_{key}.npy" if cache_dir else None
        if cache and cache.exists():
            mat = np.load(cache)
        else:
            mat = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, mat)
        self._mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        self._embedder = embedder

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        sims = self._mat @ (q / np.linalg.norm(q))
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


def rrf(rank_lists: list[list[tuple[int, float]]], k: int = RRF_K,
        weights: Optional[list[float]] = None) -> list[tuple[int, float]]:
    """支持自定义路权重的 Reciprocal Rank Fusion（默认等权）。"""
    ws = weights if weights is not None else [1.0] * len(rank_lists)
    fused: dict[int, float] = {}
    for lst, w in zip(rank_lists, ws):
        for rank, (idx, _) in enumerate(lst, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _union_candidates(rank_lists: list[list[tuple[int, float]]]) -> list[int]:
    """保持各路优先顺序的去重并集，专供重排器读取完整候选池。"""
    seen: dict[int, None] = {}
    for lst in rank_lists:
        for idx, _ in lst:
            seen.setdefault(idx, None)
    return list(seen)


# ---------------- 重排器 ----------------
RERANK_PROMPT = """你是检索结果的相关性评审。给定用户问题和若干候选政策条款，为每个条款打 0～10 分：
10 = 直接回答了问题；5 = 相关但不能直接回答；0 = 无关。只依据条款内容判断。"""


class LLMReranker:
    """使用对话模型结构化输出批量对候选条款相关性打分。"""

    def __init__(self, llm):
        from pydantic import BaseModel, Field

        class Scores(BaseModel):
            scores: list[float] = Field(description="按候选顺序给出的分数，个数与候选相同")

        self._chain = llm.with_structured_output(Scores, method="function_calling")

    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        from langchain_core.messages import HumanMessage, SystemMessage

        body = "\n\n".join(f"候选 {i}：【{c.title}】{c.text}" for i, c in enumerate(chunks, 1))
        out = self._chain.invoke([SystemMessage(RERANK_PROMPT), HumanMessage(f"问题：{query}\n\n{body}")])
        if out is None:
            raise ValueError("重排模型未返回分数")
        scores = list(out.scores)[: len(chunks)]
        return scores + [0.0] * (len(chunks) - len(scores))


class DashScopeReranker:
    """专用 Cross-Encoder 重排模型（DashScope text-rerank），延迟约 100～200ms。"""

    URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(self, api_key: str, model: str, timeout: float = 10.0):
        import httpx

        self._client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self._model = model

    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        body = {"model": self._model, "input": {"query": query, "documents": [f"{c.title}：{c.text}" for c in chunks]},
                "parameters": {"top_n": len(chunks), "return_documents": False}}
        resp = self._client.post(self.URL, json=body)
        resp.raise_for_status()
        scores = [0.0] * len(chunks)
        for r in resp.json()["output"]["results"]:
            scores[r["index"]] = float(r["relevance_score"])
        return scores


# ---------------- 混合检索器 ----------------
class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Optional[Embedder] = None, reranker=None,
                 cache_dir: Optional[str] = None, model: str = "", recall_k: int = 10,
                 min_rerank_score: float = 0.0):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.vector: Optional[VectorIndex] = None
        if embedder is not None:
            try:
                self.vector = VectorIndex(chunks, embedder, cache_dir, model)
            except Exception:
                log.exception("向量索引构建失败，降级使用 BM25")
        self.reranker = reranker
        self.recall_k = recall_k
        self.min_rerank_score = min_rerank_score
        # 无重排器时的排序策略：vector（向量优先，加权 RRF 辅排）/ weighted_rrf / rrf（等权融合，评测对照用）
        self.order_without_rerank = "vector"

    @property
    def mode(self) -> str:
        return ("hybrid" if self.vector else "bm25") + ("+rerank" if self.reranker else "")

    def search(self, query: str, k: int = 3, *, use_vector: bool = True, use_bm25: bool = True,
               use_rerank: bool = True, category: Optional[str] = None) -> list[Hit]:
        lists, names = [], []
        if use_vector and self.vector is not None:
            try:
                lists.append(self.vector.search(query, self.recall_k))
                names.append("vector")
            except Exception:
                log.exception("向量检索失败，本次降级使用 BM25")
        if use_bm25:
            lists.append(self.bm25.search(query, self.recall_k))
            names.append("bm25")
        if not lists:
            return []

        ranks = [{i: r for r, (i, _) in enumerate(lst, 1)} for lst in lists]

        # 启用重排时，直接取两路召回的去重并集送入 Cross-Encoder，避免无意义的中间 RRF 计算
        if use_rerank and self.reranker is not None:
            cand_ids = _union_candidates(lists)
            if category:
                cand_ids = [i for i in cand_ids if self.chunks[i].category == category]
            if len(cand_ids) > 1:
                try:
                    cand_chunks = [self.chunks[i] for i in cand_ids]
                    scores = self.reranker.rerank(query, cand_chunks)
                    hits = [
                        Hit(self.chunks[i], float(s), {**{n: rk.get(i) for n, rk in zip(names, ranks)}, "rerank": s})
                        for i, s in zip(cand_ids, scores)
                        if float(s) >= self.min_rerank_score
                    ]
                    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
                except Exception:
                    log.exception("重排失败，退回无重排融合排序")

        # 无重排（或重排失败降级）路径
        weights = [0.85 if n == "vector" else 0.15 for n in names] if self.order_without_rerank != "rrf" else None
        fused = rrf(lists, weights=weights)
        if category:
            fused = [(i, s) for i, s in fused if self.chunks[i].category == category]
        hits = [Hit(self.chunks[i], s, {n: rk.get(i) for n, rk in zip(names, ranks)}) for i, s in fused]
        if self.order_without_rerank == "vector" and "vector" in names:
            hits.sort(key=lambda h: (h.detail.get("vector") or 10**6, -h.score))
        return hits[:k]


def make_reranker(settings, llm=None):
    """RAG_RERANK：0 不重排；llm 用对话模型批量打分；model 用 DashScope 专用重排模型。"""
    if settings.rag_rerank == "model" and settings.embedding_api_key:
        return DashScopeReranker(settings.embedding_api_key, settings.rerank_model)
    if settings.rag_rerank == "llm" and llm is not None:
        return LLMReranker(llm)
    return None


def build_retriever(settings, llm=None) -> HybridRetriever:
    """按配置组装：配置向量模型密钥时启用混合检索，否则仅用 BM25；重排方式由 RAG_RERANK 决定。"""
    embedder = None
    if settings.embedding_api_key:
        from langchain_openai import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.embedding_api_key, base_url=settings.embedding_base_url,
            check_embedding_ctx_length=False,
            chunk_size=10,
        )
    reranker = make_reranker(settings, llm)
    t0 = time.perf_counter()
    r = HybridRetriever(load_chunks(settings.kb_dir), embedder, reranker,
                        cache_dir=settings.kb_index_dir, model=settings.embedding_model,
                        min_rerank_score=settings.rag_min_score)
    log.info("知识库检索就绪：%s，%d 条，用时 %.1fs", r.mode, len(r.chunks), time.perf_counter() - t0)
    return r
