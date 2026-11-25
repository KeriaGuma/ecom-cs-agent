"""全局运行时配置：统一从环境变量（`.env`）读取，不在代码中硬编码密钥与服务地址。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> str:
    p = Path(os.getenv(env, default))
    return str(p if p.is_absolute() else ROOT / p)


@dataclass(frozen=True)
class Settings:
    # 模型配置：统一走 OpenAI 兼容接口（DeepSeek / 通义千问 / LiteLLM 网关等）
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    agent_model: str = os.getenv("AGENT_MODEL") or os.getenv("WORKER_MODEL", "deepseek-flash")
    router_model: str = os.getenv("ROUTER_MODEL", "deepseek-flash")  # 供多 Agent 对照实验使用
    extra_body: dict = field(default_factory=lambda: json.loads(
        os.getenv("LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}') or "{}"
    ))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "30"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # 存储与检索目录
    db_path: str = _path("DB_PATH", "data/shop.db")
    kb_dir: str = _path("KB_DIR", "data/kb")
    kb_index_dir: str = _path("KB_INDEX_DIR", "data/kb_index")

    # 混合检索（BM25 + 向量召回 + Cross-Encoder 重排）
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
    rag_rerank: str = os.getenv("RAG_RERANK", "model")  # 0 不重排 / llm 对话模型重排 / model 专用重排模型
    rerank_model: str = os.getenv("RERANK_MODEL", "qwen3.7-text-rerank")
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.0"))

    # 多模态视觉核验
    vision: bool = os.getenv("VISION", "1") == "1"
    vision_api_key: str = os.getenv("VISION_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    vision_base_url: str = os.getenv("VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    vision_model: str = os.getenv("VISION_MODEL", "qwen3.7-flash")
    vision_extra_body: dict = field(default_factory=lambda: json.loads(
        os.getenv("VISION_EXTRA_BODY", '{"enable_thinking": false}') or "{}"))

    # 业务规则与安全护栏开关（支持消融实验对照）
    refund_auto_approve_limit: float = float(os.getenv("REFUND_AUTO_APPROVE_LIMIT", "500"))
    memory_summary: bool = os.getenv("MEMORY_SUMMARY", "1") == "1"
    claim_check: bool = os.getenv("CLAIM_CHECK", "1") == "1"
    refund_rule: int = int(os.getenv("REFUND_RULE", "2"))
    pii_mask: bool = os.getenv("PII_MASK", "1") == "1"
    address_guard: bool = os.getenv("ADDRESS_GUARD", "1") == "1"
    unsupported_rule: bool = os.getenv("UNSUPPORTED_RULE", "1") == "1"
    ops_log: bool = os.getenv("OPS_LOG", "1") == "1"
    claim_note_version: int = int(os.getenv("CLAIM_NOTE_V", "2"))

    # 订单工具部署形态：local / mcp (stdio) / mcp-http (Streamable HTTP + 双层鉴权)
    order_tools: str = os.getenv("ORDER_TOOLS", "local")
    mcp_url: str = os.getenv("MCP_URL", "http://127.0.0.1:8765/mcp")
    mcp_token: str = field(default=os.getenv("MCP_TOKEN", ""), repr=False)
    mcp_user_key: str = field(default=os.getenv("MCP_USER_KEY", ""), repr=False)
    demo_secret_key: str = field(default=os.getenv("DEMO_SECRET_KEY", ""), repr=False)

    # 多 Agent 对照与扩展实验参数
    max_hops: int = int(os.getenv("MAX_HOPS", "5"))
    plan_routing: bool = os.getenv("PLAN_ROUTING", "1") == "1"
    direct_reply: bool = os.getenv("DIRECT_REPLY", "1") == "1"
    extra_tools: int = int(os.getenv("EXTRA_TOOLS", "0"))
    domains: str = os.getenv("DOMAINS", "")
    domain_data: bool = os.getenv("DOMAIN_DATA", "0") == "1"

    @property
    def worker_model(self) -> str:
        return self.agent_model

    @property
    def domain_list(self) -> tuple:
        from .domains import parse_domains

        return parse_domains(self.domains)


settings = Settings()
