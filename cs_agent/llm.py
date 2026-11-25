"""模型工厂：所有 LLM 都从这里创建，方便统一换模型、换网关。

供应商按模型名推断：qwen 系走 DashScope（用另一个 Key，关思考的参数也不同），其余走主供应商。
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import settings


def get_llm(model: str, temperature: float = 0.2) -> ChatOpenAI:
    if model.startswith("qwen"):  # 通义：另一个 Key / 网关 / 关思考参数
        api_key = settings.embedding_api_key
        base_url = settings.embedding_base_url
        extra = {"enable_thinking": False}
        if not api_key:
            raise RuntimeError("模型以 qwen 开头但未设置 DASHSCOPE_API_KEY")
    else:
        if not settings.api_key:
            raise RuntimeError("未设置 LLM_API_KEY：请复制 .env.example 为 .env 并填入密钥")
        api_key, base_url = settings.api_key, settings.base_url
        extra = settings.extra_body or None
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        extra_body=extra,
    )
