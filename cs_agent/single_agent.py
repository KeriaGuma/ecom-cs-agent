"""向后兼容导出模块：核心单 Agent 已提升为主线引擎 `cs_agent.agent`。"""
from __future__ import annotations

from .agent import (
    AGENT_MAX_STEPS as SINGLE_MAX_STEPS,
    CONTACT_RULE,
    PHOTO_RULE,
    REFUND_DECISION,
    SINGLE_PROMPT,
    SINGLE_PROMPT_V1,
    build_agent_prompt as build_single_prompt,
    build_agent_runner as build_single_runner,
    build_agent_tools as build_single_tools,
)

__all__ = [
    "SINGLE_MAX_STEPS",
    "CONTACT_RULE",
    "PHOTO_RULE",
    "REFUND_DECISION",
    "SINGLE_PROMPT",
    "SINGLE_PROMPT_V1",
    "build_single_prompt",
    "build_single_runner",
    "build_single_tools",
]
