"""Supervisor 多 Agent 对照基线（用于与主线单 Agent 进行同工程水平消融对比）。"""
from __future__ import annotations

from .router import ACTIONS, SUPERVISOR_PROMPT, WORKERS, LLMRouter, RouteDecision, Router, build_supervisor_prompt
from .supervisor import RESPOND_PROMPT, build_multi_runner, direct_reply, make_respond, make_supervisor
from .workers import KB_PROMPT, ORDER_PROMPT, REFUND_PROMPT, WorkerFn, build_workers

__all__ = [
    "ACTIONS",
    "WORKERS",
    "SUPERVISOR_PROMPT",
    "RESPOND_PROMPT",
    "ORDER_PROMPT",
    "REFUND_PROMPT",
    "KB_PROMPT",
    "RouteDecision",
    "Router",
    "LLMRouter",
    "WorkerFn",
    "build_supervisor_prompt",
    "build_workers",
    "make_supervisor",
    "make_respond",
    "direct_reply",
    "build_multi_runner",
]
