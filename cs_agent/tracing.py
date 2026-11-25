"""运行轨迹记录：模型调用次数、Token 用量、工具调用。

对话执行器用 LangChain 的运行上下文传递回调，Python 和 Pi 两种工具循环按同一口径记录。
"""
from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .claims import CLAIM_NOTES


class RunTrace(BaseCallbackHandler):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tools: list[dict[str, Any]] = []
        self._open_tools: dict[UUID, dict[str, Any]] = {}
        self.claim_notes: list[str] = []  # “谎报完成”检查拦下的回答草稿（检查触发一次记一条）

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self.llm_calls += 1
            for msgs in messages or []:  # 最后一条是检查提醒 → 这次调用是检查触发后的重新决定，前一条就是被拦下的草稿
                if len(msgs) >= 2 and getattr(msgs[-1], "content", None) in CLAIM_NOTES:
                    self.claim_notes.append(str(msgs[-2].content))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        inp = out = 0
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                inp += usage.get("input_tokens", 0)
                out += usage.get("output_tokens", 0)
        if not (inp or out):  # 部分接口只在 llm_output 里给用量
            usage = (response.llm_output or {}).get("token_usage") or {}
            inp, out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        with self._lock:
            self.input_tokens += inp
            self.output_tokens += out

    def on_tool_start(self, serialized, input_str: str, *, run_id: UUID, inputs=None, **kwargs: Any) -> None:
        rec = {"tool": (serialized or {}).get("name") or kwargs.get("name", "?"), "args": inputs or input_str, "output": None}
        with self._lock:
            self._open_tools[run_id] = rec
            self.tools.append(rec)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            rec = self._open_tools.pop(run_id, None)
        if rec is not None:
            rec["output"] = str(getattr(output, "content", output))

    @property
    def tool_names(self) -> list[str]:
        return [t["tool"] for t in self.tools]
