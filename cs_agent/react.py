"""可控 ReAct 工具执行引擎（支持同步调用与原生异步 Token 级流式输出）。

核心机制：
1. **步数上限与终止工具短路**：达到 `max_steps` 自动安全收敛；命中 `stop_tools`（如转人工）立即跳出循环，省去多余模型调用。
2. **流式输出与防谎报守卫（Claim Guard）协同**：
   - 工具调用轮次：静默组装工具参数并向前端发射细粒度处理阶段状态（`on_tool`），不向用户暴露中间推理碎片；
   - 最终文本轮次：若本轮已执行写操作工具（无“未办谎报”风险）或关闭检查，零缓冲逐 Token 实时流式推送；
     若本轮未调用任何写工具，先对短文本回复做轻量缓冲校验 `unbacked_claim`，若检测到无依据承诺则静默打回重跑一次，
     校验通过后再流式输出，兼顾首字响应速度与零虚假承诺。
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .claims import CLAIM_NOTE, WRITE_TOOLS, unbacked_claim

MAX_STEPS = 6
STEP_LIMIT_REPLY = "处理步骤过多，未能完成，请转人工处理。"

TOOL_STAGE_MAP = {
    "list_my_orders": "order_agent",
    "get_order": "order_agent",
    "get_logistics": "order_agent",
    "change_address": "order_agent",
    "check_refund": "refund_agent",
    "prepare_refund": "refund_agent",
    "submit_refund": "refund_agent",
    "search_policy": "kb_agent",
    "transfer_to_human": "handoff_human",
}


@dataclass
class ReactResult:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    claim_retries: int = 0


def run_react(
    llm: BaseChatModel,
    system_prompt: str,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    max_steps: int = MAX_STEPS,
    stop_tools: frozenset[str] = frozenset(),
    claim_check: bool = False,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    on_tool: Optional[Callable[[str], None]] = None,
) -> ReactResult:
    """同步 ReAct 循环（用于 CLI、离线评测与多 Agent 对照基线）。"""
    if max_steps <= 0:
        return ReactResult(STEP_LIMIT_REPLY)
    tool_map = {t.name: t for t in tools}
    model = llm.bind_tools(tools)
    convo: list[BaseMessage] = [SystemMessage(system_prompt), *messages]
    calls: list[dict[str, Any]] = []

    retries = 0
    for _ in range(max_steps):
        ai = model.invoke(convo)
        convo.append(ai)
        if not ai.tool_calls:
            text = str(ai.content)
            if claim_check and not retries and unbacked_claim(text, calls):
                retries = 1
                convo.append(HumanMessage(CLAIM_NOTE))
                continue
            if on_token and text:
                on_token(text)
            return ReactResult(text, calls, retries)
        for tc in ai.tool_calls:
            name = tc["name"]
            if on_tool:
                on_tool(TOOL_STAGE_MAP.get(name, "agent"))
            t = tool_map.get(name)
            if t is None:
                output = f"错误：不存在工具 {name}"
            else:
                try:
                    output = t.invoke(tc["args"])
                except Exception as e:
                    output = f"工具执行出错：{e}"
            calls.append({"tool": name, "args": tc["args"], "output": str(output)})
            convo.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
        if any(tc["name"] in stop_tools for tc in ai.tool_calls):
            return ReactResult(str(ai.content), calls, retries)

    return ReactResult(STEP_LIMIT_REPLY, calls, retries)


async def _invoke_tool_async(tool: BaseTool, args: dict[str, Any]) -> str:
    """在异步事件循环中安全执行同步业务工具，避免阻塞其他并发会话。"""
    return str(await asyncio.to_thread(tool.invoke, args))


async def arun_react_stream(
    llm: BaseChatModel,
    system_prompt: str,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    max_steps: int = MAX_STEPS,
    stop_tools: frozenset[str] = frozenset(),
    claim_check: bool = False,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    on_tool: Optional[Callable[[str], None]] = None,
    suppress_tokens: Optional[Callable[[], bool]] = None,
) -> ReactResult:
    """原生异步流式 ReAct 循环。"""
    if max_steps <= 0:
        return ReactResult(STEP_LIMIT_REPLY)
    tool_map = {t.name: t for t in tools}
    model = llm.bind_tools(tools)
    convo: list[BaseMessage] = [SystemMessage(system_prompt), *messages]
    calls: list[dict[str, Any]] = []

    retries = 0
    for _ in range(max_steps):
        wrote_already = any(c.get("tool") in WRITE_TOOLS for c in calls)
        can_stream_live = on_token is not None and (not claim_check or wrote_already or retries > 0)

        merged: Optional[AIMessageChunk] = None
        buffered_pieces: list[str] = []
        saw_tool_chunk = False

        async for chunk in model.astream(convo):
            merged = chunk if merged is None else merged + chunk
            if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
                saw_tool_chunk = True
            piece = chunk.text if hasattr(chunk, "text") else str(chunk.content or "")
            if piece and not saw_tool_chunk:
                if can_stream_live and not (suppress_tokens and suppress_tokens()):
                    on_token(piece)
                else:
                    buffered_pieces.append(piece)

        if merged is None:
            ai = AIMessage(content="")
        else:
            ai = AIMessage(
                content=merged.content,
                tool_calls=list(merged.tool_calls or []),
                usage_metadata=getattr(merged, "usage_metadata", None),
            )
        convo.append(ai)

        if not ai.tool_calls:
            text = str(ai.content)
            if claim_check and not retries and unbacked_claim(text, calls):
                retries = 1
                convo.append(HumanMessage(CLAIM_NOTE))
                continue
            if on_token and buffered_pieces and not (suppress_tokens and suppress_tokens()):
                for piece in buffered_pieces:
                    on_token(piece)
            return ReactResult(text, calls, retries)

        for tc in ai.tool_calls:
            name = tc["name"]
            if on_tool:
                on_tool(TOOL_STAGE_MAP.get(name, "agent"))
            t = tool_map.get(name)
            if t is None:
                output = f"错误：不存在工具 {name}"
            else:
                try:
                    output = await _invoke_tool_async(t, tc["args"])
                except Exception as e:
                    output = f"工具执行出错：{e}"
            calls.append({"tool": name, "args": tc["args"], "output": output})
            convo.append(ToolMessage(content=output, tool_call_id=tc["id"]))
        if any(tc["name"] in stop_tools for tc in ai.tool_calls):
            return ReactResult(str(ai.content), calls, retries)

    return ReactResult(STEP_LIMIT_REPLY, calls, retries)
