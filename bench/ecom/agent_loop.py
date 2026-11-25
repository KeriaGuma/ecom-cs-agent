"""ECom-Bench 评测专用的纯 Python 异步 ReAct 工具循环（替代原 Node Pi 桥接层）。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage


async def run_agent(llm, system_prompt: str, messages: list[BaseMessage], tools: list, *, max_steps: int = 15) -> tuple[str, list[BaseMessage]]:
    """异步运行工具调用循环，返回 `(最终文本回复, 本轮新增消息列表)` 供上游评分器检查工具轨迹。"""
    tool_map = {t.name: t for t in tools}
    bound = llm.bind_tools(tools) if tools else llm
    convo: list[BaseMessage] = [SystemMessage(content=system_prompt), *messages]
    produced: list[BaseMessage] = []

    for _ in range(max_steps):
        ai: AIMessage = await bound.ainvoke(convo)
        convo.append(ai)
        produced.append(ai)
        if not ai.tool_calls:
            return str(ai.content or ""), produced
        for tc in ai.tool_calls:
            tool_obj = tool_map.get(tc["name"])
            if tool_obj is None:
                out = f"错误：不存在工具 {tc['name']}"
            else:
                try:
                    res = await tool_obj.ainvoke(tc["args"])
                    out = str(res)
                except Exception as exc:
                    out = f"工具执行出错：{exc}"
            tm = ToolMessage(content=out, tool_call_id=tc["id"])
            convo.append(tm)
            produced.append(tm)

    return "", produced
