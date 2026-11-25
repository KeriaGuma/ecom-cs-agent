"""每轮对话状态、跨轮关键事实提取与上下文视图渲染。

设计要点：
- `messages` 只保存用户消息和最终客服回复，工具调用中间轨迹留在当轮循环内部，避免污染长对话上下文；
- `facts` 由确定性代码从真实工具返回中提取（最近焦点订单 `order_id` + 涉及订单列表 `recent_orders`），绝不由 LLM 自由生成，防止误记越权或虚构订单号；
- `summary` 保存较早历史对话的增量压缩摘要，配合最近原文滑动窗口控制上下文体积。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing_extensions import TypedDict


class WorkerResult(TypedDict):
    agent: str
    content: str


@dataclass(frozen=True)
class TurnStep:
    goto: str
    update: dict


class CSState(TypedDict, total=False):
    messages: list[BaseMessage]
    user_id: str
    facts: dict[str, Any]              # 跨轮保留的可信关键事实（焦点订单与近期涉及订单列表）
    hops: int                          # 多 Agent 对照基线：本轮已分派次数
    turn_results: list[WorkerResult]   # 本轮处理结论
    route_log: list[dict]              # 路由与降级轨迹日志
    final_action: Optional[str]        # finish / clarify / handoff / faq / human
    refund_proposed: bool              # 本轮是否生成了待用户核对的退款确认单
    handoff_ticket: Optional[dict]     # 转人工交接工单
    degraded: bool                     # 本轮是否因模型故障进入降级链路
    plan: Optional[list[str]]          # 多 Agent 对照基线：剩余计划步骤
    unsupported: list[str]             # 系统暂不支持的附加要求清单
    summary: str                       # 较早对话的增量摘要
    images: list[str]                  # 本轮上传的待核验图片（核验后清空）
    photos: list[dict]                 # 跨轮保留的图片核验结果


RECENT_MESSAGES = 6        # 摘要压缩后保留的最近原文条数
CONTEXT_MAX_MESSAGES = 12  # 给模型展示的原文上限（最近窗口 + 尚未达到压缩批次的消息）
MAX_RECENT_ORDERS = 5


def extract_facts(existing: dict[str, Any] | None, tool_calls: list[dict[str, Any]], not_found_text: str) -> dict[str, Any]:
    """从工具调用记录中提取可信订单号事实，忽略不存在或不属于当前用户的订单。"""
    facts = dict(existing or {})
    recent = list(facts.get("recent_orders") or [])
    if facts.get("order_id") and facts["order_id"] not in recent:
        recent.insert(0, facts["order_id"])

    for call in tool_calls:
        args = call.get("args")
        oid = args.get("order_id") if isinstance(args, dict) else None
        if oid and call.get("output") != not_found_text:
            clean_id = str(oid).strip().upper()
            if clean_id:
                facts["order_id"] = clean_id
                recent = [clean_id, *[x for x in recent if x != clean_id]][:MAX_RECENT_ORDERS]

    if recent:
        facts["recent_orders"] = recent
    return facts


def session_state(state: dict) -> dict:
    fields = ("user_id", "facts", "summary", "photos")
    saved = {field: state[field] for field in fields if field in state}
    saved["messages"] = [
        {"id": m.id, "role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))
    ]
    return saved


def restore_session(saved: dict) -> dict:
    if not saved:
        return {}
    messages = [
        (HumanMessage if row["role"] == "user" else AIMessage)(content=row["content"], id=row["id"])
        for row in saved.get("messages", [])
    ]
    return {**saved, "messages": messages}


def render_facts(facts: dict[str, Any] | None) -> str:
    if not facts:
        return ""
    items = []
    if facts.get("order_id"):
        items.append(f"order_id={facts['order_id']}")
    recent = [x for x in (facts.get("recent_orders") or []) if x != facts.get("order_id")]
    if recent:
        items.append(f"历史提及订单={','.join(recent)}")
    for k, v in facts.items():
        if k not in ("order_id", "recent_orders"):
            items.append(f"{k}={v}")
    return "【关键事实】" + "；".join(items) if items else ""


def render_state(state: dict, *, include_turn_results: bool = True) -> str:
    """将会话状态渲染为统一的模型输入文本视图。"""
    lines: list[str] = []

    fact_line = render_facts(state.get("facts"))
    if fact_line:
        lines.append(fact_line)
    if state.get("summary"):
        lines.append("【较早对话摘要】" + state["summary"])

    if state.get("photos"):
        from .vision import render_photos

        lines.append(render_photos(state["photos"]))

    lines.append("【最近对话】")
    for m in _recent(state.get("messages") or []):
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        lines.append(f"{role}：{m.content}")

    if include_turn_results:
        results = state.get("turn_results") or []
        lines.append("【本轮已完成的处理】" + ("" if results else "无"))
        for r in results:
            lines.append(f"- {r['agent']}：{r['content']}")

    return "\n".join(lines)


def _recent(messages: list[BaseMessage]) -> list[BaseMessage]:
    keep = [m for m in messages if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    return keep[-CONTEXT_MAX_MESSAGES:]


def last_user_text(state: dict) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""
