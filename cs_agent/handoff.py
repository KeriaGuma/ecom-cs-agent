"""转人工交接：生成结构化交接摘要、代码计算优先级、写入坐席队列。

设计要点：
- 摘要提取意图、涉及订单、用户情绪、系统已做处理、待办事项与下一步建议，坐席接手无需让用户重复描述；
- 工单优先级由确定性代码计算（投诉/12315/曝光/欺诈等情绪激烈词强制兜底为最高优先级）；
- 因模型故障触发转人工时自动切换规则摘要，不再调用模型白等超时。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .pii import mask_obj
from .state import last_user_text, render_facts

log = logging.getLogger(__name__)

HISTORY_FOR_HANDOFF = 20

ANGRY_WORDS = ("投诉", "12315", "消协", "曝光", "骗子", "举报", "起诉", "律师", "破服务", "垃圾")
UNHAPPY_WORDS = ("怎么还", "太慢", "到底", "几次", "又是", "不满意", "失望", "催")
ORDER_ID = re.compile(r"O\d{4}", re.IGNORECASE)

PRIORITY_NAME = {1: "高", 2: "中", 3: "低"}


class HandoffSummary(BaseModel):
    intent: str = Field(description="用户想办的事，一句话")
    order_ids: list[str] = Field(default_factory=list, description="涉及的订单号")
    emotion: Literal["平静", "不满", "愤怒"] = Field(description="用户当前情绪")
    done: list[str] = Field(default_factory=list, description="系统已经完成或尝试过的处理")
    pending: str = Field(description="需要人工客服处理的问题")
    suggestion: str = Field(description="给人工客服的建议下一步，一句话")


SUMMARY_PROMPT = """你是客服主管，要把一段机器人客服对话交接给人工客服。根据对话、关键事实和已完成的处理，提取交接摘要。
只依据给出的信息，不要编造订单号、金额或处理结果；不确定的字段留空或写“未知”。"""


def summarize(llm: Optional[BaseChatModel], state: dict, reason: str, use_llm: bool = True) -> dict:
    """同步生成交接摘要；模型不可用或结构化输出为空时自动退回规则摘要。"""
    if llm is not None and use_llm:
        try:
            chain = llm.with_structured_output(HandoffSummary, method="function_calling")
            out: HandoffSummary = chain.invoke([SystemMessage(SUMMARY_PROMPT), HumanMessage(_context(state, reason))])
            if out is None:
                log.warning("交接摘要的结构化输出为空，改用规则摘要")
            else:
                return {**out.model_dump(), "summary_by": "llm"}
        except Exception:
            log.exception("交接摘要生成失败，改用规则摘要")
    return {**_rule_summary(state, reason), "summary_by": "rule"}


async def asummarize_handoff(llm: Optional[BaseChatModel], state: dict, reason: str, use_llm: bool = True) -> dict:
    """异步生成交接摘要。"""
    if llm is not None and use_llm:
        try:
            chain = llm.with_structured_output(HandoffSummary, method="function_calling")
            out: HandoffSummary = await chain.ainvoke([SystemMessage(SUMMARY_PROMPT), HumanMessage(_context(state, reason))])
            if out is None:
                log.warning("交接摘要的结构化输出为空，改用规则摘要")
            else:
                return {**out.model_dump(), "summary_by": "llm"}
        except Exception:
            log.exception("交接摘要生成失败，改用规则摘要")
    return {**_rule_summary(state, reason), "summary_by": "rule"}


def priority_of(summary: dict, state: dict, reason: str) -> int:
    """1 高 / 2 中 / 3 低。愤怒或投诉类强制高优；不满、系统故障、超范围业务为中优。"""
    text = _user_texts(state) + reason
    if summary.get("emotion") == "愤怒" or any(w in text for w in ANGRY_WORDS):
        return 1
    if summary.get("emotion") == "不满" or any(w in text for w in UNHAPPY_WORDS) or "失败" in reason or "超过" in reason:
        return 2
    return 3


def _persist_ticket(store, state: dict, summary: dict, reason: str, mask: bool, thread_id: Optional[str]) -> dict:
    priority = priority_of(summary, state, reason)
    now = datetime.now().isoformat(timespec="seconds")
    payload = {**summary, "reason": reason, "facts": state.get("facts") or {},
               "handled_this_turn": state.get("turn_results") or []}
    if mask:
        payload, reason = mask_obj(payload), mask_obj(reason)
    ticket_id = store.create_ticket(state.get("user_id", ""), priority, payload, now, thread_id=thread_id)
    return {"ticket_id": ticket_id, "priority": priority, "queue_position": store.queue_position(ticket_id),
            "user_id": state.get("user_id"), "reason": reason, "summary": payload, "created_at": now}


def create_ticket(store, llm: Optional[BaseChatModel], state: dict, reason: str, use_llm: bool = True,
                  mask: Optional[bool] = None, *, thread_id: Optional[str] = None) -> dict:
    if mask is None:
        from .config import settings

        mask = settings.pii_mask
    summary = summarize(llm, state, reason, use_llm=use_llm)
    return _persist_ticket(store, state, summary, reason, mask, thread_id)


async def acreate_ticket(store, llm: Optional[BaseChatModel], state: dict, reason: str, use_llm: bool = True,
                         mask: Optional[bool] = None, *, thread_id: Optional[str] = None) -> dict:
    if mask is None:
        from .config import settings

        mask = settings.pii_mask
    summary = await asummarize_handoff(llm, state, reason, use_llm=use_llm)
    return _persist_ticket(store, state, summary, reason, mask, thread_id)


def handoff_reply(ticket: dict) -> str:
    ahead = ticket["queue_position"]
    queue = f"前面还有 {ahead} 位用户在等待" if ahead else "您是下一位"
    return (f"已为您转接人工客服（工单号 #{ticket['ticket_id']}，{queue}）。"
            "我已把问题和处理进度同步给客服同事，接通后无需重复描述，请稍候。")


def _history(state: dict) -> list:
    msgs = [m for m in state.get("messages") or [] if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    return msgs[-HISTORY_FOR_HANDOFF:]


def _user_texts(state: dict) -> str:
    return " ".join(str(m.content) for m in _history(state) if isinstance(m, HumanMessage))


def _context(state: dict, reason: str) -> str:
    lines = [f"【转人工原因】{reason or '未知'}"]
    fact_line = render_facts(state.get("facts"))
    if fact_line:
        lines.append(fact_line)
    if state.get("summary"):
        lines.append("【较早对话摘要】" + state["summary"])
    lines.append("【对话】")
    for m in _history(state):
        lines.append(("用户：" if isinstance(m, HumanMessage) else "客服：") + str(m.content))
    results = state.get("turn_results") or []
    lines.append("【本轮已完成的处理】" + ("" if results else "无"))
    lines += [f"- {r['agent']}：{r['content']}" for r in results]
    return "\n".join(lines)


def _rule_summary(state: dict, reason: str) -> dict:
    texts = _user_texts(state)
    ids = {m.upper() for m in ORDER_ID.findall(texts)}
    facts = state.get("facts") or {}
    if facts.get("order_id"):
        ids.add(facts["order_id"])
    for oid in facts.get("recent_orders") or ():
        ids.add(oid)
    emotion = "愤怒" if any(w in texts for w in ANGRY_WORDS) else "不满" if any(w in texts for w in UNHAPPY_WORDS) else "平静"
    failed = "失败" in reason
    return {
        "intent": "未识别（系统故障时转入，请查看用户原话）" if failed else (reason or "用户要求人工服务"),
        "order_ids": sorted(ids),
        "emotion": emotion,
        "done": [f"{r['agent']}：{r['content']}" for r in state.get("turn_results") or []],
        "pending": last_user_text(state),
        "suggestion": "查看对话记录，确认用户诉求后处理",
    }
