"""三层会话记忆管理：代码级关键事实 + 增量滚动摘要 + 最近原文滑动窗口。

分层职责：
- **关键事实（`state.facts`）**：由确定性代码从工具调用结果中提取（最近操作订单与历史涉及订单），可信度最高；
- **滚动摘要（`state.summary`）**：当窗口外旧消息累积达到批次阈值时，将“旧摘要 + 待压缩消息”合并为新摘要并裁剪上下文；
  在线 HTTP 服务中，摘要压缩在向用户推送 `done` 事件后异步执行，避免阻塞当轮尾延迟；
- **最近对话（`state.messages`）**：保留所有尚未被摘要的原文，确保多轮指代信息不丢失；
- **完整存档（`transcripts` 表）**：全量消息经 PII 脱敏后独立持久化，供坐席接管、审计与历史回看。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .pii import mask_text
from .state import RECENT_MESSAGES, restore_session, session_state
from .store import ShopStore

log = logging.getLogger(__name__)

SUMMARY_BATCH = 6  # 最近窗口之外累计达到该条数时触发一次增量摘要压缩

SUMMARY_PROMPT = """你负责维护一段客服对话的滚动摘要，供后续对话参考。把【旧摘要】和【新对话】合并成新的摘要。
要求：
1. 保留：用户的诉求和原因（如退款原因、质量问题的具体表现）、涉及的订单和商品、已经办结的事项及结果、
   尚未解决的问题、用户明确表达的偏好或不满。
2. 去掉寒暄和重复内容；只依据给出的内容，不要编造；不超过 200 字。
3. 直接输出摘要正文。"""


def _format_summary_input(old_summary: str, messages: list) -> str:
    lines = [("用户：" if isinstance(m, HumanMessage) else "客服：") + str(m.content) for m in messages]
    return f"【旧摘要】{old_summary or '无'}\n【新对话】\n" + "\n".join(lines)


def summarize(llm: BaseChatModel, old_summary: str, messages: list) -> str:
    body = _format_summary_input(old_summary, messages)
    return str(llm.invoke([SystemMessage(SUMMARY_PROMPT), HumanMessage(body)]).content).strip()


async def asummarize(llm: BaseChatModel, old_summary: str, messages: list) -> str:
    body = _format_summary_input(old_summary, messages)
    res = await llm.ainvoke([SystemMessage(SUMMARY_PROMPT), HumanMessage(body)])
    return str(res.content).strip()


def persist_turn_state(store: ShopStore, state: dict, thread_id: Optional[str], *, mask: bool = True) -> None:
    """快速同步落库当轮上下文与脱敏对话存档（不调模型，毫秒级完成）。"""
    if not thread_id:
        return
    clean = mask_text if mask else (lambda s: s)
    msgs = [m for m in state.get("messages") or [] if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    archive = [(m.id, "user" if isinstance(m, HumanMessage) else "assistant", clean(str(m.content)))
               for m in msgs if m.id]
    store.save_session(thread_id, session_state(state), archive)


async def compact_session_async(
    store: ShopStore,
    llm: Optional[BaseChatModel],
    thread_id: str,
    *,
    enabled: bool = True,
    recent: int = RECENT_MESSAGES,
    batch: int = SUMMARY_BATCH,
) -> bool:
    """在会话锁保护下异步执行滚动摘要压缩，按消息 ID 精准移除已压缩项，绝不覆盖新消息。"""
    if not enabled or llm is None or not thread_id:
        return False
    state = restore_session(store.load_session(thread_id))
    msgs = [m for m in state.get("messages") or [] if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    old = msgs[:-recent] if len(msgs) > recent else []
    if len(old) < batch:
        return False
    compacted_ids = {m.id for m in old if m.id}
    try:
        new_summary = await asummarize(llm, state.get("summary") or "", old)
    except Exception:
        log.exception("异步对话摘要失败 thread_id=%s", thread_id)
        return False

    # 重新读取最新持久化状态，仅移除本次已纳入摘要的消息 ID
    latest = restore_session(store.load_session(thread_id))
    if not latest:
        return False
    latest_msgs = [m for m in latest.get("messages") or [] if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    remaining = [m for m in latest_msgs if m.id not in compacted_ids]
    latest.update(summary=new_summary, messages=remaining)
    store.save_session(thread_id, session_state(latest), [])
    return True


def make_memory_node(store: ShopStore, llm: Optional[BaseChatModel], enabled: bool = True,
                     recent: int = RECENT_MESSAGES, batch: int = SUMMARY_BATCH, mask: Optional[bool] = None) -> Callable:
    """同步记忆保存与摘要节点（供 CLI 与离线评测使用）。"""
    if mask is None:
        from .config import settings

        mask = settings.pii_mask
    clean = mask_text if mask else (lambda s: s)

    def memory(state: dict, config: RunnableConfig, compact: bool = True) -> dict:
        msgs = [m for m in state.get("messages") or [] if isinstance(m, (HumanMessage, AIMessage)) and m.content]
        old = msgs[:-recent] if len(msgs) > recent else []
        update = {}
        if compact and enabled and llm is not None and not state.get("degraded") and len(old) >= batch:
            try:
                summary = summarize(llm, state.get("summary") or "", old)
                update = {"summary": summary, "messages": msgs[-recent:]}
            except Exception:
                log.exception("对话摘要失败")

        thread_id = (config.get("configurable") or {}).get("thread_id")
        if thread_id:
            archive = [(m.id, "user" if isinstance(m, HumanMessage) else "assistant", clean(str(m.content)))
                       for m in msgs if m.id]
            store.save_session(thread_id, session_state({**state, **update}), archive)
        return update

    return memory
