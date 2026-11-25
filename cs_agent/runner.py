"""会话单轮执行编排器（支持同步执行与原生异步流式执行）。

职责：
1. 每轮开始前执行前置多模态图片核验（未带图时零开销跳过）并初始化当轮临时状态；
2. 驱动核心单 Agent（或多 Agent 对照基线）完成工具调用与回复生成；
3. 异常时执行安全降级链路：未涉及写操作的纯咨询走本地零依赖 FAQ 匹配，其余情况走规则摘要转人工；
4. 持久化三层会话记忆与 PII 脱敏对话存档。
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Optional
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import set_config_context, var_child_runnable_config

from . import handoff
from .config import Settings, settings as default_settings
from .kb import PolicyKB
from .memory import compact_session_async, make_memory_node, persist_turn_state
from .refunds import RefundError
from .state import TurnStep, last_user_text, restore_session
from .store import ShopStore
from .vision import get_vlm, make_async_perceive_node, make_perceive_node

TurnEvent = Callable[[str, dict], None]

FAQ_THRESHOLD = 0.25
ACTION_WORDS = (
    "帮我", "给我", "退了", "退掉", "我要退", "申请退", "取消", "改成", "改一下", "查一下", "查下", "到哪", "物流", "签收",
)
FAQ_SAFE_FAILURES = ("supervisor", "kb_agent", "single_agent", "cs_agent")


def start_turn(state: dict) -> dict:
    """新的一轮用户输入：重置本轮临时执行状态，保留跨轮关键事实、摘要与历史图片核验记录。"""
    return {
        "hops": 0,
        "turn_results": [],
        "route_log": [],
        "final_action": None,
        "refund_proposed": False,
        "degraded": False,
        "handoff_ticket": None,
        "plan": None,
        "unsupported": [],
        "facts": state.get("facts") or {},
    }


def make_fallback(kb: PolicyKB, threshold: float = FAQ_THRESHOLD) -> Callable[[dict], TurnStep]:
    """故障降级节点：未涉及业务办理且匹配本地高频政策条款时直接答复，否则安全转接人工。"""
    def fallback(state: dict) -> TurnStep:
        last = (state.get("route_log") or [{}])[-1]
        text = last_user_text(state)
        needs_action = bool(handoff.ORDER_ID.search(text)) or any(w in text for w in ACTION_WORDS)
        hit = kb.best_match(text)
        if (last.get("failed") in FAQ_SAFE_FAILURES and not needs_action and not state.get("turn_results")
                and hit and hit[0] >= threshold):
            sec = hit[1]
            reply = (f"抱歉，智能客服暂时繁忙。以下是可能相关的政策说明：\n【{sec.title}】{sec.text}\n"
                     f"（来源：{sec.source} · {sec.title}）\n如果没有解决您的问题，请回复“转人工”。")
            return TurnStep(goto="memory", update={"final_action": "faq", "degraded": True,
                                                  "messages": [AIMessage(content=reply)]})
        return TurnStep(goto="handoff_human", update={"final_action": "handoff", "degraded": True})

    return fallback


def make_handoff(store: ShopStore, llm: Optional[BaseChatModel] = None) -> Callable[[dict, dict], dict]:
    def handoff_human(state: dict, config: dict) -> dict:
        reason = (state.get("route_log") or [{}])[-1].get("reason", "")
        thread_id = (config.get("configurable") or {}).get("thread_id")
        ticket = handoff.create_ticket(store, llm, state, reason, use_llm=not state.get("degraded"), thread_id=thread_id)
        return {"handoff_ticket": ticket, "final_action": "handoff",
                "messages": [AIMessage(content=handoff.handoff_reply(ticket))]}

    return handoff_human


def make_async_handoff(store: ShopStore, llm: Optional[BaseChatModel] = None):
    async def handoff_human(state: dict, config: dict) -> dict:
        reason = (state.get("route_log") or [{}])[-1].get("reason", "")
        thread_id = (config.get("configurable") or {}).get("thread_id")
        ticket = await handoff.acreate_ticket(
            store, llm, state, reason, use_llm=not state.get("degraded"), thread_id=thread_id,
        )
        return {"handoff_ticket": ticket, "final_action": "handoff",
                "messages": [AIMessage(content=handoff.handoff_reply(ticket))]}

    return handoff_human


def _accepts_emit(fn: Callable) -> bool:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(params) >= 2 or any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)


class TurnRunner:
    """统一的会话轮次执行器，兼容主线异步单 Agent 与多步对照基线。"""

    def __init__(
        self,
        store: ShopStore,
        dispatch: Callable[..., TurnStep],
        *,
        async_dispatch: Optional[Callable[[dict, Optional[TurnEvent]], Awaitable[TurnStep]]] = None,
        llm: Optional[BaseChatModel] = None,
        settings: Settings = default_settings,
        kb: Optional[PolicyKB] = None,
        vlm=None,
        workers: Optional[dict[str, Callable[[dict], TurnStep]]] = None,
        respond: Optional[Callable] = None,
        direct_reply_fn: Optional[Callable[[dict], dict]] = None,
    ):
        self.store = store
        self.dispatch = dispatch
        self._dispatch_takes_emit = _accepts_emit(dispatch)
        self.async_dispatch = async_dispatch
        self.workers = workers or {}
        self.llm = llm
        self.settings = settings
        kb_inst = kb or PolicyKB(settings.kb_dir)
        resolved_vlm = vlm if vlm is not None else (get_vlm(settings) if llm is not None else None)
        self.perceive = make_perceive_node(resolved_vlm)
        self.aperceive = make_async_perceive_node(resolved_vlm)
        self.respond = respond
        self.direct_reply_fn = direct_reply_fn
        self.fallback = make_fallback(kb_inst)
        self.handoff = make_handoff(store, llm)
        self.ahandoff = make_async_handoff(store, llm)
        self.remember = make_memory_node(store, llm, enabled=settings.memory_summary, mask=settings.pii_mask)

    def snapshot(self, thread_id: str) -> dict:
        return restore_session(self.store.load_session(thread_id))

    @staticmethod
    def _apply(state: dict, update: dict) -> None:
        if "messages" in update:
            added = [m if m.id else m.model_copy(update={"id": uuid4().hex}) for m in update["messages"]]
            update = {**update, "messages": [*state.get("messages", []), *added]}
        state.update(update)

    def _call_dispatch(self, state: dict, emit: TurnEvent | None) -> TurnStep:
        return self.dispatch(state, emit) if self._dispatch_takes_emit else self.dispatch(state)

    def run_turn(self, thread_id: str, user_id: str, message: HumanMessage, *, images=None,
                 callbacks=None, emit: TurnEvent | None = None) -> dict:
        state = self.snapshot(thread_id)
        if state.get("user_id", user_id) != user_id:
            raise RefundError("会话不存在", 404)
        message = message if message.id else message.model_copy(update={"id": uuid4().hex})
        state.update(user_id=user_id, messages=[*state.get("messages", []), message], images=images or [])
        config = {"configurable": {"thread_id": thread_id}, "callbacks": callbacks,
                  "metadata": {"turn_id": message.id}}
        with set_config_context(config) as context:
            return context.run(self._start_sync, state, config, emit)

    def _start_sync(self, state: dict, config: dict, emit: TurnEvent | None) -> dict:
        if emit and state["images"]:
            emit("status", {"stage": "perceive"})
        state.update(self.perceive(state))
        state.update(start_turn(state))
        return self._run_sync(state, self._call_dispatch(state, emit), config, emit)

    def _run_sync(self, state: dict, step: TurnStep, config: dict, emit: TurnEvent | None) -> dict:
        while True:
            self._apply(state, step.update)
            stage = step.goto
            if stage == "supervisor":
                step = self._call_dispatch(state, emit)
            elif stage in self.workers:
                if emit:
                    emit("status", {"stage": stage})
                step = self.workers[stage](state)
            elif stage == "fallback":
                if emit:
                    emit("status", {"stage": stage})
                step = self.fallback(state)
            else:
                break

        if stage == "respond" and self.respond is not None:
            self._apply(state, self.respond(state, (lambda text: emit("token", {"text": text})) if emit else None))
        elif stage == "direct_reply" and self.direct_reply_fn is not None:
            self._apply(state, self.direct_reply_fn(state))
        elif stage == "handoff_human":
            if emit:
                emit("status", {"stage": stage})
            self._apply(state, self.handoff(state, config))
        elif stage != "memory":
            raise ValueError(f"未知的对话步骤：{stage}")

        if emit:
            reply = next((str(m.content) for m in reversed(state["messages"]) if isinstance(m, AIMessage)), "")
            emit("message", {"text": reply})
        state.update(self.remember(state, config))
        return state

    async def arun_turn(self, thread_id: str, user_id: str, message: HumanMessage, *, images=None,
                        callbacks=None, emit: TurnEvent | None = None, defer_summary: bool = False) -> dict:
        """原生异步流式轮次执行；若当前实例未配置 `async_dispatch`（如测试桩），自动委派至工作线程执行。"""
        if self.async_dispatch is None:
            return await asyncio.to_thread(
                self.run_turn, thread_id, user_id, message, images=images, callbacks=callbacks, emit=emit,
            )

        state = self.snapshot(thread_id)
        if state.get("user_id", user_id) != user_id:
            raise RefundError("会话不存在", 404)
        message = message if message.id else message.model_copy(update={"id": uuid4().hex})
        state.update(user_id=user_id, messages=[*state.get("messages", []), message], images=images or [])
        config = {"configurable": {"thread_id": thread_id}, "callbacks": callbacks,
                  "metadata": {"turn_id": message.id}}

        token = var_child_runnable_config.set(config)
        try:
            if emit and state["images"]:
                emit("status", {"stage": "perceive"})
            state.update(await self.aperceive(state))
            state.update(start_turn(state))
            step = await self.async_dispatch(state, emit)
            while True:
                self._apply(state, step.update)
                stage = step.goto
                if stage == "fallback":
                    if emit:
                        emit("status", {"stage": stage})
                    step = self.fallback(state)
                else:
                    break

            if stage == "handoff_human":
                if emit:
                    emit("status", {"stage": stage})
                self._apply(state, await self.ahandoff(state, config))
            elif stage != "memory":
                raise ValueError(f"未知的异步对话步骤：{stage}")

            if emit:
                reply = next((str(m.content) for m in reversed(state["messages"]) if isinstance(m, AIMessage)), "")
                emit("message", {"text": reply})

            if defer_summary:
                persist_turn_state(self.store, state, thread_id, mask=self.settings.pii_mask)
            else:
                persist_turn_state(self.store, state, thread_id, mask=self.settings.pii_mask)
                if not state.get("degraded"):
                    await compact_session_async(
                        self.store, self.llm, thread_id, enabled=self.settings.memory_summary,
                    )
            return state
        finally:
            var_child_runnable_config.reset(token)

    async def acompact(self, thread_id: str) -> bool:
        """在推送 `done` 事件后执行异步摘要压缩。"""
        return await compact_session_async(
            self.store, self.llm, thread_id, enabled=self.settings.memory_summary,
        )
