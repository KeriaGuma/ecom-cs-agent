"""Supervisor 编排与汇总回复节点（多 Agent 架构对照基线）。"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cs_agent.config import Settings, settings as default_settings
from cs_agent.kb import PolicyKB
from cs_agent.refunds import RefundService
from cs_agent.runner import TurnRunner
from cs_agent.state import TurnStep, render_state
from cs_agent.store import ShopStore
from cs_agent.vision import get_vlm

from .router import WORKERS, Router
from .workers import WorkerFn, build_workers

log = logging.getLogger(__name__)


def make_supervisor(router: Router, max_hops: int, plan_routing: bool = True,
                    direct_reply_enabled: bool = True, workers: tuple = WORKERS) -> Callable[[dict], TurnStep]:
    def goto(nxt: str, state: dict, update: dict) -> TurnStep:
        if nxt in workers:
            return TurnStep(goto=nxt, update={**update, "hops": state.get("hops", 0) + 1})
        if nxt == "FINISH":
            results = state.get("turn_results") or []
            unsupported = update.get("unsupported") or state.get("unsupported")
            if direct_reply_enabled and len(results) == 1 and results[0]["agent"] in workers and not unsupported:
                return TurnStep(goto="direct_reply", update={**update, "final_action": "finish"})
            return TurnStep(goto="respond", update={**update, "final_action": "finish"})
        if nxt == "clarify":
            return TurnStep(goto="respond", update={**update, "final_action": "clarify"})
        return TurnStep(goto="handoff_human", update={**update, "final_action": "handoff"})

    def supervisor(state: dict) -> TurnStep:
        hops = state.get("hops", 0)
        route_log = list(state.get("route_log") or [])

        if hops >= max_hops:
            route_log.append({"next": "handoff_human", "reason": f"超过最大分派次数 {max_hops}"})
            return TurnStep(goto="handoff_human", update={"route_log": route_log, "final_action": "handoff"})

        remaining = state.get("plan")
        if plan_routing and remaining is not None:
            nxt = remaining[0] if remaining else "FINISH"
            route_log.append({"next": nxt, "reason": "按本轮计划执行", "planned": True, "ms": 0})
            return goto(nxt, state, {"route_log": route_log, "plan": remaining[1:]})

        t0 = time.perf_counter()
        try:
            d = router.decide(state)
        except Exception as e:
            log.exception("调度失败")
            route_log.append({"next": "fallback", "reason": f"调度失败：{type(e).__name__}", "failed": "supervisor"})
            return TurnStep(goto="fallback", update={"route_log": route_log})
        ms = round((time.perf_counter() - t0) * 1000)
        route_log.append({"next": d.next, "reason": d.reason, "confidence": d.confidence, "ms": ms, "plan": list(d.plan)})
        update: dict = {"route_log": route_log}
        if getattr(d, "unsupported", ()):
            update["unsupported"] = list(dict.fromkeys([*(state.get("unsupported") or []), *d.unsupported]))
        if plan_routing and d.next in workers and d.plan:
            update["plan"] = list(d.plan[1:])
        return goto(d.next, state, update)

    return supervisor


def direct_reply(state: dict) -> dict:
    return {"messages": [AIMessage(content=state["turn_results"][0]["content"])]}


RESPOND_PROMPT = """你是电商平台的客服，负责把本轮的处理结果整理成给用户的最终回复。
要求：
1. 只使用【本轮已完成的处理】里的信息，不得编造订单、物流、金额等信息。
2. 不得承诺处理结果以外的补偿、赔付或时效。
3. 用户有多个问题时逐条回答，简洁友好。
4. 如果任务是“追问”，只礼貌地询问缺少的信息。
5. 如果本轮没有任何处理结果（比如打招呼），简短回应，并说明可以帮忙查订单物流、办理退款、解答售后政策。
6. 用纯文本回复，不要使用 Markdown 格式（加粗、标题等），需要分点时用“1.”“2.”或换行。"""

UNSUPPORTED_SECTION = "【系统办不了的要求】"
RESPOND_UNSUPPORTED = (
    f"\n7. 有{UNSUPPORTED_SECTION}时，逐项明确告诉用户这里办不了，可以建议其他途径（比如联系快递公司、人工客服）；"
    "不能说已经办好、已登记、已备注，也不能略过不提。能办的部分照常按处理结果回答。"
)
if default_settings.unsupported_rule:
    RESPOND_PROMPT += RESPOND_UNSUPPORTED


def make_respond(llm: Optional[BaseChatModel]) -> Callable:
    def respond(state: dict, on_token: Callable[[str], None] | None = None) -> dict:
        clarify = state.get("final_action") == "clarify"
        reason = (state.get("route_log") or [{}])[-1].get("reason", "")
        task = f"追问用户缺少的信息（调度员的判断：{reason}）" if clarify else "生成给用户的最终回复"
        text = None
        if llm is not None and not state.get("refund_proposed"):
            try:
                extra = ""
                if state.get("unsupported"):
                    extra = f"\n{UNSUPPORTED_SECTION}" + "；".join(state["unsupported"])
                prompt = [SystemMessage(RESPOND_PROMPT), HumanMessage(render_state(state) + extra + f"\n\n任务：{task}")]
                if on_token is None:
                    text = llm.invoke(prompt).content
                else:
                    chunks = []
                    for chunk in llm.stream(prompt):
                        piece = chunk.text if hasattr(chunk, "text") else str(chunk.content or "")
                        if piece:
                            chunks.append(piece)
                            on_token(piece)
                    text = "".join(chunks)
            except Exception:
                log.exception("生成回复失败，使用模板")
        if not text:
            text = _template_reply(state, clarify, reason)
        return {"messages": [AIMessage(content=str(text))]}

    return respond


def _template_reply(state: dict, clarify: bool, reason: str) -> str:
    if clarify:
        return f"为了帮您处理，还需要您补充一些信息：{reason}"
    results = state.get("turn_results") or []
    note = ("\n" + "、".join(state["unsupported"]) + "这里办不了，需要的话可以转人工客服。") if state.get("unsupported") else ""
    if results:
        return "\n".join(r["content"] for r in results) + note
    if note:
        return note.strip()
    return "您好，我可以帮您查询订单和物流、办理退款，或解答售后政策问题。"


def build_multi_runner(
    router: Router,
    *,
    llm: Optional[BaseChatModel] = None,
    store: Optional[ShopStore] = None,
    workers: Optional[dict[str, WorkerFn]] = None,
    settings: Settings = default_settings,
    retriever=None,
    vlm=None,
    refund_service: Optional[RefundService] = None,
) -> TurnRunner:
    store = store or ShopStore(settings.db_path)
    kb = PolicyKB(settings.kb_dir)
    if workers is None:
        if llm is None:
            raise ValueError("未传入 workers 时必须提供 llm")
        from cs_agent.rag import build_retriever

        workers = build_workers(
            llm, store, retriever or build_retriever(settings, llm), settings, refund_service=refund_service,
        )
    dispatch = make_supervisor(
        router, settings.max_hops, settings.plan_routing, settings.direct_reply, workers=tuple(workers),
    )
    return TurnRunner(
        store, dispatch, llm=llm, settings=settings, kb=kb,
        vlm=vlm if vlm is not None else (get_vlm(settings) if llm is not None else None),
        workers=workers, respond=make_respond(llm), direct_reply_fn=direct_reply,
    )
