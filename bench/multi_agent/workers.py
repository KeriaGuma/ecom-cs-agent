"""三个领域专员 Worker（多 Agent 架构对照基线，与主线单 Agent 共用业务规则与工具实现）。"""
from __future__ import annotations

import logging
from datetime import date
from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from cs_agent.agent import CONTACT_RULE, PHOTO_RULE, REFUND_DECISION, prepare_refund_tool, refund_proposal_reply
from cs_agent.config import Settings, settings as _settings
from cs_agent.distractors import WORKER_DOMAIN, distractor_tools
from cs_agent.react import ReactResult, run_react
from cs_agent.refunds import RefundService
from cs_agent.state import TurnStep, extract_facts, render_state
from cs_agent.store import ShopStore
from cs_agent.tools import NOT_FOUND, make_kb_tools, make_refund_tools, order_tools_for

log = logging.getLogger(__name__)

_SUFFIX = (
    "\n\n只处理本轮用户问题中属于你职责的部分；其他问题直接忽略，结论里也不要提及，也不要提你的职责范围或内部分工。"
    "最后用一两句话给出处理结论，直接写给用户（可能原样发给用户，也可能和其他结论一起汇总），"
    "礼貌简洁，不要寒暄，不要用 Markdown 格式。"
    "【关键事实】里的订单号是之前提到过的订单，只在用户用“它”“这单”等指代时使用；"
    "用户说了具体商品名时以商品名为准，先用 list_my_orders 找到对应的订单。"
)

ORDER_PROMPT = (
    "你是订单物流专员，可以查询订单详情、物流轨迹，修改未发货订单的收货地址。"
    "所有信息以工具返回为准，不要编造。" + (CONTACT_RULE if _settings.address_guard else "") + _SUFFIX
)

REFUND_PROMPT_V1 = (
    "你是售后退款专员。先用 check_refund 查询规则判断结果；只有用户明确表示要退款时才调用 submit_refund。"
    "用户已明确要求退款（如“帮我退了”“不要了，退款”“帮我取消”）时，查到可退后直接提交，不需要再征求确认；"
    "用户只是询问能不能退时，只查询不提交。"
    "退款原因：不想要了、不喜欢、买错了 → no_reason；坏了、有瑕疵、质量问题 → quality；"
    "无法判断原因时不要提交，在结论里说明需要用户确认原因。"
    "能否退款、金额、运费由谁承担一律以工具返回为准，不得自行承诺。" + _SUFFIX
)

_WORKER_REFUND_DECISION = REFUND_DECISION.replace("是否发起退款申请", "是否调用 submit_refund").replace("直接调用退款工具", "直接提交")

REFUND_PROMPT_V2 = (
    "你是售后退款专员。先用 check_refund 查询规则判断结果。\n" + _WORKER_REFUND_DECISION + "\n"
    "你不能直接问用户，缺信息时不要为了把事办完而替用户补全（比如用户没说原因就当作无理由提交），"
    "只调用 check_refund 查询，在结论里说明需要用户补充什么，由前台去问。"
    "能否退款、金额、运费由谁承担一律以工具返回为准，不得自行承诺。" + _SUFFIX
)

REFUND_PROMPT = REFUND_PROMPT_V2 if _settings.refund_rule == 2 else REFUND_PROMPT_V1
if _settings.vision:
    REFUND_PROMPT = (
        REFUND_PROMPT.replace(_SUFFIX, "\n" + PHOTO_RULE + _SUFFIX)
        if _SUFFIX in REFUND_PROMPT else REFUND_PROMPT + "\n" + PHOTO_RULE
    )

KB_PROMPT = (
    "你是售后政策顾问。先用 search_policy 检索，再根据检索到的条款回答，并注明出处（文件 · 条款标题）。"
    "知识库里没有的内容不要编造，直接说明没有找到相关规定。" + _SUFFIX
)

WorkerFn = Callable[[dict], TurnStep]


def build_workers(
    llm: BaseChatModel,
    store: ShopStore,
    kb,
    settings: Settings,
    today: Callable[[], date] = date.today,
    refund_service: Optional[RefundService] = None,
) -> dict[str, WorkerFn]:
    def extra(agent: str) -> list:
        return distractor_tools(settings.extra_tools, WORKER_DOMAIN[agent]) if settings.extra_tools and agent in WORKER_DOMAIN else []

    def order_agent(state: dict) -> TurnStep:
        try:
            res = run_react(
                llm, ORDER_PROMPT, _task(state),
                order_tools_for(store, state["user_id"], settings.order_tools) + extra("order_agent"),
                claim_check=settings.claim_check,
            )
        except Exception as e:
            return _degrade(state, "order_agent", e)
        return _done(state, "order_agent", res)

    def refund_agent(state: dict) -> TurnStep:
        proposals: dict[str, dict] = {}
        tools = make_refund_tools(
            store, state["user_id"], settings.refund_auto_approve_limit, today=today,
            prepare=prepare_refund_tool(state, refund_service, proposals),
        )
        prompt = REFUND_PROMPT
        if refund_service is not None:
            prompt = prompt.replace("submit_refund", "prepare_refund") + (
                "\nprepare_refund 只生成确认单。请让用户核对并确认；草稿阶段尚未提交申请、审核或付款。"
            )
        try:
            res = run_react(llm, prompt, _task(state), tools + extra("refund_agent"), claim_check=settings.claim_check)
        except Exception as e:
            return _degrade(state, "refund_agent", e)
        if proposals:
            res.answer = refund_proposal_reply(proposals)
        return _done(state, "refund_agent", res, extra={"refund_proposed": True} if proposals else None)

    def kb_agent(state: dict) -> TurnStep:
        try:
            res = run_react(llm, KB_PROMPT, _task(state), make_kb_tools(kb) + extra("kb_agent"), claim_check=settings.claim_check)
        except Exception as e:
            return _degrade(state, "kb_agent", e)
        return _done(state, "kb_agent", res)

    workers = {"order_agent": order_agent, "refund_agent": refund_agent, "kb_agent": kb_agent}

    from cs_agent.domains import SPECS, domain_tools

    def make_domain_worker(domain: str, name: str) -> WorkerFn:
        def worker(state: dict) -> TurnStep:
            try:
                res = run_react(
                    llm, SPECS[domain]["prompt"] + _SUFFIX, _task(state),
                    domain_tools(domain, store, state["user_id"], today), claim_check=settings.claim_check,
                )
            except Exception as e:
                return _degrade(state, name, e)
            return _done(state, name, res)

        return worker

    for d in settings.domain_list:
        workers[SPECS[d]["worker"]] = make_domain_worker(d, SPECS[d]["worker"])
    return workers


def _task(state: dict) -> list[HumanMessage]:
    return [HumanMessage(render_state(state) + "\n\n请处理其中属于你职责范围的问题。")]


def _done(state: dict, agent: str, res: ReactResult, goto: str = "supervisor", extra: Optional[dict] = None) -> TurnStep:
    facts = extract_facts(state.get("facts"), res.tool_calls, NOT_FOUND)
    results = list(state.get("turn_results") or []) + [{"agent": agent, "content": res.answer}]
    return TurnStep(goto=goto, update={"turn_results": results, "facts": facts, **(extra or {})})


def _degrade(state: dict, agent: str, err: Exception) -> TurnStep:
    log.exception("%s 执行失败", agent)
    route_log = list(state.get("route_log") or []) + [
        {"next": "fallback", "reason": f"{agent} 执行失败：{type(err).__name__}", "failed": agent}
    ]
    return TurnStep(goto="fallback", update={"route_log": route_log})
