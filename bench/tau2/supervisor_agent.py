"""τ²-bench 零售领域的 Supervisor 多 Agent 实现（与 τ²-bench 自带的单 Agent `llm_agent` 对比）。

结构和 cs_agent 的多 Agent 一致：调度员只决定由谁处理，Worker 各自只拿自己职责内的工具和政策章节。
- account   ：身份验证、查询用户 / 订单 / 商品、修改默认地址
- pending   ：取消或修改待发货订单（地址、支付方式、商品规格）
- delivered ：已签收订单的退货、换货
每收到一条用户消息，调度员决定交给谁（用户在回答当前 Worker 的确认问题时保持不变）；
工具调用的结果总是回到发起调用的 Worker。工具由 τ²-bench 的环境执行，保证轨迹和数据库评估口径一致。

调度员的模型调用不在 τ²-bench 记录的对话轨迹里，这里把它的 Token 和费用并入紧随其后的那条 Worker 消息，保证成本统计公平。
"""
from __future__ import annotations

import re
from typing import Optional

from tau2.agent.llm_agent import AGENT_INSTRUCTION, LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, UserMessage
from tau2.environment.tool import as_tool
from tau2.utils.llm_utils import generate

MAX_HANDOFFS = 2  # 同一步内最多在同事之间转交几次，防止来回踢皮球
# Worker 用文字宣布“交给同事”而没有调用转交工具时的特征（模拟用户会把它当成转人工而结束对话）
HANDOFF_TEXT = re.compile(r"specialist|colleague|hand (?:it|this|that|you) (?:off|over)|pass (?:it|this|that) (?:along|on)"
                          r"|transfer you to (?:the|our|a) (?!human)", re.I)


def handoff_to_colleague(colleague: str, reason: str) -> str:
    """Hand the conversation over to a colleague in this same chat when the user's current request is outside your scope.
    This is NOT a transfer to a human agent; the user will not notice the switch.

    Args:
        colleague: one of "account", "pending", "delivered".
        reason: what the colleague should take care of.
    """
    return "handed off"

READ_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details",
              "get_product_details", "get_item_details", "list_all_product_types", "calculate", "transfer_to_human_agents"}
WORKERS = {
    "account": dict(
        role="account and information specialist: authenticate the user, look up the user's profile, orders, products "
             "and items, answer information questions, and modify the user's default address",
        tools=READ_TOOLS | {"modify_user_address"},
        sections=(),
    ),
    "pending": dict(
        role="pending-order specialist: cancel pending orders, and modify a pending order's shipping address, "
             "payment method or item options",
        tools=READ_TOOLS | {"cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items",
                            "modify_pending_order_payment"},
        sections=("Generic action rules", "Cancel pending order", "Modify pending order"),
    ),
    "delivered": dict(
        role="delivered-order specialist: return or exchange items of delivered orders",
        tools=READ_TOOLS | {"return_delivered_order_items", "exchange_delivered_order_items"},
        sections=("Generic action rules", "Return delivered order", "Exchange delivered order"),
    ),
}

ROUTER_PROMPT = """You are the dispatcher of a retail customer service team. Decide which specialist should reply to the user now.

Specialists:
- account: authenticating the user, looking up the user's profile, orders, products and items, answering information questions, modifying the user's default address.
- pending: cancelling a pending order, or modifying a pending order's shipping address, payment method or item options.
- delivered: returning or exchanging items of a delivered order.

Rules:
1. If the user is answering a question from, or confirming an action proposed by, the current specialist, keep the current specialist.
   But if the current specialist said the request belongs to another specialist (e.g. "that's handled by our delivered-order specialist"), choose that specialist.
2. Otherwise choose by what the user wants to do now. If the user mentions several requests, choose the one being handled now.
3. Authentication and pure information questions go to account. If unsure, keep the current specialist.

Answer with exactly one word: account, pending, or delivered."""


def split_policy(policy: str) -> tuple[str, dict[str, str]]:
    """把政策按二级标题拆开：通用部分（开头的总则 + Domain basic）和各业务章节。"""
    parts = re.split(r"(?m)^## ", policy)
    common, sections = parts[0], {}
    for p in parts[1:]:
        title = p.split("\n", 1)[0].strip()
        if title == "Domain basic":
            common += "## " + p
        else:
            sections[title] = "## " + p
    return common, sections


class SupervisorState(LLMAgentState):
    active: str = "account"
    routes: list[str] = []


class SupervisorAgent(LLMAgent):
    def __init__(self, tools, domain_policy: str, llm: str, llm_args: Optional[dict] = None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        common, sections = split_policy(domain_policy)
        self.worker_tools, self.worker_prompts = {}, {}
        for name, spec in WORKERS.items():
            self.worker_tools[name] = [t for t in tools if t.name in spec["tools"]]
            policy = common + "\n".join(sections[s] for s in spec["sections"] if s in sections)
            others = "; ".join(f"{n}: {w['role'].split(':')[0]}" for n, w in WORKERS.items() if n != name)
            self.worker_prompts[name] = (
                f"<instructions>\n{AGENT_INSTRUCTION}\n\nYou are the {spec['role']}, working in one chat with colleagues "
                f"({others}). Only handle requests within your scope. When the user's current request is outside your scope, "
                f"call handoff_to_colleague immediately (do not tell the user to wait and do not ask for confirmation); "
                f"your colleague continues in the same chat. Never use transfer_to_human_agents to reach a colleague: "
                f"it transfers the user to a human and ends this chat, so use it only when the policy requires a human agent.\n"
                f"</instructions>\n<policy>\n{policy}\n</policy>"
            )

    HANDOFF_TOOL = as_tool(handoff_to_colleague)

    def get_init_state(self, message_history=None) -> SupervisorState:
        base = super().get_init_state(message_history)
        return SupervisorState(system_messages=base.system_messages, messages=base.messages)

    def _route(self, state: SupervisorState) -> tuple[str, AssistantMessage]:
        lines = []
        for m in state.messages[-16:]:
            if isinstance(m, UserMessage) and m.content:
                lines.append(f"user: {m.content}")
            elif isinstance(m, AssistantMessage):
                if m.content:
                    lines.append(f"agent: {m.content}")
                for tc in m.tool_calls or []:
                    lines.append(f"agent called tool: {tc.name}")
        body = f"Current specialist: {state.active}\n\nConversation (latest last):\n" + "\n".join(lines)
        msg = generate(model=self.llm, messages=[SystemMessage(role="system", content=ROUTER_PROMPT),
                                                 UserMessage(role="user", content=body)],
                       call_name="supervisor_route", **self.llm_args)
        words = re.findall(r"account|pending|delivered", (msg.content or "").lower())
        return (words[0] if words else state.active), msg

    def generate_next_message(self, message, state: SupervisorState):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        router_msg = None
        if isinstance(message, UserMessage):  # 新的一轮用户输入：由调度员决定交给谁；工具结果则回到原来的 Worker
            state.active, router_msg = self._route(state)
            state.routes.append(state.active)

        msg = self._worker_step(state)
        if router_msg is not None:  # 调度员的用量并入这条消息
            _merge_usage(msg, router_msg)
        state.messages.append(msg)
        return msg, state

    def _worker_step(self, state: SupervisorState) -> AssistantMessage:
        """当前 Worker 生成下一步。它调用 handoff_to_colleague 时在内部换人重新生成（不交给环境执行，也不写进轨迹）。"""
        extra: list[AssistantMessage] = []
        for _ in range(MAX_HANDOFFS + 1):
            worker = state.active
            msg = generate(model=self.llm, tools=self.worker_tools[worker] + [self.HANDOFF_TOOL],
                           messages=[SystemMessage(role="system", content=self.worker_prompts[worker])] + state.messages,
                           call_name=f"worker_{worker}", **self.llm_args)
            hand = [tc for tc in (msg.tool_calls or []) if tc.name == "handoff_to_colleague"]
            target = (hand[0].arguments or {}).get("colleague") if hand else None
            if not hand and not msg.tool_calls and HANDOFF_TEXT.search(msg.content or ""):
                # 交接检测：文字里说要交给同事，先不发给用户。优先从文字里认出是哪位同事，认不出再请调度员判断
                target = _mentioned_colleague(msg.content or "", worker)
                if target is None:
                    probe = state.model_copy(deep=True)
                    probe.messages.append(msg)
                    target, router_msg = self._route(probe)
                    extra.append(router_msg)
                hand = [None] if target != worker else []
            if not hand or target not in WORKERS or target == worker:
                if hand and hand[0] is not None:  # 无效的转交：去掉这个调用，按普通回复处理
                    msg.tool_calls = [tc for tc in msg.tool_calls if tc.name != "handoff_to_colleague"] or None
                    if not msg.tool_calls and not msg.content:
                        msg.content = "Could you tell me a bit more about what you need?"
                break
            state.active = target
            state.routes.append(f"handoff:{target}")
            extra.append(msg)
        for m in extra:  # 被转交掉的那几次生成，用量也要计入
            _merge_usage(msg, m)
        return msg


def _mentioned_colleague(text: str, current: str) -> Optional[str]:
    """从“交给 xx 专员”这类说法里认出目标同事；认不出或指向自己时返回 None。"""
    t = text.lower()
    hits = {name for name, pat in (("pending", r"pending[- ]order|pending order"),
                                   ("delivered", r"delivered[- ]order|return|exchange"),
                                   ("account", r"account")) if re.search(pat, t)}
    hits.discard(current)
    return hits.pop() if len(hits) == 1 else None


def _merge_usage(msg: AssistantMessage, other: AssistantMessage) -> None:
    if msg.usage and other.usage:
        msg.usage = {k: (msg.usage.get(k) or 0) + (other.usage.get(k) or 0) for k in msg.usage}
    if msg.cost is not None and other.cost is not None:
        msg.cost += other.cost


def create_supervisor_agent(tools, domain_policy, **kwargs):
    return SupervisorAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))
