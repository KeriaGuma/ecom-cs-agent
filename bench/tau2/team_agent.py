"""τ²-bench 上的“调度员 + Worker 汇报”多 Agent，支持零售和航空两个领域，两种确认方式：

- relay（转述确认）：和 plan_agent.py 相同。Worker 自己持有写工具，确认内容由调度员用自然语言转述，
  用户确认后再由 Worker 自己调用写工具。
- structured（结构化方案）：Worker 手里没有写工具，只能调用 propose_action 提交“工具名 + 完整参数”。
  代码按该工具的参数定义校验（缺支付方式等必填项会被打回），登记为方案 P1、P2……
  调度员把方案给用户看，用户明确确认后调用 execute_proposal，由代码把登记的参数原样发给环境执行。
  代码还会把关：方案必须先在调度员发给用户的消息之后、且用户回复过，才允许执行。

领域由政策文本判断（τ²-bench 也会把 task 传给工厂函数，这里刻意不用，避免看到标准答案）。
内部调用（调度、委派、汇报、方案登记）不在轨迹里，它们的 Token 并入随后发出的那条消息。
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from pydantic import BaseModel, ValidationError

from bench.tau2.supervisor_agent import _merge_usage
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (AssistantMessage, MultiToolMessage, SystemMessage, ToolCall, ToolMessage,
                                     UserMessage)
from tau2.environment.tool import as_tool
from tau2.utils.llm_utils import generate

MAX_INTERNAL_CALLS = 16
MAX_TASK_STEPS = 10
HUMAN = "transfer_to_human_agents"

# ---------------------------------------------------------------- 领域配置
TEAMS = {
    "retail": dict(
        read={"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details",
              "get_product_details", "get_item_details", "list_all_product_types", "calculate"},
        first_step="Authentication comes first: delegate it to account with the identity details the customer gave "
                   "(email, or name + zip code).",
        workers={
            "account": dict(role="account and information specialist: authenticate the customer (by email, or by name "
                                 "+ zip code), look up the customer's profile, orders, products and items, answer "
                                 "information questions, and modify the customer's default address",
                            write={"modify_user_address"}, sections=()),
            "pending": dict(role="pending-order specialist: cancel pending orders, and modify a pending order's shipping "
                                 "address, payment method or item options",
                            write={"cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items",
                                   "modify_pending_order_payment"},
                            sections=("Generic action rules", "Cancel pending order", "Modify pending order")),
            "delivered": dict(role="delivered-order specialist: return or exchange items of delivered orders",
                              write={"return_delivered_order_items", "exchange_delivered_order_items"},
                              sections=("Generic action rules", "Return delivered order", "Exchange delivered order")),
        },
    ),
    "airline": dict(
        read={"get_user_details", "get_reservation_details", "get_flight_status", "list_all_airports", "calculate"},
        first_step="Specialists need the customer's user id; ask the customer for it first.",
        workers={
            "booking": dict(role="booking specialist: search flights and book new reservations",
                            write={"book_reservation"}, extra_read={"search_direct_flight", "search_onestop_flight"},
                            sections=("Book flight",)),
            "change": dict(role="reservation-change specialist: change the flights, cabin, checked bags or passengers "
                                "of an existing reservation, and compensation related to delayed flights",
                           write={"update_reservation_flights", "update_reservation_baggages",
                                  "update_reservation_passengers", "send_certificate"},
                           extra_read={"search_direct_flight", "search_onestop_flight"},
                           sections=("Modify flight", "Refunds and Compensation")),
            "cancel": dict(role="cancellation and compensation specialist: cancel reservations, refunds, and "
                                "compensation for cancelled or delayed flights",
                           write={"cancel_reservation", "send_certificate"},
                           sections=("Cancel flight", "Refunds and Compensation")),
        },
    ),
}


def detect_domain(policy: str) -> str:
    head = policy.lstrip().split("\n", 1)[0].lower()
    for name in TEAMS:
        if name in head:
            return name
    raise ValueError(f"无法从政策标题判断领域：{head!r}")


def split_policy(policy: str) -> tuple[str, dict[str, str]]:
    """通用部分（开头的总则 + Domain basic）和各业务章节。"""
    parts = re.split(r"(?m)^## ", policy)
    common, sections = parts[0], {}
    for p in parts[1:]:
        title = p.split("\n", 1)[0].strip()
        if title.lower() == "domain basic":
            common += "## " + p
        else:
            sections[title] = "## " + p
    return common, sections


# ---------------------------------------------------------------- 内部工具（不交给环境执行）
def delegate(specialist: str, task: str) -> str:
    """Delegate one task to a specialist. The specialist looks things up (or prepares the action) with its tools and reports back to you.
    The specialist cannot talk to the customer, so include everything it needs: the customer's identity details or user id,
    order or reservation ids, items, passengers, payment method, and what the customer has said.

    Args:
        specialist: the specialist's name.
        task: what the specialist should do, with all relevant details.
    """
    return "report"


def propose_action(action: str, arguments: dict, summary: str) -> str:
    """Propose one database-changing action. It is NOT executed now: the dispatcher shows it to the customer and,
    after the customer explicitly confirms, the system executes exactly these arguments.

    Args:
        action: the name of the action tool to call, e.g. one listed in your instructions.
        arguments: the complete arguments for that tool, exactly as they should be executed.
        summary: a short plain-language description of the action for the customer (including amounts and payment method if any).
    """
    return "proposed"


def execute_proposal(proposal_id: str) -> str:
    """Execute a proposed action exactly as proposed. Only call this after the customer has seen the proposal's details
    and explicitly confirmed (yes) this exact proposal.

    Args:
        proposal_id: the proposal id, e.g. "P1".
    """
    return "executed"


# ---------------------------------------------------------------- 提示词
DISPATCHER = """You are the front-desk agent of a {domain} customer service team and the only one who talks to the customer.
You have no database tools. Specialists do all lookups and actions and report back to you; they cannot talk to the customer.
{team}

How to work:
1. Use the delegate tool to give a specialist one task at a time, with all details it needs. {first_step}
2. If a report asks for information, ask the customer.
{confirm}
4. If the customer has several requests, handle them one by one. Never say an action was done unless it was reported done.
5. Deny requests that are against the policy. Transfer to a human agent only if the request cannot be handled within the scope of the team's actions: call transfer_to_human_agents, then send 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'

In each turn you can either send a message to the customer or make a tool call, not both. Always make sure you generate valid JSON only."""

CONFIRM_RELAY = """3. Before any database change, a specialist reports the exact action details. List them to the customer and ask for explicit confirmation (yes).
   After the customer explicitly confirms, delegate to the same specialist again, saying the customer explicitly confirmed and repeating the exact details."""

CONFIRM_STRUCT = """3. Database changes are prepared by specialists as proposals (P1, P2, ...) with exact arguments. Show the customer every detail of a proposal,
   including amounts and the payment method, and ask for explicit confirmation (yes). Only after the customer explicitly confirms that exact proposal,
   call execute_proposal with its id; the system executes it exactly as proposed. If the customer wants something different, delegate again for a new proposal."""

WORKER = """You are the {role}, working behind a front-desk agent (the dispatcher) in a {domain} customer service team.
The dispatcher gives you a task; you cannot talk to the customer. Use your tools to do the task, following the policy.
When you stop calling tools, your text message is your report to the dispatcher. Keep it short but include every fact needed (ids, items, options, prices, statuses, payment methods, amounts).

Rules:
{action_rule}
- Every argument must come from the customer or from tool results. Never choose a payment method or option the customer has not chosen; if one is missing, report what to ask.
- If the policy does not allow the request, report why.
- If the task is outside your scope, report which specialist should handle it: {others}.
- Make at most one tool call at a time."""

ACTION_RELAY = """- Before any action that updates the database, the customer must have explicitly confirmed the exact action details.
  If the task does not say the customer explicitly confirmed exactly this action, do NOT take it: report the exact action details
  (and anything the policy says the customer must be reminded of) for the dispatcher to confirm with the customer."""

ACTION_STRUCT = """- You cannot change the database yourself. When an action is needed and allowed, call propose_action with the action tool name
  ({actions}) and its complete arguments. The dispatcher will show it to the customer, and the system executes it after the customer confirms.
  Then report briefly, including anything the policy says the customer must be reminded of.
- Argument schemas of your action tools:
{schemas}"""


def _render(m) -> Optional[str]:
    if isinstance(m, UserMessage) and m.content:
        return f"customer: {m.content}"
    if isinstance(m, AssistantMessage) and m.content:
        return f"you (front desk): {m.content}"
    return None


class TeamState(BaseModel):
    dialog: list = []
    log: list[str] = []
    queue: list[dict] = []
    task: Optional[dict] = None
    task_msgs: list = []
    task_steps: int = 0
    routes: list[str] = []
    pending_usage: list = []
    proposals: dict = {}         # P1 -> {worker, action, arguments, summary, status, presented, replied}
    executing: Optional[dict] = None  # {call_id, pid}

    model_config = {"arbitrary_types_allowed": True}


class TeamAgent(LLMAgent):
    def __init__(self, tools, domain_policy: str, llm: str, llm_args: Optional[dict] = None, structured: bool = True):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.structured = structured
        self.domain = detect_domain(domain_policy)
        spec = TEAMS[self.domain]
        self.by_name = {t.name: t for t in tools}
        common, _ = split_policy(domain_policy)  # 调度员只看总则和基础信息
        team = "\n".join(f"- {n}: {w['role'].split(':', 1)[1].strip()}" for n, w in spec["workers"].items())
        internal = [as_tool(delegate)] + ([as_tool(execute_proposal)] if structured else [])
        self.dispatcher_tools = internal + [t for t in tools if t.name == HUMAN]
        self.dispatcher_prompt = (
            "<instructions>\n" + DISPATCHER.format(domain=self.domain, team=team, first_step=spec["first_step"],
                                                   confirm=CONFIRM_STRUCT if structured else CONFIRM_RELAY)
            + f"\n</instructions>\n<policy>\n{common}\n</policy>")
        self.workers = spec["workers"]
        self.worker_tools, self.worker_prompts, self.worker_writes = {}, {}, {}
        for name, w in self.workers.items():
            names = spec["read"] | w.get("extra_read", set())
            writes = [t for t in tools if t.name in w["write"]]
            self.worker_writes[name] = {t.name for t in writes}
            if structured:
                schemas = "\n".join(f"  {t.name}: {json.dumps(t.params.model_json_schema(), ensure_ascii=False)}"
                                    for t in writes)
                rule = ACTION_STRUCT.format(actions=", ".join(sorted(self.worker_writes[name])), schemas=schemas)
                self.worker_tools[name] = [t for t in tools if t.name in names] + [as_tool(propose_action)]
            else:
                rule = ACTION_RELAY
                self.worker_tools[name] = [t for t in tools if t.name in names] + writes
            # 每个 Worker 都拿完整政策：航空政策的章节互相引用（如保险的赔付范围、会员免费行李额写在“订票”一章，
            # 取消和改签都要用到），按章节拆分会让 Worker 缺规则（训练集开发第 1 轮发现）。职责只靠工具划分。
            policy = domain_policy
            others = "; ".join(f"{n} ({o['role'].split(':')[0]})" for n, o in self.workers.items() if n != name)
            self.worker_prompts[name] = ("<instructions>\n" + WORKER.format(role=w["role"], domain=self.domain,
                                                                            action_rule=rule, others=others)
                                         + f"\n</instructions>\n<policy>\n{policy}\n</policy>")

    def get_init_state(self, message_history=None) -> TeamState:
        return TeamState()

    # ---- 主循环 ----
    def generate_next_message(self, message, state: TeamState):
        tool_msgs = message.tool_messages if isinstance(message, MultiToolMessage) else (
            [message] if isinstance(message, ToolMessage) else [])
        if isinstance(message, UserMessage):
            state.dialog.append(message)
            for p in state.proposals.values():
                if p["presented"]:
                    p["replied"] = True
        elif state.executing is not None:  # 代码代为执行的方案，结果交回调度员
            ex, state.executing = state.executing, None
            res = tool_msgs[0] if tool_msgs else None
            content = res.content if res else "(no result)"
            state.proposals[ex["pid"]]["status"] = "failed" if (res and res.error) else "executed"
            state.log.append(f"[system executed {ex['pid']}] result: {content}")
            state.dialog.append(ToolMessage(id=ex["call_id"], role="tool", content=content))
        elif state.task is not None:
            state.task_msgs.extend(tool_msgs)
            for t in tool_msgs:
                state.log.append(f"  result: {t.content}")
        else:
            state.dialog.extend(tool_msgs)

        for _ in range(MAX_INTERNAL_CALLS):
            if state.task is None and state.queue:
                self._start_task(state, state.queue.pop(0))
            out = self._worker_step(state) if state.task is not None else self._dispatcher_step(state)
            if out is not None:
                return self._emit(out, state), state
        msg = AssistantMessage(role="assistant", content="Sorry, could you please repeat your request?")
        state.dialog.append(msg)
        return self._emit(msg, state), state

    def _emit(self, msg: AssistantMessage, state: TeamState) -> AssistantMessage:
        for u in state.pending_usage:
            _merge_usage(msg, u)
        state.pending_usage = []
        return msg

    def _call(self, system: str, messages: list, tools, name: str) -> AssistantMessage:
        return generate(model=self.llm, tools=tools, messages=[SystemMessage(role="system", content=system)] + messages,
                        call_name=name, **self.llm_args)

    # ---- 调度员 ----
    def _dispatcher_step(self, state: TeamState) -> Optional[AssistantMessage]:
        msg = self._call(self.dispatcher_prompt, state.dialog, self.dispatcher_tools, "dispatcher")
        calls = msg.tool_calls or []
        execs = [c for c in calls if c.name == "execute_proposal"]
        delegations = [c for c in calls if c.name == "delegate"]
        if execs:
            call = execs[0]
            msg.tool_calls, msg.content = [call], None
            state.dialog.append(msg)
            state.pending_usage.append(msg)
            return self._execute(state, call)
        if delegations:
            msg.tool_calls, msg.content = delegations, None
            state.dialog.append(msg)
            state.pending_usage.append(msg)
            for c in delegations:
                args = c.arguments or {}
                state.queue.append({"call_id": c.id, "worker": str(args.get("specialist", "")).strip().lower(),
                                    "task": str(args.get("task", ""))})
            return None
        if calls:  # 转人工：交给环境执行
            msg.tool_calls, msg.content = [c for c in calls if c.name == HUMAN][:1] or None, None
            if msg.tool_calls:
                state.dialog.append(msg)
                return msg
            msg.content = "Could you tell me a bit more about what you need?"
        state.dialog.append(msg)
        for p in state.proposals.values():  # 回复用户：之前登记、尚未展示的方案视为已展示
            if p["status"] == "proposed" and not p["presented"]:
                p["presented"] = True
        return msg

    def _execute(self, state: TeamState, call) -> Optional[AssistantMessage]:
        pid = str((call.arguments or {}).get("proposal_id", "")).strip().upper()
        p = state.proposals.get(pid)
        if p is None or p["status"] != "proposed":
            err = f"Not executed: there is no open proposal {pid!r}. Open proposals: " + (
                ", ".join(k for k, v in state.proposals.items() if v["status"] == "proposed") or "none") + "."
        elif not (p["presented"] and p["replied"]):
            err = (f"Not executed: the customer has not yet seen and confirmed {pid}. Show the customer its exact details "
                   "and ask for explicit confirmation first.")
        else:
            p["status"] = "executing"
            state.executing = {"call_id": call.id, "pid": pid}
            state.log.append(f"[dispatcher executes {pid}] {p['action']}({json.dumps(p['arguments'], ensure_ascii=False)})")
            return AssistantMessage(role="assistant", content=None, tool_calls=[
                ToolCall(id=f"call_{uuid.uuid4().hex[:24]}", name=p["action"], arguments=p["arguments"])])
        state.dialog.append(ToolMessage(id=call.id, role="tool", content=err, error=True))
        return None

    # ---- Worker ----
    def _start_task(self, state: TeamState, job: dict) -> None:
        if job["worker"] not in self.workers:
            self._finish_task(state, job, f"Unknown specialist '{job['worker']}'. Use one of: {', '.join(self.workers)}.")
            return
        state.task, state.task_msgs, state.task_steps = job, [], 0
        state.routes.append(job["worker"])
        state.log.append(f"[dispatcher -> {job['worker']}] {job['task']}")
        dialog = "\n".join(filter(None, (_render(m) for m in state.dialog))) or "(none)"
        log = "\n".join(state.log[:-1]) or "(none)"
        brief = (f"## Conversation between the front desk and the customer so far\n{dialog}\n\n"
                 f"## Team work log so far (tasks, tool calls, results, reports)\n{log}\n\n"
                 f"## Your task from the dispatcher\n{job['task']}")
        state.task_msgs.append(UserMessage(role="user", content=brief))

    def _worker_step(self, state: TeamState) -> Optional[AssistantMessage]:
        job = state.task
        worker = job["worker"]
        tools = self.worker_tools[worker] if state.task_steps < MAX_TASK_STEPS else None
        msg = self._call(self.worker_prompts[worker], state.task_msgs, tools, f"worker_{worker}")
        proposals = [c for c in (msg.tool_calls or []) if c.name == "propose_action"]
        if proposals:  # 方案登记在 Agent 内部完成，不交给环境
            state.task_steps += 1
            msg.tool_calls, msg.content = proposals, None
            state.task_msgs.append(msg)
            state.pending_usage.append(msg)
            for c in proposals:
                state.task_msgs.append(ToolMessage(id=c.id, role="tool", content=self._register(state, worker, c)))
            return None
        if msg.tool_calls:
            state.task_steps += 1
            state.task_msgs.append(msg)
            for c in msg.tool_calls:
                state.log.append(f"  {worker} called {c.name}({json.dumps(c.arguments, ensure_ascii=False)})")
            msg.content = None
            return msg
        state.pending_usage.append(msg)
        report = msg.content or "(empty report)"
        mine = [k for k, v in state.proposals.items() if v["status"] == "proposed" and v["task"] == job["call_id"]]
        if mine:  # 方案的参数由代码附在汇报后面，调度员看到的是登记的原样参数
            report += "\n\nProposals awaiting the customer's explicit confirmation:\n" + "\n".join(
                f"- {k}: {state.proposals[k]['action']} {json.dumps(state.proposals[k]['arguments'], ensure_ascii=False)}"
                f" — {state.proposals[k]['summary']}" for k in mine)
        self._finish_task(state, job, report)
        return None

    def _register(self, state: TeamState, worker: str, call) -> str:
        args = call.arguments or {}
        action, arguments, summary = args.get("action"), args.get("arguments"), str(args.get("summary", ""))
        if action not in self.worker_writes[worker]:
            return f"Rejected: '{action}' is not one of your action tools ({', '.join(sorted(self.worker_writes[worker]))})."
        if not isinstance(arguments, dict):
            return "Rejected: arguments must be an object."
        try:
            self.by_name[action].params.model_validate(arguments)
        except ValidationError as e:
            missing = [".".join(map(str, er["loc"])) + f" ({er['msg']})" for er in e.errors()]
            return "Rejected: invalid arguments: " + "; ".join(missing) + ". If a value is unknown, report what to ask the customer."
        key = next((arguments.get(k) for k in ("order_id", "reservation_id", "user_id") if arguments.get(k)), None)
        for p in state.proposals.values():  # 同一对象上的旧方案作废
            if p["status"] == "proposed" and p["action"] == action and p["key"] == key:
                p["status"] = "superseded"
        pid = f"P{len(state.proposals) + 1}"
        state.proposals[pid] = dict(worker=worker, action=action, arguments=arguments, summary=summary, key=key,
                                    task=state.task["call_id"], status="proposed", presented=False, replied=False)
        state.log.append(f"  {worker} proposed {pid}: {action}({json.dumps(arguments, ensure_ascii=False)}) — {summary}")
        return f"Recorded as proposal {pid}. It will be shown to the customer and executed only after explicit confirmation."

    def _finish_task(self, state: TeamState, job: dict, report: str) -> None:
        state.log.append(f"[{job['worker']} report] {report}")
        state.dialog.append(ToolMessage(id=job["call_id"], role="tool", content=report))
        state.task, state.task_msgs, state.task_steps = None, [], 0


def create_team_relay(tools, domain_policy, **kwargs):
    return TeamAgent(tools, domain_policy, kwargs.get("llm"), kwargs.get("llm_args"), structured=False)


def create_team_struct(tools, domain_policy, **kwargs):
    return TeamAgent(tools, domain_policy, kwargs.get("llm"), kwargs.get("llm_args"), structured=True)
