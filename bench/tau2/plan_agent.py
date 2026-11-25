"""τ²-bench 零售领域的“调度员规划、Worker 汇报”多 Agent（与本项目 cs_agent 的多 Agent 结构一致）。

和 supervisor_agent.py（Worker 直接面对用户、对话中途互相转手）的区别：
- 只有调度员和用户对话。调度员没有数据库工具，只能把任务委派给 Worker（delegate 工具）；
- Worker 不和用户说话。它用自己的工具办事，最后一条文字消息就是交回调度员的汇报；
- 需要用户补充信息或确认操作时，Worker 在汇报里写明，由调度员去问；用户确认后，调度员再委派 Worker 执行；
- Worker 之间从不直接交接，跨领域的请求由调度员按顺序分别委派。

Worker 的工具调用照常发给 τ²-bench 的环境执行，评估口径和单 Agent 一致。
调度员的委派、Worker 的汇报都是 Agent 内部的模型调用，不在轨迹里，它们的 Token 并入随后发出的那条消息。
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel

from bench.tau2.supervisor_agent import WORKERS, _merge_usage, split_policy
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage,
                                     UserMessage)
from tau2.environment.tool import as_tool
from tau2.utils.llm_utils import generate

MAX_INTERNAL_CALLS = 16  # 一次 generate_next_message 里最多几次内部模型调用，防止死循环
MAX_TASK_STEPS = 10      # 一个委派任务里 Worker 最多调用几轮工具
HUMAN = "transfer_to_human_agents"


def delegate(specialist: str, task: str) -> str:
    """Delegate one task to a specialist. The specialist looks things up or takes the action with its tools and reports back to you.
    The specialist cannot talk to the customer, so include everything it needs: the customer's identity details
    or user id, order ids, items, payment method, and whether the customer has explicitly confirmed the exact action.

    Args:
        specialist: one of "account", "pending", "delivered".
        task: what the specialist should do, with all relevant details.
    """
    return "report"


DISPATCHER_INSTRUCTION = """You are the front-desk agent of a retail customer service team and the only one who talks to the customer.
You have no database tools. Specialists do all lookups and actions and report back to you; they cannot talk to the customer.
- account: authenticate the customer (by email, or by name + zip code), look up the customer's profile, orders, products and items, answer information questions, modify the customer's default address.
- pending: cancel a pending order, or modify a pending order's shipping address, payment method or item options.
- delivered: return or exchange items of a delivered order.

How to work:
1. Use the delegate tool to give a specialist one task at a time, with all details it needs. Authentication comes first: delegate it to account with the details the customer gave.
2. If a report asks for information or confirmation, ask the customer. When asking for confirmation, list the exact action details from the report.
3. After the customer explicitly confirms (yes), delegate to the same specialist again, saying the customer explicitly confirmed and repeating the exact details.
4. If the customer has several requests, handle them one by one. Never say an action was done unless a specialist reported it done.
5. Deny requests that are against the policy. Transfer to a human agent only if the request cannot be handled within the scope of the team's actions: call transfer_to_human_agents, then send 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'

In each turn you can either send a message to the customer or make a tool call, not both. Always make sure you generate valid JSON only."""

WORKER_INSTRUCTION = """You are the {role}, working behind a front-desk agent (the dispatcher) in a retail customer service team.
The dispatcher gives you a task; you cannot talk to the customer. Use your tools to do the task, following the policy.
When you stop calling tools, your text message is your report to the dispatcher. Keep it short but include every fact needed (ids, items, options, prices, statuses, payment methods, refund details).

Rules:
- Before any action that updates the database, the customer must have explicitly confirmed the exact action details.
  If the task does not say the customer explicitly confirmed exactly this action, do NOT take it: report the exact action details
  (and anything the policy says the customer must be reminded of) for the dispatcher to confirm with the customer.
- If information is missing (e.g. which items, which payment method, the cancellation reason), report exactly what to ask.
- If the policy does not allow the request, report why.
- If the task is outside your scope, report which specialist should handle it: {others}.
- Make at most one tool call at a time."""


def _render(m) -> Optional[str]:
    if isinstance(m, UserMessage) and m.content:
        return f"customer: {m.content}"
    if isinstance(m, AssistantMessage) and m.content:
        return f"you (front desk): {m.content}"
    return None


class PlanState(BaseModel):
    dialog: list = []        # 调度员视角的对话：用户消息、调度员回复、委派调用和汇报
    log: list[str] = []      # 共享工作记录：每次委派的任务、Worker 的工具调用和结果、汇报
    queue: list[dict] = []   # 待执行的委派 {call_id, worker, task}
    task: Optional[dict] = None   # 正在执行的委派
    task_msgs: list = []     # 当前委派里 Worker 的消息（工具调用与结果）
    task_steps: int = 0
    routes: list[str] = []
    pending_usage: list = []  # 内部调用的用量，并入下一条发出的消息

    model_config = {"arbitrary_types_allowed": True}


class PlanAgent(LLMAgent):
    def __init__(self, tools, domain_policy: str, llm: str, llm_args: Optional[dict] = None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        common, sections = split_policy(domain_policy)
        self.delegate_tool = as_tool(delegate)
        self.dispatcher_tools = [self.delegate_tool] + [t for t in tools if t.name == HUMAN]
        self.dispatcher_prompt = f"<instructions>\n{DISPATCHER_INSTRUCTION}\n</instructions>\n<policy>\n{common}\n</policy>"
        self.worker_tools, self.worker_prompts = {}, {}
        for name, spec in WORKERS.items():
            self.worker_tools[name] = [t for t in tools if t.name in spec["tools"] and t.name != HUMAN]
            policy = common + "\n".join(sections[s] for s in spec["sections"] if s in sections)
            others = "; ".join(f"{n} ({w['role'].split(':')[0]})" for n, w in WORKERS.items() if n != name)
            self.worker_prompts[name] = (f"<instructions>\n{WORKER_INSTRUCTION.format(role=spec['role'], others=others)}\n"
                                         f"</instructions>\n<policy>\n{policy}\n</policy>")

    def get_init_state(self, message_history=None) -> PlanState:
        return PlanState()

    # ---- 主循环：内部反复调用调度员和 Worker，直到产生一条要发出去的消息（回复用户，或交给环境执行的工具调用）----
    def generate_next_message(self, message, state: PlanState):
        tool_msgs = message.tool_messages if isinstance(message, MultiToolMessage) else (
            [message] if isinstance(message, ToolMessage) else [])
        if isinstance(message, UserMessage):
            state.dialog.append(message)
        elif state.task is not None:  # Worker 调用的工具结果
            state.task_msgs.extend(tool_msgs)
            for t in tool_msgs:
                state.log.append(f"  result: {t.content}")
        else:  # 调度员自己调用的转人工工具
            state.dialog.extend(tool_msgs)

        for _ in range(MAX_INTERNAL_CALLS):
            if state.task is None and state.queue:
                self._start_task(state, state.queue.pop(0))
            if state.task is not None:
                out = self._worker_step(state)
            else:
                out = self._dispatcher_step(state)
            if out is not None:
                return self._emit(out, state), state
        msg = AssistantMessage(role="assistant", content="Sorry, could you please repeat your request?")
        state.dialog.append(msg)
        return self._emit(msg, state), state

    def _emit(self, msg: AssistantMessage, state: PlanState) -> AssistantMessage:
        for u in state.pending_usage:
            _merge_usage(msg, u)
        state.pending_usage = []
        return msg

    def _call(self, state: PlanState, system: str, messages: list, tools, name: str) -> AssistantMessage:
        return generate(model=self.llm, tools=tools, messages=[SystemMessage(role="system", content=system)] + messages,
                        call_name=name, **self.llm_args)

    # ---- 调度员 ----
    def _dispatcher_step(self, state: PlanState) -> Optional[AssistantMessage]:
        msg = self._call(state, self.dispatcher_prompt, state.dialog, self.dispatcher_tools, "dispatcher")
        calls = msg.tool_calls or []
        delegations = [c for c in calls if c.name == "delegate"]
        if delegations:
            msg.tool_calls, msg.content = delegations, None
            state.dialog.append(msg)
            state.pending_usage.append(msg)
            for c in delegations:
                worker = str((c.arguments or {}).get("specialist", "")).strip().lower()
                state.queue.append({"call_id": c.id, "worker": worker, "task": str((c.arguments or {}).get("task", ""))})
            return None
        if calls:  # 转人工：交给环境执行
            msg.tool_calls, msg.content = [c for c in calls if c.name == HUMAN][:1] or None, None
            if msg.tool_calls:
                state.dialog.append(msg)
                return msg
            msg.content = "Could you tell me a bit more about what you need?"
        state.dialog.append(msg)
        return msg

    # ---- Worker ----
    def _start_task(self, state: PlanState, job: dict) -> None:
        if job["worker"] not in WORKERS:
            self._finish_task(state, job, f"Unknown specialist '{job['worker']}'. Use account, pending or delivered.")
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

    def _worker_step(self, state: PlanState) -> Optional[AssistantMessage]:
        job = state.task
        worker = job["worker"]
        tools = self.worker_tools[worker] if state.task_steps < MAX_TASK_STEPS else None
        msg = self._call(state, self.worker_prompts[worker], state.task_msgs, tools, f"worker_{worker}")
        if msg.tool_calls:
            state.task_steps += 1
            state.task_msgs.append(msg)
            for c in msg.tool_calls:
                state.log.append(f"  {worker} called {c.name}({json.dumps(c.arguments, ensure_ascii=False)})")
            msg.content = None
            return msg  # 交给环境执行，结果会回到这个 Worker
        state.pending_usage.append(msg)
        self._finish_task(state, job, msg.content or "(empty report)")
        return None

    def _finish_task(self, state: PlanState, job: dict, report: str) -> None:
        state.log.append(f"[{job['worker']} report] {report}")
        state.dialog.append(ToolMessage(id=job["call_id"], role="tool", content=report))
        state.task, state.task_msgs, state.task_steps = None, [], 0


def create_plan_agent(tools, domain_policy, **kwargs):
    return PlanAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))
