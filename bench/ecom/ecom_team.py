"""ECom-Bench 上的“调度员 + Worker 汇报”多 Agent（结构与本项目 cs_agent、bench/tau2/team_agent.py 的转述确认版一致）。

- 只有调度员（前台）和用户对话。它没有业务工具，只能用 delegate 把任务委派给专员（以及转专员）；
- 专员（Worker）不和用户说话：用自己的工具办事，最后的文字就是交回前台的汇报；
- 需要用户确认的操作，专员先把明细写进汇报，由前台向用户确认；确认后前台再委派同一专员执行；
- 专员按工具分工；每位专员都拿完整政策（政策章节之间互相引用，比如取消订单后要用 E 卡工具退款）。

接口与 ECom-Bench 的 AgentLangChain 相同：load_system_prompt(政策) / async call(用户消息) -> 回复，
detail_messages 每轮追加本轮产生的全部消息（环境从中提取工具调用，用于“必要查询”评分）。
"""
from __future__ import annotations

import json
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from bench.ecom.agent_loop import run_agent
from utils import LLM  # ECom-Bench 的 utils（模型由 bench/ecom/run.py 注入）

# 每位专员都能查用户和订单：政策要求退货、换货、预约服务前先查询订单，而查询订单只能用 manage_order（查询和取消、
# 修改、下单是同一个工具）。所以 manage_order 也给售后和商品专员，但职责里写明只用来查询（开发集第 1 轮发现：
# 售后专员没有查询工具，为了“查状态”调用了 manage_return，直接提交了退货）。
BASIC = {"get_user_info_tool", "get_user_orders_info_tool", "manage_order_tool"}
QUERY_ONLY = "（manage_order 只用来查询订单，取消、修改订单和下单交给 order 专员）"
WORKERS = {
    "order": dict(role="订单与物流专员：查询订单和物流，取消、修改订单和下单，催单，开发票，以及订单相关的京东E卡支付和退款",
                  tools=BASIC | {"get_logistics_info_tool", "manage_urgent_tool",
                                 "manage_invoice_tool", "manage_ecard_tool"}),
    "aftersale": dict(role="售后与返现专员：退货、换货，京东E卡的查询、使用、退款和价保退款，晒单返现，识别小主发来的图片" + QUERY_ONLY,
                      tools=BASIC | {"manage_return_tool", "manage_exchange_tool", "manage_ecard_tool",
                                     "register_cashback_by_review_tool", "get_image_info_tool"}),
    "product": dict(role="商品与服务专员：商品详情和对比，优惠、赠品、配件，安装、故障码、维修，以及预约上门服务" + QUERY_ONLY,
                    tools=BASIC | {"get_product_info_tool", "compare_products_info_tool", "get_discount_info_tool",
                                   "get_gift_info_tool", "get_auxiliary_materials_info_tool",
                                   "get_installation_service_info_tool", "get_fault_code_info_tool",
                                   "get_repair_info_tool", "schedule_service_tool"}),
}
TRANSFER = "transfer_to_specialist_tool"
MAX_DISPATCH_STEPS = 8
MAX_WORKER_STEPS = 15  # 原图评测的 recursion_limit=30 折合 15 次模型调用
# 前台在本轮没有委派任何专员、回复里却说“已办好”（开发集第 2 轮发现：小主确认加急后，前台没有委派就回复“已加急”）
CLAIM = re.compile(r"已(?:经)?(?:为您|帮您|给您)?(?:成功)?(?:提交|办理|加急|取消|下单|退货|退款|换货|开具|开票|登记|预约|修改|扣款|设置|安排)"
                   r"|(?:办理|处理|提交|下单|退款|加急)(?:成功|完成)")
CLAIM_NOTE = ("（系统提示：本轮你没有委派任何专员，也没有专员汇报办成任何事，不能对小主说已经办好。"
              "如果小主已经确认了某项操作，请先委派对应专员执行；如果你说的是之前已经办完的事，可以照常回复。）")
LOG_RESULT_CHARS = 1500  # 工作记录里每条工具结果最多保留的字数（完整结果只在调用它的专员手里）


@tool
def delegate(specialist: str, task: str) -> str:
    """把一项任务委派给专员。专员用自己的工具查询或办理，然后向你汇报；专员不能和小主说话。

    Args:
        specialist: 专员名称：order、aftersale 或 product。
        task: 要专员做什么，写清楚需要的全部信息（订单号、商品链接或ID、小主的具体要求、图片链接、小主是否已明确确认等）。
    """
    return "report"


DISPATCHER = """

## 你在团队中的角色
你是前台客服，是唯一和小主对话的人。你没有业务工具，所有查询和办理都由专员完成，你用 delegate 工具委派：
{team}

工作方式：
1. 小主一次提出多件事时，先想清楚要办哪几件，再分别委派；互不相关的事可以在同一步里委派给不同专员。
2. 委派时写清楚任务和需要的全部信息（订单号、商品链接或ID、小主的原话要求、图片链接等）。专员能看到对话记录。
3. 专员汇报需要小主确认的操作时，把操作明细完整列给小主，请小主明确回复确认。小主确认后，再委派同一位专员执行，写明“小主已明确确认”和完整明细。
4. 专员汇报需要补充信息时，向小主询问。
5. 只根据专员的汇报回复，不编造；专员没有汇报办成的事，不能说已经办成。
6. 回复遵守上面的回复要求。"""

WORKER = """

## 你在团队中的角色
你是{role}。你在一个客服团队里工作：前台客服负责和小主对话，并把具体任务交给你。你不能直接和小主说话。
- 用你的工具完成任务，严格遵守上面的规则。
- 按规则需要先获得小主确认的操作：只有任务里写明“小主已明确确认”、且确认的内容与要执行的完全一致时才执行；否则不要执行，把将要执行的完整操作明细写进汇报，由前台向小主确认。
- 缺少信息时，在汇报里写明需要向小主问什么。
- 超出你职责的事，写明应由哪位同事处理：{others}。
- 完成后，用一段文字向前台汇报：结论，以及小主需要知道的全部事实（订单号、商品、状态、金额、支付方式、物流等）。前台会据此回复小主。"""


class TeamAgent(LLM):
    def __init__(self, agent_model: str, verbose=False, mcp_tools=(), temperature=0.3):
        super().__init__(model_name=agent_model, verbose=verbose, mcp_tools=list(mcp_tools), temperature=temperature)
        by_name = {t.name: t for t in self.mcp_tools}
        self.workers = {n: [by_name[x] for x in sorted(w["tools"]) if x in by_name] for n, w in WORKERS.items()}
        self.transfer = by_name.get(TRANSFER)
        self.dispatcher = self.llm.bind_tools([delegate] + ([self.transfer] if self.transfer else []))
        self.detail_messages: list[list] = []
        self.dialog: list = []   # 用户和前台之间的对话
        self.log: list[str] = []  # 团队工作记录
        self.routes: list[str] = []
        self.claim_checks = 0

    def load_system_prompt(self, system_prompt: str):
        policy = system_prompt
        team = "\n".join(f"- {n}：{w['role'].split('：', 1)[1]}" for n, w in WORKERS.items())
        self.dispatcher_prompt = policy + DISPATCHER.format(team=team)
        self.worker_prompts = {
            n: policy + WORKER.format(role=w["role"], others="；".join(
                f"{o}（{ow['role'].split('：')[0]}）" for o, ow in WORKERS.items() if o != n))
            for n, w in WORKERS.items()}

    async def call(self, message: str) -> str:
        self.dialog.append(HumanMessage(content=message))
        msgs = [SystemMessage(content=self.dispatcher_prompt)] + self.dialog
        produced: list = []
        reply = ""
        delegated = checked = False
        for _ in range(MAX_DISPATCH_STEPS):
            ai = await self.dispatcher.ainvoke(msgs)
            produced.append(ai)
            if not ai.tool_calls:
                reply = ai.content or ""
                if not delegated and not checked and CLAIM.search(reply):  # 没人办却说办好了：先不发，让前台重新决定
                    checked = True
                    self.claim_checks += 1
                    msgs += [ai, HumanMessage(content=CLAIM_NOTE)]
                    continue
                break
            delegated = delegated or any(tc["name"] != TRANSFER for tc in ai.tool_calls)
            msgs.append(ai)
            for tc in ai.tool_calls:
                if tc["name"] == "delegate":
                    out, worker_msgs = await self._run_worker(tc["args"].get("specialist", ""), tc["args"].get("task", ""))
                    produced += worker_msgs
                elif self.transfer is not None and tc["name"] == TRANSFER:
                    out = str(await self.transfer.ainvoke(tc["args"]))
                    self.log.append(f"[前台转专员] {json.dumps(tc['args'], ensure_ascii=False)} -> {out}")
                else:
                    out = f"没有工具 {tc['name']}。"
                msgs.append(ToolMessage(content=out, tool_call_id=tc["id"]))
        if not reply:
            reply = "小主，麻烦您再说一下具体需要什么帮助呢？"
        self.dialog.append(AIMessage(content=reply))
        self.detail_messages.append(produced)
        self._debug_dump()
        return reply

    def _debug_dump(self):
        """设置 ECOM_DEBUG_DIR 时，把对话和团队工作记录写到文件里（只用于开发集排查，不影响运行）。"""
        d = os.getenv("ECOM_DEBUG_DIR")
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        uid = (re.search(r"user_id是(\S+?)。", self.dispatcher_prompt) or [None, "unknown"])[1]
        with open(os.path.join(d, f"{uid}_{id(self)}.json"), "w", encoding="utf-8") as f:
            json.dump({"user_id": uid, "dialog": [(type(m).__name__, m.content) for m in self.dialog],
                       "log": self.log, "routes": self.routes,
                       "claim_checks": self.claim_checks}, f, ensure_ascii=False, indent=1)

    async def _run_worker(self, name: str, task: str) -> tuple[str, list]:
        name = str(name).strip().lower()
        if name not in self.workers:
            return f"没有专员 {name!r}，可选：{', '.join(self.workers)}。", []
        self.routes.append(name)
        dialog = "\n".join(f"{'小主' if isinstance(m, HumanMessage) else '前台'}：{m.content}" for m in self.dialog)
        brief = (f"## 前台和小主到目前为止的对话\n{dialog}\n\n"
                 f"## 团队工作记录（之前的委派、工具调用、结果和汇报）\n{chr(10).join(self.log) or '（无）'}\n\n"
                 f"## 前台交给你的任务\n{task}")
        self.log.append(f"[前台 -> {name}] {task}")
        try:
            answer, new = await run_agent(self.llm, self.worker_prompts[name], [HumanMessage(content=brief)],
                                          self.workers[name], max_steps=MAX_WORKER_STEPS)
        except Exception as e:  # 专员出错：如实汇报给前台
            report = f"专员处理出错：{type(e).__name__}: {e}"
            self.log.append(f"[{name} 汇报] {report}")
            return report, []
        for m in new:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    self.log.append(f"  {name} 调用 {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
            elif isinstance(m, ToolMessage):
                self.log.append(f"  结果：{str(m.content)[:LOG_RESULT_CHARS]}")
        report = answer or "（专员没有汇报内容）"
        self.log.append(f"[{name} 汇报] {report}")
        return report, new


AGENTS = {"team": TeamAgent}
