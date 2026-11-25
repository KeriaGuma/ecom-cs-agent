"""ECom-Bench 上“我们的单 Agent”：和多 Agent（ecom_team.py）共用全部与架构无关的工程手段，只是不拆分角色。

用来区分多 Agent 的提升来自架构，还是来自提示词和代码兜底：
- 工作方式提示：多个诉求逐件处理、按规则先确认再办、缺信息就问、不编造（与多 Agent 前台和专员的要求对应）；
- “谎报完成”检查：本轮没有调用任何工具、回复却说“已办好”时，先不发出，提醒一次（与多 Agent 的检查相同，
  只是把“没有委派”换成“没有调用工具”）；
- 跨轮保留完整的工具调用和结果（多 Agent 靠团队工作记录做到这一点；ECom-Bench 自带的单 Agent 每轮只保留文字回复）。
工具是全部 21 个，政策是完整政策，模型和温度相同。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from bench.ecom.ecom_team import CLAIM
from bench.ecom.agent_loop import run_agent
from utils import LLM  # ECom-Bench 的 utils（模型由 bench/ecom/run.py 注入）

MAX_STEPS = 20  # 原图评测的 recursion_limit=40 折合 20 次模型调用

SINGLE = """

## 工作方式
1. 小主一次提出多件事时，先想清楚要办哪几件，逐件处理，不要遗漏。
2. 按规则需要先获得小主确认的操作：先把完整操作明细列给小主，小主明确确认、且确认的内容与要执行的完全一致后再执行。
3. 缺少信息时，向小主询问。
4. 只根据工具返回的结果回复，不编造；没有通过工具办成的事，不能说已经办成。
5. 回复遵守上面的回复要求。"""

CLAIM_NOTE = ("（系统提示：本轮你没有调用任何工具，不能对小主说已经办好。"
              "如果小主已经确认了某项操作，请先调用对应工具执行；如果你说的是之前已经办完的事，可以照常回复。）")


class SingleAgent(LLM):
    def __init__(self, agent_model: str, verbose=False, mcp_tools=(), temperature=0.3):
        super().__init__(model_name=agent_model, verbose=verbose, mcp_tools=list(mcp_tools), temperature=temperature)
        self.detail_messages: list[list] = []
        self.history: list = []  # 完整历史：用户消息、工具调用和结果、回复
        self.claim_checks = 0

    def load_system_prompt(self, system_prompt: str):
        self.system = system_prompt + SINGLE

    async def call(self, message: str) -> str:
        user_message = HumanMessage(content=message)
        msgs = [*self.history, user_message]
        reply, new = await run_agent(self.llm, self.system, msgs, self.mcp_tools, max_steps=MAX_STEPS)
        used_tools = any(isinstance(m, AIMessage) and m.tool_calls for m in new)
        if not used_tools and CLAIM.search(reply or ""):  # 没调工具却说办好了：先不发，提醒一次
            self.claim_checks += 1
            retry_in = [*msgs, *new, HumanMessage(content=CLAIM_NOTE)]
            reply, extra = await run_agent(self.llm, self.system, retry_in, self.mcp_tools, max_steps=MAX_STEPS)
            # 提醒只用于本轮重新决定，不写进历史：历史里保留用户消息和最终的处理过程
            new = new[:-1] + extra
        self.history += [user_message] + new
        self.detail_messages.append(new)
        return reply or "小主，麻烦您再说一下具体需要什么帮助呢？"


AGENTS = {"single": SingleAgent}
