"""验证主线单 Agent 与对照 Worker 的 ReAct 循环、多订单关键事实提取、越权隔离与异常降级。"""
import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from bench.multi_agent import build_multi_runner, build_workers
from cs_agent.agent import build_agent_runner
from cs_agent.config import settings
from cs_agent.kb import PolicyKB
from cs_agent.refunds import RefundService
from tests.conftest import TODAY
from tests.test_graph import ScriptedRouter


class ScriptedLLM(BaseChatModel):
    """按顺序返回预设 AIMessage，同时支持同步 `_generate` 与异步流式 `_astream`。"""

    replies: list[Any]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        msg: AIMessage = self.replies.pop(0)
        if msg.tool_calls:
            tc_chunks = [
                {"name": tc["name"], "args": json.dumps(tc["args"], ensure_ascii=False),
                 "id": tc["id"], "index": idx}
                for idx, tc in enumerate(msg.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content=str(msg.content or ""), tool_call_chunks=tc_chunks))
            return
        text = str(msg.content or "")
        if len(text) <= 2:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            return
        mid = len(text) // 2
        yield ChatGenerationChunk(message=AIMessageChunk(content=text[:mid]))
        yield ChatGenerationChunk(message=AIMessageChunk(content=text[mid:]))


def call(name, args, i=0):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def test_agent_extracts_multi_order_facts_and_creates_refund_draft(store):
    refunds = RefundService(store, 500, today=lambda: TODAY)
    llm = ScriptedLLM(replies=[
        call("get_logistics", {"order_id": "O1003"}, 0),
        call("check_refund", {"order_id": "O1002", "reason_type": "no_reason"}, 1),
        call("prepare_refund", {"order_id": "O1002", "reason_type": "no_reason"}, 2),
        AIMessage(content="已生成确认单"),
    ])
    runner = build_agent_runner(llm, store=store, retriever=PolicyKB(settings.kb_dir),
                                today=lambda: TODAY, refund_service=refunds)

    runner.run_turn("w1", "u001", HumanMessage("查下 O1003，另外 O1002 不想要了帮我退了"))
    snap = runner.snapshot("w1")
    assert snap["facts"]["order_id"] == "O1002"
    assert snap["facts"]["recent_orders"] == ["O1002", "O1003"]
    assert store.get_order("O1002")["status"] == "delivered"
    drafts = refunds.for_thread("w1", "u001")
    assert len(drafts) == 1 and drafts[0]["status"] == "draft"


def test_worker_cannot_touch_other_users_order(store):
    llm = ScriptedLLM(replies=[
        call("submit_refund", {"order_id": "O2001", "reason_type": "no_reason"}),
        AIMessage(content="未找到该订单"),
    ])
    workers = build_workers(llm, store, PolicyKB(settings.kb_dir), settings, today=lambda: TODAY)
    runner = build_multi_runner(ScriptedRouter(["refund_agent", "FINISH"]), store=store, workers=workers)
    out = runner.run_turn("w2", "u001", HumanMessage("把 O2001 退了"))
    assert store.get_order("O2001")["status"] == "delivered"
    assert "order_id" not in out["facts"]


def test_worker_error_degrades_to_human(store):
    class BrokenLLM(ScriptedLLM):
        def _generate(self, *a, **k):
            raise TimeoutError("timeout")

    workers = build_workers(BrokenLLM(replies=[]), store, PolicyKB(settings.kb_dir), settings)
    runner = build_multi_runner(ScriptedRouter(["order_agent"]), store=store, workers=workers)
    out = runner.run_turn("w3", "u001", HumanMessage("查物流"))
    assert "order_agent 执行失败" in out["handoff_ticket"]["reason"]
