"""三层记忆：代码提取关键事实 + 异步/同步滚动摘要 + 上下文裁剪 + 完整脱敏存档。"""
from dataclasses import replace

from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from cs_agent.agent import build_agent_runner
from cs_agent.api import create_app
from cs_agent.config import settings
from cs_agent.kb import PolicyKB
from cs_agent.state import RECENT_MESSAGES, render_state


class FakeLLM(BaseChatModel):
    fail_summary: bool = False
    summary_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _reply_text(self, messages) -> str:
        if "滚动摘要" in str(messages[0].content):
            self.summary_calls += 1
            if self.fail_summary:
                raise TimeoutError
            return f"摘要#{self.summary_calls}：用户说耳机右耳没声音"
        return "好的"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply_text(messages)))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply_text(messages)))])


def _runner(store, llm, **cfg):
    kb = PolicyKB(settings.kb_dir)
    return build_agent_runner(llm, store=store, kb=kb, retriever=kb, settings=replace(settings, **cfg))


def _chat(runner, n, thread="m"):
    out = None
    for i in range(n):
        out = runner.run_turn(thread, "u001", HumanMessage(f"第{i + 1}句"))
    return out


def test_old_messages_summarized_and_trimmed_but_archived(store):
    llm = FakeLLM()
    out = _chat(_runner(store, llm), 8)
    assert llm.summary_calls >= 1
    assert out["summary"].startswith("摘要#")
    assert len(out["messages"]) <= RECENT_MESSAGES + 6
    assert out["messages"][-2].content == "第8句"
    archived = store.transcript("m")
    assert len(archived) == 16 and archived[0] == {"role": "user", "content": "第1句"}
    assert "【较早对话摘要】摘要#" in render_state(out)


def test_summary_failure_keeps_messages(store):
    out = _chat(_runner(store, FakeLLM(fail_summary=True)), 8)
    assert not out.get("summary")
    assert len(out["messages"]) == 16


def test_memory_summary_can_be_disabled(store):
    llm = FakeLLM()
    out = _chat(_runner(store, llm, memory_summary=False), 8)
    assert llm.summary_calls == 0 and len(out["messages"]) == 16
    assert len(store.transcript("m")) == 16


def test_async_chat_endpoint_compacts_context_and_keeps_full_archive(store):
    llm = FakeLLM()
    runner = _runner(store, llm)
    c = TestClient(create_app(runner=runner, store=store))
    for i in range(8):
        c.post("/chat", json={"session_id": "s", "message": f"第{i + 1}句"}, headers={"X-User-Id": "u001"})
    hist = c.get("/sessions/s/messages", headers={"X-User-Id": "u001"}).json()["messages"]
    assert len(hist) == 16 and hist[0]["content"] == "第1句"
    values = runner.snapshot("u001:s")
    assert len(values["messages"]) < 16
    assert values["summary"].startswith("摘要#")


def test_messages_outside_window_but_not_yet_summarized_stay_visible():
    msgs = []
    for i in range(1, 6):
        msgs += [HumanMessage(f"第{i}轮" + ("：新地址是莫干山路 100 号" if i == 2 else "")), AIMessage("好的")]
    msgs.append(HumanMessage("把键盘改到我刚才说的新地址"))
    assert "莫干山路 100 号" in render_state({"messages": msgs})
