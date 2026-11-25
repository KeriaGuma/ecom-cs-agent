"""降级链路（FAQ 兜底）与转人工交接（结构化摘要、规则降级、优先级队列）。"""
from langchain_core.messages import AIMessage, HumanMessage

from bench.multi_agent import build_multi_runner
from bench.multi_agent.workers import _degrade
from cs_agent import handoff
from tests.test_graph import WORKERS, ScriptedRouter
from tests.test_workers import ScriptedLLM, call


class BrokenRouter:
    def decide(self, state):
        raise TimeoutError("model down")


def _broken_worker(name):
    def run(state):
        return _degrade(state, name, TimeoutError("model down"))

    return run


def _ask(runner, text, thread="h", user="u001"):
    return runner.run_turn(thread, user, HumanMessage(text))


def test_router_down_policy_question_answered_by_faq(store):
    out = _ask(build_multi_runner(BrokenRouter(), store=store, workers=WORKERS), "退款一般几天到账")
    assert out["final_action"] == "faq"
    assert "1–3 个工作日" in out["messages"][-1].content
    assert store.open_tickets() == []


def test_router_down_action_request_goes_to_human_with_rule_summary(store):
    llm = ScriptedLLM(replies=[])
    runner = build_multi_runner(BrokenRouter(), llm=llm, store=store, workers=WORKERS)
    out = _ask(runner, "O1001 不想要了帮我退了")
    t = out["handoff_ticket"]
    assert out["final_action"] == "handoff" and out["degraded"]
    assert t["summary"]["summary_by"] == "rule"
    assert t["summary"]["order_ids"] == ["O1001"]
    assert f"#{t['ticket_id']}" in out["messages"][-1].content


def test_write_worker_failure_never_answered_by_faq(store):
    workers = {**WORKERS, "refund_agent": _broken_worker("refund_agent")}
    runner = build_multi_runner(ScriptedRouter(["refund_agent"]), store=store, workers=workers)
    out = _ask(runner, "退款一般几天到账")
    assert out["final_action"] == "handoff"


def test_complaint_gets_high_priority_and_jumps_queue(store):
    runner = build_multi_runner(ScriptedRouter(["handoff_human", "handoff_human"]), store=store, workers=WORKERS)
    calm = _ask(runner, "我想咨询企业采购", thread="a")["handoff_ticket"]
    angry = _ask(runner, "什么破服务，我要去 12315 投诉", thread="b")["handoff_ticket"]
    assert calm["priority"] == 3 and angry["priority"] == 1
    assert angry["queue_position"] == 0
    assert store.queue_position(calm["ticket_id"]) == 1
    assert [t["ticket_id"] for t in store.open_tickets()] == [angry["ticket_id"], calm["ticket_id"]]


def test_llm_summary_used_when_available_and_priority_set_by_code(store):
    summary = {"intent": "退款进度", "order_ids": ["O1002"], "emotion": "平静", "done": [],
               "pending": "退款审核太慢", "suggestion": "查询审批进度"}
    llm = ScriptedLLM(replies=[call("HandoffSummary", summary)])
    runner = build_multi_runner(ScriptedRouter(["handoff_human"]), llm=llm, store=store, workers=WORKERS)
    t = _ask(runner, "退款拖了一周了，我要投诉")["handoff_ticket"]
    assert t["summary"]["summary_by"] == "llm" and t["summary"]["intent"] == "退款进度"
    assert t["priority"] == 1


def test_summary_falls_back_to_rules_when_llm_fails():
    class Broken(ScriptedLLM):
        def _generate(self, *a, **k):
            raise TimeoutError

    state = {"messages": [HumanMessage("O1003 怎么还没到"), AIMessage("运输中")],
             "facts": {"order_id": "O1003", "recent_orders": ["O1003", "O1001"]},
             "turn_results": [], "user_id": "u001"}
    s = handoff.summarize(Broken(replies=[]), state, "用户要求人工")
    assert s["summary_by"] == "rule" and s["order_ids"] == ["O1001", "O1003"] and s["emotion"] == "不满"


def test_summary_falls_back_to_rules_when_llm_returns_none(store):
    class Empty(ScriptedLLM):
        replies: list = []

        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages, *a, **k):
            return None

    state = {"messages": [HumanMessage("O1003 怎么还没到"), AIMessage("运输中")], "facts": {"order_id": "O1003"},
             "turn_results": [], "user_id": "u001"}
    s = handoff.summarize(Empty(), state, "用户要求人工")
    assert s["summary_by"] == "rule" and s["order_ids"] == ["O1003"]
    t = handoff.create_ticket(store, Empty(), state, "用户要求人工")
    assert t["summary"]["summary_by"] == "rule" and t["priority"] == 2
    assert store.open_tickets()[-1]["ticket_id"] == t["ticket_id"]
