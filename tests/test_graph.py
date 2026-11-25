"""验证会话编排器与 `bench/multi_agent` 对照基线的多步分派、状态重置、跳转上限与降级。"""
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from bench.multi_agent import RouteDecision, build_multi_runner
from cs_agent.config import settings
from cs_agent.state import TurnStep


class ScriptedRouter:
    """按顺序返回预设的调度决策，并记录每次调度时看到的 State。"""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.seen = []

    def decide(self, state):
        self.seen.append({"turn_results": list(state.get("turn_results") or [])})
        return RouteDecision(self.decisions.pop(0), reason="scripted")


def fake_worker(agent, content, goto="supervisor", extra=None):
    def run(state):
        results = list(state.get("turn_results") or []) + [{"agent": agent, "content": content}]
        return TurnStep(goto=goto, update={"turn_results": results, **(extra or {})})

    return run


WORKERS = {
    "order_agent": fake_worker("order_agent", "订单 O1003 已到达杭州转运中心"),
    "kb_agent": fake_worker("kb_agent", "无理由退货运费由买家承担"),
    "refund_agent": fake_worker("refund_agent", "已生成退货退款确认单"),
}


def _run(runner, text, thread="t1", user="u001"):
    return runner.run_turn(thread, user, HumanMessage(text))


def test_multi_intent_dispatches_in_sequence(store):
    router = ScriptedRouter(["order_agent", "kb_agent", "FINISH"])
    runner = build_multi_runner(router, store=store, workers=WORKERS)

    out = _run(runner, "查下 O1003 的物流，顺便问下无理由退货运费谁出")

    assert [r["agent"] for r in out["turn_results"]] == ["order_agent", "kb_agent"]
    assert router.seen[1]["turn_results"][0]["agent"] == "order_agent"
    assert isinstance(out["messages"][-1], AIMessage)
    assert "转运中心" in out["messages"][-1].content


def test_turn_state_resets_between_turns(store):
    router = ScriptedRouter(["order_agent", "FINISH", "kb_agent", "FINISH"])
    runner = build_multi_runner(router, store=store, workers=WORKERS)

    _run(runner, "查物流")
    out = _run(runner, "运费谁出")

    assert [r["agent"] for r in out["turn_results"]] == ["kb_agent"]
    assert len([m for m in out["messages"] if isinstance(m, HumanMessage)]) == 2


def test_max_hops_hands_off_to_human(store):
    router = ScriptedRouter(["kb_agent"] * 10)
    cfg = replace(settings, max_hops=3)
    runner = build_multi_runner(router, store=store, workers=WORKERS, settings=cfg)

    out = _run(runner, "随便问问")

    assert len(out["turn_results"]) == 3
    assert out["handoff_ticket"]["reason"].startswith("超过最大分派次数")
    assert "人工" in out["messages"][-1].content


def test_router_error_degrades_to_human(store):
    class BrokenRouter:
        def decide(self, state):
            raise TimeoutError("model timeout")

    runner = build_multi_runner(BrokenRouter(), store=store, workers=WORKERS)
    out = _run(runner, "你好")
    assert "调度失败" in out["handoff_ticket"]["reason"]
