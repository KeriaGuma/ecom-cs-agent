"""多 Agent 对照基线的延迟消融验证：计划路由与单 Worker 直接回复开关。"""
from dataclasses import replace

from langchain_core.messages import HumanMessage

from bench.multi_agent import RouteDecision, build_multi_runner
from bench.multi_agent.router import normalize_plan
from cs_agent.config import settings
from tests.test_graph import WORKERS


class PlanRouter:
    def __init__(self, plan, fallback="FINISH"):
        self.plan, self.fallback, self.calls = list(plan), fallback, 0

    def decide(self, state):
        self.calls += 1
        if self.calls == 1:
            return RouteDecision(self.plan[0], reason="plan", plan=tuple(self.plan))
        return RouteDecision(self.fallback, reason="step")


def _run(router, store, text="查物流再问运费", workers=WORKERS, **cfg):
    runner = build_multi_runner(router, store=store, workers=workers, settings=replace(settings, **cfg))
    return runner, runner.run_turn("l", "u001", HumanMessage(text))


def test_plan_runs_all_steps_with_one_router_call(store):
    router = PlanRouter(["order_agent", "kb_agent"])
    _, out = _run(router, store)
    assert router.calls == 1
    assert [r["agent"] for r in out["turn_results"]] == ["order_agent", "kb_agent"]
    assert [r["next"] for r in out["route_log"]] == ["order_agent", "kb_agent", "FINISH"]
    assert "转运中心" in out["messages"][-1].content and "买家" in out["messages"][-1].content


def test_single_worker_result_is_sent_directly(store):
    router = PlanRouter(["order_agent"])
    _, out = _run(router, store, "羽绒服到哪了")
    assert router.calls == 1
    assert out["messages"][-1].content == "订单 O1003 已到达杭州转运中心"


def test_switches_off_restore_step_by_step_routing(store):
    router = PlanRouter(["order_agent", "kb_agent"], fallback="FINISH")
    _, out = _run(router, store, plan_routing=False, direct_reply=False)
    assert router.calls == 2
    assert [r["agent"] for r in out["turn_results"]] == ["order_agent"]


def test_normalize_plan():
    assert normalize_plan("order_agent", ["order_agent", "kb_agent", "FINISH", "kb_agent"]) == ("order_agent", "kb_agent")
    assert normalize_plan("kb_agent", []) == ("kb_agent",)
    assert normalize_plan("handoff_human", ["order_agent"]) == ("handoff_human",)
