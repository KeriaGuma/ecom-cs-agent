"""系统不支持的附加要求（加备注、催快递、换快递等）：不误转人工，能办的照常办，明确告知办不了的事项。"""
import os
import subprocess
import sys
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from bench.multi_agent import LLMRouter, RouteDecision, build_multi_runner, build_supervisor_prompt
from bench.multi_agent.router import WORKERS as WORKER_NAMES
from bench.multi_agent.router import RouteOutputV2
from bench.multi_agent.supervisor import UNSUPPORTED_SECTION
from cs_agent.agent import unsupported_features
from cs_agent.config import settings
from tests.test_graph import WORKERS
from tests.test_workers import ScriptedLLM


class OnceRouter:
    def __init__(self, nxt, plan=(), unsupported=()):
        self.decision = RouteDecision(nxt, reason="plan", plan=tuple(plan) or (nxt,), unsupported=tuple(unsupported))
        self.calls = 0

    def decide(self, state):
        self.calls += 1
        return self.decision if self.calls == 1 else RouteDecision("FINISH", reason="step")


class RecordingLLM(ScriptedLLM):
    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


def _run(router, store, text, llm=None):
    runner = build_multi_runner(router, llm=llm, store=store, workers=WORKERS, settings=settings)
    return runner.run_turn("u", "u001", HumanMessage(text))


def test_single_worker_with_unsupported_goes_through_respond(store):
    out = _run(OnceRouter("order_agent", unsupported=["催快递"]), store, "羽绒服到哪了，帮我催一下")
    assert out["final_action"] == "finish"
    reply = out["messages"][-1].content
    assert "订单 O1003 已到达杭州转运中心" in reply and "催快递这里办不了" in reply


def test_respond_prompt_lists_unsupported_items(store):
    llm = RecordingLLM(replies=[AIMessage(content="已查到物流；加备注这里办不了。")], seen=[])
    out = _run(OnceRouter("order_agent", unsupported=["加备注"]), store, "改地址，再加个备注", llm=llm)
    system, human = llm.seen[-1]
    assert "【系统办不了的要求】时，逐项明确告诉用户这里办不了" in system.content
    assert f"{UNSUPPORTED_SECTION}加备注" in human.content
    assert out["messages"][-1].content == "已查到物流；加备注这里办不了。"


def test_without_unsupported_single_worker_is_still_sent_directly(store):
    out = _run(OnceRouter("order_agent"), store, "羽绒服到哪了")
    assert out["messages"][-1].content == "订单 O1003 已到达杭州转运中心"


def test_only_unsupported_request_finishes_with_explanation(store):
    out = _run(OnceRouter("FINISH", unsupported=["换快递公司"]), store, "给我换成顺丰发")
    assert out["final_action"] == "finish" and not out.get("handoff_ticket")
    assert "换快递公司这里办不了" in out["messages"][-1].content


def test_unsupported_is_cleared_next_turn(store):
    router = OnceRouter("order_agent", unsupported=["加备注"])
    runner = build_multi_runner(router, store=store, workers=WORKERS, settings=settings)
    runner.run_turn("c", "u001", HumanMessage("改地址再加备注"))
    router.calls, router.decision = 0, RouteDecision("order_agent", reason="plan", plan=("order_agent",))
    out = runner.run_turn("c", "u001", HumanMessage("羽绒服到哪了"))
    assert out["unsupported"] == [] and out["messages"][-1].content == "订单 O1003 已到达杭州转运中心"


class _Chain:
    def __init__(self, out):
        self.out = out

    def invoke(self, messages):
        return self.out


def _decide(out):
    router = LLMRouter.__new__(LLMRouter)
    router.workers, router.prompt, router._chain = WORKER_NAMES, "x", _Chain(out)
    return router.decide({"messages": [HumanMessage("x")]})


def test_router_parses_unsupported_and_drops_it_on_handoff():
    d = _decide(RouteOutputV2(next="order_agent", plan=["order_agent"], reason="r", unsupported=[" 加备注 ", ""]))
    assert d.unsupported == ("加备注",)
    d = _decide(RouteOutputV2(next="handoff_human", plan=[], reason="要人工", unsupported=["加备注"]))
    assert d.unsupported == ()


def test_feature_list_follows_open_domains():
    assert "开发票或改发票" in unsupported_features() and "开发票或改发票" not in unsupported_features(("invoice",))
    prompt = build_supervisor_prompt(("invoice", "points", "coupon"))
    rule = prompt[prompt.index("7. "):]
    assert "开发票或改发票" not in rule
    assert "发优惠券" not in rule and "发积分" not in rule and "催快递" in rule


def test_prompts_and_schema_toggle_when_rule_off():
    code = (
        "from bench.multi_agent.router import SUPERVISOR_PROMPT, SUPERVISOR_PROMPT_V1;"
        "from bench.multi_agent.supervisor import RESPOND_PROMPT;"
        "from cs_agent.agent import SINGLE_PROMPT;"
        "print(int(SUPERVISOR_PROMPT == SUPERVISOR_PROMPT_V1), int('系统办不了' in RESPOND_PROMPT),"
        " int('系统没有的功能' in SINGLE_PROMPT))"
    )
    out_off = subprocess.run([sys.executable, "-c", code], env={**os.environ, "UNSUPPORTED_RULE": "0"},
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    out_on = subprocess.run([sys.executable, "-c", code], env={**os.environ, "UNSUPPORTED_RULE": "1"},
                            capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    assert out_off == "1 0 0"
    assert out_on == "0 1 1"


def test_explicit_handoff_rules_are_kept():
    prompt = build_supervisor_prompt()
    assert "投诉快递员" in prompt and "仍按 handoff_human 处理" in prompt
    assert replace(settings).unsupported_rule
