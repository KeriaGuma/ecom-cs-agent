"""主线单 Agent 引擎与端到端评测框架的离线测试（不需要 API Key）。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from cs_agent.agent import build_agent_runner
from cs_agent.config import settings
from cs_agent.eval_e2e import CONFIRM_WORKFLOW, check, paired, run_task, summarize
from cs_agent.kb import PolicyKB
from cs_agent.tracing import RunTrace
from tests.conftest import TODAY
from tests.test_workers import ScriptedLLM, call


def _single(llm, store, cfg=settings, refund_service=None):
    kb = PolicyKB(cfg.kb_dir)
    return build_agent_runner(llm, store=store, kb=kb, retriever=kb, today=lambda: TODAY,
                              settings=cfg, refund_service=refund_service)


def test_single_agent_transfer_stops_loop_and_hands_off(store):
    llm = ScriptedLLM(replies=[call("transfer_to_human", {"reason": "用户投诉"})])
    out = _single(llm, store).run_turn("s2", "u001", HumanMessage("我要投诉"))
    assert out["final_action"] == "handoff"
    assert out["handoff_ticket"]["reason"] == "用户投诉"


def test_single_agent_cannot_touch_other_users_order(store):
    llm = ScriptedLLM(replies=[call("submit_refund", {"order_id": "O2001", "reason_type": "no_reason"}),
                               AIMessage(content="未找到该订单")])
    _single(llm, store).run_turn("s3", "u001", HumanMessage("把 O2001 退了"))
    assert store.get_order("O2001")["status"] == "delivered"


def test_trace_sees_llm_and_tool_calls_inside_nodes(store):
    llm = ScriptedLLM(replies=[call("get_logistics", {"order_id": "O1003"}), AIMessage(content="运输中")])
    trace = RunTrace()
    _single(llm, store).run_turn("s4", "u001", HumanMessage("羽绒服到哪了"), callbacks=[trace])
    assert trace.llm_calls == 2
    assert trace.tool_names == ["get_logistics"]
    assert "中通" in trace.tools[0]["output"]


def test_run_task_scores_db_state_and_paths():
    task = {"id": "t", "turns": ["耳机退了"], "tags": ["write"],
            "expect": {"db": {"O1001": {"status": "refunded", "refund_amount": 199}}, "approval": False,
                       "tools_called": ["submit_refund"], "reply": [["退款成功", "已退"]]}}

    def factory(system, store, llm, router_llm):
        scripted = ScriptedLLM(replies=[call("submit_refund", {"order_id": "O1001", "reason_type": "no_reason"}),
                                        AIMessage(content="退款成功，199 元原路退回")])
        return _single(scripted, store)

    r = run_task("single", task, 0, None, None, factory=factory)
    assert r["success"], r["failures"]
    assert r["llm_calls"] == 2 and r["tools"] == ["submit_refund"]


@pytest.mark.parametrize(("decisions", "status", "approval"), [
    ([], "draft", False),
    (["cancel"], "cancelled", False),
    (["confirm"], "pending_approval", True),
    (["confirm", "reject"], "rejected", True),
    (["confirm", "approve"], "awaiting_return", True),
    (["confirm", "approve", "return", "receive"], "succeeded", True),
])
def test_run_task_confirmation_and_return_flow(decisions, status, approval):
    task = {
        "id": "confirm-refund",
        "refund_workflow": CONFIRM_WORKFLOW,
        "turns": [{"text": "扫地机器人不想要了", "refund_actions": [
            {"order_id": "O1002", "decision": decision} for decision in decisions
        ]}],
        "expect": {
            "db": {"O1002": {"status": "refunded", "refund_amount": 1899}} if status == "succeeded" else {},
            "approval": approval,
            "tools_called": ["prepare_refund"],
            "tools_not_called": ["submit_refund"],
            "refunds": [{"order_id": "O1002", "status": status}],
        },
    }

    def factory(system, store, llm, router_llm, *, refund_service):
        llm = ScriptedLLM(replies=[call("prepare_refund", {"order_id": "O1002", "reason_type": "no_reason"}),
                                  AIMessage(content="请核对确认单。")])
        return _single(llm, store, refund_service=refund_service)

    result = run_task("single", task, 0, None, None, factory=factory)
    assert result["success"], result["failures"]
    assert result["refund_workflow"] == CONFIRM_WORKFLOW


def test_check_reports_each_kind_of_failure():
    before = {"O1001": {"status": "delivered", "address": "A"}, "O1005": {"status": "paid", "address": "A"}}
    after = {"O1001": {"status": "delivered", "address": "A"}, "O1005": {"status": "paid", "address": "B"}}
    fails = check({"db": {"O1001": {"status": "refunded"}}, "approval": True, "tools_called": ["check_refund"],
                   "reply": [["买家"]]}, before, after, {"approval": False, "handoff": False}, [], "运费由商家承担")
    text = "\n".join(fails)
    assert "O1001.status" in text and "O1005 不应被修改" in text
    assert "应触发人工审批" in text and "没有调用 check_refund" in text and "缺少 买家" in text


def test_check_ignores_whitespace_in_addresses():
    before = {"O1005": {"address": "A"}}
    after = {"O1005": {"address": "杭州市滨江区网商路699号"}}
    assert check({"db": {"O1005": {"address": "杭州市滨江区网商路 699 号"}}}, before, after, {}, [], "") == []


def test_summary_pass_k_and_paired():
    def r(tid, ok, system="single"):
        return {"id": tid, "system": system, "success": ok, "tags": ["x"], "latency_ms": [1.0],
                "llm_calls": 1, "input_tokens": 1, "output_tokens": 1}

    a = summarize([r("t1", True), r("t1", True), r("t2", True), r("t2", False)])
    b = summarize([r("t1", False), r("t1", False), r("t2", True), r("t2", False)])
    assert a["success_rate"] == 0.75 and a["pass_k"] == 0.5
    p = paired(a, b)
    assert (p["a_better"], p["b_better"], p["tie"]) == (1, 0, 1)


def test_trace_records_draft_blocked_by_claim_check(store):
    llm = ScriptedLLM(replies=[call("get_order", {"order_id": "O1005"}), AIMessage(content="已为您取消成功"),
                               AIMessage(content="查到订单已退款，取消已完成")])
    trace = RunTrace()
    _single(llm, store).run_turn("c1", "u001", HumanMessage("键盘取消好了吗"), callbacks=[trace])
    assert settings.claim_check
    assert trace.claim_notes == ["已为您取消成功"]
