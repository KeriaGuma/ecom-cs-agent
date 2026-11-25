import json

from cs_agent.config import settings
from cs_agent.kb import PolicyKB
from cs_agent.refunds import RefundService
from cs_agent.tools import NOT_FOUND, make_kb_tools, make_order_tools, make_refund_tools
from tests.conftest import TODAY


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


def test_cannot_read_other_users_order(store):
    tools = make_order_tools(store, "u001")
    assert _tool(tools, "get_order").invoke({"order_id": "O2001"}) == NOT_FOUND
    assert json.loads(_tool(tools, "get_order").invoke({"order_id": "o1001"}))["item"] == "蓝牙耳机"


def test_change_address_only_before_shipping(store):
    tools = make_order_tools(store, "u001")
    change = _tool(tools, "change_address")
    assert "无法修改" in change.invoke({"order_id": "O1003", "new_address": "新地址"})
    assert "已将订单" in change.invoke({"order_id": "O1005", "new_address": "新地址"})
    assert store.get_order("O1005")["address"] == "新地址"


def test_small_refund_compat_tool_executes_directly(store):
    pending = {}
    tools = make_refund_tools(store, "u001", 500, pending.update, today=lambda: TODAY)
    out = _tool(tools, "submit_refund").invoke({"order_id": "O1001", "reason_type": "no_reason"})
    assert "退款成功" in out and not pending
    assert store.get_order("O1001")["status"] == "refunded"


def test_large_refund_goes_to_pending_not_executed(store):
    pending = {}
    tools = make_refund_tools(store, "u001", 500, pending.update, today=lambda: TODAY)
    out = _tool(tools, "submit_refund").invoke({"order_id": "O1002", "reason_type": "no_reason"})
    assert "人工审核" in out
    assert pending["order_id"] == "O1002" and pending["amount"] == 1899.0
    assert store.get_order("O1002")["status"] == "delivered"


def test_kb_search_finds_policy():
    out = _tool(make_kb_tools(PolicyKB(settings.kb_dir)), "search_policy").invoke({"query": "七天无理由退货运费谁出"})
    assert "七天无理由退货" in out


def test_check_refund_unknown_reason_returns_both_outcomes_and_refund_type(store):
    tools = make_refund_tools(store, "u001", 500, lambda d: None, today=lambda: TODAY, unknown_reason=True)
    out = json.loads(_tool(tools, "check_refund").invoke({"order_id": "O1004", "reason_type": "unknown"}))
    assert set(out) == {"无理由", "质量问题", "提示"}
    assert out["无理由"]["eligible"] is False
    assert store.get_order("O1004")["status"] == "delivered"

    valid = json.loads(_tool(tools, "check_refund").invoke({"order_id": "O1001", "reason_type": "no_reason"}))
    assert valid["refund_type"] == "return_refund"
    unshipped = json.loads(_tool(tools, "check_refund").invoke({"order_id": "O1005", "reason_type": "no_reason"}))
    assert unshipped["refund_type"] == "refund_only"


def test_old_refund_rule_keeps_two_reasons(store):
    tools = make_refund_tools(store, "u001", 500, lambda d: None, today=lambda: TODAY, unknown_reason=False)
    assert _tool(tools, "check_refund").args["reason_type"]["enum"] == ["no_reason", "quality"]


def test_refund_decision_shared_by_single_and_worker():
    from bench.multi_agent.workers import REFUND_PROMPT
    from cs_agent.agent import REFUND_DECISION, SINGLE_PROMPT

    expected = REFUND_DECISION.replace("是否发起退款申请", "是否调用 submit_refund").replace("直接调用退款工具", "直接提交")
    if settings.refund_rule == 2:
        assert expected in REFUND_PROMPT
        assert expected.replace("\n", "\n  ") in SINGLE_PROMPT


def test_rejected_and_awaiting_return_refunds_leave_records_visible_to_query_tools(store):
    refunds = RefundService(store, 500, today=lambda: TODAY)
    rid = refunds.prepare(user_id="u001", thread_id="u001:t", message_id="m1", order_id="O1002", reason="no_reason")["refund_id"]
    refunds.confirm(rid, "u001", True)
    tools = {t.name: t for t in make_refund_tools(store, "u001", 500, today=lambda: TODAY)}
    assert "等待人工审核" in tools["get_order"].invoke({"order_id": "O1002"})

    refunds.decide(rid, "a01", False, "已超范围")
    assert store.get_order("O1002")["status"] == "delivered"
    order = json.loads(tools["get_order"].invoke({"order_id": "O1002"}))
    assert order["操作记录"][-1]["结果"].startswith("人工审核驳回") and order["操作记录"][-1]["金额"] == 1899

    rid2 = refunds.prepare(user_id="u001", thread_id="u001:t", message_id="m2", order_id="O1001", reason="no_reason")["refund_id"]
    refunds.confirm(rid2, "u001", True)
    order1 = json.loads(tools["get_order"].invoke({"order_id": "O1001"}))
    assert "等待买家寄回商品" in order1["操作记录"][-1]["结果"]


def test_auto_refund_and_address_change_are_recorded(store):
    tools = {t.name: t for t in make_refund_tools(store, "u001", 500, today=lambda: TODAY)}
    tools["submit_refund"].invoke({"order_id": "O1001", "reason_type": "no_reason"})
    assert "退款成功" in json.loads(tools["get_order"].invoke({"order_id": "O1001"}))["操作记录"][0]["结果"]
    order_tools = {t.name: t for t in make_order_tools(store, "u001", ops_log=True)}
    order_tools["change_address"].invoke({"order_id": "O1005", "new_address": "杭州市滨江区网商路 699 号"})
    rec = json.loads(order_tools["get_order"].invoke({"order_id": "O1005"}))["操作记录"][0]
    assert rec["操作"] == "修改收货地址" and rec["新地址"] == "杭州市滨江区网商路 699 号"


def test_ops_log_off_keeps_old_tool_output(store):
    tools = {t.name: t for t in make_refund_tools(store, "u001", 500, today=lambda: TODAY, ops_log=False)}
    tools["submit_refund"].invoke({"order_id": "O1001", "reason_type": "no_reason"})
    assert "操作记录" not in tools["get_order"].invoke({"order_id": "O1001"})
    assert "申请记录" not in tools["check_refund"].invoke({"order_id": "O1002", "reason_type": "no_reason"})


def test_orders_without_operations_have_no_record_field(store):
    tools = {t.name: t for t in make_refund_tools(store, "u001", 500, today=lambda: TODAY)}
    assert "操作记录" not in tools["get_order"].invoke({"order_id": "O1003"})
