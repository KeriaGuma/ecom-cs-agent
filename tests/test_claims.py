"""防“谎报完成”代码守卫：覆盖 `prepare_refund` / `submit_refund` / `change_address` 及条件分句免误伤。"""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from cs_agent.claims import claims_done, unbacked_claim
from cs_agent.react import run_react
from tests.test_workers import ScriptedLLM, call


@tool
def prepare_refund(order_id: str) -> str:
    """生成退款确认单。"""
    return f"{order_id} 确认单草稿已创建"


@tool
def search_policy(query: str) -> str:
    """查政策。"""
    return "支付宝 1-3 个工作日到账"


def test_claims_detection_and_prepare_refund_backing():
    assert claims_done("已为您提交退款申请") and claims_done("地址修改成功")
    assert claims_done("已为您生成退款确认单，请核对") and claims_done("退款确认单已创建")
    assert not claims_done("没能提交成功") and not claims_done("退款到账一般 1-3 天")
    assert not claims_done("如果您已经提交过申请，请等待审核")
    assert unbacked_claim("已为您生成退款确认单", [{"tool": "search_policy"}])
    assert not unbacked_claim("已为您生成退款确认单", [{"tool": "prepare_refund"}])
    assert not unbacked_claim("已为您提交退款申请", [{"tool": "submit_refund"}])


def test_unbacked_claim_gets_one_retry_and_then_tool_is_called():
    llm = ScriptedLLM(replies=[
        call("search_policy", {"query": "到账时间"}, 0),
        AIMessage(content="已为您生成退款确认单，支付宝 1-3 天到账"),   # 谎报：未调 prepare_refund
        call("prepare_refund", {"order_id": "O1001"}, 1),               # 被提醒后真正执行
        AIMessage(content="已为您生成退款确认单，请确认；支付宝 1-3 天到账"),
    ])
    res = run_react(llm, "sys", [HumanMessage("退了，几天到账")], [prepare_refund, search_policy], claim_check=True)
    assert res.claim_retries == 1 and [c["tool"] for c in res.tool_calls] == ["search_policy", "prepare_refund"]


def test_retry_happens_at_most_once():
    llm = ScriptedLLM(replies=[AIMessage(content="已为您提交退款申请"), AIMessage(content="您的退款已提交")])
    res = run_react(llm, "sys", [HumanMessage("之前那单退款怎么样了")], [prepare_refund], claim_check=True)
    assert res.claim_retries == 1 and res.answer == "您的退款已提交" and res.tool_calls == []


def test_disabled_or_backed_claims_are_not_touched():
    llm = ScriptedLLM(replies=[AIMessage(content="已为您提交退款申请")])
    res = run_react(llm, "sys", [HumanMessage("退了")], [prepare_refund], claim_check=False)
    assert res.claim_retries == 0 and res.answer == "已为您提交退款申请"
    llm = ScriptedLLM(replies=[call("prepare_refund", {"order_id": "O1001"}), AIMessage(content="已为您生成确认单")])
    res = run_react(llm, "sys", [HumanMessage("退了")], [prepare_refund], claim_check=True)
    assert res.claim_retries == 0
