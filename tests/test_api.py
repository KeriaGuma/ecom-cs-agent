"""HTTP 网关测试：原生异步真流式 SSE、防谎报缓冲、会话锁零泄漏回收、身份隔离、退货/退款与工单接口。"""
import json

from fastapi import Header, HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from cs_agent.agent import build_agent_runner
from cs_agent.api import create_app
from cs_agent.config import settings
from cs_agent.kb import PolicyKB
from cs_agent.refunds import RefundService
from tests.conftest import TODAY
from tests.test_workers import ScriptedLLM, call

U1 = {"X-User-Id": "u001"}
STAFF = {"X-Staff-Id": "s01"}


def _events(resp) -> list[tuple[str, dict]]:
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _client(store, llm, refund_service=None):
    refunds = refund_service or RefundService(store, 500, today=lambda: TODAY)
    kb = PolicyKB(settings.kb_dir)
    runner = build_agent_runner(llm, store=store, kb=kb, retriever=kb, today=lambda: TODAY, refund_service=refunds)
    app = create_app(runner=runner, store=store, refund_service=refunds)
    return TestClient(app), app


def test_chat_streams_status_tokens_then_final_message_and_cleans_locks(store):
    llm = ScriptedLLM(replies=[
        call("get_logistics", {"order_id": "O1003"}),
        AIMessage(content="您的羽绒服已到达杭州转运中心，预计明天送达"),
    ])
    c, app = _client(store, llm)
    evs = _events(c.post("/chat", json={"session_id": "s1", "message": "羽绒服到哪了"}, headers=U1))
    kinds = [e for e, _ in evs]
    assert kinds[0] == "status" and evs[0][1]["stage"] == "order_agent"
    assert kinds.count("token") >= 2  # 单 Agent 最终回复实现真流式分段推送
    assert "".join(d["text"] for e, d in evs if e == "token") == "您的羽绒服已到达杭州转运中心，预计明天送达"
    assert kinds[-2:] == ["message", "done"] and evs[-1][1]["final_action"] == "finish"
    # 请求结束后，引用计数锁必须自动回收，零驻留泄漏
    assert app.state.session_locks.active_keys == 0

    hist = c.get("/sessions/s1/messages", headers=U1).json()["messages"]
    assert [m["role"] for m in hist] == ["user", "assistant"]


def test_streaming_claim_guard_suppresses_false_claim_before_emitting_tokens(store):
    """未调写工具却谎称已改好地址时，流式引擎先拦截草稿重跑，前端绝不收到错误谎报片段。"""
    llm = ScriptedLLM(replies=[
        AIMessage(content="已为您修改好收货地址！"),  # 被 Claim Guard 静默拦下
        call("change_address", {"order_id": "O1005", "new_address": "杭州市滨江区网商路 699 号"}),
        AIMessage(content="已将键盘订单地址修改为网商路 699 号。"),
    ])
    c, _ = _client(store, llm)
    evs = _events(c.post("/chat", json={"session_id": "cg", "message": "把键盘地址改到杭州市滨江区网商路 699 号"}, headers=U1))
    streamed = "".join(d["text"] for e, d in evs if e == "token")
    assert "修改好收货地址" not in streamed
    assert streamed == "已将键盘订单地址修改为网商路 699 号。"
    assert store.get_order("O1005")["address"] == "杭州市滨江区网商路 699 号"


def test_identity_required_and_sessions_isolated(store):
    c, _ = _client(store, ScriptedLLM(replies=[AIMessage(content="您好")]))
    assert c.post("/chat", json={"session_id": "s1", "message": "你好"}).status_code == 401
    assert c.post("/chat", json={"session_id": "s1", "message": "你好"}, headers={"X-User-Id": "u001;drop"}).status_code == 401
    c.post("/chat", json={"session_id": "s1", "message": "你好"}, headers=U1)
    assert c.get("/sessions/s1/messages", headers={"X-User-Id": "u002"}).json()["messages"] == []


def test_injected_identities_replace_demo_headers(store):
    def logged_in(authorization: str | None = Header(None)) -> str:
        if authorization != "Bearer test-customer":
            raise HTTPException(401)
        return "u001"

    def staff_member(authorization: str | None = Header(None)) -> str:
        if authorization != "Bearer test-staff":
            raise HTTPException(403)
        return "s01"

    kb = PolicyKB(settings.kb_dir)
    runner = build_agent_runner(ScriptedLLM(replies=[AIMessage(content="您好")]), store=store, kb=kb, retriever=kb)
    client = TestClient(create_app(runner=runner, store=store, user_identity=logged_in, staff_identity=staff_member))
    assert client.get("/sessions/s1/messages", headers=U1).status_code == 401
    assert client.get("/approvals", headers=STAFF).status_code == 403
    customer = {"Authorization": "Bearer test-customer", "X-User-Id": "u002"}
    assert client.post("/chat", json={"session_id": "s1", "message": "你好"}, headers=customer).status_code == 200
    assert store.transcript("u001:s1")[0]["content"] == "你好"
    assert not store.transcript("u002:s1")
    assert client.get("/approvals", headers={"Authorization": "Bearer test-staff"}).status_code == 200


def test_return_refund_requires_shipment_and_warehouse_receipt_while_chat_stays_open(store):
    llm = ScriptedLLM(replies=[
        call("prepare_refund", {"order_id": "O1002", "reason_type": "no_reason"}),
        AIMessage(content="已生成确认单"),
        AIMessage(content="审批期间会话依然畅通，您可以随时咨询其他问题。"),
    ])
    c, _ = _client(store, llm)

    evs = _events(c.post("/chat", json={"session_id": "r", "message": "扫地机器人退了"}, headers=U1))
    draft = dict(evs)["done"]["refunds"][0]
    rid = draft["refund_id"]
    assert draft["status"] == "draft" and draft["refund_type"] == "return_refund"
    assert c.post(f"/refunds/{rid}/confirm", json={"confirmed": True}, headers=U1).json()["status"] == "pending_approval"

    # 大额退款待审期间会话不锁死，买家仍可正常继续对话
    follow = c.post("/chat", json={"session_id": "r", "message": "还在审核吗"}, headers=U1)
    assert follow.status_code == 200

    # 审批通过后：已签收订单进入 awaiting_return，不能未退货先打款
    decided = c.post(f"/refunds/{rid}/decision", json={"approved": True}, headers=STAFF).json()
    assert decided["status"] == "awaiting_return"
    assert store.get_order("O1002")["status"] == "delivered"

    # 买家提交退货物流单号 -> 仓库确认收货验货 -> 触发支付打款完成退款
    shipped = c.post(f"/refunds/{rid}/return", json={"tracking_no": "SF12345678"}, headers=U1).json()
    assert shipped["return_tracking_no"] == "SF12345678" and "SF12345678" in shipped["status_text"]
    assert c.get("/approvals", headers=STAFF).json()["returns"][0]["refund_id"] == rid

    received = c.post(f"/refunds/{rid}/receive", json={}, headers=STAFF).json()
    assert received["status"] == "succeeded"
    assert store.get_order("O1002")["status"] == "refunded"


def test_handoff_ticket_visible_to_staff_and_closable(store):
    llm = ScriptedLLM(replies=[call("transfer_to_human", {"reason": "用户投诉"})])
    c, _ = _client(store, llm)
    done = _events(c.post("/chat", json={"session_id": "h", "message": "我要投诉"}, headers=U1))[-1][1]
    assert done["final_action"] == "handoff" and done["queue_position"] == 0
    open_ = c.get("/tickets", headers=STAFF).json()["open"]
    assert open_[0]["ticket_id"] == done["ticket_id"] and open_[0]["priority"] == 1
    assert c.post(f"/tickets/{done['ticket_id']}/close", headers=STAFF).json()["status"] == "closed"
    assert c.get("/tickets", headers=STAFF).json()["open"] == []
