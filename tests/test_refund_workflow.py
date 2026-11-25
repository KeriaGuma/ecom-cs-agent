"""通过 HTTP 与业务层验证确认隔离、仅退款 vs 退货验货双分支、支付恢复与工单接手。"""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from cs_agent.agent import build_agent_runner
from cs_agent.api import create_app
from cs_agent.config import settings
from cs_agent.kb import PolicyKB
from cs_agent.refunds import PaymentResult, RefundError, RefundService, SimulatedPaymentGateway
from cs_agent.store import ShopStore
from tests.conftest import TODAY
from tests.test_api import STAFF, U1, _events
from tests.test_workers import ScriptedLLM, call


def service_for(store, gateway=None):
    return RefundService(store, 500, gateway=gateway, today=lambda: TODAY)


def proposal(refunds, order="O1002", session="refund", reason="no_reason"):
    return refunds.prepare(user_id="u001", thread_id=f"u001:{session}", message_id=session,
                           order_id=order, reason=reason)["refund_id"]


def test_web_confirmation_return_receipt_and_staff_handoff(store):
    refunds = service_for(store)
    llm = ScriptedLLM(replies=[
        call("submit_refund", {"order_id": "O1001", "reason_type": "no_reason"}),  # 未暴露该工具，安全忽略
        call("prepare_refund", {"order_id": "O1002", "reason_type": "no_reason"}, 1),
        AIMessage(content="请核对确认单"),
        call("transfer_to_human", {"reason": "用户要求人工"}),
    ])
    kb = PolicyKB(settings.kb_dir)
    runner = build_agent_runner(llm, store=store, kb=kb, retriever=kb, today=lambda: TODAY, refund_service=refunds)
    c = TestClient(create_app(runner=runner, store=store, refund_service=refunds))
    reply = c.post("/chat", json={"session_id": "refund", "message": "O1002 我想退货，电话 13812345678"}, headers=U1)
    events = dict(_events(reply))
    draft = events["done"]["refunds"][0]
    refund_id = draft["refund_id"]
    endpoint = f"/refunds/{refund_id}"

    assert draft["status"] == "draft" and draft["amount"] == 1899 and draft["refund_type"] == "return_refund"
    assert "尚未提交" in events["message"]["text"]
    assert not store.list_ops("O1001") and not store.list_ops("O1002")
    assert c.post(f"{endpoint}/confirm", json={"confirmed": True}, headers={"X-User-Id": "u002"}).status_code == 404
    assert c.post(f"{endpoint}/decision", json={"approved": True}, headers=STAFF).status_code == 409
    assert c.post(f"{endpoint}/confirm", json={"confirmed": True}, headers=U1).json()["status"] == "pending_approval"
    assert c.get("/approvals", headers=STAFF).json()["refunds"][0]["refund_id"] == refund_id
    assert store.get_order("O1002")["status"] == "delivered"

    reply = c.post("/chat", json={"session_id": "refund", "message": "转人工"}, headers=U1)
    ticket_id = dict(_events(reply))["done"]["ticket_id"]
    ticket_url = f"/tickets/{ticket_id}"
    assert c.get(ticket_url, headers=U1).status_code == 403
    ticket = c.post(f"{ticket_url}/claim", headers=STAFF).json()
    assert ticket["thread_id"] == "u001:refund" and ticket["assignee"] == "s01"
    assert ticket["refunds"][0]["message_id"] == ticket["messages"][0]["message_id"]
    assert "13812345678" not in ticket["messages"][0]["content"]
    assert c.post(f"{ticket_url}/claim", headers={"X-Staff-Id": "s02"}).status_code == 409
    assert c.post(f"{ticket_url}/close", headers={"X-Staff-Id": "s02"}).status_code == 409
    assert c.post(f"{endpoint}/decision", json={"approved": True}, headers=U1).status_code == 403

    # 审批通过后进入 awaiting_return，买家寄回并由后台验货后才打款
    assert c.post(f"{endpoint}/decision", json={"approved": True}, headers=STAFF).json()["status"] == "awaiting_return"
    assert store.get_order("O1002")["status"] == "delivered"
    assert c.post(f"{endpoint}/return", json={"tracking_no": "SF889900"}, headers=U1).json()["return_tracking_no"] == "SF889900"
    assert c.post(f"{endpoint}/receive", json={}, headers=STAFF).json()["status"] == "succeeded"
    assert c.get(endpoint, headers=U1).json()["status"] == "succeeded"

    completed = c.get(ticket_url, headers=STAFF).json()["refunds"][0]
    assert [event["actor"] for event in completed["events"]] == [
        "agent", "user:u001", "staff:s01", "user:u001", "staff:s01", "payment",
    ]
    assert completed["payment_ref"] == f"sim-{refund_id}"
    assert store.get_order("O1002")["status"] == "refunded"


def test_unshipped_order_refund_only_settles_immediately_without_return(store):
    """未发货订单（O1005，¥459）属于仅退款，用户确认后直接原路打款，无需退货入库。"""
    refunds = service_for(store)
    refund_id = proposal(refunds, order="O1005")
    res = refunds.confirm(refund_id, "u001", True)
    assert res["refund_type"] == "refund_only" and res["status"] == "succeeded"
    assert store.get_order("O1005")["status"] == "refunded"


def test_recovers_payment_after_ack_lost_and_restart(store):
    class LostAcknowledgment(SimulatedPaymentGateway):
        def submit(self, *args):
            super().submit(*args)
            raise TimeoutError("payment committed; response lost")

    refunds = service_for(store, LostAcknowledgment(store))
    refund_id = proposal(refunds, order="O1005")
    uncertain = refunds.confirm(refund_id, "u001", True)
    assert uncertain["status"] == "unknown"
    assert store.get_order("O1005")["status"] == "paid"
    assert store.list_ops("O1005")[0]["status"] == "payment_unknown"

    restarted_store = ShopStore(store.path)
    restarted = service_for(restarted_store)
    assert restarted.status(refund_id, "u001")["status"] == "succeeded"
    assert restarted.confirm(refund_id, "u001", True)["status"] == "succeeded"
    assert restarted.reconcile(refund_id, "s01")["status"] == "succeeded"
    assert restarted_store.get_order("O1005")["status"] == "refunded"
    assert len(restarted_store.query("SELECT * FROM simulated_refund_payments")) == 1
    assert [row["status"] for row in restarted_store.list_ops("O1005")] == ["payment_succeeded"]


def test_local_finalization_is_atomic_and_replay_uses_payment_record(store):
    store.execute("CREATE TRIGGER reject_success BEFORE INSERT ON refund_events WHEN NEW.status='succeeded'"
                  " BEGIN SELECT RAISE(ABORT, 'test audit failure'); END")
    refunds = service_for(store)
    refund_id = proposal(refunds, order="O1005")
    with pytest.raises(sqlite3.IntegrityError):
        refunds.confirm(refund_id, "u001", True)
    assert store.get_order("O1005")["status"] == "paid"
    assert store.list_ops("O1005")[0]["status"] == "payment_ready"
    assert len(store.query("SELECT * FROM simulated_refund_payments")) == 1
    store.execute("DROP TRIGGER reject_success")
    assert service_for(ShopStore(store.path)).status(refund_id, "u001")["status"] == "succeeded"
    assert len(store.query("SELECT * FROM simulated_refund_payments")) == 1


def test_payment_processing_is_distinct_from_success(store):
    class AsynchronousPayment:
        result = None

        def submit(self, *args):
            self.result = PaymentResult("processing", "async-ref")
            return self.result

        def lookup(self, refund_id):
            return self.result

    gateway = AsynchronousPayment()
    refunds = service_for(store, gateway)
    refund_id = proposal(refunds, order="O1005")
    assert refunds.confirm(refund_id, "u001", True)["status"] == "processing"
    assert store.get_order("O1005")["status"] == "paid"
    gateway.result = PaymentResult("failed", "async-ref")
    assert refunds.status(refund_id, "u001")["status"] == "failed"
    assert store.get_order("O1005")["status"] == "paid"


def test_only_one_active_confirmation_across_sessions(store):
    refunds = service_for(store)
    first, second = [proposal(refunds, session=session) for session in ("a", "b")]

    def confirm(refund_id):
        try:
            return refunds.confirm(refund_id, "u001", True)["status"]
        except RefundError as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(confirm, (first, second)))
    assert results.count("pending_approval") == 1 and results.count(409) == 1
    assert len(store.list_ops("O1002")) == 1


def test_legacy_ticket_schema_upgrades_without_replacing_records(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE tickets (ticket_id INTEGER PRIMARY KEY, user_id TEXT, priority INTEGER,"
                     " status TEXT, summary TEXT, created_at TEXT)")
        conn.execute("INSERT INTO tickets VALUES (7,'u001',2,'open','{}','2026-11-09')")
    store = ShopStore(str(path))
    assert store.get_ticket(7)["thread_id"] is None
    assert store.claim_ticket(7, "s01")
    assert service_for(store).ticket_details(7)["messages"] == []
    assert store.get_ticket(7)["assignee"] == "s01"
