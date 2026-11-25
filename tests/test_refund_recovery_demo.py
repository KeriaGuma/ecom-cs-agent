import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from cs_agent.refunds import RefundError, RefundService, SIMULATED_PROCESSING_SECONDS, SimulatedPaymentGateway
from cs_agent.store import ShopStore
from tests.conftest import TODAY


def prepare(refunds: RefundService, order_id: str = "O1005") -> str:
    return refunds.prepare(user_id="u001", thread_id="u001:demo", message_id=order_id,
                           order_id=order_id, reason="no_reason")["refund_id"]


def get_events(store: ShopStore, refund_id: str) -> list[dict]:
    return store.query("SELECT status,actor,detail FROM refund_events WHERE refund_id=? ORDER BY event_id", (refund_id,))


def test_ack_lost_recovers_from_a_fresh_worker_process(store):
    refunds = RefundService(store, 500, today=lambda: TODAY, enable_simulated_faults=True)
    refund_id = prepare(refunds, "O1001")
    assert refunds.confirm(refund_id, "u001", True)["status"] == "awaiting_return"
    refunds.submit_return(refund_id, "u001", "SF123456")
    refunds.set_simulated_fault(refund_id, "ack_lost", "a01")
    refunds.set_simulated_fault(refund_id, "ack_lost", "a01")
    assert sum("ack_lost" in e["detail"] for e in get_events(store, refund_id)) == 1
    with pytest.raises(RefundError, match="已绑定"):
        refunds.set_simulated_fault(refund_id, "failed", "a01")

    assert refunds.confirm_return(refund_id, "a01")["status"] == "unknown"
    assert store.get_order("O1001")["status"] == "delivered"
    assert len(store.query("SELECT refund_id FROM simulated_refund_payments")) == 1

    cmd = [sys.executable, "-m", "cs_agent.refund_jobs", "--db", store.path]
    recovered = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
    assert json.loads(recovered.stdout) == {"checked": 1, "succeeded": 1}
    again = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
    assert json.loads(again.stdout) == {"checked": 0}
    assert store.get_order("O1001")["status"] == "refunded"
    assert len(store.query("SELECT refund_id FROM simulated_refund_payments")) == 1
    assert [(e["status"], e["actor"]) for e in get_events(store, refund_id)][-2:] == [
        ("unknown", "payment"), ("succeeded", "payment:recovery"),
    ]


def test_processing_depends_on_persisted_time_and_concurrent_checks_finalize_once(store):
    clock = [1_000]
    gateway = SimulatedPaymentGateway(store, enable_faults=True, now=lambda: clock[0])
    refunds = RefundService(store, 500, gateway=gateway, today=lambda: TODAY)
    refund_id = prepare(refunds, "O1002")
    assert refunds.confirm(refund_id, "u001", True)["status"] == "pending_approval"
    assert refunds.decide(refund_id, "a01", True)["status"] == "awaiting_return"
    assert refunds.set_simulated_fault(refund_id, "processing", "a01")["status"] == "awaiting_return"
    assert refunds.confirm_return(refund_id, "a01", "SF778899")["status"] == "processing"
    with pytest.raises(RefundError, match="只能在"):
        refunds.set_simulated_fault(refund_id, "processing", "a01")

    original_events = get_events(store, refund_id)
    restarted_store = ShopStore(store.path)
    restarted = RefundService(restarted_store, 500, gateway=SimulatedPaymentGateway(restarted_store, now=lambda: clock[0]))
    for _ in range(3):
        assert restarted.status(refund_id, "u001")["status"] == "processing"
    assert restarted.recover_simulated_payments() == {"checked": 1, "processing": 1}
    assert get_events(store, refund_id) == original_events
    assert store.get_order("O1002")["status"] == "delivered"

    clock[0] += SIMULATED_PROCESSING_SECONDS
    with ThreadPoolExecutor(max_workers=2) as pool:
        recovered = list(pool.map(lambda _: restarted.recover_simulated_payments(), range(2)))
    assert any(summary.get("succeeded") == 1 for summary in recovered)
    assert len(store.query("SELECT refund_id FROM simulated_refund_payments")) == 1
    assert [op["status"] for op in store.list_ops("O1002")] == ["payment_succeeded"]
    assert [event["status"] for event in get_events(store, refund_id)].count("succeeded") == 1
    assert store.get_order("O1002")["status"] == "refunded"


def test_failed_simulation_has_no_order_effect_and_demo_gate_controls_assignment(store):
    demo = RefundService(store, 500, today=lambda: TODAY, enable_simulated_faults=True)
    ordinary = RefundService(store, 500, today=lambda: TODAY)
    failed_id = prepare(demo, "O1005")
    with pytest.raises(RefundError) as error:
        ordinary.set_simulated_fault(failed_id, "failed", "a01")
    assert error.value.status_code == 404
    demo.set_simulated_fault(failed_id, "failed", "a01")
    assert demo.confirm(failed_id, "u001", True)["status"] == "failed"
    assert store.get_order("O1005")["status"] == "paid"

    normal_id = prepare(demo, "O1001")
    demo.set_simulated_fault(normal_id, "failed", "a01")
    assert ordinary.confirm(normal_id, "u001", True)["status"] == "awaiting_return"
    assert ordinary.confirm_return(normal_id, "a01", "SF654321")["status"] == "succeeded"


def test_recovery_does_not_submit_when_ledger_has_no_payment(store):
    class TimeoutBeforeSubmission(SimulatedPaymentGateway):
        def submit(self, refund_id, order_id, amount):
            raise TimeoutError("模拟请求未到达")

    interrupted = RefundService(store, 500, gateway=TimeoutBeforeSubmission(store), today=lambda: TODAY)
    refund_id = prepare(interrupted, "O1005")
    assert interrupted.confirm(refund_id, "u001", True)["status"] == "unknown"
    restarted = RefundService(ShopStore(store.path), 500)
    assert restarted.recover_simulated_payments() == {"checked": 0}
    assert restarted.status(refund_id, "u001")["status"] == "unknown"
    assert not store.query("SELECT refund_id FROM simulated_refund_payments")
    assert store.get_order("O1005")["status"] == "paid"


def test_old_simulated_ledger_is_migrated_as_success(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE simulated_refund_payments (refund_id TEXT PRIMARY KEY, order_id TEXT, amount REAL)")
        conn.execute("INSERT INTO simulated_refund_payments VALUES ('old-refund','O1001',199)")
    store = ShopStore(str(path))
    result = SimulatedPaymentGateway(store).lookup("old-refund")
    assert result is not None
    assert result.status == "succeeded" and result.reference == "sim-old-refund"
