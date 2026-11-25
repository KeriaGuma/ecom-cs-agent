from pathlib import Path

import pytest

from cs_agent.refunds import RefundService
from cs_agent.sandbox_db import copy_database
from cs_agent.store import ShopStore
from tests.conftest import TODAY


def test_restore_preserves_session_and_refund_ledger(store, tmp_path):
    refunds = RefundService(store, 500, today=lambda: TODAY)
    proposal = refunds.prepare(user_id="u001", thread_id="u001:t", message_id="m", order_id="O1001", reason="no_reason")
    result = refunds.confirm(proposal["refund_id"], "u001", True)
    assert result["status"] == "awaiting_return"
    result = refunds.confirm_return(proposal["refund_id"], "a01", "SF123456")
    assert result["status"] == "succeeded"
    store.save_session("u001:t", {"user_id": "u001", "messages": []}, [("m", "user", "申请退款")])
    backup = tmp_path / "snapshot.db"
    restored = tmp_path / "restored.db"

    copy_database(Path(store.path), backup)
    copy_database(backup, restored)
    recovered = ShopStore(str(restored))
    assert recovered.transcript("u001:t") == [{"role": "user", "content": "申请退款"}]
    assert RefundService(recovered, 500).status(proposal["refund_id"], "u001")["status"] == "succeeded"
    assert len(recovered.query("SELECT * FROM simulated_refund_payments")) == 1
    with pytest.raises(FileExistsError):
        copy_database(backup, restored)
