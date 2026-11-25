from cs_agent.rules import evaluate_refund
from tests.conftest import TODAY

LIMIT = 500


def test_no_reason_within_7_days_buyer_pays(store):
    d = evaluate_refund(store.get_order("O1001"), "no_reason", TODAY, LIMIT)  # 签收 3 天，199 元
    assert d.eligible and d.refund_amount == 199.0
    assert d.return_shipping_paid_by == "buyer"
    assert not d.needs_approval


def test_large_amount_needs_approval(store):
    d = evaluate_refund(store.get_order("O1002"), "no_reason", TODAY, LIMIT)  # 1899 元
    assert d.eligible and d.needs_approval


def test_no_reason_after_7_days_rejected(store):
    d = evaluate_refund(store.get_order("O1004"), "no_reason", TODAY, LIMIT)  # 签收 20 天
    assert not d.eligible


def test_quality_after_15_days_rejected(store):
    d = evaluate_refund(store.get_order("O1004"), "quality", TODAY, LIMIT)
    assert not d.eligible and "保修" in d.reason


def test_quality_within_15_days_seller_pays(store):
    d = evaluate_refund(store.get_order("O1001"), "quality", TODAY, LIMIT)
    assert d.eligible and d.return_shipping_paid_by == "seller"


def test_unshipped_order_full_refund_no_return(store):
    d = evaluate_refund(store.get_order("O1005"), "no_reason", TODAY, LIMIT)
    assert d.eligible and d.return_shipping_paid_by is None


def test_in_transit_not_refundable(store):
    d = evaluate_refund(store.get_order("O1003"), "no_reason", TODAY, LIMIT)
    assert not d.eligible


def test_already_refunded(store):
    store.mark_refunded("O1001", 199.0)
    d = evaluate_refund(store.get_order("O1001"), "no_reason", TODAY, LIMIT)
    assert not d.eligible
