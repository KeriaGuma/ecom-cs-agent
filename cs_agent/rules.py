"""售后退款规则引擎。

核心原则：确定性逻辑不交给 LLM。
LLM 只负责从用户表达与图片核验结果中判断退款原因类型（无理由 / 质量问题），
能不能退、属于仅退款还是退货退款、退多少、运费谁出、要不要人工审批，全部由代码计算。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal, Optional

ReasonType = Literal["no_reason", "quality"]
RefundType = Literal["refund_only", "return_refund"]

NO_REASON_DAYS = 7   # 7 天无理由退货期
QUALITY_DAYS = 15    # 质量问题 15 天退货期


@dataclass(frozen=True)
class RefundDecision:
    eligible: bool
    refund_amount: float
    return_shipping_paid_by: Optional[str]  # buyer / seller / None（无需退货）
    needs_approval: bool
    reason: str
    refund_type: Optional[RefundType] = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_refund(order: dict, reason_type: ReasonType, today: date, auto_approve_limit: float) -> RefundDecision:
    status = order["status"]
    amount = float(order["amount"])

    if status == "refunded":
        return RefundDecision(False, 0.0, None, False, "该订单已退款，无需重复申请")

    if status == "paid":
        # 未发货：仅退款（直接取消订单全额原路退款，无需寄回商品）
        return _decide(amount, None, auto_approve_limit, "订单未发货，可直接取消并全额仅退款", "refund_only")

    if status == "shipped":
        return RefundDecision(False, 0.0, None, False, "订单运输中，可拒收后自动退款，或签收后再申请售后")

    if status != "delivered" or not order.get("delivered_at"):
        return RefundDecision(False, 0.0, None, False, f"订单状态为 {status}，暂不支持退款")

    days = (today - date.fromisoformat(order["delivered_at"])).days
    if reason_type == "quality":
        if days <= QUALITY_DAYS:
            return _decide(
                amount, "seller", auto_approve_limit,
                f"签收 {days} 天，质量问题在 {QUALITY_DAYS} 天内可退货退款，退货运费由商家承担",
                "return_refund",
            )
        return RefundDecision(False, 0.0, None, False, f"签收已 {days} 天，超过质量问题 {QUALITY_DAYS} 天退货期，可申请保修")

    if days <= NO_REASON_DAYS:
        return _decide(
            amount, "buyer", auto_approve_limit,
            f"签收 {days} 天，符合 {NO_REASON_DAYS} 天无理由退货退款，退货运费由买家承担",
            "return_refund",
        )
    return RefundDecision(False, 0.0, None, False, f"签收已 {days} 天，超过 {NO_REASON_DAYS} 天无理由退货期")


def _decide(amount: float, shipping: Optional[str], limit: float, reason: str,
            refund_type: RefundType) -> RefundDecision:
    needs_approval = amount > limit
    if needs_approval:
        reason += f"；金额超过 {limit:.0f} 元，需人工审批"
    return RefundDecision(True, amount, shipping, needs_approval, reason, refund_type)
