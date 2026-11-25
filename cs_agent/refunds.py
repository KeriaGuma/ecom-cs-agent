"""售后申请、用户确认、退货验货、人工审批与支付结果的持久化业务状态机。

核心业务区分：
- **未发货仅退款（`refund_only`）**：用户确认（且超额审批通过）后直接进入 `ready`，提交支付网关原路退款；
- **已签收退货退款（`return_refund`）**：用户确认（且超额审批通过）后进入 `awaiting_return`（待寄回商品），
  买家提交退货物流单号、仓库/坐席确认验货收货后，才进入 `ready` 提交支付网关原路退款，杜绝“未退货先打款”的资损漏洞。
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from time import time
from typing import Callable, Literal, Protocol
from uuid import uuid4

from .pii import mask_text
from .rules import ReasonType, evaluate_refund
from .store import ShopStore


STATUS_TEXT = {
    "draft": "待用户确认，尚未申请退款",
    "pending_approval": "已提交，等待人工审核",
    "awaiting_return": "已同意退货，请寄回商品并填写退货单号",
    "ready": "申请已授权，等待执行原路退款",
    "processing": "支付方正在处理退款",
    "unknown": "支付结果尚未确认，请勿重复申请",
    "succeeded": "支付方已确认退款成功",
    "failed": "支付方确认退款失败",
    "rejected": "人工审核未通过",
    "cancelled": "已取消退款草稿",
}
EXECUTABLE = ("ready", "processing", "unknown")
ACTIVE_OP_STATUSES = ("pending", "awaiting_return", "payment_ready", "payment_pending", "payment_unknown")
SimulatedFault = Literal["ack_lost", "processing", "failed"]
SIMULATED_PROCESSING_SECONDS = 10
TRACKING_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,32}$")
ORDER_OP_STATUS = {
    "pending_approval": "pending",
    "awaiting_return": "awaiting_return",
    "ready": "payment_ready",
    "processing": "payment_pending",
    "unknown": "payment_unknown",
    "succeeded": "payment_succeeded",
    "failed": "payment_failed",
    "rejected": "rejected",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RefundError(RuntimeError):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PaymentResult:
    status: Literal["processing", "succeeded", "failed"]
    reference: str


class PaymentGateway(Protocol):
    """以退款申请 ID 为幂等键。响应超时不能推断支付失败。"""

    def submit(self, refund_id: str, order_id: str, amount: float) -> PaymentResult: ...

    def lookup(self, refund_id: str) -> PaymentResult | None: ...


class SimulatedPaymentGateway:
    """模拟支付方账本，每次写入独立于客服本地业务事务。"""

    def __init__(self, store: ShopStore, *, enable_faults: bool = False, now: Callable[[], float] = time):
        self.store = store
        self.enable_faults = enable_faults
        self.now = now

    def submit(self, refund_id: str, order_id: str, amount: float) -> PaymentResult:
        with self.store.transaction() as conn:
            fault = conn.execute("SELECT fault FROM simulated_refund_faults WHERE refund_id=?", (refund_id,)).fetchone()
            fault = fault["fault"] if enable_faults_flag(self.enable_faults) and fault is not None else None
            outcome = "failed" if fault == "failed" else "succeeded"
            settles_at = self.now() + SIMULATED_PROCESSING_SECONDS if fault == "processing" else None
            inserted = conn.execute(
                "INSERT INTO simulated_refund_payments (refund_id,order_id,amount,outcome,settles_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(refund_id) DO NOTHING", (refund_id, order_id, amount, outcome, settles_at),
            ).rowcount
            existing = conn.execute("SELECT * FROM simulated_refund_payments WHERE refund_id=?",
                                    (refund_id,)).fetchone()
            if existing["order_id"] != order_id or existing["amount"] != amount:
                raise ValueError("同一退款申请的支付参数发生变化")
        if inserted and fault == "ack_lost":
            raise TimeoutError("模拟支付已落账，响应丢失")
        return self._result(existing)

    def lookup(self, refund_id: str) -> PaymentResult | None:
        rows = self.store.query("SELECT * FROM simulated_refund_payments WHERE refund_id=?", (refund_id,))
        return self._result(rows[0]) if rows else None

    def _result(self, row: dict | sqlite3.Row) -> PaymentResult:
        settles_at = row["settles_at"]
        status = "processing" if settles_at is not None and self.now() < settles_at else row["outcome"]
        return PaymentResult(status, f"sim-{row['refund_id']}")


def enable_faults_flag(enabled: bool) -> bool:
    return bool(enabled)


class RefundService:
    def __init__(self, store: ShopStore, limit: float, gateway: PaymentGateway | None = None,
                 today: Callable[[], date] = date.today, *, enable_simulated_faults: bool = False):
        self.store = store
        self.limit = limit
        self.gateway = gateway if gateway is not None else SimulatedPaymentGateway(store, enable_faults=enable_simulated_faults)
        self.today = today

    @staticmethod
    def _request(conn: sqlite3.Connection, refund_id: str, user_id: str | None = None) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM refund_requests WHERE refund_id=?", (refund_id,)).fetchone()
        if row is None or (user_id is not None and row["user_id"] != user_id):
            raise RefundError("退款申请不存在", 404)
        return row

    @staticmethod
    def _order(conn: sqlite3.Connection, user_id: str, order_id: str) -> dict:
        row = conn.execute("SELECT * FROM orders WHERE order_id=? AND user_id=?", (order_id, user_id)).fetchone()
        if row is None:
            raise RefundError("订单不存在或不属于当前用户", 404)
        return dict(row)

    @staticmethod
    def _event(conn: sqlite3.Connection, refund_id: str, status: str, actor: str, detail: str = "") -> None:
        conn.execute("INSERT INTO refund_events (refund_id,status,actor,detail,created_at) VALUES (?,?,?,?,?)",
                     (refund_id, status, actor, detail, _now()))

    def _set_status(self, conn: sqlite3.Connection, row: sqlite3.Row, status: str, actor: str, detail: str = "",
                    *, payment_ref: str | None = None, decision: str | None = None,
                    tracking_no: str | None = None) -> None:
        same_status = row["status"] == status
        same_ref = payment_ref is None or row["payment_ref"] == payment_ref
        same_track = tracking_no is None or row["return_tracking_no"] == tracking_no
        if same_status and same_ref and same_track:
            return
        now = _now()
        conn.execute(
            "UPDATE refund_requests SET status=?,updated_at=?,payment_ref=COALESCE(?,payment_ref),"
            " staff_decision=COALESCE(?,staff_decision),return_tracking_no=COALESCE(?,return_tracking_no)"
            " WHERE refund_id=?",
            (status, now, payment_ref, decision, tracking_no, row["refund_id"]),
        )
        if row["op_id"] is not None and status in ORDER_OP_STATUS:
            conn.execute("UPDATE order_ops SET status=?, decided_at=? WHERE op_id=?",
                         (ORDER_OP_STATUS[status], now, row["op_id"]))
        self._event(conn, row["refund_id"], status, actor, detail)

    @staticmethod
    def view(row: dict | sqlite3.Row) -> dict:
        fields = (
            "refund_id", "order_id", "item", "amount", "reason_type", "refund_type",
            "shipping_paid_by", "return_tracking_no", "status", "updated_at",
        )
        status_text = STATUS_TEXT[row["status"]]
        if row["status"] == "awaiting_return" and row["return_tracking_no"]:
            status_text = f"退货已寄出（{row['return_tracking_no']}），等待仓库验货退款"
        return {
            **{key: row[key] for key in fields},
            "needs_approval": bool(row["needs_approval"]),
            "status_text": status_text,
        }

    @staticmethod
    def _post_auth_status(row: sqlite3.Row | dict) -> str:
        """确认或审批通过后的下一状态：已签收退货单先等寄回验货，未发货仅退款单直接进入待打款。"""
        return "awaiting_return" if row["refund_type"] == "return_refund" else "ready"

    def prepare(self, *, user_id: str, thread_id: str, message_id: str, order_id: str, reason: ReasonType) -> dict:
        order_id = order_id.strip().upper()
        with self.store.transaction() as conn:
            order = self._order(conn, user_id, order_id)
            existing = conn.execute("SELECT * FROM refund_requests WHERE user_id=? AND thread_id=? AND message_id=?"
                                    " AND order_id=? AND reason_type=?", (user_id, thread_id, message_id, order_id, reason)).fetchone()
            if existing:
                return self.view(existing)
            quote = evaluate_refund(order, reason, self.today(), self.limit)
            if not quote.eligible:
                raise RefundError(quote.reason)
            refund_type = quote.refund_type or ("refund_only" if order["status"] == "paid" else "return_refund")
            refund_id, now = uuid4().hex, _now()
            conn.execute(
                "INSERT INTO refund_requests (refund_id,user_id,thread_id,message_id,order_id,item,amount,"
                "reason_type,refund_type,shipping_paid_by,needs_approval,status,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,'draft',?,?)",
                (refund_id, user_id, thread_id, message_id, order_id, order["item"], quote.refund_amount,
                 reason, refund_type, quote.return_shipping_paid_by, int(quote.needs_approval), now, now),
            )
            self._event(conn, refund_id, "draft", "agent")
            return self.view(self._request(conn, refund_id))

    def confirm(self, refund_id: str, user_id: str, confirmed: bool) -> dict:
        with self.store.transaction() as conn:
            row = self._request(conn, refund_id, user_id)
            actor = f"user:{user_id}"
            if row["status"] == "draft":
                if not confirmed:
                    self._set_status(conn, row, "cancelled", actor)
                    return self.view(self._request(conn, refund_id))
                order = self._order(conn, user_id, row["order_id"])
                decision = evaluate_refund(order, row["reason_type"], self.today(), self.limit)
                if not decision.eligible:
                    raise RefundError(decision.reason)
                if (decision.refund_amount != row["amount"] or decision.return_shipping_paid_by != row["shipping_paid_by"]
                        or (decision.needs_approval and not row["needs_approval"])):
                    raise RefundError("退款条件已变化，请让客服生成新的确认单")
                placeholders = ",".join("?" for _ in ACTIVE_OP_STATUSES)
                active = conn.execute(
                    f"SELECT 1 FROM order_ops WHERE order_id=? AND op='refund' AND status IN ({placeholders})",
                    (row["order_id"], *ACTIVE_OP_STATUSES),
                ).fetchone()
                if active:
                    raise RefundError("该订单已有退款在处理中")
                status = "pending_approval" if row["needs_approval"] else self._post_auth_status(row)
                detail = {
                    "amount": row["amount"], "reason_type": row["reason_type"],
                    "refund_type": row["refund_type"], "refund_id": refund_id,
                }
                op_id = self.store.insert_op(conn, row["order_id"], user_id, "refund", detail, ORDER_OP_STATUS[status])
                conn.execute("UPDATE refund_requests SET op_id=? WHERE refund_id=?", (op_id, refund_id))
                row = self._request(conn, refund_id)
                self._set_status(conn, row, status, actor, "用户确认退款单上的订单、金额与原因")
            elif confirmed and row["status"] == "cancelled":
                raise RefundError("退款确认单已处理")
            elif not confirmed and row["status"] != "cancelled":
                raise RefundError("这笔退款已提交，无法取消确认单")
        return self._process(refund_id, submit=confirmed)

    def decide(self, refund_id: str, staff: str, approved: bool, comment: str = "") -> dict:
        decision = "approved" if approved else "rejected"
        with self.store.transaction() as conn:
            row = self._request(conn, refund_id)
            if row["status"] == "pending_approval":
                nxt = self._post_auth_status(row) if approved else "rejected"
                self._set_status(conn, row, nxt, f"staff:{staff}", mask_text(comment), decision=decision)
            elif row["staff_decision"] != decision:
                raise RefundError("此申请没有可处理的审批，或已记录其他审批结果")
        return self._process(refund_id, submit=approved)

    def submit_return(self, refund_id: str, user_id: str, tracking_no: str) -> dict:
        """买家回填退货物流单号。"""
        tracking_no = tracking_no.strip().upper()
        if not TRACKING_PATTERN.match(tracking_no):
            raise RefundError("请填写合法的退货物流单号（4～32 位字母或数字）", 422)
        with self.store.transaction() as conn:
            row = self._request(conn, refund_id, user_id)
            if row["status"] != "awaiting_return":
                raise RefundError("当前申请不在待寄回退货状态")
            self._set_status(
                conn, row, "awaiting_return", f"user:{user_id}",
                f"买家提交退货物流单号：{tracking_no}", tracking_no=tracking_no,
            )
            return self.view(self._request(conn, refund_id))

    def confirm_return(self, refund_id: str, staff: str, tracking_no: str = "") -> dict:
        """仓库或坐席核验退货商品无误后放行打款。"""
        clean_track = tracking_no.strip().upper() if tracking_no.strip() else None
        if clean_track and not TRACKING_PATTERN.match(clean_track):
            raise RefundError("退货物流单号格式不合法", 422)
        with self.store.transaction() as conn:
            row = self._request(conn, refund_id)
            if row["status"] == "awaiting_return":
                track = clean_track or row["return_tracking_no"]
                note = f"退货包裹已验收入库（单号 {track}）" if track else "退货商品已验收入库"
                self._set_status(conn, row, "ready", f"staff:{staff}", note, tracking_no=track)
            elif row["refund_type"] != "return_refund" or row["status"] not in (*EXECUTABLE, "succeeded", "failed"):
                raise RefundError("当前申请不在待验货退货状态")
        return self._process(refund_id, submit=True)

    def set_simulated_fault(self, refund_id: str, fault: SimulatedFault, staff: str) -> dict:
        if not isinstance(self.gateway, SimulatedPaymentGateway) or not self.gateway.enable_faults:
            raise RefundError("模拟故障功能未启用", 404)
        if fault not in ("ack_lost", "processing", "failed"):
            raise RefundError("不支持的模拟故障", 400)
        with self.store.transaction() as conn:
            row = self._request(conn, refund_id)
            if row["status"] not in ("draft", "pending_approval", "awaiting_return"):
                raise RefundError("只能在打款执行前（草稿、待审批或待退货阶段）设置模拟故障")
            bound = conn.execute("SELECT fault FROM simulated_refund_faults WHERE refund_id=?", (refund_id,)).fetchone()
            if bound is None:
                conn.execute("INSERT INTO simulated_refund_faults VALUES (?,?)", (refund_id, fault))
                self._event(conn, refund_id, row["status"], f"staff:{staff}", f"选择模拟支付场景：{fault}")
            elif bound["fault"] != fault:
                raise RefundError("此退款已绑定其他模拟故障")
            return {**self.view(row), "fault": fault}

    def recover_simulated_payments(self) -> dict[str, int]:
        """以模拟支付账本已落账记录为准主动对账恢复。"""
        if not isinstance(self.gateway, SimulatedPaymentGateway):
            raise RefundError("模拟账本核对只能使用模拟支付", 400)
        rows = self.store.query(
            "SELECT r.refund_id FROM refund_requests r JOIN simulated_refund_payments p USING(refund_id) "
            "WHERE r.status IN ('ready','processing','unknown') ORDER BY r.created_at,r.rowid",
        )
        counts = {"checked": len(rows)}
        for row in rows:
            status = self._process(row["refund_id"], submit=False, actor="payment:recovery")["status"]
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _process(self, refund_id: str, *, submit: bool, actor: str = "payment") -> dict:
        row = self._find(refund_id)
        if row["status"] not in EXECUTABLE:
            return self.view(row)
        try:
            result = self.gateway.lookup(refund_id)
            if result is None and submit:
                result = self.gateway.submit(refund_id, row["order_id"], row["amount"])
        except (TimeoutError, OSError):
            with self.store.transaction() as conn:
                current = self._request(conn, refund_id)
                if current["status"] == "ready":
                    self._set_status(conn, current, "unknown", actor, "调用中断，等待核对支付结果")
            return self.view(self._find(refund_id))
        if result is None:
            return self.view(self._find(refund_id))
        with self.store.transaction() as conn:
            current = self._request(conn, refund_id)
            if current["status"] in EXECUTABLE:
                if result.status == "succeeded":
                    self.store.apply_refund(conn, current["order_id"], current["amount"])
                self._set_status(conn, current, result.status, actor, payment_ref=result.reference)
            return self.view(self._request(conn, refund_id))

    def _find(self, refund_id: str, user_id: str | None = None) -> dict:
        rows = self.store.query("SELECT * FROM refund_requests WHERE refund_id=?", (refund_id,))
        if not rows or (user_id is not None and rows[0]["user_id"] != user_id):
            raise RefundError("退款申请不存在", 404)
        return rows[0]

    def status(self, refund_id: str, user_id: str) -> dict:
        self._find(refund_id, user_id)
        return self._process(refund_id, submit=False)

    def reconcile(self, refund_id: str, staff: str) -> dict:
        with self.store.transaction() as conn:
            row = self._request(conn, refund_id)
            if row["status"] in EXECUTABLE:
                self._event(conn, refund_id, row["status"], f"staff:{staff}", "核对支付结果，必要时以相同申请编号重试")
        return self._process(refund_id, submit=True)

    def for_thread(self, thread_id: str, user_id: str) -> list[dict]:
        return self.store.query("SELECT * FROM refund_requests WHERE thread_id=? AND user_id=? ORDER BY created_at,rowid",
                                (thread_id, user_id))

    def refresh_thread(self, thread_id: str, user_id: str) -> None:
        for row in self.for_thread(thread_id, user_id):
            if row["status"] in EXECUTABLE:
                self._process(row["refund_id"], submit=False)

    def refresh_user(self, user_id: str) -> None:
        rows = self.store.query("SELECT refund_id FROM refund_requests WHERE user_id=?"
                                " AND status IN ('ready','processing','unknown')", (user_id,))
        for row in rows:
            self._process(row["refund_id"], submit=False)

    def pending_approvals(self) -> list[dict]:
        rows = self.store.query("SELECT * FROM refund_requests WHERE status='pending_approval' ORDER BY created_at,rowid")
        return [{**self.view(row), "user_id": row["user_id"], "thread_id": row["thread_id"]} for row in rows]

    def awaiting_returns(self) -> list[dict]:
        rows = self.store.query("SELECT * FROM refund_requests WHERE status='awaiting_return' ORDER BY created_at,rowid")
        return [{**self.view(row), "user_id": row["user_id"], "thread_id": row["thread_id"]} for row in rows]

    def ticket_details(self, ticket_id: int) -> dict:
        ticket = self.store.get_ticket(ticket_id)
        if ticket is None:
            raise RefundError("工单不存在", 404)
        thread_id = ticket["thread_id"]
        messages = self.store.transcript(thread_id, include_ids=True) if thread_id else []
        refunds = []
        if thread_id:
            self.refresh_thread(thread_id, ticket["user_id"])
            for row in self.for_thread(thread_id, ticket["user_id"]):
                events = self.store.query("SELECT status,actor,detail,created_at FROM refund_events WHERE refund_id=?"
                                          " ORDER BY event_id", (row["refund_id"],))
                refunds.append({**self.view(row), "payment_ref": row["payment_ref"], "op_id": row["op_id"],
                                "message_id": row["message_id"], "events": events})
        return {**ticket, "messages": messages, "refunds": refunds}
