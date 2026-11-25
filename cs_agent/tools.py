"""受控业务工具集。

安全与确定性原则：
- `user_id` 在构造工具时由宿主闭包或 MCP `_meta` 绑定，不暴露为 LLM 参数，从根源阻断越权冒充；
- 订单归属在工具层强校验；“订单不存在”与“无权访问”返回完全一致的提示，防止枚举探测；
- 退款金额、售后类型、运费承担方由规则引擎计算，工具不接受模型传入的金额；
- 修改地址工具内置收件人/电话守卫（含带分隔符手机号检测），拦截超范围参数混入。
"""
from __future__ import annotations

import re
from datetime import date
from typing import Callable, Literal, Optional

from langchain_core.tools import BaseTool, tool

from .refunds import RefundError
from .rules import ReasonType, evaluate_refund
from .store import ShopStore, to_json

NOT_FOUND = "未找到该订单，或该订单不属于当前用户"

_CONTACT_WORDS = re.compile(r"收件人|收货人|联系人|联系电话|联系方式|电话|手机|姓名")
_PHONE_LIKE = re.compile(r"(?<!\d)(?:(?:\+?86[\s-]?)?1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}|0\d{2,3}-?\d{7,8})(?!\d)")
ADDRESS_REJECTED = (
    "未修改：new_address 里带了收件人或联系电话。这个工具只能修改收货地址本身，"
    "订单上没有可以修改的收件人和联系电话。请只把地址部分传给 new_address 重新调用，"
    "并如实告诉用户：收件人和联系电话这里改不了，不能说已经更新或登记。"
)


def contact_in_address(address: str) -> bool:
    return bool(_CONTACT_WORDS.search(address) or _PHONE_LIKE.search(address))


def _owned(store: ShopStore, user_id: str, order_id: str) -> Optional[dict]:
    order = store.get_order(order_id.strip().upper())
    if not order or order["user_id"] != user_id:
        return None
    return order


OP_STATUS = {
    ("refund", "pending"): "退款申请已提交，等待人工审核",
    ("refund", "approved"): "人工审核通过，已退款",
    ("refund", "rejected"): "人工审核驳回，未退款，订单保持原状",
    ("refund", "failed"): "人工审核通过，但执行退款时系统出错，暂未退款",
    ("refund", "done"): "退款成功（未超过自动审批额度，直接退款）",
    ("refund", "awaiting_return"): "已同意退货退款，等待买家寄回商品验货后原路退款",
    ("refund", "payment_ready"): "已确认退款申请，等待执行原路退款",
    ("refund", "payment_pending"): "支付方正在处理退款",
    ("refund", "payment_unknown"): "支付结果尚未确认，请勿重复申请",
    ("refund", "payment_succeeded"): "支付方已确认退款成功",
    ("refund", "payment_failed"): "支付方确认退款失败",
    ("address", "done"): "收货地址已修改",
}


def _ops_enabled(ops_log: Optional[bool]) -> bool:
    if ops_log is None:
        from .config import settings

        return settings.ops_log
    return ops_log


def op_records(store: ShopStore, order_id: str, op: Optional[str] = None) -> list[dict]:
    """返回订单历史操作留痕，弥补跨轮对话中工具调用历史被裁剪后的状态可见性。"""
    out = []
    for r in store.list_ops(order_id):
        if op and r["op"] != op:
            continue
        rec = {"时间": r["created_at"], "操作": "退款申请" if r["op"] == "refund" else "修改收货地址",
               "结果": OP_STATUS.get((r["op"], r["status"]), r["status"])}
        if r["op"] == "refund":
            rec["金额"] = r["detail"].get("amount")
            if r["detail"].get("refund_id"):
                rec["申请编号"] = r["detail"]["refund_id"]
        else:
            rec["新地址"] = r["detail"].get("address")
        if r["decided_at"]:
            rec["审核时间"] = r["decided_at"]
        out.append(rec)
    return out


def _order_view(store: ShopStore, order: dict, ops_log: Optional[bool]) -> str:
    recs = op_records(store, order["order_id"]) if _ops_enabled(ops_log) else []
    return to_json({**order, "操作记录": recs} if recs else order)


def make_order_tools(store: ShopStore, user_id: str, ops_log: Optional[bool] = None,
                     address_guard: Optional[bool] = None) -> list[BaseTool]:
    if address_guard is None:
        from .config import settings

        address_guard = settings.address_guard

    @tool
    def list_my_orders() -> str:
        """列出当前用户的全部订单（订单号、商品、金额、状态）。用户没说订单号时先调用它。"""
        return to_json(store.list_orders(user_id))

    @tool
    def get_order(order_id: str) -> str:
        """查询订单详情，包括商品、金额、状态、下单和签收日期、收货地址，以及退款申请、改地址等操作记录。"""
        order = _owned(store, user_id, order_id)
        return _order_view(store, order, ops_log) if order else NOT_FOUND

    @tool
    def get_logistics(order_id: str) -> str:
        """查询订单的物流轨迹。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        events = store.get_logistics(order["order_id"])
        return to_json({"status": order["status"], "events": events}) if events else "该订单暂无物流信息（可能尚未发货）"

    @tool
    def change_address(order_id: str, new_address: str) -> str:
        """修改订单收货地址。仅未发货（status=paid）的订单可以修改。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        if order["status"] != "paid":
            return "订单已发货或已完成，无法修改地址；可联系快递改派或拒收"
        if address_guard and contact_in_address(new_address):
            return ADDRESS_REJECTED
        store.update_address(order["order_id"], new_address)
        store.add_op(order["order_id"], user_id, "address", {"address": new_address}, "done")
        return f"已将订单 {order['order_id']} 的收货地址修改为：{new_address}"

    if address_guard:
        change_address.description += "new_address 只填地址本身，不含收件人和电话。"
    return [list_my_orders, get_order, get_logistics, change_address]


def order_tools_for(store: ShopStore, user_id: str, source: str = "local") -> list[BaseTool]:
    """订单工具统一入口：进程内直接调用，或通过 MCP Server（stdio / Streamable HTTP）调用。"""
    if source == "mcp":
        from .mcp_client import order_tools

        return order_tools(store.path, user_id)
    if source == "mcp-http":
        from .config import settings
        from .mcp_client import order_tools_http

        return order_tools_http(settings.mcp_url, settings.mcp_token, settings.mcp_user_key, user_id)
    return make_order_tools(store, user_id)


def make_refund_tools(
    store: ShopStore,
    user_id: str,
    auto_approve_limit: float,
    on_pending: Optional[Callable[[dict], None]] = None,
    today: Callable[[], date] = date.today,
    unknown_reason: Optional[bool] = None,
    ops_log: Optional[bool] = None,
    prepare: Optional[Callable[[str, ReasonType], dict]] = None,
) -> list[BaseTool]:
    """构造售后退款工具：
    - 线上服务传入 `prepare` 时挂载 `prepare_refund`（生成确认单草稿，由用户核对确认后进入退货/退款状态机）；
    - 评测兼容模式未传 `prepare` 时挂载 `submit_refund` 供历史评测集回归对比。
    """
    if unknown_reason is None:
        from .config import settings

        unknown_reason = settings.refund_rule == 2

    def _with_history(order: dict, result: dict) -> str:
        recs = op_records(store, order["order_id"], "refund") if _ops_enabled(ops_log) else []
        return to_json({**result, "该订单的退款申请记录": recs} if recs else result)

    @tool
    def check_refund(order_id: str, reason_type: Literal["no_reason", "quality"]) -> str:
        """查询订单能否退款、可退金额、运费由谁承担、是否需要人工审批。只查询，不提交。
        reason_type：no_reason 表示不想要了/无理由，quality 表示商品有质量问题。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        return _with_history(order, evaluate_refund(order, reason_type, today(), auto_approve_limit).to_dict())

    @tool("check_refund")
    def check_refund_v2(order_id: str, reason_type: Literal["no_reason", "quality", "unknown"]) -> str:
        """查询订单能否退款、售后类型（仅退款/退货退款）、可退金额、运费由谁承担、是否需要人工审批。只查询，不提交。
        reason_type：no_reason 表示不想要了/无理由，quality 表示商品有质量问题；
        用户没说原因、无法判断时填 unknown，会同时返回两种原因下的结果。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        if reason_type != "unknown":
            return _with_history(order, evaluate_refund(order, reason_type, today(), auto_approve_limit).to_dict())
        return _with_history(order, {
            "无理由": evaluate_refund(order, "no_reason", today(), auto_approve_limit).to_dict(),
            "质量问题": evaluate_refund(order, "quality", today(), auto_approve_limit).to_dict(),
            "提示": "用户尚未说明退款原因，提交前需要先问清原因",
        })

    @tool
    def submit_refund(order_id: str, reason_type: Literal["no_reason", "quality"]) -> str:
        """提交退款申请（评测兼容工具）。只有用户明确表示要退款时才调用。金额由系统规则计算。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        decision = evaluate_refund(order, reason_type, today(), auto_approve_limit)
        if not decision.eligible:
            return f"无法退款：{decision.reason}"
        if decision.needs_approval:
            op_id = store.add_op(order["order_id"], user_id, "refund",
                                 {"amount": decision.refund_amount, "reason_type": reason_type}, "pending")
            if on_pending is not None:
                on_pending({
                    "op_id": op_id,
                    "order_id": order["order_id"],
                    "item": order["item"],
                    "amount": decision.refund_amount,
                    "reason_type": reason_type,
                    "rule_reason": decision.reason,
                })
            return f"退款申请已提交，金额 {decision.refund_amount:.2f} 元超过自动审批额度，需人工审核"
        detail = {"amount": decision.refund_amount, "reason_type": reason_type}
        if not store.mark_refunded(order["order_id"], decision.refund_amount, operation=(user_id, detail)):
            return f"订单 {order['order_id']} 已退款，无需重复处理"
        return f"退款成功：{decision.refund_amount:.2f} 元将原路退回。{decision.reason}"

    @tool
    def prepare_refund(order_id: str, reason_type: ReasonType) -> str:
        """生成给用户核对的退款确认单。用户在界面核对确认后，才会进入退货或退款流程。"""
        assert prepare is not None
        try:
            return to_json(prepare(order_id, reason_type))
        except RefundError as exc:
            return str(exc)

    @tool
    def get_order(order_id: str) -> str:
        """查询订单详情，包括商品、金额、状态、签收日期，以及退款申请等操作记录。"""
        order = _owned(store, user_id, order_id)
        return _order_view(store, order, ops_log) if order else NOT_FOUND

    @tool
    def list_my_orders() -> str:
        """列出当前用户的全部订单。用户没说订单号时先调用它。"""
        return to_json(store.list_orders(user_id))

    return [list_my_orders, get_order, check_refund_v2 if unknown_reason else check_refund,
            prepare_refund if prepare is not None else submit_refund]


def make_kb_tools(kb) -> list[BaseTool]:
    @tool
    def search_policy(query: str) -> str:
        """检索售后政策知识库（退换货、维修保修、运费物流、价保优惠、支付发票、会员积分、账户安全、特殊商品等规则），
        返回最相关的条款和出处。query 写成完整的检索问题，指代要替换成具体对象。"""
        hits = kb.search(query, k=3)
        if not hits:
            return "知识库中没有找到相关条款"
        return "\n\n".join(f"[{h.source} · {h.title}]\n{h.text}" for h in hits)

    return [search_policy]
