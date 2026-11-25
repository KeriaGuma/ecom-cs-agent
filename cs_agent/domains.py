"""领域扩展实验：会员积分、发票、优惠券、维修、价格保护五个新业务领域。

每个领域包含：带权限校验的工具（能不能办、办多少由代码判断）、Worker 提示词、给调度员看的职责说明、
以及单 Agent 提示词里对应的使用说明。两种架构拿到的业务信息相同。

档位：T0 = 现有 3 个领域；T1 = + 积分、发票；T2 = + 优惠券、维修、价保。
"""
from __future__ import annotations

import re
from datetime import date
from typing import Callable, Literal

from langchain_core.tools import BaseTool, tool

from .store import ShopStore, to_json
from .tools import NOT_FOUND, _owned

DOMAINS = ("points", "invoice", "coupon", "repair", "price")
TIERS = {"T0": (), "T1": ("points", "invoice"), "T2": DOMAINS}
TAX_ID = re.compile(r"^[0-9A-Z]{15,20}$")


def parse_domains(value: str) -> tuple[str, ...]:
    """DOMAINS 配置：档位名（T0/T1/T2）或逗号分隔的领域列表。"""
    value = (value or "").strip()
    if value in TIERS:
        return TIERS[value]
    picked = tuple(d.strip() for d in value.split(",") if d.strip())
    unknown = set(picked) - set(DOMAINS)
    if unknown:
        raise ValueError(f"未知领域：{unknown}")
    return tuple(d for d in DOMAINS if d in picked)


def _days_since_delivery(order: dict, today: date) -> int:
    return (today - date.fromisoformat(order["delivered_at"])).days


def _common(store: ShopStore, user_id: str) -> list[BaseTool]:
    @tool
    def list_my_orders() -> str:
        """列出当前用户的全部订单。用户没说订单号时先调用它，按商品名找到订单。"""
        return to_json(store.list_orders(user_id))

    @tool
    def get_order(order_id: str) -> str:
        """查询订单详情，包括商品、金额、状态、签收日期。"""
        order = _owned(store, user_id, order_id)
        return to_json(order) if order else NOT_FOUND

    return [list_my_orders, get_order]


# ---------------- 会员积分 ----------------
def points_tools(store: ShopStore, user_id: str, today: Callable[[], date]) -> list[BaseTool]:
    def balance() -> int:
        rows = store.query("SELECT balance FROM points WHERE user_id = ?", (user_id,))
        return int(rows[0]["balance"]) if rows else 0

    @tool
    def get_points() -> str:
        """查询当前用户的积分余额。"""
        return to_json({"balance": balance()})

    @tool
    def reissue_points(order_id: str) -> str:
        """为签收后漏发积分的订单补发积分（每元 1 分）。只能补发已签收、未退款、确实漏发的订单。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        if order["status"] != "delivered":
            return f"订单状态为 {order['status']}，只有已签收且未退款的订单才能发放积分"
        if store.query("SELECT 1 FROM points_credit WHERE order_id = ?", (order["order_id"],)):
            return "该订单的积分已经发放过，无需补发"
        pts = int(order["amount"])
        store.execute("INSERT INTO points_credit (order_id, points, reissued) VALUES (?,?,1)", (order["order_id"], pts))
        store.execute("UPDATE points SET balance = balance + ? WHERE user_id = ?", (pts, user_id))
        return f"已为订单 {order['order_id']} 补发 {pts} 积分，当前余额 {balance()}"

    @tool
    def exchange_points(value: Literal[5, 10, 20]) -> str:
        """用积分兑换无门槛店铺券：5 元券 500 积分，10 元券 1000 积分，20 元券 2000 积分。"""
        cost = {5: 500, 10: 1000, 20: 2000}.get(int(value))
        if cost is None:
            return "只能兑换 5、10、20 元的券"
        if balance() < cost:
            return f"积分不足：兑换 {value} 元券需要 {cost} 积分，当前余额 {balance()}"
        n = len(store.query("SELECT 1 FROM coupons WHERE coupon_id LIKE 'P%'")) + 1
        store.execute("UPDATE points SET balance = balance - ? WHERE user_id = ?", (cost, user_id))
        store.execute("INSERT INTO coupons VALUES (?,?,?,?,?,?,?)",
                      (f"P{n}", user_id, f"无门槛 {value} 元（积分兑换）", float(value), "unused", None,
                       date.fromordinal(today().toordinal() + 30).isoformat()))
        return f"兑换成功：获得 {value} 元无门槛券，消耗 {cost} 积分，当前余额 {balance()}"

    return _common(store, user_id) + [get_points, reissue_points, exchange_points]


# ---------------- 发票 ----------------
def invoice_tools(store: ShopStore, user_id: str, today: Callable[[], date]) -> list[BaseTool]:
    @tool
    def issue_invoice(order_id: str, invoice_type: Literal["normal", "special"], title: str, tax_id: str = "") -> str:
        """为订单开发票。invoice_type：normal 电子普通发票（抬头可以是“个人”），special 增值税专用发票
        （必须提供公司抬头和税号）。只有已签收、未退款的订单能开，每单只能开一次。不要编造用户没提供的抬头或税号。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        if order["status"] != "delivered":
            return f"订单状态为 {order['status']}，只有已签收、未退款的订单才能开票"
        if store.query("SELECT 1 FROM invoices WHERE order_id = ?", (order["order_id"],)):
            return "该订单已经开过发票；如需修改抬头或税号，可以申请换开一次"
        title, tax_id = (title or "").strip(), (tax_id or "").strip().upper()
        if not title:
            return "缺少发票抬头，请向用户确认"
        if invoice_type == "special":
            if title == "个人":
                return "增值税专用发票的抬头必须是公司名称"
            if not TAX_ID.match(tax_id):
                return "增值税专用发票需要有效的税号（15～20 位字母或数字），请向用户确认"
        store.execute("INSERT INTO invoices (order_id, user_id, type, title, tax_id) VALUES (?,?,?,?,?)",
                      (order["order_id"], user_id, invoice_type, title, tax_id))
        kind = "增值税专用发票" if invoice_type == "special" else "电子普通发票"
        return f"已为订单 {order['order_id']} 开具{kind}，抬头：{title}" + (f"，税号：{tax_id}" if tax_id else "")

    @tool
    def reissue_invoice(order_id: str, title: str, tax_id: str = "") -> str:
        """换开已开具的发票（修改抬头或税号），每个订单只能换开一次。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        rows = store.query("SELECT * FROM invoices WHERE order_id = ?", (order["order_id"],))
        if not rows:
            return "该订单还没有开过发票，请直接开票"
        inv = rows[0]
        if inv["reissued"]:
            return "该订单的发票已经换开过一次，不能再换开"
        tax_id = (tax_id or "").strip().upper()
        if inv["type"] == "special" and not TAX_ID.match(tax_id):
            return "增值税专用发票换开需要有效的税号，请向用户确认"
        store.execute("UPDATE invoices SET title = ?, tax_id = ?, reissued = 1 WHERE order_id = ?",
                      ((title or "").strip() or inv["title"], tax_id, order["order_id"]))
        return f"已为订单 {order['order_id']} 换开发票，新抬头：{(title or '').strip() or inv['title']}"

    return _common(store, user_id) + [issue_invoice, reissue_invoice]


# ---------------- 优惠券 ----------------
def coupon_tools(store: ShopStore, user_id: str, today: Callable[[], date]) -> list[BaseTool]:
    @tool
    def list_my_coupons() -> str:
        """列出当前用户的优惠券（券号、名称、状态、用于哪个订单、有效期）。"""
        return to_json(store.query("SELECT coupon_id, name, status, used_order, expire FROM coupons WHERE user_id = ?",
                                   (user_id,)))

    @tool
    def return_coupon(order_id: str) -> str:
        """退还订单使用的优惠券。只有订单已经全额退款、且券没过期时才能退还。退款本身不在这里办理。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        rows = store.query("SELECT * FROM coupons WHERE used_order = ? AND user_id = ?", (order["order_id"], user_id))
        if not rows:
            return "该订单没有使用优惠券"
        c = rows[0]
        if order["status"] != "refunded":
            return "订单还没有退款，优惠券不能退还；全额退款后才能退券"
        if c["expire"] < today().isoformat():
            return f"优惠券 {c['coupon_id']}（{c['name']}）已过期，不予退还"
        store.execute("UPDATE coupons SET status = 'unused', used_order = NULL WHERE coupon_id = ?", (c["coupon_id"],))
        return f"已退还优惠券 {c['coupon_id']}（{c['name']}），有效期至 {c['expire']}"

    return _common(store, user_id) + [list_my_coupons, return_coupon]


# ---------------- 维修 ----------------
def repair_tools(store: ShopStore, user_id: str, today: Callable[[], date]) -> list[BaseTool]:
    @tool
    def apply_repair(order_id: str, issue: str, cause: Literal["quality", "human"]) -> str:
        """为已签收的电子产品申请维修。cause：quality 质量问题；human 人为损坏（进水、摔坏、私自拆机等，付费维修）。
        签收 15 天内的质量问题不受理维修，应走退换货。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        if order["status"] != "delivered" or not order.get("delivered_at"):
            return f"订单状态为 {order['status']}，只有已签收的商品才能申请维修"
        prod = store.query("SELECT electronic FROM products WHERE item = ?", (order["item"],))
        if not prod or not prod[0]["electronic"]:
            return f"{order['item']}不是电子产品，不提供保修维修服务"
        days = _days_since_delivery(order, today())
        if cause == "quality" and days <= 15:
            return f"签收 {days} 天，15 天内的质量问题请申请退换货，不受理维修"
        if days > 365:
            return f"签收已 {days} 天，超过 1 年保修期"
        if store.query("SELECT 1 FROM repairs WHERE order_id = ? AND status = 'pending'", (order["order_id"],)):
            return "该订单已有处理中的维修单，不能重复申请"
        paid = cause == "human"
        store.execute("INSERT INTO repairs (order_id, user_id, issue, cause, paid, status) VALUES (?,?,?,?,?,?)",
                      (order["order_id"], user_id, issue, cause, int(paid), "pending"))
        tid = store.query("SELECT MAX(ticket_id) AS t FROM repairs")[0]["t"]
        return (f"已为订单 {order['order_id']} 创建维修单 #{tid}"
                + ("（人为损坏，不在免费保修范围内，按付费维修处理）" if paid else "（保修期内免费维修）"))

    @tool
    def get_repair_status(order_id: str) -> str:
        """查询订单的维修单状态。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        rows = store.query("SELECT ticket_id, status, paid FROM repairs WHERE order_id = ?", (order["order_id"],))
        return to_json(rows) if rows else "该订单没有维修单"

    return _common(store, user_id) + [apply_repair, get_repair_status]


# ---------------- 价格保护 ----------------
def price_tools(store: ShopStore, user_id: str, today: Callable[[], date]) -> list[BaseTool]:
    @tool
    def apply_price_protect(order_id: str) -> str:
        """申请价格保护：签收 7 天内、当前售价低于实付价、非秒杀订单，退还差价，每单一次。"""
        order = _owned(store, user_id, order_id)
        if not order:
            return NOT_FOUND
        if order["status"] != "delivered" or not order.get("delivered_at"):
            return f"订单状态为 {order['status']}，只有已签收的订单才能申请价保"
        days = _days_since_delivery(order, today())
        if days > 7:
            return f"签收已 {days} 天，超过 7 天价保期"
        if store.query("SELECT 1 FROM order_flags WHERE order_id = ? AND flash_sale = 1", (order["order_id"],)):
            return "秒杀购入的订单不在价保范围内"
        if store.query("SELECT 1 FROM price_protect WHERE order_id = ?", (order["order_id"],)):
            return "该订单已经申请过价保"
        prod = store.query("SELECT price FROM products WHERE item = ?", (order["item"],))
        now = float(prod[0]["price"]) if prod else float(order["amount"])
        diff = round(float(order["amount"]) - now, 2)
        if diff <= 0:
            return f"{order['item']}当前售价 {now:.0f} 元，没有低于实付价 {order['amount']:.0f} 元，无需价保"
        store.execute("INSERT INTO price_protect VALUES (?,?)", (order["order_id"], diff))
        return f"价保成功：订单 {order['order_id']} 退还差价 {diff:.2f} 元，原路退回"

    return _common(store, user_id) + [apply_price_protect]


# ---------------- Worker、调度员、单 Agent 的说明 ----------------
SPECS = {
    "points": dict(
        worker="points_agent", tools=points_tools,
        route="points_agent：会员积分——查询积分余额、订单积分漏发补发、积分兑换优惠券",
        prompt="你是会员积分专员，可以查询积分余额、为漏发积分的订单补发积分、用积分兑换无门槛券（5/10/20 元）。"
               "能否补发、能否兑换一律以工具返回为准。",
        single="会员积分：get_points 查余额；reissue_points 为漏发积分的订单补发；exchange_points 用积分兑换无门槛券（5/10/20 元）。",
    ),
    "invoice": dict(
        worker="invoice_agent", tools=invoice_tools,
        route="invoice_agent：为具体订单开具或换开发票（只是询问开票政策、多久能开好，仍交给 kb_agent）",
        prompt="你是发票专员，可以为已签收的订单开具电子普通发票或增值税专用发票，以及换开一次。"
               "专用发票必须有公司抬头和税号；用户没提供的信息不要编造，在结论里请用户补充。",
        single="发票：issue_invoice 为订单开电子普通发票或增值税专用发票（专票必须有公司抬头和税号，用户没提供不要编造）；"
               "reissue_invoice 换开一次。只是询问开票政策时用 search_policy。",
    ),
    "coupon": dict(
        worker="coupon_agent", tools=coupon_tools,
        route="coupon_agent：查询优惠券、为已退款订单退还优惠券（用户同时要求退款时，先 refund_agent 再 coupon_agent）",
        prompt="你是优惠券专员，可以查询用户的优惠券，以及为已全额退款的订单退还未过期的优惠券。退款本身不归你处理。",
        single="优惠券：list_my_coupons 查券；return_coupon 为已全额退款的订单退还未过期的券。用户要求“退款并退券”时，先退款再退券。",
    ),
    "repair": dict(
        worker="repair_agent", tools=repair_tools,
        route="repair_agent：电子产品申请维修、查询维修单（用户要求“修”时交给它；要求“退”时交给 refund_agent）",
        prompt="你是维修专员，可以为已签收的电子产品申请维修、查询维修单。进水、摔坏、私自拆机等属于人为损坏（human），"
               "其余为质量问题（quality）。能否受理以工具返回为准；签收 15 天内的质量问题会被拒绝，此时告诉用户可以申请退换货。",
        single="维修：apply_repair 为已签收的电子产品申请维修（进水、摔坏、私自拆机为 human 人为损坏，其余为 quality），"
               "签收 15 天内的质量问题不受理维修、应走退换货；get_repair_status 查维修单。",
    ),
    "price": dict(
        worker="price_agent", tools=price_tools,
        route="price_agent：价格保护——商品降价后申请退差价（不是退款）",
        prompt="你是价格保护专员，可以为签收 7 天内、降价了的订单申请价保并退还差价。能否价保、差价多少以工具返回为准。",
        single="价格保护：apply_price_protect 为签收 7 天内、降价了的非秒杀订单退还差价（降价补差价走价保，不是退款）。",
    ),
}


def domain_workers(domains: tuple[str, ...]) -> list[str]:
    return [SPECS[d]["worker"] for d in domains]


def domain_tools(domain: str, store: ShopStore, user_id: str, today: Callable[[], date]) -> list[BaseTool]:
    return SPECS[domain]["tools"](store, user_id, today)
