"""工具数量扩展实验用的干扰工具。

模拟真实电商客服平台上“挂着很多工具，但和当前任务无关”的情况：会员、优惠券、积分、购物车、商品、门店……
- 全部只读，返回固定的演示数据，不会改动订单。
- 一部分是“近似干扰”：名字或用途和真实工具接近（如售后工单进度、发票抬头、价格走势），用来观察模型会不会选错工具。
- 每个工具标注所属领域（order / refund / kb）：单 Agent 一次挂上全部；多 Agent 只挂到对应领域的 Worker 上。

按固定顺序取前 N 个，领域交错排列，保证每一档的领域分布大致均匀。不改动任何提示词。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model

# (名称, 描述, 参数 [(名, 说明)], 领域, 返回数据)
CATALOG: list[tuple[str, str, list[tuple[str, str]], str, Any]] = [
    ("get_after_sale_ticket_progress", "查询售后工单（维修、换货申请）的处理进度。", [("ticket_id", "售后工单号")], "refund",
     {"status": "未找到该售后工单"}),
    ("list_cart_items", "查看当前用户购物车里的商品。", [], "order",
     [{"item": "无线鼠标", "price": 89, "qty": 1}]),
    ("get_policy_update_log", "查看售后政策的历史修订记录（修订日期和修订摘要）。", [], "kb",
     [{"date": "2026-06-01", "summary": "调整大促期间发货时效说明"}]),
    ("list_my_coupons", "列出当前用户账户里可用的优惠券。", [], "refund",
     [{"name": "满 199 减 20", "expire": "2026-12-31"}]),
    ("query_product_stock", "查询某个商品在售规格的库存。", [("product_name", "商品名称")], "order",
     {"in_stock": True, "variants": 3}),
    ("get_points_balance", "查询当前用户的积分余额和即将过期的积分。", [], "kb",
     {"points": 1260, "expiring_this_month": 0}),
    ("get_invoice_titles", "列出当前用户保存的发票抬头。", [], "refund",
     [{"title": "个人", "type": "电子普通发票"}]),
    ("list_favorites", "查看当前用户收藏的商品。", [], "order",
     [{"item": "空气炸锅", "price": 299}]),
    ("get_member_level", "查询当前用户的会员等级和近 12 个月实付金额。", [], "kb",
     {"level": "银卡", "paid_12m": 2310}),
    ("get_wallet_balance", "查询当前用户的账户余额（不含银行卡）。", [], "refund",
     {"balance": 0.0}),
    ("get_product_specs", "查询商品的规格参数。", [("product_name", "商品名称")], "order",
     {"specs": "以商品详情页为准"}),
    ("get_product_reviews", "查看商品的用户评价摘要。", [("product_name", "商品名称")], "kb",
     {"rating": 4.7, "count": 1832}),
    ("get_price_history", "查询商品近 30 天的价格走势。", [("product_name", "商品名称")], "refund",
     {"min": 0, "max": 0, "note": "暂无价格变动"}),
    ("list_saved_addresses", "列出当前用户保存的收货地址簿。", [], "order",
     [{"tag": "家", "address": "杭州市西湖区文三路 1 号"}]),
    ("get_product_qa", "查看商品详情页的买家问答。", [("product_name", "商品名称")], "kb",
     [{"q": "续航多久", "a": "约 20 小时"}]),
    ("list_gift_cards", "列出当前用户绑定的礼品卡。", [], "refund", []),
    ("find_nearby_stores", "按城市查询线下门店。", [("city", "城市")], "order",
     [{"store": "杭州湖滨店", "hours": "10:00-22:00"}]),
    ("get_security_settings", "查看账户的安全设置（登录保护、绑定设备数）。", [], "kb",
     {"login_protection": True, "devices": 2}),
    ("get_installment_plans", "查询某个金额可用的分期方案。", [("amount", "金额")], "refund",
     [{"periods": 3, "fee_rate": "0%"}]),
    ("list_browse_history", "查看当前用户最近浏览的商品。", [], "order",
     [{"item": "降噪耳机", "viewed_at": "昨天"}]),
    ("get_flash_sales", "查询正在进行的限时秒杀活动。", [], "kb", [{"item": "保温杯", "price": 49}]),
    ("get_refund_account_info", "查看退款默认退回的支付账户（脱敏）。", [], "refund",
     {"account": "支付宝 138****0000"}),
    ("list_delivery_stations", "按城市查询快递自提点。", [("city", "城市")], "order",
     [{"station": "文三路菜鸟驿站"}]),
    ("get_live_streams", "查看店铺正在进行的直播。", [], "kb", []),
    ("get_price_protect_records", "查看历史价保申请记录。", [], "refund", []),
    ("get_shipping_template", "查询某个商品适用的运费模板。", [("product_name", "商品名称")], "order",
     {"template": "满 99 包邮"}),
    ("list_announcements", "查看店铺公告。", [], "kb", [{"title": "双十一发货时间说明"}]),
    ("get_payment_records", "查看当前用户最近的支付流水（不含订单详情）。", [], "refund",
     [{"amount": 459, "channel": "微信支付"}]),
    ("get_warranty_card", "查询商品的电子保修卡信息。", [("product_name", "商品名称")], "order",
     {"warranty": "以品牌保修卡为准"}),
    ("get_sign_in_record", "查看当前用户本月的签到记录。", [], "kb", {"days": 7}),
    ("list_subscriptions", "查看当前用户订阅的降价提醒和到货提醒。", [], "refund", []),
    ("get_order_comments", "查看当前用户对已完成订单的评价。", [], "order", []),
    ("get_customer_service_hours", "查询人工客服的在线时间。", [], "kb", {"hours": "9:00-22:00"}),
    ("list_trade_in_offers", "查询以旧换新的回收报价。", [("product_name", "商品名称")], "refund",
     {"offer": 0, "note": "暂无回收报价"}),
    ("get_bundle_offers", "查询商品可参与的组合套餐。", [("product_name", "商品名称")], "order", []),
    ("get_lottery_chances", "查询当前用户的抽奖次数。", [], "kb", {"chances": 0}),
    ("get_credit_limit", "查询店铺白条的可用额度。", [], "refund", {"limit": 0}),
    ("get_size_guide", "查询服饰类商品的尺码表。", [("product_name", "商品名称")], "order",
     {"guide": "以商品详情页尺码表为准"}),
    ("get_faq_categories", "列出帮助中心的问题分类目录。", [], "kb", ["订单", "售后", "支付", "会员"]),
    ("get_referral_rewards", "查询邀请好友获得的奖励。", [], "refund", []),
]


def _make(name: str, desc: str, params: list[tuple[str, str]], data: Any) -> BaseTool:
    fields = {p: (str, Field(description=d)) for p, d in params}
    schema = create_model(f"{name}_args", **fields)
    payload = json.dumps(data, ensure_ascii=False)
    return StructuredTool.from_function(func=lambda **_: payload, name=name, description=desc, args_schema=schema)


def distractor_tools(n: int, domain: str | None = None) -> list[BaseTool]:
    """前 n 个干扰工具；指定 domain 时只返回其中属于该领域的（多 Agent 按领域分给 Worker）。"""
    picked = CATALOG[: max(0, min(n, len(CATALOG)))]
    return [_make(name, desc, params, data) for name, desc, params, d, data in picked if domain is None or d == domain]


WORKER_DOMAIN = {"order_agent": "order", "refund_agent": "refund", "kb_agent": "kb"}
