"""命令行坐席与售后审批工作台：python -m cs_agent.desk

支持按优先级查看并领取人工工单、审批大额售后、核验退货入库及触发主动支付对账。
"""
from __future__ import annotations

import argparse
import json

from .config import settings
from .handoff import PRIORITY_NAME
from .refunds import RefundError, RefundService
from .store import ShopStore


def render(ticket: dict) -> str:
    s = ticket["summary"]
    lines = [
        f"#{ticket['ticket_id']}  [{PRIORITY_NAME.get(ticket['priority'], '?')}优先级]  用户 {ticket['user_id']}  "
        f"{ticket['created_at']}  情绪：{s.get('emotion', '未知')}  摘要来源：{s.get('summary_by', '?')}",
        f"    意图：{s.get('intent', '')}",
        f"    订单：{'、'.join(s.get('order_ids') or []) or '无'}    转人工原因：{s.get('reason', '')}",
    ]
    for d in s.get("done") or []:
        lines.append(f"    已处理：{d}")
    lines.append(f"    待处理：{s.get('pending', '')}")
    lines.append(f"    建议：{s.get('suggestion', '')}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="坐席与售后审批工作台")
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--close", type=int, help="标记工单已处理")
    action.add_argument("--ticket", type=int, help="查看工单、原始会话和退款处理记录")
    action.add_argument("--claim", type=int, help="领取并查看工单")
    action.add_argument("--approvals", action="store_true", help="查看待审批退款与待验货退货单")
    action.add_argument("--approve", metavar="申请编号", help="通过退款审批")
    action.add_argument("--reject", metavar="申请编号", help="驳回退款申请")
    action.add_argument("--receive", metavar="申请编号", help="确认退货包裹已验收入库并执行原路退款")
    action.add_argument("--reconcile", metavar="申请编号", help="核对支付结果，必要时幂等重试")
    ap.add_argument("--staff", default="s01", help="坐席或审批员编号")
    ap.add_argument("--comment", default="", help="退款审核备注或退货快递单号")
    args = ap.parse_args()
    store = ShopStore(settings.db_path)
    refunds = RefundService(store, settings.refund_auto_approve_limit)
    try:
        if args.ticket is not None or args.claim is not None:
            ticket_id = args.ticket if args.ticket is not None else args.claim
            if args.claim is not None and not store.claim_ticket(ticket_id, args.staff):
                raise RefundError("工单不可领取")
            print(json.dumps(refunds.ticket_details(ticket_id), ensure_ascii=False, indent=2))
            return
        if args.approvals or args.approve or args.reject or args.receive or args.reconcile:
            if args.approvals:
                result = {"pending_approvals": refunds.pending_approvals(),
                          "awaiting_returns": refunds.awaiting_returns()}
            elif args.receive:
                result = refunds.confirm_return(args.receive, args.staff, args.comment)
            elif args.reconcile:
                result = refunds.reconcile(args.reconcile, args.staff)
            else:
                result = refunds.decide(args.approve or args.reject, args.staff, bool(args.approve), args.comment)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
    except RefundError as exc:
        ap.exit(1, f"操作失败：{exc}\n")
    if args.close is not None:
        closed = store.close_ticket(args.close, staff=args.staff)
        print(f"工单 #{args.close} " + ("已关闭" if closed else "无法关闭，请核对工单和坐席"))
        return
    tickets = store.open_tickets()
    print(f"待处理工单 {len(tickets)} 张（按优先级、创建时间排序）\n")
    for t in tickets:
        print(render(t) + "\n")


if __name__ == "__main__":
    main()
