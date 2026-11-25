"""命令行客服入口：python -m cs_agent.cli --user u001 --session demo

会话上下文与售后申请持久化存储；用户确认退款草稿后按业务类型进入仅退款、退货验货或人工审批状态机。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from .agent import build_agent_runner
from .config import settings
from .desk import render as render_ticket
from .llm import get_llm
from .refunds import RefundError, RefundService
from .store import ShopStore


def main() -> None:
    ap = argparse.ArgumentParser(description="电商智能客服 Agent（命令行）")
    ap.add_argument("--user", default="u001", help="当前登录用户（演示数据有 u001、u002）")
    ap.add_argument("--session", default="demo", help="会话 ID，相同 ID 会接着上次的对话")
    ap.add_argument("--init-db", action="store_true", help="重置演示订单数据")
    ap.add_argument("--verbose", action="store_true", help="打印内部轨迹与交接工单")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if settings.pii_mask:
        from .pii import install_log_masking

        install_log_masking()

    store = ShopStore(settings.db_path)
    if args.init_db or not Path(settings.db_path).exists():
        store.init()
        print(f"[已初始化演示数据] {settings.db_path}")

    try:
        llm = get_llm(settings.agent_model)
    except RuntimeError as e:
        raise SystemExit(str(e))

    refunds = RefundService(store, settings.refund_auto_approve_limit)
    runner = build_agent_runner(llm, store=store, refund_service=refunds)
    thread_id = f"{args.user}:{args.session}"

    print(f"用户 {args.user} 已登录，会话 {args.session}。输入 exit 退出。")
    refunds.refresh_user(args.user)
    _confirm_drafts(refunds, thread_id, args.user)
    while True:
        try:
            text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("exit", "quit", "q"):
            break
        if not text:
            continue
        refunds.refresh_user(args.user)
        message = HumanMessage(text, id=uuid4().hex)
        state = runner.run_turn(thread_id, args.user, message)
        _finish_turn(state, args.verbose)
        _confirm_drafts(refunds, thread_id, args.user, message.id)


def _finish_turn(state: dict, verbose: bool) -> None:
    if verbose:
        for r in state.get("route_log") or []:
            print(f"  [控制流] → {r['next']}  {r.get('reason', '')}")
        if state.get("facts"):
            print(f"  [关键事实] {state['facts']}")
        if state.get("handoff_ticket"):
            print("  [交接单] " + render_ticket(state["handoff_ticket"]))
    reply = next((m.content for m in reversed(state.get("messages") or []) if isinstance(m, AIMessage)), "")
    print(f"\n客服：{reply}")


def _confirm_drafts(refunds: RefundService, thread_id: str, user_id: str, message_id: str | None = None) -> None:
    for row in refunds.for_thread(thread_id, user_id):
        if row["status"] != "draft" or (message_id is not None and message_id != row["message_id"]):
            continue
        reason = "质量问题" if row["reason_type"] == "quality" else "无理由退货"
        kind = "退货退款" if row["refund_type"] == "return_refund" else "仅退款"
        print(f"\n{kind}确认单 {row['refund_id']}｜{row['order_id']} {row['item']}｜¥{row['amount']:.2f}｜{reason}")
        shipping = {"buyer": "买家承担", "seller": "商家承担", None: "无需退货"}[row["shipping_paid_by"]]
        print(f"退货运费：{shipping}" + ("；确认后提交人工审核" if row["needs_approval"] else ""))
        try:
            choice = input("确认 y，取消 n，回车稍后处理：").strip().lower()
            if choice in {"y", "n"}:
                result = refunds.confirm(row["refund_id"], user_id, choice == "y")
                print(result["status_text"])
                if result["status"] == "awaiting_return":
                    track = input("可输入退货快递单号（回车稍后填写）：").strip()
                    if track:
                        print(refunds.submit_return(row["refund_id"], user_id, track)["status_text"])
        except (EOFError, KeyboardInterrupt):
            return
        except RefundError as exc:
            print(exc)


if __name__ == "__main__":
    main()
