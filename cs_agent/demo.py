"""零外部 API Key 的可交互客服沙箱入口（纯 Python 异步单 Agent 运行时）。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import closing
from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from .agent import build_agent_runner
from .api import create_app
from .claims import CLAIM_NOTES
from .config import ROOT, settings
from .kb import PolicyKB
from .refunds import RefundService
from .store import ShopStore

DEMO_DATE = date(2026, 11, 23)
DEMO_APPROVAL_LIMIT = 500
DEMO_APPLICATION_ID = 0x4543444D
ORDER_ID = re.compile(r"O\d{4}(?!\d)", re.IGNORECASE | re.ASCII)
REFUND_ACTION = re.compile(r"帮我.*(?:退款|退货|退了|退掉|取消)|申请.*(?:退款|退货)|我要(?:退货|退款)|帮我退")
REFUND_TOPIC = re.compile(r"退款|退货|能退|退了|退掉|取消")
REFUND_PROGRESS = ("退款进度", "退款记录", "退款状态", "审核进度", "到账了吗")
HANDOFF_TERMS = ("转人工", "人工客服", "找人工", "投诉")
POLICY_TERMS = ("政策", "运费", "到账", "发票", "保修", "无理由", "积分")


def _db_path() -> Path:
    return ROOT / os.getenv("DEMO_DB_PATH", "data/demo.db")


def bootstrap_demo_db(path: str | Path) -> ShopStore:
    """初始化独立演示业务库；重启时复用已有订单、会话与对账记录。"""
    path = Path(path).resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temp:
            seed = Path(temp) / "seed.db"
            ShopStore(str(seed)).init(today=DEMO_DATE)
            with closing(sqlite3.connect(seed)) as conn:
                conn.execute(f"PRAGMA application_id = {DEMO_APPLICATION_ID}")
            try:
                os.link(seed, path)
            except FileExistsError:
                pass
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
        if conn.execute("PRAGMA application_id").fetchone()[0] != DEMO_APPLICATION_ID:
            raise ValueError("请为 DEMO_DB_PATH 指定新文件或已有的演示库")
    return ShopStore(str(path))


def _tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"demo-{name}"}])


def _latest_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content not in CLAIM_NOTES:
            return str(msg.content)
    return ""


def _refund_reason(text: str) -> str | None:
    if "不是质量问题" not in text and any(term in text for term in ("质量", "故障", "坏了", "破损", "没声音")):
        return "quality"
    if any(term in text for term in ("不要", "不想要", "用不上", "买错", "买重", "无理由", "不喜欢")):
        return "no_reason"
    return None


def _wants_refund(text: str) -> bool:
    return bool(REFUND_ACTION.search(text)) and not any(
        term in text for term in ("先别退", "不要退款", "不退款", "不退了", "只问", "只查")
    )


def _order_action(text: str, order_id: str) -> AIMessage:
    if any(term in text for term in REFUND_PROGRESS):
        return _tool_call("get_order", order_id=order_id)
    if REFUND_ACTION.search(text) or REFUND_TOPIC.search(text):
        reason = _refund_reason(text)
        if reason is None:
            return AIMessage(content="请告诉我退款原因，例如买错型号或商品出现质量问题。")
        return _tool_call("check_refund", order_id=order_id, reason_type=reason)
    if any(term in text for term in ("物流", "快递", "运到哪", "发货了吗", "到哪")):
        return _tool_call("get_logistics", order_id=order_id)
    return _tool_call("get_order", order_id=order_id)


class DemoModel(BaseChatModel):
    """驱动真实单 Agent 工具循环的确定性脚本模型（支持同步与原生异步流式调用）。"""

    delay_ms: int = Field(default=0, ge=0)

    @property
    def _llm_type(self) -> str:
        return "offline-ecommerce-demo"

    def bind_tools(self, tools, **kwargs):
        names = tuple(
            getattr(tool, "name", None)
            or getattr(tool, "__name__", None)
            or (tool.get("name") if isinstance(tool, dict) else str(tool))
            for tool in tools
        )
        return self.bind(tool_names=names)

    def _decide_message(self, messages: list, tool_names: tuple[str, ...]) -> AIMessage:
        text = _latest_user_text(messages)
        if "HandoffSummary" in tool_names:
            return _tool_call(
                "HandoffSummary",
                intent=text or "用户要求人工服务",
                order_ids=[m.upper() for m in ORDER_ID.findall(text)],
                emotion="愤怒" if "投诉" in text else "平静",
                done=[],
                pending=text or "待人工客服跟进",
                suggestion="人工核实订单与诉求后跟进处理",
            )
        if isinstance(messages[-1], ToolMessage):
            return self._after_tool(messages, text)
        if messages[-1].content in CLAIM_NOTES and len(messages) >= 3 and isinstance(messages[-3], ToolMessage):
            return self._after_tool(messages[:-2], text)
        if any(term in text for term in HANDOFF_TERMS) and "transfer_to_human" in tool_names:
            return _tool_call("transfer_to_human", reason="用户要求转接人工客服")
        if match := ORDER_ID.search(text):
            return _order_action(text, match[0].upper())
        if any(term in text for term in ("我的订单", "查订单")) or REFUND_ACTION.search(text):
            return _tool_call("list_my_orders")
        if any(term in text for term in POLICY_TERMS) and "search_policy" in tool_names:
            return _tool_call("search_policy", query=text)
        return AIMessage(content="您好，我可以帮您查询订单和物流、修改未发货订单地址、申请退货退款，或解答售后政策问题。")

    def _generate(self, messages, stop=None, run_manager=None, *, tool_names=(), **kwargs) -> ChatResult:
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        answer = self._decide_message(messages, tool_names)
        return ChatResult(generations=[ChatGeneration(message=answer)])

    async def _agenerate(self, messages, stop=None, run_manager=None, *, tool_names=(), **kwargs) -> ChatResult:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        answer = self._decide_message(messages, tool_names)
        return ChatResult(generations=[ChatGeneration(message=answer)])

    async def _astream(self, messages, stop=None, run_manager=None, *, tool_names=(), **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        answer = self._decide_message(messages, tool_names)
        if answer.tool_calls:
            tc = answer.tool_calls[0]
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": tc["name"], "args": json.dumps(tc["args"], ensure_ascii=False),
                                   "id": tc["id"], "index": 0}],
            )
            yield ChatGenerationChunk(message=chunk)
            return
        content = str(answer.content)
        mid = max(1, len(content) // 2)
        for piece in (content[:mid], content[mid:]):
            if piece:
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    @staticmethod
    def _after_tool(messages, text: str) -> AIMessage:
        previous = messages[-2].tool_calls[0]
        name, args = previous["name"], previous["args"]
        output = str(messages[-1].content)
        if name == "search_policy":
            return AIMessage(content=output.split("\n\n", 1)[0])

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return AIMessage(content=output)

        if name == "list_my_orders":
            matches = [row for row in data if row["item"][-2:] in text]
            if len(matches) == 1:
                return _order_action(text, matches[0]["order_id"])
            if "我的订单" in text or "查订单" in text:
                orders = "、".join(f"{row['order_id']} {row['item']}" for row in data)
                return AIMessage(content=orders or "您还没有订单。")
            return AIMessage(content="请提供要查询或办理的订单号。")
        if name == "get_logistics":
            event = data["events"][-1]["event"]
            return AIMessage(content=f"订单 {args['order_id']} 的最新物流：{event}。")
        if name == "get_order":
            if any(term in text for term in REFUND_PROGRESS):
                last_refund = next((row for row in reversed(data.get("操作记录", [])) if row["操作"] == "退款申请"), None)
                if last_refund:
                    return AIMessage(content=last_refund["结果"])
                if data["status"] == "refunded":
                    return AIMessage(content="订单状态显示退款成功。")
                return AIMessage(content="当前没有查到这笔订单的退款处理记录。")
            statuses = {"paid": "待发货", "shipped": "运输中", "delivered": "已签收", "refunded": "已退款"}
            return AIMessage(content=f"订单 {data['order_id']} 的商品是{data['item']}，状态为{statuses[data['status']]}。")
        if name == "check_refund":
            if data.get("eligible") and _wants_refund(text):
                return _tool_call("prepare_refund", **args)
            return AIMessage(content=data.get("reason", output))
        return AIMessage(content="已生成确认单，请核对后选择确认或取消。")


def create_demo_app(db_path: str | Path | None = None) -> FastAPI:
    cfg = replace(
        settings, kb_dir=str(ROOT / "data/kb"), order_tools="local", vision=False, memory_summary=False,
        claim_check=True, extra_tools=0, domains="", refund_auto_approve_limit=DEMO_APPROVAL_LIMIT,
    )
    store = bootstrap_demo_db(db_path if db_path is not None else _db_path())
    refunds = RefundService(store, DEMO_APPROVAL_LIMIT, today=lambda: DEMO_DATE, enable_simulated_faults=True)
    model = DemoModel(delay_ms=int(os.getenv("DEMO_MODEL_DELAY_MS", "0")))
    kb = PolicyKB(cfg.kb_dir)
    runner = build_agent_runner(
        model, store=store, kb=kb, retriever=kb, settings=cfg, today=lambda: DEMO_DATE, refund_service=refunds,
    )
    app = create_app(runner=runner, store=store, refund_service=refunds, demo_auth=True)

    @app.get("/ready")
    def ready() -> dict:
        try:
            with closing(sqlite3.connect(f"{Path(store.path).as_uri()}?mode=ro", uri=True)) as conn:
                conn.execute("SELECT 1 FROM orders LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            raise HTTPException(503, "演示数据库不可用") from exc
        return {"ok": True, "runtime": "python-async"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化并保留独立的合成客服演示库。")
    parser.add_argument("--db", type=Path, default=_db_path())
    store = bootstrap_demo_db(parser.parse_args().db)
    print(store.path)


if __name__ == "__main__":
    main()
