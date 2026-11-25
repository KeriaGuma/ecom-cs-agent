"""全异步 HTTP 网关与 SSE 流式接口：uvicorn --factory cs_agent.api:create_app

核心工程特性：
- **原生异步 SSE 真流式（`POST /chat`）**：逐段推送工具执行进度（`status`）、模型输出片段（`token`）、完整回复（`message`）与结果回执（`done`）；
- **无泄漏会话互斥锁（`AsyncSessionLockManager`）**：按 `thread_id = user_id:session_id` 串行化同一会话并发请求，
  通过引用计数在请求结束时自动清理空闲锁对象，消除长期运行下的字典内存泄漏（多副本部署可平滑替换为 Redis `SET NX PX` 锁）；
- **异步记忆压缩**：先向客户端发出 `done` 事件与退款确认卡片，再执行历史消息摘要压缩，不阻塞用户感知的尾延迟；
- **完整售后状态机接口**：支持买家确认草稿、买家回填退货物流单号、审批员审核大额申请、坐席/仓库核验退货入库及支付掉单主动对账。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator

from .config import settings
from .desk_service import DemoIdentity, DeskService
from .refunds import RefundError, RefundService
from .runner import TurnRunner
from .store import ShopStore
from .tracing import RunTrace

log = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
STATUS_TEXT = {
    "perceive": "正在识别图片…",
    "order_agent": "正在查询订单和物流…",
    "refund_agent": "正在核实售后退款规则…",
    "kb_agent": "正在查询售后政策…",
    "handoff_human": "正在为您转接人工客服…",
    "human_service": "人工客服已接入，消息已转交",
    "fallback": "系统繁忙，正在为您切换备用方案…",
}

IMAGE_DATA_URL = re.compile(r"^data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$")
MAX_IMAGE_CHARS = 7_000_000  # base64 约 5MB 的图片


class _LockEntry:
    __slots__ = ("lock", "refs")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.refs = 0


class SessionLockManager:
    """带引用计数自动回收的会话级非阻塞互斥锁管理器，彻底杜绝海量历史 `thread_id` 导致的锁表内存泄漏。
    （多实例水平扩展时，可实现相同 `try_acquire / release` 协议接入 Redis `SET lock:{thread_id} token NX PX 60000`。）
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, _LockEntry] = {}

    def try_acquire(self, key: str) -> bool:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = _LockEntry()
            if not entry.lock.acquire(blocking=False):
                return False
            entry.refs += 1
            return True

    def release(self, key: str) -> None:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refs -= 1
            entry.lock.release()
            if entry.refs <= 0:
                self._entries.pop(key, None)

    @property
    def active_keys(self) -> int:
        with self._guard:
            return len(self._entries)

    @contextmanager
    def hold(self, key: str, detail: str = "该会话正在处理中"):
        if not self.try_acquire(key):
            raise HTTPException(409, detail)
        try:
            yield
        finally:
            self.release(key)


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=ID_PATTERN.pattern)
    message: str = Field(min_length=1, max_length=2000)
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    images: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("images")
    @classmethod
    def _check_images(cls, v: list[str]) -> list[str]:
        for img in v:
            if len(img) > MAX_IMAGE_CHARS or not IMAGE_DATA_URL.match(img):
                raise ValueError("图片须为 png / jpeg / webp 的 data URL，且不超过约 5MB")
        return v


class ApprovalRequest(BaseModel):
    approved: bool
    comment: str = Field(default="", max_length=1000)


class RefundConfirmation(BaseModel):
    confirmed: bool = Field(strict=True)


class ReturnShipmentRequest(BaseModel):
    tracking_no: str = Field(min_length=4, max_length=32)


class ReturnReceiveRequest(BaseModel):
    tracking_no: str = Field(default="", max_length=32)


class DeskReply(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def _clean_message(cls, message: str) -> str:
        if not (text := message.strip()):
            raise ValueError("回复不能为空")
        return text


class DemoLogin(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class DemoLogout(BaseModel):
    role: Literal["customer", "staff"]


class DemoRefundFault(BaseModel):
    fault: Literal["ack_lost", "processing", "failed"]


def _turn_result(state: dict, t0: float) -> dict:
    result: dict[str, Any] = {
        "final_action": state.get("final_action"),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }
    if (ticket := state.get("handoff_ticket")) and state.get("final_action") == "handoff":
        result.update(ticket_id=ticket["ticket_id"], queue_position=ticket["queue_position"])
    return result


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(runner: TurnRunner | None = None, store: ShopStore | None = None,
               refund_service: RefundService | None = None, *, user_identity: Callable | None = None,
               staff_identity: Callable | None = None, approver_identity: Callable | None = None,
               demo_auth: bool = False, max_active_turns: int = 16) -> FastAPI:
    """创建客服 HTTP 应用；支持注入部署侧真实鉴权依赖、支付服务与自定义编排器。"""
    if (user_identity is None) != (staff_identity is None):
        raise ValueError("部署自定义鉴权时必须提供用户和坐席身份解析器")
    if demo_auth and any((user_identity, staff_identity, approver_identity)):
        raise ValueError("演示身份与外部身份解析器需分别启用")
    if max_active_turns < 1:
        raise ValueError("max_active_turns 必须为正数")
    if store is None and runner is not None:
        store = runner.store
    if store is None:
        store = ShopStore(settings.db_path)
        if not Path(settings.db_path).exists():
            store.init()
    refunds = refund_service if refund_service is not None else RefundService(store, settings.refund_auto_approve_limit)
    if runner is None:
        runner = _build_real_runner(store, refunds)

    if settings.pii_mask:
        from .pii import install_log_masking

        install_log_masking()

    app = FastAPI(title="电商智能客服 Agent")
    desk = DeskService(store, pii_mask=settings.pii_mask)
    demo = DemoIdentity(settings.demo_secret_key, seed_path=store.path) if demo_auth else None
    strict_assignment = demo_auth or staff_identity is not None
    session_locks = SessionLockManager()
    active_turns = threading.BoundedSemaphore(max_active_turns)
    app.state.session_locks = session_locks

    @app.exception_handler(RefundError)
    def refund_error(request, error: RefundError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content={"detail": str(error)})

    def user_of(x_user_id: Optional[str] = Header(None)) -> str:
        if not x_user_id or not ID_PATTERN.match(x_user_id):
            raise HTTPException(401, "缺少或非法的用户身份（X-User-Id）")
        return x_user_id

    def staff_of(x_staff_id: Optional[str] = Header(None)) -> str:
        if not x_staff_id or not ID_PATTERN.match(x_staff_id):
            raise HTTPException(403, "仅限客服人员（X-Staff-Id）")
        return x_staff_id

    user_dependency = Depends(demo.customer if demo else user_identity or user_of)
    staff_dependency = Depends(demo.staff if demo else staff_identity or staff_of)
    approver_dependency = Depends(demo.approver if demo else approver_identity or staff_identity or staff_of)

    if demo:
        @app.post("/demo/login")
        def demo_login(req: DemoLogin, request: Request, response: Response) -> dict:
            response.headers["Cache-Control"] = "no-store"
            return demo.login(request, response, req.username, req.password)

        @app.get("/demo/session")
        def demo_session(request: Request, response: Response) -> dict:
            response.headers["Cache-Control"] = "no-store"
            return {"customer": demo.read(request, "customer"), "operator": demo.read(request, "staff")}

        @app.post("/demo/logout")
        def demo_logout(req: DemoLogout, response: Response) -> dict:
            demo.logout(response, req.role)
            return {"ok": True}

        @app.post("/demo/refunds/{refund_id}/fault")
        def demo_refund_fault(refund_id: str, req: DemoRefundFault, staff: str = approver_dependency) -> dict:
            return refunds.set_simulated_fault(refund_id, req.fault, staff)

        @app.get("/desk", response_class=HTMLResponse)
        def desk_page() -> str:
            return (Path(__file__).parent / "static" / "desk.html").read_text(encoding="utf-8")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.post("/chat")
    async def chat(req: ChatRequest, user: str = user_dependency) -> StreamingResponse:
        thread_id = f"{user}:{req.session_id}"
        if not session_locks.try_acquire(thread_id):
            raise HTTPException(409, "上一条消息还在处理中，请稍候")

        loop = asyncio.get_running_loop()
        events_queue: asyncio.Queue[str | None] = asyncio.Queue()
        turn_id = uuid5(NAMESPACE_URL, f"ecom-chat:{thread_id}:{req.request_id}") if req.request_id else uuid4()
        message = HumanMessage(req.message, id=turn_id.hex)
        terminal_frames: list[str] = []
        stream_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        seen_stages: set[str] = set()

        def emit(event: str, data: dict) -> None:
            if event == "status":
                stage = data["stage"]
                if stage in seen_stages and stage != "perceive":
                    return
                seen_stages.add(stage)
                data = {**data, "text": STATUS_TEXT.get(stage, "正在处理您的请求…")}
            frame = _sse(event, data)
            if req.request_id and event in ("message", "done", "error"):
                terminal_frames.append(frame)
                if event in ("done", "error"):
                    desk.finish(thread_id, req.request_id, terminal_frames)
            loop.call_soon_threadsafe(events_queue.put_nowait, frame)

        async def run_turn_task() -> None:
            t0 = time.perf_counter()
            trace = RunTrace()
            need_compact = False
            try:
                if serving:
                    await asyncio.to_thread(desk.append, thread_id, user, message.id, "user", req.message)
                    emit("status", {"stage": "human_service"})
                    emit("done", {**_turn_result({"final_action": "human"}, t0), "ticket_id": serving["ticket_id"],
                                  "turn_id": message.id})
                    return
                await asyncio.to_thread(refunds.refresh_user, user)
                state = await runner.arun_turn(
                    thread_id, user, message, images=req.images, callbacks=[trace], emit=emit, defer_summary=True,
                )
                need_compact = runner.async_dispatch is not None and not state.get("degraded")
                done = _turn_result(state, t0)
                done["turn_id"] = message.id
                current = [refunds.view(row) for row in refunds.for_thread(thread_id, user)
                           if row["message_id"] == message.id]
                if current:
                    done["refunds"] = current
                log.info("对话结束 turn_id=%s action=%s latency_ms=%s llm_calls=%d tools=%s refund_ids=%s",
                         message.id, done["final_action"], done["latency_ms"], trace.llm_calls, trace.tool_names,
                         [row["refund_id"] for row in current])
                # 先推出 done 事件与退款卡片，再在会话锁内完成滚动摘要压缩
                emit("done", done)
                if need_compact:
                    await runner.acompact(thread_id)
            except Exception as e:
                log.exception("对话处理失败 turn_id=%s", message.id)
                emit("error", {"text": "系统繁忙，请稍后再试", "type": type(e).__name__})
            finally:
                active_turns.release()
                session_locks.release(thread_id)
                loop.call_soon_threadsafe(events_queue.put_nowait, None)

        admitted = False
        try:
            fingerprint = ""
            if req.request_id:
                fingerprint = desk.fingerprint(req.message, req.images)
                if replay := desk.replay(thread_id, req.request_id, fingerprint):
                    session_locks.release(thread_id)
                    return StreamingResponse(_iter_frames(replay), media_type="text/event-stream", headers=stream_headers)
            ticket = desk.handoff(thread_id)
            serving = ticket if ticket and ticket["assignee"] else None
            if serving and req.images:
                raise HTTPException(422, "人工接待期间请发送文字消息")
            if not active_turns.acquire(blocking=False):
                raise HTTPException(503, "客服接待已满，请稍后重试", headers={"Retry-After": "1"})
            admitted = True
            if req.request_id and (replay := desk.reserve(thread_id, req.request_id, fingerprint)):
                active_turns.release()
                session_locks.release(thread_id)
                return StreamingResponse(_iter_frames(replay), media_type="text/event-stream", headers=stream_headers)
            # 独立后台任务持有会话锁直至整轮闭环，防止客户端中途断开 SSE 造成半成品并发竞态
            asyncio.create_task(run_turn_task())
        except Exception:
            if admitted:
                active_turns.release()
            session_locks.release(thread_id)
            raise

        async def event_stream() -> AsyncIterator[str]:
            while (frame := await events_queue.get()) is not None:
                yield frame

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=stream_headers)

    @app.get("/sessions/{session_id}/messages")
    def history(session_id: str, user: str = user_dependency) -> dict:
        if not ID_PATTERN.match(session_id):
            raise HTTPException(400, "非法的会话 ID")
        thread_id = f"{user}:{session_id}"
        ticket = desk.handoff(thread_id)
        handoff = {"ticket_id": ticket["ticket_id"], "active": bool(ticket["assignee"])} if ticket else None
        return {"session_id": session_id, "messages": store.transcript(thread_id), "handoff": handoff}

    @app.delete("/sessions/{session_id}/messages")
    def erase_history(session_id: str, user: str = user_dependency) -> dict:
        if not ID_PATTERN.match(session_id):
            raise HTTPException(400, "非法的会话 ID")
        thread_id = f"{user}:{session_id}"
        with session_locks.hold(thread_id):
            ticket = desk.handoff(thread_id)
            if ticket and ticket["assignee"]:
                raise HTTPException(409, "请在人工接待结束后清理记录")
            desk.erase_conversation(thread_id)
        return {"cleared": True}

    @app.get("/sessions/{session_id}/refunds")
    def session_refunds(session_id: str, user: str = user_dependency) -> dict:
        if not ID_PATTERN.match(session_id):
            raise HTTPException(400, "非法的会话 ID")
        thread_id = f"{user}:{session_id}"
        refunds.refresh_thread(thread_id, user)
        return {"refunds": [refunds.view(row) for row in refunds.for_thread(thread_id, user)]}

    @app.get("/refunds/{refund_id}")
    def refund_status(refund_id: str, user: str = user_dependency) -> dict:
        return refunds.status(refund_id, user)

    @app.post("/refunds/{refund_id}/confirm")
    def confirm_refund(refund_id: str, req: RefundConfirmation, user: str = user_dependency) -> dict:
        return refunds.confirm(refund_id, user, req.confirmed)

    @app.post("/refunds/{refund_id}/return")
    def submit_return_shipment(refund_id: str, req: ReturnShipmentRequest, user: str = user_dependency) -> dict:
        return refunds.submit_return(refund_id, user, req.tracking_no)

    @app.post("/refunds/{refund_id}/decision")
    def decide_refund(refund_id: str, req: ApprovalRequest, staff: str = approver_dependency) -> dict:
        return refunds.decide(refund_id, staff, req.approved, req.comment)

    @app.post("/refunds/{refund_id}/receive")
    def receive_return(refund_id: str, req: ReturnReceiveRequest | None = None,
                       staff: str = approver_dependency) -> dict:
        return refunds.confirm_return(refund_id, staff, req.tracking_no if req else "")

    @app.post("/refunds/{refund_id}/reconcile")
    def reconcile_refund(refund_id: str, staff: str = approver_dependency) -> dict:
        return refunds.reconcile(refund_id, staff)

    @app.get("/approvals")
    def approvals(staff: str = approver_dependency) -> dict:
        return {"refunds": refunds.pending_approvals(), "returns": refunds.awaiting_returns()}

    @app.get("/tickets")
    def tickets(staff: str = staff_dependency) -> dict:
        return {"open": [row for row in store.open_tickets()
                         if not strict_assignment or row["assignee"] in (None, staff)]}

    def get_ticket(ticket_id: int) -> dict:
        ticket = store.get_ticket(ticket_id)
        if not ticket:
            raise HTTPException(404, "工单不存在")
        return ticket

    @app.get("/tickets/{ticket_id}")
    def ticket_detail(ticket_id: int, staff: str = staff_dependency) -> dict:
        if strict_assignment and get_ticket(ticket_id)["assignee"] not in (None, staff):
            raise HTTPException(403, "该工单由其他坐席负责")
        return {**refunds.ticket_details(ticket_id), "events": store.ticket_events(ticket_id)}

    @app.post("/tickets/{ticket_id}/claim")
    def claim_ticket(ticket_id: int, staff: str = staff_dependency) -> dict:
        ticket = get_ticket(ticket_id)
        with session_locks.hold(ticket["thread_id"] or f"ticket:{ticket_id}"):
            active = desk.handoff(ticket["thread_id"]) if ticket["thread_id"] else None
            if active and active["assignee"] and active["ticket_id"] != ticket_id:
                raise HTTPException(409, "该会话已有坐席接待")
            if not store.claim_ticket(ticket_id, staff):
                raise HTTPException(409, "工单不可领取，可能已关闭或由其他坐席接手")
        return refunds.ticket_details(ticket_id)

    @app.post("/tickets/{ticket_id}/messages")
    def reply_to_ticket(ticket_id: int, req: DeskReply, staff: str = staff_dependency) -> dict:
        thread_id = get_ticket(ticket_id)["thread_id"]
        if not thread_id:
            raise HTTPException(409, "此历史工单没有可回复的会话")
        with session_locks.hold(thread_id):
            ticket = get_ticket(ticket_id)
            if ticket["status"] != "open" or ticket["assignee"] != staff:
                raise HTTPException(409, "请由接手的坐席回复未关闭的工单")
            message_id = uuid4().hex
            content = desk.append(thread_id, ticket["user_id"], message_id, "staff", req.message,
                                  staff_reply=(ticket_id, staff, message_id))
        return {"message_id": message_id, "role": "staff", "content": content}

    @app.post("/tickets/{ticket_id}/close")
    def close_ticket(ticket_id: int, staff: str = staff_dependency) -> dict:
        ticket = get_ticket(ticket_id)
        with session_locks.hold(ticket["thread_id"] or f"ticket:{ticket_id}"):
            ticket = get_ticket(ticket_id)
            if ticket["status"] != "open":
                raise HTTPException(404, "工单不存在或已关闭")
            if (strict_assignment and ticket["assignee"] != staff) or not store.close_ticket(ticket_id, staff=staff):
                raise HTTPException(409, "请由接手的坐席关闭工单")
        return {"ticket_id": ticket_id, "status": "closed"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "chat.html").read_text(encoding="utf-8")

    return app


async def _iter_frames(frames: list[str]) -> AsyncIterator[str]:
    for frame in frames:
        yield frame


def _build_real_runner(store: ShopStore, refunds: RefundService) -> TurnRunner:
    from .agent import build_agent_runner
    from .llm import get_llm

    return build_agent_runner(get_llm(settings.agent_model), store=store, refund_service=refunds)
