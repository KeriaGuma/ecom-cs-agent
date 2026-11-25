"""坐席接待、演示会话身份与 HTTP 请求幂等回执。"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time

from fastapi import HTTPException, Request, Response

from .pii import mask_text
from .refunds import RefundError
from .store import ShopStore, to_json


DEMO_ROLES = {"u001": "customer", "u002": "customer", "s01": "staff", "s02": "staff", "a01": "approver"}
DEMO_SESSION_SECONDS = 8 * 60 * 60


class DemoIdentity:
    """离线沙箱演示 Cookie 签名鉴权。支持显式配置密钥或基于数据库路径派生稳定密钥，避免服务重启或多进程下全员掉线。"""

    def __init__(self, secret: str | bytes | None = None, *, seed_path: str = "") -> None:
        if isinstance(secret, bytes) and secret:
            self.key = secret
        elif isinstance(secret, str) and secret:
            self.key = hashlib.sha256(secret.encode()).digest()
        elif seed_path and seed_path != ":memory:":
            self.key = hmac.new(b"ecom-cs-demo-cookie-v1", seed_path.encode(), hashlib.sha256).digest()
        else:
            self.key = secrets.token_bytes(32)

    @staticmethod
    def _cookie(slot: str) -> str:
        return f"ecom_demo_{slot}"

    def _signature(self, payload: str) -> str:
        return hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()

    def read(self, request: Request, slot: str) -> dict | None:
        token = request.cookies.get(self._cookie(slot), "")
        parts = token.split(".")
        if len(parts) != 3:
            return None
        username, expires, signature = parts
        role = DEMO_ROLES.get(username)
        if role is None or slot != ("customer" if role == "customer" else "staff"):
            return None
        if not expires.isascii() or not expires.isdigit() or len(expires) > 12 or int(expires) <= time.time():
            return None
        if not signature.isascii() or not hmac.compare_digest(signature, self._signature(f"{username}.{expires}")):
            return None
        return {"username": username, "role": role}

    def login(self, request: Request, response: Response, username: str, password: str) -> dict:
        role = DEMO_ROLES.get(username)
        if role is None or not hmac.compare_digest(password.encode(), b"demo"):
            raise HTTPException(401, "演示账号或口令有误")
        slot = "customer" if role == "customer" else "staff"
        payload = f"{username}.{int(time.time()) + DEMO_SESSION_SECONDS}"
        response.set_cookie(self._cookie(slot), f"{payload}.{self._signature(payload)}", max_age=DEMO_SESSION_SECONDS,
                            httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return {"username": username, "role": role}

    def logout(self, response: Response, slot: str) -> None:
        response.delete_cookie(self._cookie(slot), httponly=True, samesite="strict")

    def customer(self, request: Request) -> str:
        who = self.read(request, "customer")
        if who is None:
            raise HTTPException(401, "请登录买家账号")
        return who["username"]

    def staff(self, request: Request) -> str:
        return self._operator(request, "staff", "仅限客服坐席")

    def approver(self, request: Request) -> str:
        return self._operator(request, "approver", "仅限退款审批员")

    def _operator(self, request: Request, role: str, detail: str) -> str:
        who = self.read(request, "staff")
        if who is None or who["role"] != role:
            raise HTTPException(403, detail)
        return who["username"]


class DeskService:
    def __init__(self, store: ShopStore, *, pii_mask: bool) -> None:
        self.store = store
        self.pii_mask = pii_mask
        store.execute("CREATE TABLE IF NOT EXISTS chat_receipts (thread_id TEXT NOT NULL, request_id TEXT NOT NULL, "
                      "fingerprint TEXT NOT NULL, frames TEXT, PRIMARY KEY (thread_id, request_id))")

    def handoff(self, thread_id: str) -> dict | None:
        rows = self.store.query("SELECT ticket_id, assignee FROM tickets WHERE thread_id=? AND status='open' "
                                "ORDER BY assignee IS NULL, ticket_id LIMIT 1", (thread_id,))
        return rows[0] if rows else None

    def append(self, thread_id: str, user_id: str, message_id: str, role: str, content: str, *,
               staff_reply: tuple[int, str, str] | None = None) -> str:
        """在调用方会话锁内，把人工接待消息写进原有会话和存档。"""
        saved = self.store.load_session(thread_id)
        if saved.get("user_id", user_id) != user_id:
            raise RefundError("会话不存在", 404)
        context_role = "assistant" if role == "staff" else "user"
        saved.update(user_id=user_id, messages=[*saved.get("messages", []),
                                                {"id": message_id, "role": context_role, "content": content}])
        visible = mask_text(content) if self.pii_mask else content
        self.store.save_session(thread_id, saved, [(message_id, role, visible)], staff_reply=staff_reply)
        return visible

    @staticmethod
    def fingerprint(text: str, images: list[str]) -> str:
        return hashlib.sha256(to_json([text, images]).encode()).hexdigest()

    def replay(self, thread_id: str, request_id: str, fingerprint: str) -> list[str] | None:
        rows = self.store.query("SELECT fingerprint, frames FROM chat_receipts WHERE thread_id=? AND request_id=?",
                                (thread_id, request_id))
        if not rows:
            return None
        row = rows[0]
        if row["fingerprint"] != fingerprint:
            raise RefundError("同一请求编号的消息内容发生变化", 409)
        if row["frames"] is None:
            raise RefundError("原请求状态尚未确认，请先查看会话和业务记录", 409)
        frames = json.loads(row["frames"])
        if not frames:
            raise RefundError("原请求记录已清理", 409)
        return frames

    def reserve(self, thread_id: str, request_id: str, fingerprint: str) -> list[str] | None:
        inserted = self.store.execute("INSERT OR IGNORE INTO chat_receipts (thread_id, request_id, fingerprint) "
                                      "VALUES (?, ?, ?)", (thread_id, request_id, fingerprint))
        return None if inserted else self.replay(thread_id, request_id, fingerprint)

    def finish(self, thread_id: str, request_id: str, frames: list[str]) -> None:
        self.store.execute("UPDATE chat_receipts SET frames=? WHERE thread_id=? AND request_id=?",
                           (to_json(frames), thread_id, request_id))

    def erase_conversation(self, thread_id: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM conversation_state WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM transcripts WHERE thread_id=?", (thread_id,))
            conn.execute("UPDATE chat_receipts SET frames='[]' WHERE thread_id=?", (thread_id,))
