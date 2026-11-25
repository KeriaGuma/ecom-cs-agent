"""MCP Server 独立部署成 HTTP 服务时的鉴权：服务令牌 + 用户断言，两层分别回答两个问题。

1. **谁在调用**（服务令牌，HTTP 头 `Authorization: Bearer ...`）
   - 每个调用方一个随机令牌，Server 的令牌文件里只存 SHA-256，不存原文；
   - 令牌带权限范围：查询类工具要 `orders:read`，改地址要 `orders:write`。比如报表服务只发只读令牌；
   - 令牌不对或缺失，请求在 HTTP 层就被拒绝（401），到不了工具。
2. **替哪个用户调用**（用户断言，请求 `_meta` 里的 `ecom-cs/user_assertion`）
   - stdio 模式下 Server 是客服后端拉起的子进程，`_meta` 里直接放 user_id 就够了；
     HTTP 服务谁都能连，裸 user_id 不能信，否则拿到服务令牌就能读任意用户的订单；
   - 所以 HTTP 模式只认签过名的断言：`{sub: 用户, aud: 本服务, exp: 60 秒后}`，HMAC-SHA256 签名；
   - 断言 60 秒过期，日志、抓包里看到的旧断言很快作废；`aud` 限定只能用在订单服务上。
   - 本项目由客服后端签发断言；生产中更好的做法是由登录服务签发、客服后端只转发，这样客服后端也伪造不了用户。

模型始终看不到、也填不了这两样东西：令牌在 HTTP 头里，断言由客户端代码写进 `_meta`。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Callable, Optional

from mcp.server.auth.provider import AccessToken, TokenVerifier

AUDIENCE = "ecom-orders"
ASSERTION_TTL = 60  # 秒
READ, WRITE = "orders:read", "orders:write"
TOOL_SCOPES = {"list_my_orders": READ, "get_order": READ, "get_logistics": READ, "change_address": WRITE}


class AuthError(Exception):
    """断言无效、过期、权限不够等。消息会返回给调用方，不含任何密钥内容。"""


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------- 服务令牌 ----------------
class StaticTokenVerifier(TokenVerifier):
    """按令牌文件校验服务令牌。clients：[{"name": "cs-backend", "sha256": "...", "scopes": ["orders:read", ...]}]"""

    def __init__(self, clients: list[dict]):
        for c in clients:
            unknown = set(c["scopes"]) - {READ, WRITE}
            if unknown:
                raise ValueError(f"令牌 {c['name']} 的权限范围不认识：{sorted(unknown)}")
        self.clients = clients

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        digest = hash_token(token)
        for c in self.clients:  # 逐个比对、常数时间比较
            if hmac.compare_digest(digest, c["sha256"]):
                # token 字段放摘要：AccessToken 会在 SDK 内部流转，不让令牌原文多留一份
                return AccessToken(token=digest, client_id=c["name"], scopes=list(c["scopes"]))
        return None


def load_clients(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["clients"]


def require_scope(tool: str, scopes: list[str]) -> None:
    need = TOOL_SCOPES[tool]
    if need not in scopes:
        raise AuthError(f"服务令牌没有 {need} 权限，不能调用 {tool}")


# ---------------- 用户断言 ----------------
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sig(key: str, payload: str) -> str:
    return _b64(hmac.new(key.encode(), payload.encode(), hashlib.sha256).digest())


def sign_user(user_id: str, key: str, ttl: int = ASSERTION_TTL, now: Callable[[], float] = time.time) -> str:
    payload = _b64(json.dumps({"sub": user_id, "aud": AUDIENCE, "exp": int(now()) + ttl},
                              separators=(",", ":")).encode())
    return f"{payload}.{_sig(key, payload)}"


def verify_user(assertion: object, key: str, now: Callable[[], float] = time.time) -> str:
    """校验通过返回 user_id；任何一项不对都抛 AuthError。先验签名、再解析内容。"""
    if not isinstance(assertion, str) or assertion.count(".") != 1:
        raise AuthError("缺少用户身份断言，拒绝访问订单数据")
    payload, sig = assertion.split(".")
    if not hmac.compare_digest(sig, _sig(key, payload)):
        raise AuthError("用户身份断言签名无效")
    try:
        claims = json.loads(_unb64(payload))
    except ValueError:
        raise AuthError("用户身份断言格式错误") from None
    if claims.get("aud") != AUDIENCE:
        raise AuthError("用户身份断言不是签给订单服务的")
    if not isinstance(claims.get("exp"), int) or claims["exp"] < now():
        raise AuthError("用户身份断言已过期")
    if not claims.get("sub"):
        raise AuthError("用户身份断言缺少用户")
    return str(claims["sub"])


# ---------------- 生成令牌 / 密钥 ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="生成 MCP 服务令牌或用户断言密钥")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("new-token", help="生成一个服务令牌：原文给调用方，摘要写进 Server 的令牌文件")
    t.add_argument("--name", required=True)
    t.add_argument("--scopes", default=f"{READ},{WRITE}")
    t.add_argument("--tokens", help="令牌文件路径：给了就把摘要追加进去（文件不存在会新建）")
    sub.add_parser("new-key", help="生成用户断言的签名密钥（客服后端和 Server 配同一个）")
    args = ap.parse_args()
    if args.cmd == "new-key":
        print(new_secret())
        return
    token = new_secret()
    entry = {"name": args.name, "sha256": hash_token(token), "scopes": [s for s in args.scopes.split(",") if s]}
    StaticTokenVerifier([entry])  # 校验权限范围
    if args.tokens:
        try:
            data = json.load(open(args.tokens, encoding="utf-8"))
        except FileNotFoundError:
            data = {"clients": []}
        data["clients"] = [c for c in data["clients"] if c["name"] != args.name] + [entry]
        with open(args.tokens, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写入 {args.tokens}（只存摘要）。令牌原文只显示这一次，请配到调用方的 MCP_TOKEN：")
    else:
        print("令牌文件条目：" + json.dumps(entry, ensure_ascii=False))
    print(token)


if __name__ == "__main__":
    main()
