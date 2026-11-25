"""MCP 客户端：把 MCP Server 上的工具转成 LangChain 工具，供 ReAct 循环同步调用。

订单工具使用同步调用，由客户端后台事件循环负责 MCP 通信。
这里用一个后台事件循环线程持有 MCP 连接，同步代码通过 run_coroutine_threadsafe 调用。

- 连接由一个长期运行的“持有者”任务打开和关闭：MCP SDK 基于 anyio，连接的进入和退出必须在同一个任务里；
- 一个订单库对应一个连接（一个 Server 子进程），多个用户、多个线程共用；用户身份按次放进请求 _meta；
- 连独立部署的 HTTP Server 时：连接上带服务令牌（HTTP 头），每次调用现签一个 60 秒有效的用户断言放进 _meta；
- 工具出错（Server 抛异常、超时）时抛出异常，由 ReAct 循环按“工具执行出错”交给模型处理，和本地工具一致。
"""
from __future__ import annotations

import asyncio
import atexit
import os
import sys
import threading
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from pydantic import Field, create_model

from .mcp_auth import sign_user
from .mcp_server import ASSERTION_META, USER_META

CALL_TIMEOUT = 30.0
_JSON_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}
SERVER_ENV = ("OPS_LOG", "ADDRESS_GUARD")  # 只把业务开关传给 Server 子进程，模型密钥一个都不传


class McpToolset:
    def __init__(self, target: Any, timeout: float = CALL_TIMEOUT, token: str = "", user_key: str = ""):
        """target：StdioServerParameters（拉起子进程）、HTTP 地址（独立部署的 Server，要带 token 和 user_key），
        或 MCPServer 实例（测试：进程内连接）。"""
        if isinstance(target, str) and not (token and user_key):
            raise ValueError("连接 HTTP 的 MCP Server 需要服务令牌和用户断言密钥")
        self.timeout = timeout
        self._token, self._user_key = token, user_key
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._client: Optional[Client] = None
        self._specs: list = []
        self._connected = False
        self._owner = asyncio.run_coroutine_threadsafe(self._hold(target), self._loop)
        woke = self._ready.wait(timeout)  # 连不上时 _hold 会立刻唤醒这里（比如令牌不对，HTTP 401），不用等到超时
        if not self._connected:
            err: BaseException = TimeoutError("连接 MCP Server 超时")
            if woke:
                try:
                    err = self._owner.exception(timeout) or RuntimeError("连接已关闭")
                except Exception:
                    pass
            self._shutdown_loop()
            raise RuntimeError(f"无法连接 MCP Server：{err}")

    async def _hold(self, target: Any) -> None:
        self._closing = asyncio.Event()
        try:
            await self._open(target)
        finally:
            self._ready.set()  # 连接失败也要唤醒构造函数

    async def _open(self, target: Any) -> None:
        if isinstance(target, str):  # HTTP：服务令牌放在这个连接的每个请求头里
            async with create_mcp_http_client(headers={"Authorization": f"Bearer {self._token}"}) as http:
                await self._serve(streamable_http_client(target, http_client=http))
        else:
            await self._serve(target)

    async def _serve(self, target: Any) -> None:
        async with Client(target, read_timeout_seconds=self.timeout) as client:
            self._client = client
            self._specs = (await client.list_tools()).tools
            self._connected = True
            self._ready.set()
            await self._closing.wait()

    def call(self, name: str, args: dict, user_id: str) -> str:
        # HTTP：每次调用现签断言（60 秒过期），不缓存；stdio / 进程内：直接放 user_id
        meta = {ASSERTION_META: sign_user(user_id, self._user_key)} if self._user_key else {USER_META: user_id}
        fut = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, args, meta=meta), self._loop)
        res = fut.result(self.timeout)
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            raise RuntimeError(text or f"MCP 工具 {name} 执行失败")
        return text

    def tools(self, user_id: str) -> list[BaseTool]:
        """按用户生成工具：参数说明来自 Server 的 input_schema，用户身份由这里绑定，不出现在参数里。"""
        return [self._as_tool(spec, user_id) for spec in self._specs]

    def _as_tool(self, spec, user_id: str) -> BaseTool:
        schema = spec.input_schema or {}
        required = set(schema.get("required", []))
        fields = {}
        for key, prop in (schema.get("properties") or {}).items():
            typ = _JSON_TYPES.get(prop.get("type"), str)
            default = ... if key in required else None
            fields[key] = (typ if key in required else Optional[typ], Field(default, description=prop.get("description")))
        args_schema = create_model(f"{spec.name}_args", **fields)
        name = spec.name
        return StructuredTool.from_function(func=lambda **kw: self.call(name, kw, user_id), name=name,
                                            description=spec.description or "", args_schema=args_schema)

    def close(self) -> None:
        if self._owner.done():
            self._shutdown_loop()
            return
        self._loop.call_soon_threadsafe(self._closing.set)
        try:
            self._owner.result(self.timeout)
        finally:
            self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(self.timeout)


# ---------------- 按订单库复用连接 ----------------
_POOL: dict[str, McpToolset] = {}
_LOCK = threading.Lock()


def stdio_params(db_path: str) -> StdioServerParameters:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: os.environ[k] for k in SERVER_ENV if k in os.environ}
    return StdioServerParameters(command=sys.executable, args=["-m", "cs_agent.mcp_server", "--db", db_path],
                                 env=env, cwd=root)


def order_tools(db_path: str, user_id: str) -> list[BaseTool]:
    """同一个订单库只拉起一个 Server 子进程，之后的调用复用连接。"""
    with _LOCK:
        toolset = _POOL.get(db_path)
        if toolset is None:
            if not _POOL:
                atexit.register(close)  # 进程退出时关掉所有 Server 子进程
            toolset = _POOL[db_path] = McpToolset(stdio_params(db_path))
    return toolset.tools(user_id)


def order_tools_http(url: str, token: str, user_key: str, user_id: str) -> list[BaseTool]:
    """独立部署的 HTTP Server：订单库在 Server 那边，按地址复用连接。"""
    with _LOCK:
        toolset = _POOL.get(url)
        if toolset is None:
            if not _POOL:
                atexit.register(close)
            toolset = _POOL[url] = McpToolset(url, token=token, user_key=user_key)
    return toolset.tools(user_id)


def close(db_path: Optional[str] = None) -> None:
    """关闭某个订单库的连接（评测里每个任务用一个临时库，任务结束就关）；不传则全部关闭。"""
    with _LOCK:
        paths = [db_path] if db_path else list(_POOL)
        sets = [_POOL.pop(p) for p in paths if p in _POOL]
    for s in sets:
        s.close()
