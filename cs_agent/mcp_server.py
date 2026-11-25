"""订单工具的 MCP Server：查订单列表、订单详情、物流，改收货地址。

    python -m cs_agent.mcp_server --db data/shop.db                       # stdio，由客服进程按需拉起
    MCP_USER_KEY=... python -m cs_agent.mcp_server --db data/shop.db \\
        --http --port 8765 --tokens data/mcp_tokens.json                    # 独立部署成 HTTP 服务

设计要点：
- **用户身份不是工具参数**。模型能填的只有工具参数，身份放在 MCP 请求的 `_meta` 里，由客户端代码写入
  （见 cs_agent/mcp_client.py）。和本地工具“user_id 由闭包绑定”是同一个原则：模型看不到、也改不了用户身份。
- **工具实现直接复用 cs_agent/tools.py**，权限校验、操作记录与本地工具完全一致；换协议不改业务行为。
- **最小权限**：这个进程只需要订单库，不导入 cs_agent.config，所以不会读取 .env 里的模型密钥。
- **两种部署，信任边界不同**：
  - stdio：Server 是客服后端拉起的子进程，只有父进程能和它说话，`_meta` 里直接放 user_id；
  - HTTP：谁都能连，所以要两样东西（见 cs_agent/mcp_auth.py）：HTTP 头里的服务令牌（谁在调用、能读还是能写），
    `_meta` 里签过名、60 秒过期的用户断言（替哪个用户调用）。裸 user_id 一律拒绝。
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .mcp_auth import AuthError, StaticTokenVerifier, load_clients, require_scope, verify_user
from .store import ShopStore
from .tools import make_order_tools

USER_META = "ecom-cs/user_id"  # stdio：请求 _meta 里直接携带用户身份的键
ASSERTION_META = "ecom-cs/user_assertion"  # HTTP：请求 _meta 里携带签名断言的键


def build_server(store: ShopStore, ops_log: bool = True, user_key: Optional[str] = None,
                 token_verifier: Optional[StaticTokenVerifier] = None, url: str = "",
                 address_guard: bool = True) -> MCPServer:
    """user_key 为空：stdio / 进程内，认 _meta 里的 user_id；
    user_key 非空：HTTP 模式，要服务令牌（token_verifier）+ 签名断言，url 是对外地址（用于鉴权元数据）。"""
    auth = None
    if user_key:
        if token_verifier is None:
            raise ValueError("HTTP 模式必须配置服务令牌")
        auth = AuthSettings(issuer_url=url, resource_server_url=url, validate_token_resource=False)
    srv = MCPServer("ecom-orders", instructions="电商订单服务：查询当前用户的订单、物流，修改未发货订单的收货地址。",
                    token_verifier=token_verifier, auth=auth)
    tools_of = lambda uid: make_order_tools(store, uid, ops_log=ops_log, address_guard=address_guard)  # noqa: E731
    desc = {t.name: t.description for t in tools_of("")}  # 描述与本地工具保持一致

    def user_of(ctx: Context, tool: str) -> str:
        meta = ctx.request_context.meta or {}
        if user_key:
            token = get_access_token()
            if token is None:  # HTTP 中间件已经拦过一次；进程内绕过 HTTP 直连时也不放行
                raise AuthError("缺少服务令牌")
            require_scope(tool, token.scopes)
            return verify_user(meta.get(ASSERTION_META), user_key)
        user_id = meta.get(USER_META)
        if not user_id:  # 没有身份一律拒绝，不会退回到某个默认用户
            raise AuthError("缺少用户身份，拒绝访问订单数据")
        return str(user_id)

    def run(ctx: Context, tool: str, args: dict) -> str:
        try:
            user_id = user_of(ctx, tool)
        except AuthError as e:  # 主动拒绝：原因返回给客户端；其他异常的细节只留在服务端
            raise ToolError(str(e)) from None
        return {t.name: t for t in tools_of(user_id)}[tool].invoke(args)

    @srv.tool(description=desc["list_my_orders"], structured_output=False)
    def list_my_orders(ctx: Context) -> str:
        return run(ctx, "list_my_orders", {})

    @srv.tool(description=desc["get_order"], structured_output=False)
    def get_order(order_id: str, ctx: Context) -> str:
        return run(ctx, "get_order", {"order_id": order_id})

    @srv.tool(description=desc["get_logistics"], structured_output=False)
    def get_logistics(order_id: str, ctx: Context) -> str:
        return run(ctx, "get_logistics", {"order_id": order_id})

    @srv.tool(description=desc["change_address"], structured_output=False)
    def change_address(order_id: str, new_address: str, ctx: Context) -> str:
        return run(ctx, "change_address", {"order_id": order_id, "new_address": new_address})

    return srv


def build_http_app(store: ShopStore, tokens_path: str, user_key: str, url: str, ops_log: bool = True,
                   host: str = "127.0.0.1", address_guard: bool = True):
    """HTTP 服务的 ASGI 应用（main 和测试共用）。监听本机时 SDK 自动开启 DNS 重绑定防护（校验 Host / Origin）。"""
    srv = build_server(store, ops_log=ops_log, user_key=user_key, address_guard=address_guard,
                       token_verifier=StaticTokenVerifier(load_clients(tokens_path)), url=url)
    return srv.streamable_http_app(host=host)


def main() -> None:
    ap = argparse.ArgumentParser(description="订单工具 MCP Server")
    ap.add_argument("--db", required=True, help="订单库 SQLite 文件路径")
    ap.add_argument("--http", action="store_true", help="独立部署成 HTTP 服务（默认 stdio）")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址；对外提供服务时放在 HTTPS 反向代理后面")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tokens", help="服务令牌文件（只存摘要），用 python -m cs_agent.mcp_auth new-token 生成")
    args = ap.parse_args()
    store, ops_log = ShopStore(args.db), os.getenv("OPS_LOG", "1") == "1"
    guard = os.getenv("ADDRESS_GUARD", "1") == "1"
    if not args.http:
        build_server(store, ops_log=ops_log, address_guard=guard).run("stdio")
        return
    user_key = os.getenv("MCP_USER_KEY", "")
    if not user_key or not args.tokens:
        ap.error("HTTP 模式需要 --tokens 令牌文件和 MCP_USER_KEY 环境变量（python -m cs_agent.mcp_auth new-key）")
    import uvicorn

    url = f"http://{args.host}:{args.port}/mcp"
    uvicorn.run(build_http_app(store, args.tokens, user_key, url, ops_log, args.host, guard), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
