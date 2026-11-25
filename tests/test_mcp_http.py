"""MCP Server 独立部署成 HTTP 服务：服务令牌 + 用户断言。测试里起一个真实的 uvicorn 服务，走完整的 HTTP 请求。"""
import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn

from cs_agent import mcp_auth
from cs_agent.mcp_auth import AuthError, hash_token, sign_user, verify_user
from cs_agent.mcp_client import McpToolset
from cs_agent.mcp_server import ASSERTION_META, USER_META, build_http_app
from cs_agent.tools import NOT_FOUND, make_order_tools

KEY = "test-user-key"
RW, RO = "token-read-write", "token-read-only"
ADDR = "杭州市滨江区网商路 699 号"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(store, tmp_path):
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"clients": [
        {"name": "cs-backend", "sha256": hash_token(RW), "scopes": ["orders:read", "orders:write"]},
        {"name": "report", "sha256": hash_token(RO), "scopes": ["orders:read"]},
    ]}))
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    app = build_http_app(store, str(tokens), KEY, url)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    yield url
    srv.should_exit = True
    th.join(5)


@pytest.fixture
def client(server):
    toolset = McpToolset(server, token=RW, user_key=KEY)
    yield toolset
    toolset.close()


def _raw_call(toolset: McpToolset, name: str, args: dict, meta: dict):
    """绕过 McpToolset.call 的自动签名，直接发请求，模拟攻击者手里只有服务令牌。"""
    fut = asyncio.run_coroutine_threadsafe(toolset._client.call_tool(name, args, meta=meta), toolset._loop)
    res = fut.result(10)
    return res.is_error, "\n".join(getattr(c, "text", "") for c in res.content)


# ---------------- 正常调用：和本地工具逐项一致 ----------------
def test_http_tools_return_exactly_what_local_tools_return(store, client):
    local = {t.name: t for t in make_order_tools(store, "u001", ops_log=True)}
    remote = {t.name: t for t in client.tools("u001")}
    assert set(local) == set(remote)
    for name, args in [("list_my_orders", {}), ("get_order", {"order_id": "O1001"}),
                       ("get_order", {"order_id": "O2001"}), ("get_logistics", {"order_id": "O1003"})]:
        assert remote[name].invoke(args) == local[name].invoke(args), (name, args)


def test_http_change_address_writes_to_server_db(store, client):
    out = {t.name: t for t in client.tools("u001")}["change_address"].invoke({"order_id": "O1005", "new_address": ADDR})
    assert "成功" in out or ADDR in out
    assert store.get_order("O1005")["address"] == ADDR


def test_signed_identity_still_limits_to_own_orders(client):
    u2 = {t.name: t for t in client.tools("u002")}
    assert u2["get_order"].invoke({"order_id": "O1001"}) == NOT_FOUND


# ---------------- 服务令牌 ----------------
def test_connection_with_wrong_service_token_is_refused(server):
    with pytest.raises(RuntimeError, match="无法连接"):
        McpToolset(server, token="wrong-token", user_key=KEY, timeout=5)


def test_read_only_token_cannot_change_address(store, server):
    ro = McpToolset(server, token=RO, user_key=KEY)
    try:
        tools = {t.name: t for t in ro.tools("u001")}
        assert json.loads(tools["get_order"].invoke({"order_id": "O1005"}))["order_id"] == "O1005"
        with pytest.raises(RuntimeError, match="orders:write"):
            ro.call("change_address", {"order_id": "O1005", "new_address": ADDR}, "u001")
        assert store.get_order("O1005")["address"] != ADDR
    finally:
        ro.close()


# ---------------- 用户断言 ----------------
def test_bare_user_id_is_refused_over_http(client):
    """stdio 模式认的 _meta.user_id，在 HTTP 模式下不算数：只有服务令牌伪造不了用户。"""
    err, text = _raw_call(client, "get_order", {"order_id": "O1001"}, {USER_META: "u001"})
    assert err and "断言" in text


@pytest.mark.parametrize("assertion,reason", [
    (sign_user("u001", "other-key"), "签名无效"),
    (sign_user("u001", KEY, now=lambda: time.time() - 120), "过期"),
    ("not-an-assertion", "缺少"),
])
def test_bad_assertions_are_refused(client, assertion, reason):
    err, text = _raw_call(client, "get_order", {"order_id": "O1001"}, {ASSERTION_META: assertion})
    assert err and reason in text


def test_tampered_subject_is_refused(client):
    """把断言里的用户换成别人、签名不动：验签失败。"""
    payload, sig = sign_user("u002", KEY).split(".")
    forged = mcp_auth._b64(json.dumps({"sub": "u001", "aud": mcp_auth.AUDIENCE,
                                       "exp": int(time.time()) + 60}, separators=(",", ":")).encode())
    err, text = _raw_call(client, "get_order", {"order_id": "O1001"}, {ASSERTION_META: f"{forged}.{sig}"})
    assert err and "签名无效" in text


def test_verify_user_checks_audience_and_expiry():
    assert verify_user(sign_user("u001", KEY), KEY) == "u001"
    payload = mcp_auth._b64(json.dumps({"sub": "u001", "aud": "other-service", "exp": int(time.time()) + 60}).encode())
    with pytest.raises(AuthError, match="不是签给订单服务"):
        verify_user(f"{payload}.{mcp_auth._sig(KEY, payload)}", KEY)
    with pytest.raises(AuthError, match="过期"):
        verify_user(sign_user("u001", KEY, ttl=60), KEY, now=lambda: time.time() + 61)


# ---------------- 令牌文件和配置 ----------------
def test_new_token_writes_only_digest(tmp_path, monkeypatch, capsys):
    path = tmp_path / "tokens.json"
    monkeypatch.setattr("sys.argv", ["x", "new-token", "--name", "cs-backend", "--tokens", str(path)])
    mcp_auth.main()
    token = capsys.readouterr().out.strip().splitlines()[-1]
    saved = path.read_text()
    assert token not in saved and hash_token(token) in saved


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="权限范围"):
        mcp_auth.StaticTokenVerifier([{"name": "x", "sha256": "0", "scopes": ["orders:admin"]}])


def test_http_client_requires_token_and_key():
    with pytest.raises(ValueError):
        McpToolset("http://127.0.0.1:1/mcp", token="", user_key=KEY)
