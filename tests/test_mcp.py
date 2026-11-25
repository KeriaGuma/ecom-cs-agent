"""订单工具 MCP 化：换协议不改行为，用户身份不暴露给模型。"""
import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from cs_agent import mcp_client
from cs_agent.agent import build_agent_runner
from cs_agent.config import settings
from cs_agent.kb import PolicyKB
from cs_agent.mcp_client import McpToolset, stdio_params
from cs_agent.mcp_server import build_server
from cs_agent.store import ShopStore
from cs_agent.tools import NOT_FOUND, make_order_tools
from tests.conftest import TODAY
from tests.test_workers import ScriptedLLM, call

READS = [
    ("list_my_orders", {}),
    ("get_order", {"order_id": "O1001"}),
    ("get_order", {"order_id": "o1003"}),
    ("get_order", {"order_id": "O2001"}),
    ("get_order", {"order_id": "O9999"}),
    ("get_logistics", {"order_id": "O1003"}),
    ("get_logistics", {"order_id": "O1005"}),
    ("change_address", {"order_id": "O1003", "new_address": "杭州市滨江区网商路 699 号"}),
]


@pytest.fixture
def mcp(store):
    toolset = McpToolset(build_server(store, ops_log=True))
    yield toolset
    toolset.close()


def _local(store, user="u001"):
    return {t.name: t for t in make_order_tools(store, user, ops_log=True)}


def test_mcp_tools_return_exactly_what_local_tools_return(store, mcp):
    local, remote = _local(store), {t.name: t for t in mcp.tools("u001")}
    assert set(local) == set(remote)
    for name, args in READS:
        assert remote[name].invoke(args) == local[name].invoke(args), (name, args)


def test_address_change_has_same_effect_through_mcp(tmp_path):
    results = []
    for kind in ("local", "mcp"):
        store = ShopStore(str(tmp_path / f"{kind}.db"))
        store.init(today=TODAY)
        toolset = McpToolset(build_server(store)) if kind == "mcp" else None
        tools = {t.name: t for t in (toolset.tools("u001") if toolset else make_order_tools(store, "u001", ops_log=True))}
        out = tools["change_address"].invoke({"order_id": "O1005", "new_address": "杭州市滨江区网商路 699 号"})
        ops = [{k: v for k, v in r.items() if k not in ("op_id", "created_at")} for r in store.list_ops("O1005")]
        results.append((out, store.get_order("O1005")["address"], ops))
        if toolset:
            toolset.close()
    assert results[0] == results[1]


def test_user_identity_is_not_a_tool_parameter(mcp):
    for t in mcp.tools("u001"):
        assert not any("user" in k.lower() for k in t.args), t.name
    assert {t.name: set(t.args) for t in mcp.tools("u001")} == {
        "list_my_orders": set(), "get_order": {"order_id"}, "get_logistics": {"order_id"},
        "change_address": {"order_id", "new_address"}}


def test_calls_without_identity_are_refused(mcp):
    with pytest.raises(RuntimeError, match="缺少用户身份"):
        mcp.call("get_order", {"order_id": "O1001"}, user_id="")


def test_each_user_only_sees_own_orders(mcp):
    u2 = {t.name: t for t in mcp.tools("u002")}
    assert u2["get_order"].invoke({"order_id": "O1001"}) == NOT_FOUND
    assert json.loads(u2["get_order"].invoke({"order_id": "O2001"}))["user_id"] == "u002"


def test_stdio_server_subprocess_works_and_gets_no_secrets(store, monkeypatch):
    monkeypatch.setenv("OPS_LOG", "1")
    monkeypatch.setenv("LLM_API_KEY", "should-not-leak")
    params = stdio_params(store.path)
    assert set(params.env) <= set(mcp_client.SERVER_ENV)
    toolset = McpToolset(params)
    try:
        out = {t.name: t for t in toolset.tools("u001")}["get_order"].invoke({"order_id": "O1002"})
        assert json.loads(out)["item"] == "扫地机器人"
    finally:
        toolset.close()


def test_agent_changes_address_through_mcp(store):
    llm = ScriptedLLM(replies=[
        call("change_address", {"order_id": "O1005", "new_address": "杭州市滨江区网商路 699 号"}),
        AIMessage(content="已修改"),
    ])
    kb = PolicyKB(settings.kb_dir)
    runner = build_agent_runner(llm, store=store, kb=kb, retriever=kb, today=lambda: TODAY,
                                settings=replace(settings, order_tools="mcp"))
    try:
        runner.run_turn("m1", "u001", HumanMessage("键盘地址改成杭州市滨江区网商路 699 号"))
        assert store.get_order("O1005")["address"] == "杭州市滨江区网商路 699 号"
    finally:
        mcp_client.close(store.path)
