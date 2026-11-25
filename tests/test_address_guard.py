"""改地址时收件人、电话（含带空格/短横线/+86 前缀）不能混进地址：工具拒绝修改并指导模型只传地址重试。"""
import os
import subprocess
import sys
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cs_agent.agent import build_agent_runner
from cs_agent.config import settings
from cs_agent.kb import PolicyKB
from cs_agent.mcp_client import McpToolset
from cs_agent.mcp_server import build_server
from cs_agent.tools import ADDRESS_REJECTED, contact_in_address, make_order_tools
from tests.conftest import TODAY
from tests.test_workers import ScriptedLLM, call

ADDR = "杭州市滨江区网商路 699 号"


@pytest.mark.parametrize("text,bad", [
    (ADDR, False),
    ("上海市浦东新区世纪大道 100 号 1201 室", False),
    ("北京市朝阳区建国路 88 号 SOHO 现代城 2 号楼 1503", False),
    (f"{ADDR}，电话 13812345678", True),
    (f"{ADDR} 13812345678", True),
    (f"{ADDR} 138-1234-5678", True),
    (f"{ADDR} +86 138 1234 5678", True),
    (f"{ADDR}，收件人王芳", True),
    ("杭州市余杭区文一西路 969 号 0571-88886666", True),
])
def test_contact_in_address(text, bad):
    assert contact_in_address(text) is bad


def _change(store, address, guard=True):
    tools = {t.name: t for t in make_order_tools(store, "u001", ops_log=True, address_guard=guard)}
    return tools["change_address"].invoke({"order_id": "O1005", "new_address": address})


def test_address_with_phone_is_rejected_and_nothing_changes(store):
    before = store.get_order("O1005")["address"]
    assert _change(store, f"{ADDR} 138-1234-5678") == ADDRESS_REJECTED
    assert store.get_order("O1005")["address"] == before
    assert not store.list_ops("O1005")


def test_clean_address_still_works(store):
    assert ADDR in _change(store, ADDR)
    assert store.get_order("O1005")["address"] == ADDR


def test_guard_off_keeps_original_behaviour(store):
    _change(store, f"{ADDR}，电话 13812345678", guard=False)
    assert store.get_order("O1005")["address"] == f"{ADDR}，电话 13812345678"
    desc = {g: [t.description for t in make_order_tools(store, "u001", address_guard=g) if t.name == "change_address"][0]
            for g in (True, False)}
    assert desc[False] == "修改订单收货地址。仅未发货（status=paid）的订单可以修改。"
    assert "不含收件人和电话" in desc[True]


def test_rejection_is_identical_through_mcp(store):
    toolset = McpToolset(build_server(store, ops_log=True, address_guard=True))
    try:
        remote = {t.name: t for t in toolset.tools("u001")}["change_address"]
        assert remote.invoke({"order_id": "O1005", "new_address": f"{ADDR}，电话 13812345678"}) == ADDRESS_REJECTED
        assert remote.description == [t.description for t in make_order_tools(store, "u001", address_guard=True)
                                      if t.name == "change_address"][0]
    finally:
        toolset.close()


def test_agent_retries_with_address_only(store):
    seen = []

    class Recording(ScriptedLLM):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append([m.content for m in messages if isinstance(m, ToolMessage)])
            return super()._generate(messages, stop, run_manager, **kwargs)

    llm = Recording(replies=[
        call("change_address", {"order_id": "O1005", "new_address": f"{ADDR}，电话 13812345678"}, 0),
        call("change_address", {"order_id": "O1005", "new_address": ADDR}, 1),
        AIMessage(content="地址已改好；联系电话这里改不了。"),
    ])
    kb = PolicyKB(settings.kb_dir)
    runner = build_agent_runner(llm, store=store, kb=kb, retriever=kb, today=lambda: TODAY,
                                settings=replace(settings, address_guard=True))
    runner.run_turn("g1", "u001", HumanMessage(f"键盘地址改成{ADDR}，电话 13812345678"))
    assert not llm.replies
    assert seen[1] == [ADDRESS_REJECTED]
    assert store.get_order("O1005")["address"] == ADDR
    assert len(store.list_ops("O1005")) == 1


def test_prompts_unchanged_when_guard_off():
    code = ("from bench.multi_agent.workers import ORDER_PROMPT; from cs_agent.agent import SINGLE_PROMPT;"
            "print(int('收件人和联系电话改不了' in ORDER_PROMPT), int('收件人和联系电话改不了' in SINGLE_PROMPT))")
    for guard, want in (("0", "0 0"), ("1", "1 1")):
        out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "ADDRESS_GUARD": guard},
                             capture_output=True, text=True, check=True).stdout.split("\n")[-2]
        assert out == want, guard
