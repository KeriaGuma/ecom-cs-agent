"""端到端任务文件本身的校验：字段合法、订单号和工具名有效、开发集和测试集不重复。"""
import json

import pytest

from cs_agent.config import ROOT
from cs_agent.eval_e2e import CONFIRM_WORKFLOW, load_tasks

ORDERS = {"O1001", "O1002", "O1003", "O1004", "O1005", "O2001"}
TOOLS = {"list_my_orders", "get_order", "get_logistics", "change_address", "check_refund", "submit_refund", "search_policy"}
TAGS = {"simple", "multi", "context", "approval", "safety", "handoff", "policy", "chitchat", "write", "boundary", "long"}
FILES = [p for p in (ROOT / "eval" / "e2e_dev.jsonl", ROOT / "eval" / "e2e_test.jsonl", ROOT / "eval" / "e2e_test2.jsonl",
                     ROOT / "eval" / "e2e_test8.jsonl") if p.exists()]
TASKS = [(p.name, json.loads(l)) for p in FILES for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
CONFIRM_FILE = ROOT / "eval" / "e2e_refund_confirm_dev.jsonl"
CONFIRM_TASKS = [(CONFIRM_FILE.name, t) for t in load_tasks(CONFIRM_FILE)]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: f"{t[0]}:{t[1]['id']}")
def test_task_valid(task):
    _, t = task
    assert t["turns"] and all(isinstance(x, str) and x.strip() for x in t["turns"])
    assert t.get("approval", "approve") in ("approve", "reject")
    e = t.get("expect", {})
    assert set(e) <= {"db", "approval", "handoff", "tools_called", "tools_not_called", "reply", "reply_forbidden"}
    for oid, fields in e.get("db", {}).items():
        assert oid in ORDERS and set(fields) <= {"status", "refund_amount", "address"}
    assert set(e.get("tools_called", [])) | set(e.get("tools_not_called", [])) <= TOOLS
    assert all(isinstance(g, list) and g for g in e.get("reply", []))
    assert set(t.get("tags", [])) <= TAGS


def test_ids_unique_and_no_dev_test_overlap():
    tasks = TASKS + CONFIRM_TASKS
    ids = [t["id"] for _, t in tasks]
    assert len(ids) == len(set(ids))
    turns = {}
    for name, t in tasks:
        turns.setdefault(name, set()).add(tuple(x if isinstance(x, str) else x["text"] for x in t["turns"]))
    names = list(turns)
    for i, a in enumerate(names):  # 任意两份任务文件之间不能有完全相同的对话
        for b in names[i + 1:]:
            assert not turns[a] & turns[b], (a, b)


@pytest.mark.parametrize("task", [t for _, t in CONFIRM_TASKS], ids=lambda t: t["id"])
def test_confirmation_dev_task_valid(task):
    assert task["source"] == "synthetic" and task["refund_workflow"] == CONFIRM_WORKFLOW
    assert task["turns"]
    for turn in task["turns"]:
        assert (turn if isinstance(turn, str) else turn["text"]).strip()
        if isinstance(turn, dict):
            for action in turn["refund_actions"]:
                assert action["order_id"] in ORDERS
                assert action["decision"] in {"confirm", "cancel", "approve", "reject"}
    expect = task["expect"]
    for row in expect["refunds"]:
        assert row["order_id"] in ORDERS and set(row) == {"order_id", "status"}
        assert row["status"] in {"draft", "pending_approval", "cancelled", "rejected", "succeeded"}
    assert (set(expect.get("tools_called", [])) | set(expect.get("tools_not_called", []))) <= TOOLS | {"prepare_refund"}
    assert set(task.get("tags", [])) <= TAGS
