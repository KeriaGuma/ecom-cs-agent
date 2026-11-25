"""领域扩展实验的任务文件校验：字段、领域、数据键、工具名都要合法。"""
import json
import re

import pytest

from cs_agent.config import ROOT
from cs_agent.domains import DOMAINS

FILES = [p for p in (ROOT / "eval" / "e2e_domains_dev.jsonl", ROOT / "eval" / "e2e_domains_test.jsonl") if p.exists()]
TASKS = [(p.name, json.loads(l)) for p in FILES for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
TOOLS = {"list_my_orders", "get_order", "get_logistics", "change_address", "check_refund", "submit_refund", "search_policy",
         "get_points", "reissue_points", "exchange_points", "issue_invoice", "reissue_invoice", "list_my_coupons",
         "return_coupon", "apply_repair", "get_repair_status", "apply_price_protect"}
TAGS = {"simple", "multi", "context", "approval", "safety", "handoff", "policy", "chitchat", "write", "boundary", "long",
        "domain", "cross"}
KEYS = {"points", "invoices", "coupons", "repairs", "price_protect"}


@pytest.mark.parametrize("task", TASKS, ids=lambda t: f"{t[0]}:{t[1]['id']}")
def test_domain_task_valid(task):
    _, t = task
    assert t["turns"] and all(x.strip() for x in t["turns"])
    assert set(t.get("domains", [])) <= set(DOMAINS) and set(t.get("absent", [])) <= set(DOMAINS)
    assert set(t.get("tags", [])) <= TAGS
    e = t.get("expect", {})
    for k, v in e.get("db", {}).items():
        assert re.match(r"^O(100[1-7]|2001)$", k) or k in KEYS, k
        if k.startswith("O"):
            assert set(v) <= {"status", "refund_amount", "address"}
    assert set(e.get("tools_called", [])) | set(e.get("tools_not_called", [])) <= TOOLS


def test_ids_unique():
    ids = [t["id"] for _, t in TASKS]
    assert len(ids) == len(set(ids))
