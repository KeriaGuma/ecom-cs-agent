"""检索评测问题集本身的校验：标准答案必须是知识库里真实存在的条款；开发集和测试集不能重复。"""
import json
from pathlib import Path

import pytest

from cs_agent.config import ROOT, settings
from cs_agent.kb import PolicyKB

REFS = {f"{s.source} · {s.title}" for s in PolicyKB(settings.kb_dir).sections}
FILES = [p for p in (ROOT / "eval" / "kb_dev.jsonl", ROOT / "eval" / "kb_test.jsonl") if p.exists()]
CASES = [(p.name, json.loads(line)) for p in FILES for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}:{c[1]['id']}")
def test_case_valid(case):
    _, c = case
    assert c["query"].strip() and c["expected"]
    assert set(c["expected"]) <= REFS, set(c["expected"]) - REFS
    assert c.get("tag") in ("colloquial", "term")


def test_ids_unique_and_no_overlap():
    ids = [c["id"] for _, c in CASES]
    assert len(ids) == len(set(ids))
    by_file = {}
    for name, c in CASES:
        by_file.setdefault(name, set()).add(c["query"].strip())
    if len(by_file) == 2:
        assert not set.intersection(*by_file.values())
