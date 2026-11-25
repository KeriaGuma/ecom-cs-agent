"""校验路由评测用例本身：格式、标签合法、开发集和测试集零泄漏，以及统计逻辑。"""
from collections import Counter

import pytest

from bench.multi_agent import ACTIONS, RouteDecision
from cs_agent.eval_router import EVAL_DIR, evaluate, load_cases, wilson

TAGS = {"single", "multi", "partial", "context", "hard", "colloquial", "adversarial"}
DEV = load_cases([EVAL_DIR / "route_dev.jsonl"])
TEST = load_cases([EVAL_DIR / "route_test.jsonl"])


@pytest.mark.parametrize("case", DEV + TEST, ids=lambda c: c["id"])
def test_case_schema(case):
    assert case["expected"] in ACTIONS
    assert set(case.get("also_ok", [])) <= set(ACTIONS) - {case["expected"]}
    assert case["messages"] and case["messages"][-1]["role"] == "user"
    assert all(m["role"] in ("user", "assistant") and m["content"] for m in case["messages"])
    assert case["tags"] and set(case["tags"]) <= TAGS
    for r in case.get("turn_results", []):
        assert r["agent"] in ("order_agent", "refund_agent", "kb_agent")


def test_ids_unique():
    ids = [c["id"] for c in DEV + TEST]
    assert len(ids) == len(set(ids))


def test_no_leak_between_dev_and_test():
    def key(c):
        return (tuple(m["content"] for m in c["messages"]), tuple(r["content"] for r in c.get("turn_results", [])))

    assert not {key(c) for c in DEV} & {key(c) for c in TEST}


def test_every_label_has_enough_cases():
    counts = Counter(c["expected"] for c in DEV + TEST)
    assert all(counts[a] >= 20 for a in ACTIONS), counts


class FixedRouter:
    def __init__(self, answers):
        self.answers = answers

    def decide(self, state):
        text = state["messages"][-1].content
        if text == "boom":
            raise TimeoutError
        return RouteDecision(next=self.answers[text])


def test_evaluate_counts_also_ok_and_errors():
    cases = [
        {"id": "a", "messages": [{"role": "user", "content": "x"}], "expected": "order_agent", "also_ok": ["kb_agent"], "tags": ["multi"]},
        {"id": "b", "messages": [{"role": "user", "content": "y"}], "expected": "FINISH", "tags": ["single"]},
        {"id": "c", "messages": [{"role": "user", "content": "boom"}], "expected": "clarify", "tags": ["single"]},
    ]
    report = evaluate(FixedRouter({"x": "kb_agent", "y": "clarify"}), cases, workers=2)
    assert report["correct"] == 1
    assert report["acc_by_tag"]["multi"]["acc"] == 1.0
    assert report["acc_by_tag"]["single"]["acc"] == 0.0
    assert report["confusion"]["order_agent"] == {"order_agent": 1}
    assert {e["pred"] for e in report["errors"]} == {"clarify", "ERROR:TimeoutError"}


def test_wilson_interval():
    lo, hi = wilson(21, 22)
    assert 0.77 < lo < 0.79 and 0.99 < hi <= 1.0
