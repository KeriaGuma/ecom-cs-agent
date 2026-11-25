"""工具数量扩展实验：干扰工具的数量、领域划分，以及单 Agent / 多 Agent 各自挂上的工具。"""
from dataclasses import replace

from cs_agent.config import settings
from cs_agent.distractors import CATALOG, distractor_tools
from cs_agent.kb import PolicyKB
from cs_agent.single_agent import build_single_tools
from cs_agent.tools import make_kb_tools, make_order_tools, make_refund_tools

REAL = {t.name for t in make_order_tools(None, "u") + make_refund_tools(None, "u", 500, lambda p: None)
        + make_kb_tools(PolicyKB(settings.kb_dir))}


def test_catalog_has_40_unique_names_not_clashing_with_real_tools():
    names = [c[0] for c in CATALOG]
    assert len(names) == 40 and len(set(names)) == 40 and not set(names) & REAL


def test_levels_are_prefixes_and_domains_balanced():
    for n in (10, 20, 40):
        tools = distractor_tools(n)
        assert [t.name for t in tools] == [c[0] for c in CATALOG[:n]]
        per = [len(distractor_tools(n, d)) for d in ("order", "refund", "kb")]
        assert sum(per) == n and max(per) - min(per) <= 1


def test_distractors_are_read_only_and_callable():
    for t in distractor_tools(40):
        args = {k: "测试" for k in (t.args or {})}
        assert t.invoke(args)  # 返回固定数据，不触碰数据库


def test_single_agent_gets_all_distractors(store):
    cfg = replace(settings, extra_tools=20)
    tools = build_single_tools(store, PolicyKB(settings.kb_dir), "u001", cfg, lambda p: None, lambda r: None)
    assert len(tools) == 8 + 20  # 7 个业务工具 + 转人工 + 20 个干扰工具
