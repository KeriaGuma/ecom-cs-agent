"""领域扩展实验：五个新领域的确定性规则、权限校验、单/多 Agent 提示词对齐与评测快照比对。"""
from dataclasses import replace

import pytest

from bench.multi_agent import SUPERVISOR_PROMPT, build_multi_runner, build_supervisor_prompt, build_workers
from bench.multi_agent.router import worker_names
from cs_agent.agent import SINGLE_PROMPT, build_agent_prompt
from cs_agent.config import settings
from cs_agent.domains import TIERS, coupon_tools, invoice_tools, parse_domains, points_tools, price_tools, repair_tools
from cs_agent.eval_e2e import check
from cs_agent.store import ShopStore
from tests.conftest import TODAY


@pytest.fixture
def dstore(tmp_path) -> ShopStore:
    s = ShopStore(str(tmp_path / "d.db"))
    s.init(today=TODAY, domain_data=True)
    return s


def T(factory, store, user="u001"):
    return {t.name: t for t in factory(store, user, lambda: TODAY)}


def test_points_reissue_only_missing_order_once(dstore):
    t = T(points_tools, dstore)
    assert "已经发放过" in t["reissue_points"].invoke({"order_id": "O1001"})
    assert "补发 1899 积分" in t["reissue_points"].invoke({"order_id": "O1002"})
    assert dstore.snapshot("u001")["points"] == {"balance": 3159}
    assert "已经发放过" in t["reissue_points"].invoke({"order_id": "O1002"})
    assert "只有已签收" in t["reissue_points"].invoke({"order_id": "O1005"})


def test_points_exchange_and_insufficient(dstore):
    t = T(points_tools, dstore)
    assert "积分不足" in t["exchange_points"].invoke({"value": 20})
    assert "兑换成功" in t["exchange_points"].invoke({"value": 10})
    snap = dstore.snapshot("u001")
    assert snap["points"]["balance"] == 260 and snap["coupons"]["P1"]["status"] == "unused"


def test_invoice_rules(dstore):
    t = T(invoice_tools, dstore)
    assert "只有已签收" in t["issue_invoice"].invoke({"order_id": "O1005", "invoice_type": "normal", "title": "个人"})
    assert "税号" in t["issue_invoice"].invoke({"order_id": "O1002", "invoice_type": "special", "title": "杭州云帆科技有限公司"})
    assert "已为订单 O1001" in t["issue_invoice"].invoke({"order_id": "O1001", "invoice_type": "normal", "title": "个人"})
    assert "已经开过" in t["issue_invoice"].invoke({"order_id": "O1001", "invoice_type": "normal", "title": "个人"})
    assert "换开" in t["reissue_invoice"].invoke({"order_id": "O1001", "title": "张三"})
    assert "已经换开过" in t["reissue_invoice"].invoke({"order_id": "O1001", "title": "李四"})
    inv = dstore.snapshot("u001")["invoices"]["O1001"]
    assert inv["title"] == "张三" and inv["reissued"] is True


def test_coupon_return_requires_refund_and_unexpired(dstore):
    t = T(coupon_tools, dstore)
    assert "还没有退款" in t["return_coupon"].invoke({"order_id": "O1001"})
    dstore.mark_refunded("O1001", 199)
    assert "已退还优惠券 C01" in t["return_coupon"].invoke({"order_id": "O1001"})
    dstore.mark_refunded("O1004", 89)
    assert "已过期" in t["return_coupon"].invoke({"order_id": "O1004"})
    assert "没有使用优惠券" in t["return_coupon"].invoke({"order_id": "O1002"})


def test_repair_rules(dstore):
    t = T(repair_tools, dstore)
    assert "15 天内的质量问题" in t["apply_repair"].invoke({"order_id": "O1001", "issue": "没声音", "cause": "quality"})
    assert "不是电子产品" in t["apply_repair"].invoke({"order_id": "O1004", "issue": "漏水", "cause": "quality"})
    assert "免费维修" in t["apply_repair"].invoke({"order_id": "O1006", "issue": "黑屏", "cause": "quality"})
    assert "不能重复申请" in t["apply_repair"].invoke({"order_id": "O1006", "issue": "黑屏", "cause": "quality"})
    assert dstore.snapshot("u001")["repairs"] == {"O1006": {"status": "pending", "paid": False}}


def test_repair_human_damage_is_paid(dstore):
    out = T(repair_tools, dstore)["apply_repair"].invoke({"order_id": "O1006", "issue": "进水", "cause": "human"})
    assert "付费维修" in out and dstore.snapshot("u001")["repairs"]["O1006"]["paid"] is True


def test_price_protect_rules(dstore):
    t = T(price_tools, dstore)
    assert "差价 30.00" in t["apply_price_protect"].invoke({"order_id": "O1001"})
    assert "已经申请过" in t["apply_price_protect"].invoke({"order_id": "O1001"})
    assert "无需价保" in t["apply_price_protect"].invoke({"order_id": "O1002"})
    assert "超过 7 天" in t["apply_price_protect"].invoke({"order_id": "O1006"})
    assert "秒杀" in t["apply_price_protect"].invoke({"order_id": "O1007"})
    assert dstore.snapshot("u001")["price_protect"] == {"O1001": 30.0}


def test_domain_tools_enforce_ownership(dstore):
    for factory, name, args in [(points_tools, "reissue_points", {}), (coupon_tools, "return_coupon", {}),
                                (price_tools, "apply_price_protect", {}),
                                (repair_tools, "apply_repair", {"issue": "x", "cause": "quality"}),
                                (invoice_tools, "issue_invoice", {"invoice_type": "normal", "title": "个人"})]:
        assert "未找到" in T(factory, dstore)[name].invoke({"order_id": "O2001", **args})


def test_base_prompts_unchanged_without_domains():
    assert build_supervisor_prompt(()) == SUPERVISOR_PROMPT and build_agent_prompt(()) == SINGLE_PROMPT
    p = build_supervisor_prompt(TIERS["T2"])
    assert "price_agent" in p and "企业采购、价保、" not in p
    assert "企业采购、价保、" in build_supervisor_prompt(TIERS["T1"])
    assert worker_names(TIERS["T1"])[-2:] == ("points_agent", "invoice_agent")
    assert parse_domains("T1") == ("points", "invoice") and parse_domains("") == ()


def test_check_handles_domain_collections(dstore):
    before = dstore.snapshot("u001")
    T(price_tools, dstore)["apply_price_protect"].invoke({"order_id": "O1001"})
    T(points_tools, dstore)["exchange_points"].invoke({"value": 5})
    after = dstore.snapshot("u001")
    seen = {"approval": False, "handoff": False}
    ok = check({"db": {"price_protect": {"O1001": 30}, "points": {"balance": 760}}}, before, after, seen, [], "")
    assert ok == []
    bad = check({"db": {}}, before, after, seen, [], "")
    assert any("price_protect.O1001" in f for f in bad) and any("积分不应变化" in f for f in bad)


def test_runner_routes_to_domain_worker(dstore):
    from langchain_core.messages import AIMessage, HumanMessage

    from cs_agent.kb import PolicyKB
    from tests.test_graph import ScriptedRouter
    from tests.test_workers import ScriptedLLM, call

    cfg = replace(settings, domains="T2")
    llm = ScriptedLLM(replies=[call("apply_price_protect", {"order_id": "O1001"}),
                               AIMessage(content="已退差价 30 元")])
    workers = build_workers(llm, dstore, PolicyKB(settings.kb_dir), cfg, today=lambda: TODAY)
    assert "price_agent" in workers
    runner = build_multi_runner(ScriptedRouter(["price_agent", "FINISH"]), store=dstore, workers=workers, settings=cfg)
    out = runner.run_turn("d", "u001", HumanMessage("耳机降价了"))
    assert dstore.snapshot("u001")["price_protect"] == {"O1001": 30.0}
    assert out["messages"][-1].content == "已退差价 30 元"


def test_refund_claws_back_credited_points(dstore):
    dstore.mark_refunded("O1001", 199)
    assert dstore.snapshot("u001")["points"] == {"balance": 1061}
    dstore.mark_refunded("O1001", 199)
    assert dstore.snapshot("u001")["points"] == {"balance": 1061}
    dstore.mark_refunded("O1002", 1899)
    assert dstore.snapshot("u001")["points"] == {"balance": 1061}


def test_refund_without_points_data_is_unchanged(store):
    store.mark_refunded("O1001", 199)
    assert store.get_order("O1001")["status"] == "refunded"
