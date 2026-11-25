"""离线零密钥客服沙箱（纯 Python 异步单 Agent）：验证完整售后退货验货、人工接管、冷重启 Cookie 保持与并发隔离。"""
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from cs_agent.demo import DEMO_DATE, bootstrap_demo_db, create_demo_app
from cs_agent.store import ShopStore
from tests.test_api import _events


def _login(client, username):
    response = client.post("/demo/login", json={"username": username, "password": "demo"})
    assert response.status_code == 200, response.text


def _chat(client, text, session="walkthrough"):
    response = client.post("/chat", json={"session_id": session, "message": text})
    assert response.status_code == 200, response.text
    events = dict(_events(response))
    assert "done" in events, events
    return events["message"]["text"], events["done"]


def test_bootstrap_keeps_demo_state_and_refuses_non_demo_db(tmp_path):
    path = tmp_path / "demo.db"
    with ThreadPoolExecutor(max_workers=2) as pool:
        stores = list(pool.map(bootstrap_demo_db, (path, path)))
    assert len(stores[0].get_logistics("O1003")) == 2
    assert stores[1].get_order("O1005")["created_at"] == DEMO_DATE.isoformat()
    stores[0].update_address("O1005", "演示地址")
    assert bootstrap_demo_db(path).get_order("O1005")["address"] == "演示地址"

    business_path = tmp_path / "shop.db"
    business_store = ShopStore(str(business_path))
    business_store.init(today=DEMO_DATE)
    original = business_path.read_bytes()
    with pytest.raises(ValueError, match="DEMO_DB_PATH"):
        bootstrap_demo_db(business_path)
    assert business_path.read_bytes() == original


def test_keyless_demo_walkthrough_and_restart_persists_session_cookie(tmp_path, monkeypatch):
    def reject_model_provider(*args, **kwargs):
        pytest.fail("离线演示不能初始化真实模型客户端")

    monkeypatch.setattr("cs_agent.llm.get_llm", reject_model_provider)
    monkeypatch.setenv("DEMO_MODEL_DELAY_MS", "0")
    path = tmp_path / "demo.db"
    app = create_demo_app(path)
    with TestClient(app) as customer, TestClient(app) as staff:
        assert customer.get("/ready").json() == {"ok": True, "runtime": "python-async"}
        _login(customer, "u001")
        assert "杭州转运中心" in _chat(customer, "查一下订单 O1003 的物流")[0]
        policy, _ = _chat(customer, "七天无理由退货运费谁出？")
        assert "退货运费由买家承担" in policy and "after_sales_policy.md" in policy

        # 未发货 O1005：仅退款确认后直接成功
        reply, done = _chat(customer, "订单 O1005 的机械键盘不想要了，帮我取消退款")
        refund_only = done["refunds"][0]
        assert refund_only["status"] == "draft" and refund_only["refund_type"] == "refund_only"
        assert "尚未提交" in reply
        assert customer.post(f"/refunds/{refund_only['refund_id']}/confirm", json={"confirmed": True}).json()["status"] == "succeeded"

        # 已签收大额 O1002：确认 -> 待审批 -> 审批通过进入待退货 -> 提交退货单号 -> 验货收货打款
        _, done = _chat(customer, "订单 O1002 的扫地机器人买错了，帮我申请退货退款")
        large_refund = done["refunds"][0]
        assert large_refund["status"] == "draft" and large_refund["amount"] == 1899
        assert customer.post(f"/refunds/{large_refund['refund_id']}/confirm", json={"confirmed": True}).json()["status"] == "pending_approval"
        _, done = _chat(customer, "我要转人工")
        ticket_id = done["ticket_id"]

        _login(staff, "s01")
        assert staff.post(f"/tickets/{ticket_id}/claim").json()["assignee"] == "s01"
        approval = f"/refunds/{large_refund['refund_id']}/decision"
        assert staff.post(approval, json={"approved": True}).status_code == 403

        _login(staff, "a01")
        assert staff.post(approval, json={"approved": True}).json()["status"] == "awaiting_return"
        assert customer.post(f"/refunds/{large_refund['refund_id']}/return", json={"tracking_no": "SF900123"}).json()["return_tracking_no"] == "SF900123"
        assert staff.post(f"/refunds/{large_refund['refund_id']}/receive", json={}).json()["status"] == "succeeded"

        _login(staff, "s01")
        assert staff.post(f"/tickets/{ticket_id}/close").status_code == 200
        saved_cookie = customer.cookies.get("ecom_demo_customer")

    # 重启新应用实例：派生签名密钥保持稳定，原登录 Cookie 依然有效无需重登
    with TestClient(create_demo_app(path)) as restarted:
        restarted.cookies.set("ecom_demo_customer", saved_cookie)
        history = restarted.get("/sessions/walkthrough/messages").json()["messages"]
        assert any("杭州转运中心" in item["content"] for item in history)
        reply, _ = _chat(restarted, "查一下 O1005 的退款进度")
        assert "退款成功" in reply
        assert restarted.get(f"/refunds/{large_refund['refund_id']}").json()["status"] == "succeeded"


def test_demo_keeps_concurrent_customers_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODEL_DELAY_MS", "10")
    app = create_demo_app(tmp_path / "demo.db")

    def lookup(username):
        with TestClient(app) as client:
            _login(client, username)
            return _chat(client, "查一下订单 O1003 的物流", session="same-public-id")[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        user, other = pool.map(lookup, ("u001", "u002"))
    assert "杭州转运中心" in user
    assert "不属于" in other and "杭州转运中心" not in other
