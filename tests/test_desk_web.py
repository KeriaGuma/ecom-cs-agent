"""离线演示的 HTTP 职责验证、人工接管与幂等回执重放。"""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from fastapi.testclient import TestClient

from bench.multi_agent import build_multi_runner
from cs_agent.api import create_app
from cs_agent.desk_service import DeskService
from cs_agent.refunds import RefundService
from tests.conftest import TODAY
from tests.test_api import STAFF, U1, _events
from tests.test_graph import WORKERS, ScriptedRouter


def app_for(store, routes, **kwargs):
    router = ScriptedRouter(routes)
    runner = build_multi_runner(router, store=store, workers=WORKERS)
    return create_app(runner, store, demo_auth=True, **kwargs), router


def login(client, username):
    response = client.post("/demo/login", json={"username": username, "password": "demo"})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    return response.json()


def test_demo_rejects_forged_identity_and_separates_employee_roles(store):
    refunds = RefundService(store, 500, today=lambda: TODAY, enable_simulated_faults=True)
    refund_id = refunds.prepare(user_id="u001", thread_id="u001:demo", message_id="first",
                                order_id="O1002", reason="no_reason")["refund_id"]
    assert refunds.confirm(refund_id, "u001", True)["status"] == "pending_approval"
    app, _ = app_for(store, [], refund_service=refunds)
    client = TestClient(app)
    assert client.get("/desk").status_code == 200
    assert client.get("/sessions/demo/messages", headers=U1).status_code == 401
    assert client.get("/tickets", headers=STAFF).status_code == 403
    assert client.get("/approvals", headers={"X-Staff-Id": "a01"}).status_code == 403
    assert client.post("/demo/login", json={"username": "a01", "password": "wrong"}).status_code == 401
    client.cookies.set("ecom_demo_staff", "a01.9999999999.forged")
    assert client.get("/approvals").status_code == 403
    client.cookies.clear()

    assert login(client, "u001")["role"] == "customer"
    assert login(client, "s01")["role"] == "staff"
    assert client.get("/demo/session").json() == {
        "customer": {"username": "u001", "role": "customer"},
        "operator": {"username": "s01", "role": "staff"},
    }
    fault = f"/demo/refunds/{refund_id}/fault"
    decision = f"/refunds/{refund_id}/decision"
    receive = f"/refunds/{refund_id}/receive"
    assert client.post(fault, json={"fault": "ack_lost"}).status_code == 403
    assert client.post(decision, json={"approved": True}).status_code == 403
    assert client.post(receive, json={}).status_code == 403
    login(client, "a01")
    assert client.get("/tickets").status_code == 403
    assert client.get("/approvals").json()["refunds"][0]["refund_id"] == refund_id
    result = client.post(fault, json={"fault": "ack_lost"})
    assert result.status_code == 200 and result.json()["fault"] == "ack_lost"
    assert client.post(decision, json={"approved": True}).json()["status"] == "awaiting_return"
    assert client.post(receive, json={"tracking_no": "SF10010"}).json()["status"] == "unknown"
    assert client.post("/demo/logout", json={"role": "staff"}).status_code == 200
    assert client.get("/approvals").status_code == 403
    assert client.get("/sessions/demo/messages").status_code == 200


def test_human_claim_pauses_bot_and_close_returns_full_context(store):
    app, router = app_for(store, ["handoff_human", "FINISH"])
    buyer, assigned, other = TestClient(app), TestClient(app), TestClient(app)
    login(buyer, "u001")
    login(assigned, "s01")
    login(other, "s02")
    body = {"session_id": "h", "message": "转人工", "request_id": "first"}
    first = buyer.post("/chat", json=body)
    ticket_id = dict(_events(first))["done"]["ticket_id"]
    ticket = f"/tickets/{ticket_id}"
    history = "/sessions/h/messages"
    assert buyer.get(history).json()["handoff"] == {"ticket_id": ticket_id, "active": False}
    assert assigned.post(f"{ticket}/close").status_code == 409
    assert assigned.post(f"{ticket}/claim").json()["assignee"] == "s01"
    assert other.post(f"{ticket}/claim").status_code == 409
    assert other.get(ticket).status_code == 403
    assert other.post(f"{ticket}/messages", json={"message": "越权回复"}).status_code == 409
    assert other.post(f"{ticket}/close").status_code == 409
    assert buyer.delete(history).status_code == 409

    question = {**body, "message": "帮我跟进一下", "request_id": "human-message"}
    update = buyer.post("/chat", json=question)
    assert dict(_events(update))["done"]["final_action"] == "human"
    assert dict(_events(buyer.post("/chat", json=question)))["done"] == dict(_events(update))["done"]
    assert len(router.seen) == 1
    assert assigned.post(f"{ticket}/messages", json={"message": "已人工记录"}).status_code == 200
    assert buyer.get(history).json()["handoff"]["active"]
    assert buyer.get(history).json()["messages"][-2:] == [
        {"role": "user", "content": "帮我跟进一下"}, {"role": "staff", "content": "已人工记录"},
    ]
    assert assigned.get(ticket).json()["messages"][-1]["role"] == "staff"
    assert assigned.post(f"{ticket}/close").json()["status"] == "closed"
    events = assigned.get(ticket).json()["events"]
    assert [(event["action"], event["actor"]) for event in events] == [
        ("opened", "user:u001"), ("claimed", "staff:s01"), ("replied", "staff:s01"), ("closed", "staff:s01"),
    ]
    assert events[2]["message_id"]
    assert buyer.get(history).json()["handoff"] is None
    resumed = buyer.post("/chat", json={**body, "message": "谢谢", "request_id": "resume"})
    assert dict(_events(resumed))["done"]["final_action"] == "finish"
    assert len(router.seen) == 2
    assert any(row["content"] == "已人工记录" for row in store.load_session("u001:h")["messages"])

    assert buyer.delete(history).json() == {"cleared": True}
    assert buyer.get(history).json()["messages"] == [] and not store.load_session("u001:h")
    assert store.get_ticket(ticket_id)["status"] == "closed"
    assert buyer.post("/chat", json=question).status_code == 409


def test_request_replay_survives_restart_and_is_scoped_to_buyer(store):
    app, _ = app_for(store, ["FINISH"])
    buyer = TestClient(app)
    login(buyer, "u001")
    body = {"session_id": "same", "message": "你好", "request_id": "v1"}
    original = dict(_events(buyer.post("/chat", json=body)))
    restarted_app, router = app_for(store, ["FINISH"])
    restarted = TestClient(restarted_app)
    login(restarted, "u001")
    assert dict(_events(restarted.post("/chat", json=body))) == original
    assert not router.seen
    assert restarted.post("/chat", json={**body, "message": "不同内容"}).status_code == 409
    receipts = DeskService(store, pii_mask=True)
    assert receipts.reserve("u001:same", "interrupted", receipts.fingerprint("你好", [])) is None
    assert restarted.post("/chat", json={**body, "request_id": "interrupted"}).status_code == 409
    assert not router.seen
    login(restarted, "u002")
    assert restarted.get("/sessions/same/messages").json()["messages"] == []
    other = dict(_events(restarted.post("/chat", json=body)))
    assert original["done"]["turn_id"] != other["done"]["turn_id"]


def test_chat_admission_limits_turns_but_releases_capacity(store):
    started, finished = Event(), Event()
    router = ScriptedRouter(["order_agent", "FINISH", "FINISH"])

    def blocking_worker(state):
        started.set()
        assert finished.wait(timeout=5)
        return WORKERS["order_agent"](state)

    runner = build_multi_runner(router, store=store, workers={**WORKERS, "order_agent": blocking_worker})
    client = TestClient(create_app(runner, store, max_active_turns=1))
    first = {"session_id": "active", "message": "查订单"}
    second = {"session_id": "other", "message": "你好"}
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.post, "/chat", json=first, headers=U1)
        try:
            assert started.wait(timeout=5)
            assert client.post("/chat", json=first, headers=U1).status_code == 409
            assert client.post("/chat", json=second, headers=U1).status_code == 503
        finally:
            finished.set()
        assert dict(_events(result.result(timeout=5)))["done"]["final_action"] == "finish"
    assert dict(_events(client.post("/chat", json=second, headers=U1)))["done"]["final_action"] == "finish"
