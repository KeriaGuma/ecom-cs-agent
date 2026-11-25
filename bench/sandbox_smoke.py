"""Run a real HTTP refund, return verification, and handoff scenario against a disposable sandbox."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
CONTAINER_DB = "/app/var/demo.db"
REFUND_MESSAGE = "订单 O1002 的扫地机器人买错了，帮我申请退货退款"
DESK_MESSAGE = "我已接手本次售后，您可以在退款页面查看审批和退货进展。"
LEDGER_QUERY = """
import json, sqlite3, sys
from contextlib import closing

with closing(sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)) as conn:
    conn.row_factory = sqlite3.Row
    row = conn.execute('''
        SELECT r.status, r.return_tracking_no, o.status AS order_status,
          (SELECT COUNT(*) FROM simulated_refund_payments p WHERE p.refund_id=r.refund_id) AS ledger_rows,
          (SELECT COUNT(*) FROM refund_events e WHERE e.refund_id=r.refund_id
           AND e.actor='payment:recovery' AND e.status='succeeded') AS recovered_events,
          (SELECT COUNT(*) FROM refund_events e WHERE e.refund_id=r.refund_id
           AND e.actor='staff:a01' AND e.status='awaiting_return') AS approval_events
        FROM refund_requests r JOIN orders o USING(order_id) WHERE r.refund_id=?
    ''', (sys.argv[2],)).fetchone()
    print(json.dumps(dict(row) if row else None))
"""


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise RuntimeError(explanation)


def expect(response: httpx.Response, status: int = 200) -> httpx.Response:
    require(response.status_code == status,
            f"{response.request.method} {response.request.url.path}: expected {status}, "
            f"got {response.status_code} {response.text[:200]}")
    return response


def wait_ready(client: httpx.Client) -> None:
    deadline = monotonic() + 60
    while monotonic() < deadline:
        try:
            response = client.get("/ready", timeout=2)
        except httpx.RequestError:
            pass
        else:
            if response.status_code == 200:
                ready = response.json()
                require(ready.get("ok") is True and ready.get("runtime") == "python-async",
                        f"Sandbox readiness mismatch: {ready}")
                return
        sleep(0.25)
    raise RuntimeError("Sandbox did not become ready within 60 seconds")


def login(client: httpx.Client, username: str) -> None:
    expect(client.post("/demo/login", json={"username": username, "password": "demo"}))


def chat(client: httpx.Client, session: str, message: str, request_id: str) -> dict:
    response = expect(client.post("/chat", json={"session_id": session, "message": message, "request_id": request_id}))
    require(response.headers.get("content-type", "").startswith("text/event-stream"), "Expected a chat SSE response")
    events = {}
    for frame in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines())
        events[fields["event"]] = json.loads(fields["data"])
    require("error" not in events and "done" in events, f"Chat SSE ended unexpectedly: {events}")
    return events


def compose_command(project: str | None) -> Callable[..., str]:
    prefix = ["docker", "compose"]
    if project:
        prefix.extend(("--project-name", project))

    def run(*arguments: str) -> str:
        result = subprocess.run([*prefix, *arguments], cwd=ROOT, capture_output=True, text=True, timeout=90)
        require(result.returncode == 0, f"docker compose {' '.join(arguments[:2])} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    return run


def run_scenario(base_url: str, compose: Callable[..., str] | None = None) -> dict:
    session = f"smoke-{uuid4().hex[:12]}"
    history_path = f"/sessions/{session}/messages"
    stopped = False
    with ExitStack() as clients:
        customer, other, staff, approver = [clients.enter_context(httpx.Client(base_url=base_url, timeout=20))
                                           for _ in range(4)]
        wait_ready(customer)
        if compose:
            addressed = urlsplit(base_url)
            published = compose("port", "sandbox", "8000").rsplit(":", 1)[-1]
            require(addressed.hostname in ("localhost", "127.0.0.1", "::1") and str(addressed.port or 80) == published,
                    "--base-url must match the local port published by this disposable Compose sandbox")
        try:
            if compose and "reconciliation" in compose("ps", "--status", "running", "--services", "reconciliation").splitlines():
                stopped = True
                compose("stop", "reconciliation")

            expect(customer.get(history_path, headers={"X-User-Id": "u001"}), 401)
            for client, username in ((customer, "u001"), (other, "u002"), (staff, "s01"), (approver, "a01")):
                login(client, username)

            logistics = chat(customer, session, "查一下订单 O1003 的物流", "logistics")
            require("杭州转运中心" in logistics["message"]["text"], "The agent did not return the sample shipment")
            refund_turn = chat(customer, session, REFUND_MESSAGE, "refund")
            drafts = refund_turn["done"].get("refunds", [])
            require(len(drafts) == 1, "Expected a fresh demo database with the eligible O1002 refund draft")
            draft = drafts[0]
            require(
                draft["status"] == "draft" and draft["order_id"] == "O1002"
                and draft["amount"] == 1899 and draft["refund_type"] == "return_refund",
                "The refund draft does not match the sample order",
            )
            refund_id = draft["refund_id"]
            endpoint = f"/refunds/{refund_id}"
            expect(other.get(endpoint), 404)
            confirmed = expect(customer.post(f"{endpoint}/confirm", json={"confirmed": True})).json()
            require(confirmed["status"] == "pending_approval", "Large refund must wait for approval")

            handoff = chat(customer, session, "我要转人工", "handoff")
            ticket_id = handoff["done"]["ticket_id"]
            ticket_path = f"/tickets/{ticket_id}"
            claim = expect(staff.post(f"{ticket_path}/claim")).json()
            require(claim["assignee"] == "s01", "The assigned agent is missing from the ticket")
            customer_note = "好的，我等坐席回复。"
            served = chat(customer, session, customer_note, "human-note")
            require(served["done"]["final_action"] == "human" and "message" not in served,
                    "The automated agent replied during active human handoff")
            expect(staff.post(f"{ticket_path}/messages", json={"message": DESK_MESSAGE}))
            expect(staff.post(f"{ticket_path}/close"))
            history = expect(customer.get(history_path)).json()
            require({"role": "staff", "content": DESK_MESSAGE} in history["messages"], "The customer cannot see the staff reply")
            require(not expect(other.get(history_path)).json()["messages"], "Another customer saw a private transcript")

            fault_path = f"/demo{endpoint}/fault"
            decision_path = f"{endpoint}/decision"
            receive_path = f"{endpoint}/receive"
            expect(staff.post(fault_path, json={"fault": "ack_lost"}), 403)
            expect(staff.post(decision_path, json={"approved": True}), 403)
            expect(approver.post(fault_path, json={"fault": "ack_lost"}))
            approved = expect(approver.post(decision_path, json={"approved": True})).json()
            require(approved["status"] == "awaiting_return", "Delivered order must await return shipment before payout")

            shipped = expect(customer.post(f"{endpoint}/return", json={"tracking_no": "SF900123"})).json()
            require(shipped["return_tracking_no"] == "SF900123", "Return tracking number was not persisted")

            received = expect(approver.post(receive_path, json={})).json()
            require(received["status"] == "unknown", "The simulated payment should commit with a lost acknowledgement")

            if compose:
                compose("restart", "sandbox")
                wait_ready(customer)
                before = json.loads(compose("exec", "-T", "sandbox", "python", "-c", LEDGER_QUERY, CONTAINER_DB, refund_id))
                require(before == {"status": "unknown", "return_tracking_no": "SF900123", "order_status": "delivered",
                                   "ledger_rows": 1, "recovered_events": 0, "approval_events": 1},
                        f"Unexpected post-restart state: {before}")
                recovered = json.loads(compose("exec", "-T", "sandbox", "python", "-m", "cs_agent.refund_jobs", "--db", CONTAINER_DB))
                require(recovered.get("succeeded", 0) > 0, f"The one-shot worker did not recover the refund: {recovered}")

            settled = expect(customer.get(endpoint)).json()
            require(settled["status"] == "succeeded", "The simulated ledger outcome was not recovered")
            repeated = expect(approver.post(receive_path, json={})).json()
            require(repeated["status"] == "succeeded" and repeated["refund_id"] == refund_id, "Receive replay changed the refund")
            replay = chat(customer, session, REFUND_MESSAGE, "refund")
            require(replay["done"] == refund_turn["done"], "Chat retry did not replay the original turn receipt")
            require(expect(customer.get(history_path)).json() == history, "Transcript changed after restart or request replay")

            if compose:
                final = json.loads(compose("exec", "-T", "sandbox", "python", "-c", LEDGER_QUERY, CONTAINER_DB, refund_id))
                require(final == {"status": "succeeded", "return_tracking_no": "SF900123", "order_status": "refunded",
                                  "ledger_rows": 1, "recovered_events": 1, "approval_events": 1},
                        f"Unexpected recovered state: {final}")
            return {"session_id": session, "refund_id": refund_id, "ticket_id": ticket_id, "restarted": compose is not None}
        finally:
            if stopped:
                compose("start", "reconciliation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{os.getenv('DEMO_PORT', '8000')}")
    parser.add_argument("--compose", action="store_true", help="Restart a disposable Compose sandbox during the scenario")
    parser.add_argument("--compose-project", help="Select a named disposable Compose project")
    args = parser.parse_args()
    if args.compose_project and not args.compose:
        parser.error("--compose-project requires --compose")
    try:
        result = run_scenario(args.base_url, compose_command(args.compose_project) if args.compose else None)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        parser.exit(1, f"Sandbox smoke failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
