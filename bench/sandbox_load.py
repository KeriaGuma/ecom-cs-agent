"""Measure concurrency and user-visible reply latency against the synthetic sandbox."""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx

PROMPT = "查一下订单 O1003 的物流"


def login(client: httpx.Client, username: str) -> None:
    client.post("/demo/login", json={"username": username, "password": "demo"}).raise_for_status()


def parse_event(lines: list[str]) -> tuple[str, dict] | None:
    event = next((line.removeprefix("event:").strip() for line in lines if line.startswith("event:")), None)
    data = "\n".join(line.removeprefix("data:").lstrip() for line in lines if line.startswith("data:"))
    return (event, json.loads(data)) if event and data else None


def request_chat(client: httpx.Client, session: str, message: str, request_id: str) -> dict:
    started = time.perf_counter()
    events: list[tuple[str, dict]] = []
    first_reply = None
    with client.stream("POST", "/chat", json={
        "session_id": session, "message": message, "request_id": request_id,
    }) as response:
        pieces: list[str] = []
        if response.is_success:
            for line in response.iter_lines():
                if line:
                    pieces.append(line)
                    continue
                if parsed := parse_event(pieces):
                    events.append(parsed)
                    if first_reply is None and parsed[0] in {"token", "message"} and parsed[1].get("text"):
                        first_reply = (time.perf_counter() - started) * 1000
                pieces.clear()
        status = response.status_code
    return {"status": status, "first_reply_ms": first_reply,
            "elapsed_ms": (time.perf_counter() - started) * 1000, "events": events}


def latency(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None}
    ordered = sorted(values)
    return {name: round(ordered[math.ceil(len(ordered) * percentile) - 1], 2)
            for name, percentile in (("p50", .5), ("p95", .95))}


def run(base_url: str, concurrency: int, requests: int, mode: str) -> dict:
    run_id = "load-" + uuid4().hex[:10]
    with httpx.Client(base_url=base_url, timeout=30) as client:
        login(client, "u001")

        def call(index: int) -> dict:
            session = run_id if mode == "hotspot" else f"{run_id}-{index}"
            return request_chat(client, session, PROMPT, f"{run_id}-{index}")

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(call, range(requests)))
        elapsed = time.perf_counter() - started

    completed = [row for row in rows if row["status"] == 200 and row["events"]
                 and row["events"][-1][0] == "done" and row["first_reply_ms"] is not None]
    conflicts = sum(row["status"] == 409 for row in rows)
    overloaded = sum(row["status"] == 503 for row in rows)
    handoffs = sum(row["events"][-1][1].get("final_action") == "handoff" for row in completed)
    return {
        "driver_platform": platform.platform(), "mode": mode, "requests": requests, "concurrency": concurrency,
        "completed": len(completed), "conflicts": conflicts, "overloaded": overloaded,
        "errors": requests - len(completed) - conflicts - overloaded, "handoffs": handoffs,
        "completed_per_second": round(len(completed) / elapsed, 2),
        "first_reply_ms": latency([row["first_reply_ms"] for row in completed]),
        "elapsed_ms": latency([row["elapsed_ms"] for row in completed]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--mode", choices=("isolated", "hotspot"), default="isolated")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.requests < 1 or not 1 <= args.concurrency <= 100:
        parser.error("请求数须为正数，并发须在 1 至 100 之间")
    report = run(args.base_url, args.concurrency, args.requests, args.mode)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)
    if (report["errors"] or report["handoffs"] or not report["completed"] or
            (args.mode == "isolated" and report["completed"] != args.requests)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
