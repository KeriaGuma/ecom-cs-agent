"""端到端任务评测与单/多 Agent 架构对照：python -m cs_agent.eval_e2e --system single --split dev --repeat 3

每个任务在全新隔离数据库上运行多轮固定对话，结束时自动校验三类指标：
1. 数据库最终状态：目标订单及扩展领域表是否符合预期，其余记录一律不得被误改；
2. 关键路径：是否触发人工审批、是否转人工、必调与禁调工具列表、确认单生命周期状态；
3. 回复要点：关键词组命中或双异构模型评委引用复核通过。

支持 `--system single`（线上主架构单 Agent）、`--system multi`（`bench/multi_agent` 对照基线）或 `--system both`（逐题配对 Bootstrap 对比）。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Optional

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from bench.multi_agent import (
    KB_PROMPT,
    ORDER_PROMPT,
    REFUND_PROMPT,
    RESPOND_PROMPT,
    LLMRouter,
    build_multi_runner,
    build_supervisor_prompt,
)

from .agent import build_agent_prompt, build_agent_runner
from .claims import claims_done
from .config import ROOT, settings
from .eval_router import wilson
from .llm import get_llm
from .refunds import RefundService
from .store import ShopStore
from .tracing import RunTrace

EVAL_DIR = ROOT / "eval"
TEST_LOG = EVAL_DIR / "reports" / "e2e_test_runs.log"
RETIRED_TEST_SETS = {
    "93ff531b": "2026-11-09 正式对比后，根据 e2e-t32 的失败修复了记忆窗口（未摘要的消息被截掉）",
    "2e04a9d6": "2026-11-11 LLM 评委在其中发现审核驳回时回复前后矛盾，已作为历史参考归档",
    "71533284": "2026-11-09 领域扩展实验后，按知识库一致性检查加入“退款扣回积分”，部分任务的期望不再成立",
}
SYSTEMS = ("single", "multi")
CONFIRM_WORKFLOW = "confirm-v1"
TOOL_OUTPUT_MAX = 2000


def prompt_version(system: str, domains: tuple = ()) -> str:
    if system == "single":
        texts = [build_agent_prompt(domains)]
    else:
        texts = [build_supervisor_prompt(domains), ORDER_PROMPT, REFUND_PROMPT, KB_PROMPT, RESPOND_PROMPT]
        if domains:
            from .domains import SPECS

            texts += [SPECS[d]["prompt"] for d in domains]
    if settings.vision:
        from .vision import VISION_PROMPT

        texts.append(VISION_PROMPT)
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:8]


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_RETRIEVER = None


def build_system(system: str, store: ShopStore, llm, router_llm, cfg=settings, *, refund_service=None):
    global _RETRIEVER
    if _RETRIEVER is None:
        from .rag import build_retriever

        _RETRIEVER = build_retriever(cfg, llm)
    if system == "multi":
        return build_multi_runner(LLMRouter(router_llm, cfg.domain_list), llm=llm, store=store, settings=cfg,
                                  retriever=_RETRIEVER, refund_service=refund_service)
    if system == "single":
        return build_agent_runner(llm, store=store, settings=cfg, retriever=_RETRIEVER, refund_service=refund_service)
    raise ValueError(system)


def find_image(image_dir: Path, name: str) -> Path:
    exact = image_dir / name
    if exact.exists():
        return exact
    hits = sorted(image_dir.glob(Path(name).stem + ".*"))
    if not hits:
        raise FileNotFoundError(f"找不到图片 {image_dir / name}")
    return hits[0]


def split_turn(turn, image_dir: Optional[Path]) -> tuple[str, list[str]]:
    if isinstance(turn, str):
        return turn, []
    return turn["text"], [str(find_image(image_dir, n)) for n in turn.get("images", [])]


def apply_refund_actions(refunds: RefundService, thread_id: str, user_id: str, actions: list[dict],
                         *, auto_complete_return: bool = False) -> bool:
    """按顺序执行评测脚本中的退款确认、退货回填、人工审批与验货入库动作。"""
    approval = False
    explicit_return_step = any(a.get("decision") in {"return", "receive"} for a in actions)
    for action in actions:
        order_id, decision = action["order_id"], action["decision"]
        if decision in {"confirm", "cancel"}:
            status = "draft"
        elif decision in {"approve", "reject"}:
            status = "pending_approval"
        elif decision in {"return", "receive"}:
            status = "awaiting_return"
        else:
            raise ValueError(f"未知退款动作：{decision}")
        matching = [row for row in refunds.for_thread(thread_id, user_id)
                    if row["order_id"] == order_id and row["status"] == status]
        if len(matching) != 1:
            raise ValueError(f"订单 {order_id} 的退款动作 {decision} 需要唯一的 {status} 申请，找到 {len(matching)} 笔")
        refund_id = matching[0]["refund_id"]
        if decision in {"confirm", "cancel"}:
            res = refunds.confirm(refund_id, user_id, decision == "confirm")
            approval |= res["status"] == "pending_approval"
            if auto_complete_return and not explicit_return_step and res["status"] == "awaiting_return":
                res = refunds.confirm_return(refund_id, "eval", "SF000001")
        elif decision in {"approve", "reject"}:
            res = refunds.decide(refund_id, "eval", decision == "approve")
            approval = True
            if auto_complete_return and not explicit_return_step and res["status"] == "awaiting_return":
                res = refunds.confirm_return(refund_id, "eval", "SF000001")
        elif decision == "return":
            refunds.submit_return(refund_id, user_id, action.get("tracking_no", "SF000001"))
        elif decision == "receive":
            refunds.confirm_return(refund_id, "eval", action.get("tracking_no", ""))
    return approval


def _settle_legacy_pending_ops(store: ShopStore, user_id: str, approve: bool) -> bool:
    """兼容旧评测任务（使用 `submit_refund` 测试桩）：自动对本轮新增的 `pending` 操作记录完成审批结算。"""
    rows = store.query("SELECT op_id, order_id, detail FROM order_ops WHERE user_id=? AND op='refund' AND status='pending'",
                       (user_id,))
    if not rows:
        return False
    with store.transaction() as conn:
        for row in rows:
            detail = json.loads(row["detail"])
            status = "approved" if approve else "rejected"
            if approve:
                store.apply_refund(conn, row["order_id"], float(detail["amount"]))
            conn.execute("UPDATE order_ops SET status=?, decided_at=datetime('now') WHERE op_id=?", (status, row["op_id"]))
    return True


def run_task(system: str, task: dict, run: int, llm, router_llm, factory=build_system,
             domain_data: bool = False, tool_fault: float = 0.0, image_dir: Optional[Path] = None,
             refund_limit: float = settings.refund_auto_approve_limit) -> dict[str, Any]:
    user = task.get("user_id", "u001")
    workflow = task.get("refund_workflow", "legacy")
    if workflow not in {"legacy", CONFIRM_WORKFLOW}:
        raise ValueError(f"未知退款流程版本：{workflow}")
    thread_id = f"{task['id']}-{run}"
    expect = task.get("expect", {})
    expects_settled = any(r.get("status") == "succeeded" for r in expect.get("refunds", []))

    with tempfile.TemporaryDirectory() as tmp:
        seed = int(hashlib.sha256(f"{task['id']}|{run}".encode()).hexdigest()[:8], 16)
        store = (FlakyStore(str(Path(tmp) / "shop.db"), tool_fault, seed) if tool_fault
                 else ShopStore(str(Path(tmp) / "shop.db")))
        store.init(domain_data=domain_data)
        apply_setup(store, task.get("setup"))
        before = store.snapshot(user)
        refunds = RefundService(store, refund_limit) if workflow == CONFIRM_WORKFLOW else None
        runner = (factory(system, store, llm, router_llm, refund_service=refunds) if refunds is not None
                  else factory(system, store, llm, router_llm))
        trace = RunTrace()
        approve = task.get("approval", "approve") == "approve"

        seen = {"approval": False, "handoff": False, "degraded": False}
        latencies, replies, error = [], [], None
        turn_tools: list[list[dict]] = []
        turn_claims: list[list[str]] = []
        turn_photos: list[list[dict]] = []
        store.armed = True
        try:
            for i, turn in enumerate(task["turns"]):
                text, images = split_turn(turn, image_dir)
                t0 = time.perf_counter()
                n_tools, n_claims = len(trace.tools), len(trace.claim_notes)
                n_photos = len(runner.snapshot(thread_id).get("photos") or [])
                if refunds is not None:
                    refunds.refresh_user(user)
                message = HumanMessage(text, id=f"{thread_id}:{i}")
                state = runner.run_turn(thread_id, user, message, images=images, callbacks=[trace])
                if refunds is None and _settle_legacy_pending_ops(store, user, approve):
                    seen["approval"] = True
                if refunds is not None and isinstance(turn, dict):
                    seen["approval"] |= apply_refund_actions(
                        refunds, thread_id, user, turn.get("refund_actions", []),
                        auto_complete_return=expects_settled,
                    )
                latencies.append((time.perf_counter() - t0) * 1000)
                seen["handoff"] |= state.get("final_action") == "handoff"
                seen["degraded"] |= bool(state.get("degraded"))
                replies.append(next((str(m.content) for m in reversed(state["messages"])
                                     if isinstance(m, AIMessage)), ""))
                turn_tools.append([{"tool": t["tool"], "args": t["args"], "output": (t["output"] or "")[:TOOL_OUTPUT_MAX]}
                                   for t in trace.tools[n_tools:]])
                turn_claims.append(trace.claim_notes[n_claims:])
                photos = state.get("photos") or []
                turn_photos.append(photos[n_photos:] if len(photos) > n_photos else [])
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        store.armed = False
        after = store.snapshot(user)
        refund_rows = ([{"order_id": row["order_id"], "status": row["status"]}
                        for row in refunds.for_thread(thread_id, user)] if refunds is not None else [])
        if settings.order_tools == "mcp":
            from .mcp_client import close

            close(store.path)

    failures = [f"运行出错：{error}"] if error else check(expect, before, after, seen, trace.tool_names,
                                                          replies[-1] if replies else "", all_replies=replies)
    if "refunds" in expect:
        actual = sorted((row["order_id"], row["status"]) for row in refund_rows)
        expected = sorted((row["order_id"], row["status"]) for row in expect["refunds"])
        if actual != expected:
            failures.append(f"关键路径：退款单期望 {expected}，实际 {actual}")
    return {
        "id": task["id"], "system": system, "run": run, "tags": task.get("tags", []),
        "domains": task.get("domains", []),
        "success": not failures, "failures": failures, "severity": run_severity(failures, before),
        "claimed_done": claims_done(replies[-1]) if replies else False,
        "degraded": seen["degraded"], "tool_faults": getattr(store, "faults", 0),
        "latency_ms": latencies, "llm_calls": trace.llm_calls,
        "input_tokens": trace.input_tokens, "output_tokens": trace.output_tokens,
        "tools": trace.tool_names, "replies": replies, "turn_tools": turn_tools, "turn_claims": turn_claims,
        "turn_photos": turn_photos,
        **({"refund_workflow": workflow, "refunds": refund_rows} if refunds is not None else {}),
    }


# ---------------- 故障注入 ----------------
FAULT_METHODS = ("get_order", "list_orders", "get_logistics", "update_address", "mark_refunded")


class FlakyStore(ShopStore):
    def __init__(self, path: str, p: float, seed: int):
        super().__init__(path)
        self.p, self.rng, self.armed, self.faults = p, random.Random(seed), False, 0


def _flaky(name: str):
    def method(self, *args, **kwargs):
        if self.armed and self.rng.random() < self.p:
            self.faults += 1
            raise TimeoutError("后端服务超时，请稍后重试")
        return getattr(ShopStore, name)(self, *args, **kwargs)
    method.__name__ = name
    return method


for _name in FAULT_METHODS:
    setattr(FlakyStore, _name, _flaky(_name))


class FlakyTransport(httpx.BaseTransport):
    def __init__(self, p: float, seed: int):
        self.inner, self.p, self.rng, self.lock, self.faults = httpx.HTTPTransport(), p, random.Random(seed), threading.Lock(), 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            fail = self.rng.random() < self.p
            self.faults += fail
        if fail:
            raise httpx.ConnectTimeout("模拟的模型接口超时", request=request)
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


def flaky_llm(model: str, temperature: float, p: float, seed: int):
    from langchain_openai import ChatOpenAI

    transport = FlakyTransport(p, seed)
    llm = ChatOpenAI(model=model, api_key=settings.api_key, base_url=settings.base_url, temperature=temperature,
                     timeout=settings.llm_timeout, max_retries=settings.llm_max_retries,
                     extra_body=settings.extra_body or None, http_client=httpx.Client(transport=transport))
    return llm, transport


def apply_setup(store: ShopStore, setup: dict | None) -> None:
    if not setup:
        return
    for ev in setup.get("logistics_events", []):
        store.execute("INSERT INTO logistics (order_id, ts, event) VALUES (?, ?, ?)",
                      (ev["order_id"], ev.get("ts", "9999-12-31 23:59"), ev["event"]))
    for oid, item in setup.get("order_items", {}).items():
        store.execute("UPDATE orders SET item = ? WHERE order_id = ?", (item, oid))


PROMPT_CANARIES = (
    "order_agent", "refund_agent", "kb_agent", "single_agent", "FINISH", "handoff_human",
    "【关键事实】", "【较早对话摘要】", "【最近对话】", "【本轮已完成的处理】",
    "submit_refund", "prepare_refund", "check_refund", "search_policy", "list_my_orders", "change_address", "get_logistics",
)


# ---------------- 判定 ----------------
def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return actual is not None and math.isclose(float(actual), float(expected), abs_tol=0.01)
    if isinstance(expected, str) and isinstance(actual, str):
        return "".join(actual.split()) == "".join(expected.split())
    return actual == expected


ORDER_ID = re.compile(r"^O\d+$")
COLLECTIONS = ("invoices", "coupons", "repairs", "price_protect")


def check(expect: dict, before: dict, after: dict, seen: dict, tools: list[str], reply: str,
          all_replies: list[str] | None = None) -> list[str]:
    if "orders" not in before:
        before, after = {"orders": before}, {"orders": after}
    fails: list[str] = []
    want = expect.get("db", {})

    b_orders, a_orders = before.get("orders", {}), after.get("orders", {})
    for oid, fields in want.items():
        if not ORDER_ID.match(oid):
            continue
        for k, v in fields.items():
            got = (a_orders.get(oid) or {}).get(k)
            if not _same(got, v):
                fails.append(f"数据库：{oid}.{k} 期望 {v}，实际 {got}")
    for oid, row in b_orders.items():
        if oid not in want and a_orders.get(oid) != row:
            changed = {k: a_orders[oid][k] for k in row if a_orders.get(oid, {}).get(k) != row[k]}
            fails.append(f"数据库：{oid} 不应被修改，实际改动 {changed}")

    if "points" in want:
        for k, v in want["points"].items():
            if not _same(after.get("points", {}).get(k), v):
                fails.append(f"数据库：points.{k} 期望 {v}，实际 {after.get('points', {}).get(k)}")
    elif before.get("points") != after.get("points"):
        fails.append(f"数据库：积分不应变化，实际 {before.get('points')} → {after.get('points')}")

    for c in COLLECTIONS:
        w, b, a = want.get(c, {}), before.get(c, {}), after.get(c, {})
        for key, exp in w.items():
            got = a.get(key)
            if isinstance(exp, dict):
                for f, v in exp.items():
                    if not _same((got or {}).get(f), v):
                        fails.append(f"数据库：{c}.{key}.{f} 期望 {v}，实际 {(got or {}).get(f)}")
            elif not _same(got, exp):
                fails.append(f"数据库：{c}.{key} 期望 {exp}，实际 {got}")
        for key in set(b) | set(a):
            if key in w or (c == "coupons" and key.startswith("P") and key not in b):
                continue
            if b.get(key) != a.get(key):
                fails.append(f"数据库：{c}.{key} 不应变化，实际 {b.get(key)} → {a.get(key)}")

    for key, label in (("approval", "人工审批"), ("handoff", "转人工")):
        if key in expect and seen[key] != expect[key]:
            fails.append(f"关键路径：{'应' if expect[key] else '不应'}触发{label}")
    for t in expect.get("tools_called", []):
        if t not in tools:
            fails.append(f"关键路径：没有调用 {t}")
    for t in expect.get("tools_not_called", []):
        if t in tools:
            fails.append(f"关键路径：不应调用 {t}")
    for group in expect.get("reply", []):
        if not any(s in reply for s in group):
            fails.append(f"回复：缺少 {'/'.join(group)}")
    everything = "\n".join(all_replies if all_replies is not None else [reply])
    for s in expect.get("reply_forbidden", []):
        if s in everything:
            fails.append(f"回复：不应出现 {s}")
    if expect.get("no_prompt_leak"):
        leaked = [c for c in PROMPT_CANARIES if c in everything]
        if leaked:
            fails.append(f"回复：不应出现 内部提示词或实现细节 {'、'.join(leaked)}")
    return fails


WRITE_TOOLS = {
    "prepare_refund", "submit_refund", "change_address", "reissue_points", "exchange_points",
    "issue_invoice", "reissue_invoice", "return_coupon", "apply_repair", "apply_price_protect",
}
SEVERITY_ORDER = ("运行出错", "严重", "未办成", "流程", "沟通")
_DB_EXPECT = re.compile(r"^数据库：(\S+) 期望 (.*)，实际 (.*)$")


def _before_value(before: dict, target: str) -> Any:
    path, field = target.rsplit(".", 1)
    parts = path.split(".")
    node: Any = before.get("orders", {}) if ORDER_ID.match(parts[0]) else before
    for p in parts:
        node = (node or {}).get(p) if isinstance(node, dict) else None
    return node.get(field) if isinstance(node, dict) else None


def severity_of(failure: str, before: dict) -> str:
    if failure.startswith("运行出错"):
        return "运行出错"
    if failure.startswith("数据库："):
        m = _DB_EXPECT.match(failure)
        if m and "不应" not in failure:
            if "orders" not in before:
                before = {"orders": before}
            return "未办成" if str(_before_value(before, m.group(1))) == m.group(3) else "严重"
        return "严重"
    if failure.startswith("关键路径：不应调用"):
        return "严重" if failure.split()[-1] in WRITE_TOOLS else "流程"
    if failure.startswith("关键路径：应触发"):
        return "未办成"
    if failure.startswith("回复：不应出现"):
        return "严重"
    if failure.startswith("回复："):
        return "沟通"
    return "流程"


def run_severity(failures: list[str], before: dict) -> str | None:
    levels = {severity_of(f, before) for f in failures}
    return next((lv for lv in SEVERITY_ORDER if lv in levels), None)


def _pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] if s else 0.0


def summarize(results: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r["id"]].append(r)
    n_runs = len(results)
    ok_runs = sum(r["success"] for r in results)
    pass_all = sum(all(r["success"] for r in rs) for rs in by_task.values())
    tag_runs: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for t in r["tags"]:
            tag_runs[t].append(r["success"])
    turns = [ms for r in results for ms in r["latency_ms"]]
    lo, hi = wilson(ok_runs, n_runs)
    return {
        "tasks": len(by_task), "runs": n_runs,
        "success_rate": ok_runs / n_runs, "success_ci95": [lo, hi],
        "pass_k": pass_all / len(by_task),
        "by_tag": {t: {"n": len(v), "success": sum(v) / len(v)} for t, v in tag_runs.items()},
        "turn_p50_ms": _pct(turns, 0.5), "turn_p95_ms": _pct(turns, 0.95),
        "llm_calls_per_task": statistics.mean(r["llm_calls"] for r in results),
        "tokens_per_task": statistics.mean(r["input_tokens"] + r["output_tokens"] for r in results),
        "input_tokens_per_task": statistics.mean(r["input_tokens"] for r in results),
        "output_tokens_per_task": statistics.mean(r["output_tokens"] for r in results),
        "task_success": {tid: sum(r["success"] for r in rs) / len(rs) for tid, rs in by_task.items()},
        "severity": {lv: sum(r.get("severity") == lv for r in results) for lv in SEVERITY_ORDER},
        "critical_rate": sum(r.get("severity") == "严重" for r in results) / n_runs,
        "critical_ci95": list(wilson(sum(r.get("severity") == "严重" for r in results), n_runs)),
        "false_claims": sum(r.get("severity") == "未办成" and bool(r.get("claimed_done")) for r in results),
        "degraded_runs": sum(bool(r.get("degraded")) for r in results),
        "tool_faults": sum(r.get("tool_faults", 0) for r in results),
    }


def paired(a: dict, b: dict, iters: int = 2000, seed: int = 0) -> dict:
    ids = sorted(set(a["task_success"]) & set(b["task_success"]))
    diffs = [a["task_success"][i] - b["task_success"][i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(iters))
    return {
        "a_better": sum(d > 0 for d in diffs), "b_better": sum(d < 0 for d in diffs), "tie": sum(d == 0 for d in diffs),
        "mean_diff": statistics.mean(diffs), "ci95": [boots[int(0.025 * iters)], boots[int(0.975 * iters) - 1]],
        "diff_by_task": {i: d for i, d in zip(ids, diffs) if d},
    }


def print_summary(system: str, s: dict, k: int) -> None:
    lo, hi = s["success_ci95"]
    print(f"\n[{system}] 提示词 {s.get('prompt', prompt_version(system))}  领域 {s.get('domains') or '基础'}  干扰工具 {s.get('extra_tools', 0)} 个  任务 {s['tasks']} 个 × {k} 次")
    print(f"  任务成功率 {s['success_rate']:.1%}（95% CI {lo:.1%}～{hi:.1%}）   pass^{k} {s['pass_k']:.1%}")
    print(f"  每轮延迟 p50 {s['turn_p50_ms'] / 1000:.1f}s  p95 {s['turn_p95_ms'] / 1000:.1f}s   "
          f"每任务模型调用 {s['llm_calls_per_task']:.1f} 次  Token {s['tokens_per_task']:.0f}"
          f"（输入 {s['input_tokens_per_task']:.0f} / 输出 {s['output_tokens_per_task']:.0f}）")
    if "severity" in s:
        clo, chi = s["critical_ci95"]
        print(f"  严重失败率 {s['critical_rate']:.1%}（95% CI {clo:.1%}～{chi:.1%}）   失败分级："
              + "  ".join(f"{lv} {n}" for lv, n in s["severity"].items() if n)
              + f"   谎报 {s['false_claims']} 次  降级 {s['degraded_runs']} 次  注入工具故障 {s['tool_faults']} 次")
    print("  按 tag：" + "  ".join(f"{t} {v['success']:.0%}(n={v['n']})" for t, v in sorted(s["by_tag"].items())))


def rescore(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    befores: dict[tuple, dict] = {}
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in data["results"]:
        key = (r.get("user_id", "u001"), bool(r.get("domains")) or "domains" in path.name)
        if key not in befores:
            with tempfile.TemporaryDirectory() as tmp:
                st = ShopStore(str(Path(tmp) / "s.db"))
                st.init(domain_data=key[1])
                befores[key] = st.snapshot(key[0])
        by_system[r["system"]].append({**r, "severity": run_severity(r["failures"], befores[key])})
    out = {}
    for system, rows in by_system.items():
        n = len(rows)
        levels = {lv: sum(r["severity"] == lv for r in rows) for lv in SEVERITY_ORDER}
        lo, hi = wilson(levels["严重"], n)
        out[system] = {"runs": n, "success": sum(r["success"] for r in rows) / n, "severity": levels,
                       "critical_rate": levels["严重"] / n, "critical_ci95": [lo, hi]}
        print(f"{path.name} [{system}] {n} 次：成功 {out[system]['success']:.1%}  严重失败率 {levels['严重'] / n:.1%}"
              f"（95% CI {lo:.1%}～{hi:.1%}）  分级 " + "  ".join(f"{lv} {c}" for lv, c in levels.items() if c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS + ("both",), default="single")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tasks", help="指定任务文件，覆盖 --split")
    ap.add_argument("--only", help="只跑这些任务 id（逗号分隔），调试用")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--extra-tools", type=int, default=settings.extra_tools)
    ap.add_argument("--domains", default=settings.domains)
    ap.add_argument("--out", help="完整结果写成 JSON")
    ap.add_argument("--image-dir", help="带图任务的图片目录")
    ap.add_argument("--rescore", nargs="+", metavar="REPORT")
    ap.add_argument("--tool-fault", type=float, default=0.0)
    ap.add_argument("--llm-fault", type=float, default=0.0)
    ap.add_argument("--reply-judge", default="qwen,kimi")
    args = ap.parse_args()
    if args.rescore:
        for path in args.rescore:
            rescore(Path(path))
        return

    path = Path(args.tasks) if args.tasks else EVAL_DIR / f"e2e_{args.split}.jsonl"
    tasks = load_tasks(path)
    image_dir = Path(args.image_dir) if args.image_dir else EVAL_DIR / "images" / ("test" if "image_test" in path.name else "dev")
    missing = []
    for t in tasks:
        for turn in t["turns"]:
            for name in (turn.get("images", []) if isinstance(turn, dict) else []):
                try:
                    find_image(image_dir, name)
                except FileNotFoundError:
                    missing.append(name)
    if missing:
        raise SystemExit(f"{image_dir} 下缺少 {len(set(missing))} 张图片：{sorted(set(missing))}")
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    systems = list(SYSTEMS) if args.system == "both" else [args.system]

    from .domains import parse_domains

    domains = parse_domains(args.domains)
    domain_exp = "domains" in path.name
    if domain_exp:
        tasks = [t for t in tasks if set(t.get("domains", [])) <= set(domains) and not set(t.get("absent", [])) & set(domains)]
    dom_tag = f"\tdomains={args.domains}" if domains or domain_exp else ""
    if args.tool_fault or args.llm_fault:
        dom_tag += f"\tfaults=tool{args.tool_fault}/llm{args.llm_fault}"
    if not settings.claim_check:
        dom_tag += "\tclaim_check=0"
    if not settings.ops_log:
        dom_tag += "\tops_log=0"
    if not settings.address_guard:
        dom_tag += "\taddress_guard=0"
    if settings.claim_check and settings.claim_note_version != 2:
        dom_tag += f"\tclaim_note=v{settings.claim_note_version}"
    if not settings.unsupported_rule:
        dom_tag += "\tunsupported_rule=0"
    if settings.vision and any(isinstance(turn, dict) and turn.get("images") for t in tasks for turn in t["turns"]):
        dom_tag += f"\tvision={settings.vision_model}"
    if settings.order_tools != "local":
        if settings.order_tools == "mcp-http":
            raise SystemExit("评测每个任务用一个临时订单库，请用 ORDER_TOOLS=mcp")
        if args.tool_fault:
            raise SystemExit("工具故障注入在本进程的数据层里实现，MCP Server 子进程不受影响")
        dom_tag += f"\torder_tools={settings.order_tools}"
    pv = {s: prompt_version(s, domains) for s in systems}
    runtime_tag = "\torchestrator=turn_runner\tworker_runtime=python"

    is_test = (args.split.startswith("test") or args.split.endswith("test")) and not args.tasks and not args.only
    data = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    if is_test and data in RETIRED_TEST_SETS:
        raise SystemExit(f"测试集 {data} 已退役：{RETIRED_TEST_SETS[data]}。")
    if is_test and TEST_LOG.exists():
        log = TEST_LOG.read_text(encoding="utf-8")
        used = [s for s in systems
                if f"system={s}\tprompt={pv[s]}\tdata={data}\textra={args.extra_tools}{dom_tag}{runtime_tag}\tsuccess=" in log]
        if used:
            raise SystemExit(f"{used} 的当前提示词已经在这份测试集上跑过（见 {TEST_LOG}）。")

    transports = []
    if args.llm_fault:
        llm, t1 = flaky_llm(settings.agent_model, 0.2, args.llm_fault, seed=1)
        router_llm, t2 = flaky_llm(settings.router_model, 0, args.llm_fault, seed=2)
        transports = [t1, t2]
    else:
        llm = get_llm(settings.agent_model)
        router_llm = get_llm(settings.router_model, temperature=0)
    cfg = dataclasses.replace(settings, extra_tools=args.extra_tools, domains=args.domains,
                              domain_data=settings.domain_data or domain_exp)
    factory = partial(build_system, cfg=cfg)
    jobs = [(s, t, r) for t in tasks for r in range(args.repeat) for s in systems]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: run_task(j[0], j[1], j[2], llm, router_llm, factory=factory,
                                                   domain_data=cfg.domain_data, tool_fault=args.tool_fault,
                                                   image_dir=image_dir, refund_limit=cfg.refund_auto_approve_limit), jobs))
    if transports:
        print(f"注入的模型请求超时：{sum(t.faults for t in transports)} 次")
    reply_judge = None
    if args.reply_judge != "off":
        from .reply_judge import PROMPT_VERSION as RJ_VERSION, regrade

        judges = tuple(args.reply_judge.split(","))
        results = regrade(results, {t["id"]: t for t in tasks}, judges)
        rechecked = [r for r in results if r.get("reply_rechecks")]
        print(f"回复检查复核（评委 {'+'.join(judges)}，提示词 {RJ_VERSION}）：关键词没命中 {len(rechecked)} 次，"
              f"改判通过 {sum(r['success'] for r in rechecked)} 次")
        reply_judge = {"prompt_version": RJ_VERSION, "judges": list(judges)}

    summaries = {}
    for s in systems:
        rs = [r for r in results if r["system"] == s]
        summaries[s] = {**summarize(rs), "extra_tools": args.extra_tools, "domains": args.domains, "prompt": pv[s]}
        old = [r["success"] for r in rs if not r["domains"]]
        new = [r["success"] for r in rs if r["domains"]]
        summaries[s].update(success_old_domains=sum(old) / len(old) if old else None,
                            success_new_domains=sum(new) / len(new) if new else None)
        print_summary(s, summaries[s], args.repeat)

    if len(systems) == 2:
        p = paired(summaries["single"], summaries["multi"])
        print(f"\n配对对比（单 Agent − 多 Agent，按任务成功率）：单 Agent 更好 {p['a_better']} 个，"
              f"多 Agent 更好 {p['b_better']} 个，持平 {p['tie']} 个；"
              f"平均差 {p['mean_diff']:+.1%}（95% CI {p['ci95'][0]:+.1%}～{p['ci95'][1]:+.1%}）")
        summaries["paired"] = p

    print("\n失败明细：")
    for r in results:
        if not r["success"]:
            print(f"  [{r['system']}] {r['id']} #{r['run']}：{'；'.join(r['failures'])}")
            flag = "【疑似谎报，需人工复核】" if r.get("severity") == "未办成" and r.get("claimed_done") else ""
            print(f"      {flag}工具 {r['tools']}  回复：{r['replies'][-1][:120] if r['replies'] else ''!r}")

    if is_test:
        TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TEST_LOG.open("a", encoding="utf-8") as f:
            for s in systems:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tsystem={s}\tprompt={pv[s]}\tdata={data}"
                        f"\textra={args.extra_tools}{dom_tag}{runtime_tag}\tsuccess={summaries[s]['success_rate']:.4f}"
                        f"\tpass_k={summaries[s]['pass_k']:.4f}\truns={args.repeat}\n")
    if args.out:
        Path(args.out).write_text(json.dumps({"summaries": summaries, "results": results,
                                              "prompt_versions": pv, "reply_judge": reply_judge},
                                             ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
