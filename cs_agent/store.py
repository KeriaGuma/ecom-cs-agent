"""模拟订单、物流、工单、会话与售后流水的持久化存储层（SQLite WAL）。

真实生产环境中对应订单微服务、物流中心与客服工单库；此处使用开启 WAL 与忙等待超时的 SQLite 实现事务级一致性，
订单领域工具同时支持通过 MCP Server（stdio / Streamable HTTP）解耦部署。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id     TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    item         TEXT NOT NULL,
    amount       REAL NOT NULL,
    status       TEXT NOT NULL,          -- paid / shipped / delivered / refunded
    created_at   TEXT NOT NULL,
    delivered_at TEXT,
    address      TEXT NOT NULL,
    refund_amount REAL
);
CREATE TABLE IF NOT EXISTS logistics (
    order_id TEXT NOT NULL,
    ts       TEXT NOT NULL,
    event    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (            -- 转人工工单（坐席优先级队列）
    ticket_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    priority   INTEGER NOT NULL,                -- 1 高 / 2 中 / 3 低，数字小的先处理
    status     TEXT NOT NULL DEFAULT 'open',    -- open / closed
    summary    TEXT NOT NULL,                   -- 交接摘要 JSON
    created_at TEXT NOT NULL,
    thread_id  TEXT,
    assignee   TEXT
);
CREATE TABLE IF NOT EXISTS ticket_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    message_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ticket_events_by_ticket ON ticket_events(ticket_id, event_id);
CREATE TABLE IF NOT EXISTS transcripts (        -- 完整脱敏对话存档（审计、质检、历史查询）
    thread_id  TEXT NOT NULL,
    message_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    PRIMARY KEY (thread_id, message_id)
);
CREATE TABLE IF NOT EXISTS conversation_state (
    thread_id TEXT PRIMARY KEY,
    state TEXT NOT NULL
);
-- ---------- 扩展业务域数据表（在 domain_data=True 时写入演示数据） ----------
CREATE TABLE IF NOT EXISTS points (user_id TEXT PRIMARY KEY, balance INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS points_credit (order_id TEXT PRIMARY KEY, points INTEGER NOT NULL, reissued INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS invoices (
    order_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, type TEXT NOT NULL, title TEXT NOT NULL,
    tax_id TEXT NOT NULL DEFAULT '', reissued INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS coupons (
    coupon_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, value REAL NOT NULL,
    status TEXT NOT NULL, used_order TEXT, expire TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS repairs (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, user_id TEXT NOT NULL,
    issue TEXT NOT NULL, cause TEXT NOT NULL, paid INTEGER NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS price_protect (order_id TEXT PRIMARY KEY, amount REAL NOT NULL);
CREATE TABLE IF NOT EXISTS products (item TEXT PRIMARY KEY, price REAL NOT NULL, electronic INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS order_flags (order_id TEXT PRIMARY KEY, flash_sale INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS order_ops (          -- 订单操作留痕：退款申请（含待审/驳回/待退货/打款状态）、修改地址
    op_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    op         TEXT NOT NULL,                   -- refund / address
    detail     TEXT NOT NULL,                   -- JSON：退款金额和原因、新地址等
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS refund_requests (
    refund_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    item TEXT NOT NULL,
    amount REAL NOT NULL,
    reason_type TEXT NOT NULL,
    refund_type TEXT NOT NULL DEFAULT 'refund_only',
    shipping_paid_by TEXT,
    needs_approval INTEGER NOT NULL,
    return_tracking_no TEXT,
    status TEXT NOT NULL,
    op_id INTEGER,
    staff_decision TEXT,
    payment_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(thread_id, message_id, order_id, reason_type)
);
CREATE INDEX IF NOT EXISTS refunds_by_thread ON refund_requests(thread_id, user_id, created_at);
CREATE INDEX IF NOT EXISTS refunds_by_user_status ON refund_requests(user_id, status);
DROP INDEX IF EXISTS active_refund_per_order;
CREATE UNIQUE INDEX IF NOT EXISTS active_refund_per_order ON refund_requests(order_id)
    WHERE status IN ('pending_approval', 'awaiting_return', 'ready', 'processing', 'unknown', 'succeeded');
CREATE TABLE IF NOT EXISTS refund_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    refund_id TEXT NOT NULL,
    status TEXT NOT NULL,
    actor TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS refund_events_by_request ON refund_events(refund_id, event_id);
CREATE TABLE IF NOT EXISTS simulated_refund_payments (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    amount REAL NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'succeeded',
    settles_at REAL
);
CREATE TABLE IF NOT EXISTS simulated_refund_faults (
    refund_id TEXT PRIMARY KEY,
    fault TEXT NOT NULL
);
"""


class ShopStore:
    def __init__(self, path: str):
        self.path = path
        if Path(path).exists():
            with closing(self._conn()) as conn, conn:
                self._ensure_schema(conn)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:" and not self.path.startswith("file::memory:"):
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
        return conn

    @contextmanager
    def transaction(self):
        with closing(self._conn()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            yield conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tickets)")}
        for column in ("thread_id", "assignee"):
            if column not in columns:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {column} TEXT")
        refund_columns = {row[1] for row in conn.execute("PRAGMA table_info(refund_requests)")}
        if "refund_type" not in refund_columns:
            conn.execute("ALTER TABLE refund_requests ADD COLUMN refund_type TEXT NOT NULL DEFAULT 'refund_only'")
        if "return_tracking_no" not in refund_columns:
            conn.execute("ALTER TABLE refund_requests ADD COLUMN return_tracking_no TEXT")
        payment_columns = {row[1] for row in conn.execute("PRAGMA table_info(simulated_refund_payments)")}
        if "outcome" not in payment_columns:
            conn.execute("ALTER TABLE simulated_refund_payments ADD COLUMN outcome TEXT NOT NULL DEFAULT 'succeeded'")
        if "settles_at" not in payment_columns:
            conn.execute("ALTER TABLE simulated_refund_payments ADD COLUMN settles_at REAL")

    # ---------- 初始化 ----------
    DOMAIN_TABLES = ("points", "points_credit", "invoices", "coupons", "repairs", "price_protect", "products", "order_flags")

    def init(self, today: Optional[date] = None, reset: bool = True, domain_data: bool = False) -> None:
        """建表并写入合成订单演示数据。日期按基准日往前推算，保证售后期限规则随时可演示。"""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        today = today or date.today()
        d = lambda n: (today - timedelta(days=n)).isoformat()  # noqa: E731

        with closing(self._conn()) as conn, conn:
            if reset:
                tables = (
                    "orders", "logistics", "tickets", "ticket_events", "transcripts", "conversation_state",
                    "order_ops", "refund_requests", "refund_events", "simulated_refund_payments",
                    "simulated_refund_faults", "chat_receipts", "pending_approvals", *self.DOMAIN_TABLES,
                )
                conn.executescript("".join(f"DROP TABLE IF EXISTS {table};" for table in tables))
            self._ensure_schema(conn)
            orders = [
                ("O1001", "u001", "蓝牙耳机", 199.0, "delivered", d(6), d(3), "杭州市西湖区文三路 1 号"),
                ("O1002", "u001", "扫地机器人", 1899.0, "delivered", d(8), d(5), "杭州市西湖区文三路 1 号"),
                ("O1003", "u001", "羽绒服", 699.0, "shipped", d(2), None, "杭州市西湖区文三路 1 号"),
                ("O1004", "u001", "保温杯", 89.0, "delivered", d(25), d(20), "杭州市西湖区文三路 1 号"),
                ("O1005", "u001", "机械键盘", 459.0, "paid", d(0), None, "杭州市西湖区文三路 1 号"),
                ("O2001", "u002", "手机", 3999.0, "delivered", d(4), d(2), "上海市浦东新区世纪大道 100 号"),
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO orders (order_id,user_id,item,amount,status,created_at,delivered_at,address)"
                " VALUES (?,?,?,?,?,?,?,?)",
                orders,
            )
            events = [
                ("O1001", d(5), "商家已发货，顺丰 SF123456"),
                ("O1001", d(4), "到达杭州转运中心"),
                ("O1001", d(3), "已签收"),
                ("O1003", d(1), "商家已发货，中通 ZT987654"),
                ("O1003", d(0), "到达杭州转运中心，预计明天送达"),
            ]
            conn.executemany("INSERT INTO logistics (order_id,ts,event) VALUES (?,?,?)", events)
            if domain_data:
                self._seed_domains(conn, d, today)

    def _seed_domains(self, conn, d, today: date) -> None:
        later = lambda n: (today + timedelta(days=n)).isoformat()  # noqa: E731
        conn.executemany(
            "INSERT OR REPLACE INTO orders (order_id,user_id,item,amount,status,created_at,delivered_at,address)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [("O1006", "u001", "智能手表", 599.0, "delivered", d(45), d(40), "杭州市西湖区文三路 1 号"),
             ("O1007", "u001", "吹风机", 299.0, "delivered", d(4), d(2), "杭州市西湖区文三路 1 号")],
        )
        conn.execute("INSERT OR REPLACE INTO points VALUES ('u001', 1260)")
        conn.executemany("INSERT OR REPLACE INTO points_credit (order_id, points) VALUES (?,?)",
                         [("O1001", 199), ("O1004", 89), ("O1006", 599), ("O1007", 299)])
        conn.executemany("INSERT OR REPLACE INTO coupons VALUES (?,?,?,?,?,?,?)", [
            ("C01", "u001", "满 199 减 20", 20.0, "used", "O1001", later(30)),
            ("C02", "u001", "满 89 减 10", 10.0, "used", "O1004", d(5)),
            ("C03", "u001", "无门槛 5 元", 5.0, "unused", None, later(30)),
        ])
        conn.executemany("INSERT OR REPLACE INTO products VALUES (?,?,?)", [
            ("蓝牙耳机", 169.0, 1), ("扫地机器人", 1899.0, 1), ("羽绒服", 699.0, 0), ("保温杯", 89.0, 0),
            ("机械键盘", 459.0, 1), ("智能手表", 499.0, 1), ("吹风机", 249.0, 1), ("手机", 3999.0, 1),
        ])
        conn.execute("INSERT OR REPLACE INTO order_flags VALUES ('O1007', 1)")

    def query(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

    def execute(self, sql: str, args: tuple = ()) -> int:
        with closing(self._conn()) as conn, conn:
            return conn.execute(sql, args).rowcount

    def snapshot(self, user_id: str) -> dict[str, Any]:
        """评测用的完整业务库状态快照。"""
        bal = self.query("SELECT balance FROM points WHERE user_id = ?", (user_id,))
        repairs: dict[str, dict] = {}
        for r in self.query("SELECT * FROM repairs ORDER BY ticket_id"):
            repairs[r["order_id"]] = {"status": r["status"], "paid": bool(r["paid"])}
        return {
            "orders": self.all_orders(),
            "points": {"balance": bal[0]["balance"]} if bal else {},
            "invoices": {r["order_id"]: {"type": r["type"], "title": r["title"], "tax_id": r["tax_id"],
                                          "reissued": bool(r["reissued"])} for r in self.query("SELECT * FROM invoices")},
            "coupons": {r["coupon_id"]: {"status": r["status"], "used_order": r["used_order"]}
                        for r in self.query("SELECT * FROM coupons")},
            "repairs": repairs,
            "price_protect": {r["order_id"]: r["amount"] for r in self.query("SELECT * FROM price_protect")},
        }

    # ---------- 查询 ----------
    def get_order(self, order_id: str) -> Optional[dict[str, Any]]:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return dict(row) if row else None

    def list_orders(self, user_id: str) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT order_id,item,amount,status,created_at FROM orders WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_logistics(self, order_id: str) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT ts,event FROM logistics WHERE order_id = ? ORDER BY ts", (order_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def all_orders(self) -> dict[str, dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT * FROM orders").fetchall()
        return {r["order_id"]: dict(r) for r in rows}

    # ---------- 写操作 ----------
    def update_address(self, order_id: str, address: str) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute("UPDATE orders SET address = ? WHERE order_id = ?", (address, order_id))

    def mark_refunded(self, order_id: str, amount: float, *, operation: Optional[tuple[str, dict]] = None) -> bool:
        """幂等标记订单已退款，并同步扣回该订单已发放积分。"""
        with self.transaction() as conn:
            updated = self.apply_refund(conn, order_id, amount)
            if updated and operation is not None:
                user_id, detail = operation
                self.insert_op(conn, order_id, user_id, "refund", detail, "done")
            return updated

    @staticmethod
    def apply_refund(conn: sqlite3.Connection, order_id: str, amount: float) -> bool:
        count = conn.execute(
            "UPDATE orders SET status = 'refunded', refund_amount = ? WHERE order_id = ? AND status != 'refunded'",
            (amount, order_id),
        ).rowcount
        if count == 1:
            credit = conn.execute("SELECT points FROM points_credit WHERE order_id = ?", (order_id,)).fetchone()
            if credit:
                conn.execute("UPDATE points SET balance = MAX(0, balance - ?) WHERE user_id ="
                             " (SELECT user_id FROM orders WHERE order_id = ?)", (credit["points"], order_id))
                conn.execute("DELETE FROM points_credit WHERE order_id = ?", (order_id,))
        return count == 1

    # ---------- 订单操作留痕 ----------
    def add_op(self, order_id: str, user_id: str, op: str, detail: dict, status: str) -> int:
        with closing(self._conn()) as conn, conn:
            return self.insert_op(conn, order_id, user_id, op, detail, status)

    @staticmethod
    def insert_op(conn: sqlite3.Connection, order_id: str, user_id: str, op: str, detail: dict, status: str) -> int:
        cur = conn.execute("INSERT INTO order_ops (order_id,user_id,op,detail,status,created_at) VALUES (?,?,?,?,?,?)",
                           (order_id, user_id, op, to_json(detail), status, _now()))
        return int(cur.lastrowid)

    def list_ops(self, order_id: str) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT * FROM order_ops WHERE order_id = ? ORDER BY op_id", (order_id,)).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    # ---------- 转人工工单 ----------
    def create_ticket(self, user_id: str, priority: int, summary: dict, created_at: str, *,
                      thread_id: Optional[str] = None) -> int:
        with closing(self._conn()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO tickets (user_id, priority, summary, created_at, thread_id) VALUES (?,?,?,?,?)",
                (user_id, priority, to_json(summary), created_at, thread_id),
            )
            ticket_id = int(cur.lastrowid)
            self._ticket_event(conn, ticket_id, f"user:{user_id}", "opened")
        return ticket_id

    def get_ticket(self, ticket_id: int) -> Optional[dict]:
        rows = self.query("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,))
        return {**rows[0], "summary": json.loads(rows[0]["summary"])} if rows else None

    def claim_ticket(self, ticket_id: int, staff: str) -> bool:
        with self.transaction() as conn:
            row = conn.execute("SELECT assignee FROM tickets WHERE ticket_id=? AND status='open'", (ticket_id,)).fetchone()
            if row is None or row["assignee"] not in (None, staff):
                return False
            if row["assignee"] is None:
                conn.execute("UPDATE tickets SET assignee=? WHERE ticket_id=?", (staff, ticket_id))
                self._ticket_event(conn, ticket_id, f"staff:{staff}", "claimed")
            return True

    def queue_position(self, ticket_id: int) -> int:
        """排在这张工单前面的未处理工单数：优先级更高，或同优先级但更早创建。"""
        with closing(self._conn()) as conn:
            t = conn.execute("SELECT priority FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            if not t:
                return 0
            row = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE status = 'open' AND ticket_id != ?"
                " AND (priority < ? OR (priority = ? AND ticket_id < ?))",
                (ticket_id, t["priority"], t["priority"], ticket_id),
            ).fetchone()
        return int(row[0])

    def open_tickets(self) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status = 'open' ORDER BY priority, ticket_id"
            ).fetchall()
        return [{**dict(r), "summary": json.loads(r["summary"])} for r in rows]

    def close_ticket(self, ticket_id: int, *, staff: Optional[str] = None) -> bool:
        with closing(self._conn()) as conn, conn:
            cur = conn.execute("UPDATE tickets SET status = 'closed' WHERE ticket_id = ? AND status = 'open'"
                               " AND (? IS NULL OR assignee IS NULL OR assignee=?)", (ticket_id, staff, staff))
            if cur.rowcount == 1:
                self._ticket_event(conn, ticket_id, f"staff:{staff}" if staff else "system", "closed")
        return cur.rowcount == 1

    @staticmethod
    def _ticket_event(conn: sqlite3.Connection, ticket_id: int, actor: str, action: str, message_id: str | None = None) -> None:
        conn.execute("INSERT INTO ticket_events (ticket_id,actor,action,message_id,created_at) VALUES (?,?,?,?,?)",
                     (ticket_id, actor, action, message_id, _now()))

    def ticket_events(self, ticket_id: int) -> list[dict]:
        return self.query("SELECT actor,action,message_id,created_at FROM ticket_events WHERE ticket_id=? ORDER BY event_id",
                          (ticket_id,))

    # ---------- 会话状态与脱敏存档 ----------
    def load_session(self, thread_id: str) -> dict:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT state FROM conversation_state WHERE thread_id = ?", (thread_id,)).fetchone()
        return json.loads(row["state"]) if row else {}

    def save_session(self, thread_id: str, state: dict, messages: list[tuple[str, str, str]], *,
                     staff_reply: tuple[int, str, str] | None = None) -> None:
        """原子提交模型上下文与脱敏对话档案。"""
        with self.transaction() as conn:
            conn.execute("INSERT INTO conversation_state (thread_id, state) VALUES (?, ?) "
                         "ON CONFLICT(thread_id) DO UPDATE SET state=excluded.state", (thread_id, to_json(state)))
            self._archive_messages(conn, thread_id, messages)
            if staff_reply:
                ticket_id, staff, message_id = staff_reply
                self._ticket_event(conn, ticket_id, f"staff:{staff}", "replied", message_id)

    def archive_messages(self, thread_id: str, messages: list[tuple[str, str, str]]) -> None:
        with self.transaction() as conn:
            self._archive_messages(conn, thread_id, messages)

    @staticmethod
    def _archive_messages(conn: sqlite3.Connection, thread_id: str, messages: list[tuple[str, str, str]]) -> None:
        start = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM transcripts WHERE thread_id = ?", (thread_id,)).fetchone()[0]
        rows = [(thread_id, mid, start + i, role, content) for i, (mid, role, content) in enumerate(messages, 1)]
        conn.executemany("INSERT OR IGNORE INTO transcripts VALUES (?,?,?,?,?)", rows)

    def transcript(self, thread_id: str, *, include_ids: bool = False) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            columns = "message_id, role, content" if include_ids else "role, content"
            rows = conn.execute(f"SELECT {columns} FROM transcripts WHERE thread_id = ? ORDER BY seq",
                                (thread_id,)).fetchall()
        return [dict(r) for r in rows]


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
