"""Namespaced transactional state and notification outbox.

Postgres is mandatory for deployed execution. SQLite is an explicit local-test
backend, never an automatic fallback on a Postgres outage. Legacy tables and the
Google ledger are never read, migrated or modified here.
"""

import asyncio
import copy
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class NotLeader(RuntimeError):
    pass


def initial_state():
    return {
        "settings": {"profile": "cautious", "risk_pct": None, "paused": False},
        "settings_version": 0,
        "confirmations": {},
        "opportunities": {},
        "orders": {},
        "trades": {},
        "reviews": {},
        "health": {},
        "universe": {},
        "telegram_offset": 0,
        "ai_budget": {},
        "references": {},
        "schema_version": 1,
    }


class Transaction:
    def __init__(self, state):
        self.state = state
        self.events = []

    def event(self, key, kind, payload, text=None):
        self.events.append((key, kind, payload, text))


class Store:
    def __init__(self, database_url="", sqlite_path=":memory:"):
        self.database_url = database_url
        self.owner = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.sqlite = None
        self.sqlite_path = sqlite_path
        self.pool = None

    def _connect(self):
        if self.database_url:
            if self.pool is None:
                from psycopg2.pool import ThreadedConnectionPool

                self.pool = ThreadedConnectionPool(
                    1,
                    4,
                    self.database_url,
                    connect_timeout=5,
                    options="-c statement_timeout=5000",
                )
            return self.pool.getconn()
        if self.sqlite is None:
            if self.sqlite_path != ":memory:":
                Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self.sqlite = sqlite3.connect(
                self.sqlite_path, check_same_thread=False, timeout=5
            )
        return self.sqlite

    def _execute(self, cursor, sql, params=()):
        cursor.execute(sql if self.database_url else sql.replace("%s", "?"), params)

    def _run(self, operation):
        with self.lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                if not self.database_url:
                    cur.execute("BEGIN IMMEDIATE")
                result = operation(cur)
                conn.commit()
                cur.close()
                return result
            except BaseException:
                conn.rollback()
                raise
            finally:
                if self.database_url:
                    self.pool.putconn(conn, close=bool(conn.closed))

    def _now(self, cur):
        if self.database_url:
            cur.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
            return float(cur.fetchone()[0])
        return time.time()

    async def initialize(self):
        def init(cur):
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apex_state (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apex_lease (id INTEGER PRIMARY KEY, owner TEXT NOT NULL, expires DOUBLE PRECISION NOT NULL)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apex_events (event_key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, created DOUBLE PRECISION NOT NULL)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apex_outbox (event_key TEXT PRIMARY KEY, text TEXT NOT NULL, created DOUBLE PRECISION NOT NULL, due DOUBLE PRECISION NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, delivered DOUBLE PRECISION, message_id TEXT)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apex_candles (symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time BIGINT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(symbol,interval,open_time))"
            )
            self._execute(
                cur,
                "INSERT INTO apex_state (id,value) VALUES (1,%s) ON CONFLICT(id) DO NOTHING",
                (json.dumps(initial_state()),),
            )

        await asyncio.to_thread(self._run, init)

    def _assert_leader(self, cur):
        self._execute(
            cur,
            "SELECT owner,expires FROM apex_lease WHERE id=1"
            + (" FOR UPDATE" if self.database_url else ""),
        )
        row = cur.fetchone()
        if not row or row[0] != self.owner or float(row[1]) <= self._now(cur):
            raise NotLeader("This process does not hold the cloud execution lease")

    async def lease(self, ttl=90):
        def acquire(cur):
            now = self._now(cur)
            self._execute(
                cur,
                "INSERT INTO apex_lease(id,owner,expires) VALUES(1,%s,%s) ON CONFLICT(id) DO NOTHING",
                (self.owner, now + ttl),
            )
            self._execute(
                cur,
                "UPDATE apex_lease SET owner=%s,expires=%s WHERE id=1 AND (owner=%s OR expires<=%s)",
                (self.owner, now + ttl, self.owner, now),
            )
            self._execute(cur, "SELECT owner,expires FROM apex_lease WHERE id=1")
            row = cur.fetchone()
            return row[0] == self.owner and row[1] > now

        return await asyncio.to_thread(self._run, acquire)

    async def assert_leader(self):
        await asyncio.to_thread(self._run, self._assert_leader)

    async def read(self):
        def read(cur):
            cur.execute("SELECT value FROM apex_state WHERE id=1")
            return json.loads(cur.fetchone()[0])

        return await asyncio.to_thread(self._run, read)

    async def candle_history(self, symbol, interval, bars):
        """Keep a fixed history origin; rolling API windows cannot reseed pivots.

        A revised closed candle is a material evidence change. Stop and surface
        it rather than silently rewriting the past and historical decisions.
        """
        from dataclasses import asdict
        from .models import Candle

        def record(cur):
            self._assert_leader(cur)
            self._execute(
                cur,
                "SELECT open_time,value FROM apex_candles WHERE symbol=%s AND interval=%s ORDER BY open_time",
                (symbol, interval),
            )
            history = {row[0]: json.loads(row[1]) for row in cur.fetchall()}
            inserts = []
            for bar in bars:
                value = asdict(bar)
                existing = history.get(bar.open_time)
                if existing is not None and existing != value:
                    raise ValueError(
                        "Previously stored closed candle was revised; source review required"
                    )
                if existing is None:
                    inserts.append(
                        (
                            symbol,
                            interval,
                            bar.open_time,
                            json.dumps(value, sort_keys=True, allow_nan=False),
                        )
                    )
                    history[bar.open_time] = value
            if inserts:
                for offset in range(0, len(inserts), 100):
                    batch = inserts[offset : offset + 100]
                    # Only fixed placeholder groups enter SQL; values remain
                    # bound parameters on both Postgres and local SQLite.
                    sql = (
                        "INSERT INTO apex_candles(symbol,interval,open_time,value) VALUES "
                        + ",".join(["(%s,%s,%s,%s)"] * len(batch))
                    )
                    self._execute(
                        cur, sql, tuple(value for row in batch for value in row)
                    )
            return [Candle(**history[key]) for key in sorted(history)]

        return await asyncio.to_thread(self._run, record)

    async def update(self, callback, require_leader=True):
        def change(cur):
            if require_leader:
                self._assert_leader(cur)
            cur.execute(
                "SELECT value FROM apex_state WHERE id=1"
                + (" FOR UPDATE" if self.database_url else "")
            )
            tx = Transaction(json.loads(cur.fetchone()[0]))
            result = callback(tx)
            self._execute(
                cur,
                "UPDATE apex_state SET value=%s WHERE id=1",
                (json.dumps(tx.state, allow_nan=False),),
            )
            now = self._now(cur)
            for key, kind, payload, text in tx.events:
                self._execute(
                    cur,
                    "INSERT INTO apex_events(event_key,kind,payload,created) VALUES(%s,%s,%s,%s) ON CONFLICT(event_key) DO NOTHING",
                    (key, kind, json.dumps(payload, allow_nan=False), now),
                )
                if text:
                    # Repeated processing of a semantic event cannot create a second notification.
                    self._execute(
                        cur,
                        "INSERT INTO apex_outbox(event_key,text,created,due) VALUES(%s,%s,%s,%s) ON CONFLICT(event_key) DO NOTHING",
                        (key, text, now, now),
                    )
            return copy.deepcopy(result)

        return await asyncio.to_thread(self._run, change)

    async def pending_notifications(self, limit=10):
        def pending(cur):
            self._assert_leader(cur)
            self._execute(
                cur,
                "SELECT event_key,text,created,attempts FROM apex_outbox WHERE delivered IS NULL AND due<=%s ORDER BY created,event_key LIMIT %s",
                (self._now(cur), limit),
            )
            return [
                {"key": r[0], "text": r[1], "created": r[2], "attempts": r[3]}
                for r in cur.fetchall()
            ]

        return await asyncio.to_thread(self._run, pending)

    async def notification_result(self, key, message_id=None, retry_after=30):
        def record(cur):
            self._assert_leader(cur)
            now = self._now(cur)
            if message_id is not None:
                self._execute(
                    cur,
                    "UPDATE apex_outbox SET delivered=%s,message_id=%s WHERE event_key=%s",
                    (now, str(message_id), key),
                )
            else:
                self._execute(
                    cur,
                    "UPDATE apex_outbox SET attempts=attempts+1,due=%s WHERE event_key=%s",
                    (now + max(1, retry_after), key),
                )

        await asyncio.to_thread(self._run, record)

    async def outbox_health(self):
        def health(cur):
            cur.execute(
                "SELECT COUNT(*),MIN(created) FROM apex_outbox WHERE delivered IS NULL"
            )
            count, oldest = cur.fetchone()
            return {
                "pending": count,
                "oldest_age": max(0, self._now(cur) - oldest) if oldest else 0,
            }

        return await asyncio.to_thread(self._run, health)

    async def ai_failure_diagnostic(self, archive_key):
        """Read only safe failure categories from an existing immutable AI archive.

        Older state records omit provider error codes. Never return the archived
        response, packet or arbitrary provider text to a presentation caller.
        """
        if not isinstance(archive_key, str) or not archive_key:
            return {}

        def read(cur):
            self._execute(
                cur,
                "SELECT payload FROM apex_events WHERE event_key=%s AND kind=%s",
                (archive_key, "ai_archive"),
            )
            row = cur.fetchone()
            if not row:
                return {}
            try:
                attempts = json.loads(row[0]).get("record", {}).get("attempts", [])
                failed = [a for a in attempts if a.get("valid") is False]
                if not failed:
                    return {}
                attempt = failed[-1]
                code = attempt.get("error_code")
                raw = attempt.get("raw_response")
                if not code and isinstance(raw, str):
                    error = json.loads(raw).get("error", {})
                    code = error.get("code") if isinstance(error, dict) else None
                result = {}
                if (
                    type(attempt.get("http_status")) is int
                    and 100 <= attempt["http_status"] <= 599
                ):
                    result["http_status"] = attempt["http_status"]
                if isinstance(code, str) and code in {
                    "billing_not_active",
                    "insufficient_quota",
                    "rate_limit_exceeded",
                    "invalid_api_key",
                    "model_not_found",
                }:
                    result["error_code"] = code
                return result
            except (ValueError, TypeError, AttributeError):
                return {}

        return await asyncio.to_thread(self._run, read)

    async def close(self):
        def release(cur):
            self._execute(
                cur,
                "UPDATE apex_lease SET expires=0 WHERE id=1 AND owner=%s",
                (self.owner,),
            )

        try:
            await asyncio.to_thread(self._run, release)
        finally:
            if self.pool:
                self.pool.closeall()
            if self.sqlite:
                self.sqlite.close()
