"""Capacity-only counterfactuals in an independent, durable research ledger.

No execution manager, shared trade/order writes, AI calls or notifications.
One initial price-risk R is frozen before entry. Net R includes modeled costs
and estimated settled funding; it is not a jointly fundable portfolio return.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import math

from .models import Instrument, Opportunity
from .risk import assess
from .simulation import INTERVAL_MS, advance, apply_funding, create_trade
from .storage import NotLeader

VERSION = "capacity_observer_v1"
TABLE = "apex_shadow_observer_v1"
BATCH_SIZE = 20
MAX_PRICE_BARS = 500
MAX_FUNDING_SECONDS = 7 * 86400
RETRY_SECONDS = 30
CAPACITY_REASONS = frozenset(
    {
        "SYMBOL_ALREADY_EXPOSED",
        "POSITION_CAP",
        "BUCKET_POSITION_CAP",
        "HEAT_CAP",
        "BUCKET_HEAT_CAP",
        "GROSS_NOTIONAL_CAP",
        "ROUNDED_RISK_CAP",
        "ROUNDED_BUCKET_HEAT_CAP",
    }
)
LOSS_KEYS = ("daily_loss_pct", "weekly_loss_pct", "drawdown_pct")


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _next_price(trade):
    if trade.get("last_bar") is not None:
        return trade["last_bar"] + INTERVAL_MS
    return math.ceil(trade["created_at"] * 1000 / INTERVAL_MS) * INTERVAL_MS


def _funding_cursor(trade):
    cursor = trade["opened_at"]
    for start, end in sorted(trade.get("funding_coverage", [])):
        if start <= cursor <= end:
            cursor = end
    return cursor


def _complete(trade):
    return bool(
        trade.get("status") == "CLOSED"
        and trade.get("funding_complete") is True
        and trade.get("management_complete") is True
        and not any(trade.get(k) for k in ("data_error", "data_gap", "funding_error"))
        and _finite(trade.get("net_pnl"))
    )


class ShadowObserver:
    def __init__(self, store, client, cache):
        self.store, self.client, self.cache = store, client, cache
        self._resolve_lock = asyncio.Lock()

    async def initialize(self):
        def create(cur):
            self.store._assert_leader(cur)
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                id TEXT PRIMARY KEY, symbol TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, trade TEXT NOT NULL,
                created DOUBLE PRECISION NOT NULL, updated DOUBLE PRECISION NOT NULL,
                next_due DOUBLE PRECISION NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                done INTEGER NOT NULL DEFAULT 0, outcome_complete INTEGER NOT NULL DEFAULT 0,
                net_r DOUBLE PRECISION, error TEXT
            )"""
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_due ON {TABLE}(done,next_due,updated,id)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_symbol ON {TABLE}(symbol,status)"
            )

        await asyncio.to_thread(self.store._run, create)

    async def consider(
        self,
        op,
        inst,
        equity,
        profile,
        risk_pct,
        losses,
        funding_8h,
        spread,
        now,
        rejection,
    ):
        """Insert once only when removing exposures alone makes assessment pass."""
        if not isinstance(op, Opportunity) or not isinstance(inst, Instrument):
            return False
        if (
            not isinstance(op.evidence, dict)
            or op.evidence.get("entry_style", "confirmed") != "confirmed"
        ):
            return False
        if isinstance(rejection, dict):
            if rejection.get("allowed") is not False:
                return False
            reasons = rejection.get("reasons")
        else:
            reasons = rejection
        if (
            not isinstance(reasons, (list, tuple))
            or not reasons
            or any(
                not isinstance(reason, str) or reason not in CAPACITY_REASONS
                for reason in reasons
            )
            or not isinstance(losses, dict)
            or any(key not in losses for key in LOSS_KEYS)
        ):
            return False
        sizing = assess(
            op,
            inst,
            equity,
            profile=profile,
            risk_pct=risk_pct,
            exposures=[],
            funding_rate_8h=funding_8h,
            spread_pct=spread,
            now=now,
            **{key: losses[key] for key in LOSS_KEYS},
        )
        if sizing.get("allowed") is not True:
            return False
        r_price = sizing["qty"] * abs(sizing["entry"] - sizing["stop"])
        if not _finite(r_price) or r_price <= 0:
            return False
        trade = create_trade(op, sizing, "baseline_shadow", now)
        identity = VERSION + ":" + op.id
        trade.update(id=identity, observer=VERSION, original_price_r=r_price)
        # This full capture is immutable. Only the separate trade column advances.
        payload = {
            "experiment": VERSION,
            "opportunity": op.to_dict(),
            "instrument": asdict(inst),
            "captured_at": now,
            "rejection": deepcopy(rejection),
            "reasons": list(reasons),
            "equity_reference": equity,
            "profile": profile,
            "risk_pct": risk_pct,
            "losses": deepcopy(losses),
            "funding_8h": funding_8h,
            "spread_pct": spread,
            "independent_assessment": sizing,
            "original_price_r": r_price,
            "bypass": "existing portfolio exposures only",
        }
        immutable, initial = _json(payload), _json(trade)

        def insert(cur):
            self.store._assert_leader(cur)
            self.store._execute(
                cur,
                f"""INSERT INTO {TABLE}
                (id,symbol,status,payload,trade,created,updated,next_due)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING""",
                (identity, op.symbol, "PENDING", immutable, initial, now, now, now),
            )
            return cur.rowcount == 1

        return await asyncio.to_thread(self.store._run, insert)

    async def _claim(self, now):
        def claim(cur):
            self.store._assert_leader(cur)
            self.store._execute(
                cur,
                f"""SELECT id,symbol,trade,payload,revision FROM {TABLE}
                WHERE done=0 AND next_due<=%s ORDER BY next_due,updated,id LIMIT %s""",
                (now, BATCH_SIZE),
            )
            rows = cur.fetchall()
            result = []
            for identity, symbol, trade, payload, revision in rows:
                self.store._execute(
                    cur,
                    f"UPDATE {TABLE} SET next_due=%s,updated=%s WHERE id=%s",
                    (now + RETRY_SECONDS, now, identity),
                )
                result.append(
                    dict(
                        id=identity,
                        symbol=symbol,
                        trade=json.loads(trade),
                        payload=json.loads(payload),
                        revision=revision,
                    )
                )
            return result

        return await asyncio.to_thread(self.store._run, claim)

    async def _save(self, row, trade, now, error=None):
        complete = _complete(trade)
        price_r = row["payload"]["original_price_r"]
        net_r = trade["net_pnl"] / price_r if complete else None
        if complete and not _finite(net_r):
            complete, net_r, error = False, None, "Nonfinite normalized outcome"
        error = error or next(
            (
                str(trade[k])
                for k in ("data_error", "data_gap", "funding_error")
                if trade.get(k)
            ),
            None,
        )
        done = complete or (trade["status"] == "EXPIRED" and not error)
        serialized = _json(trade)

        def save(cur):
            self.store._assert_leader(cur)
            self.store._execute(
                cur,
                f"""UPDATE {TABLE} SET status=%s,trade=%s,updated=%s,
                next_due=%s,revision=revision+1,done=%s,outcome_complete=%s,net_r=%s,error=%s
                WHERE id=%s AND revision=%s""",
                (
                    trade["status"],
                    serialized,
                    now,
                    now + RETRY_SECONDS,
                    int(done),
                    int(complete),
                    net_r,
                    str(error)[:500] if error else None,
                    row["id"],
                    row["revision"],
                ),
            )
            return cur.rowcount == 1

        return await asyncio.to_thread(self.store._run, save)

    async def _resolve_symbol(self, symbol, rows, now):
        trades = {row["id"]: deepcopy(row["trade"]) for row in rows}
        errors = {}
        active = [t for t in trades.values() if t["status"] in {"PENDING", "OPEN"}]
        if active:
            first = min(_next_price(t) for t in active)
            end = min(
                int(now * 1000 // INTERVAL_MS) * INTERVAL_MS,
                first + MAX_PRICE_BARS * INTERVAL_MS,
            )
            if end > first:
                try:
                    bars = await self.client.candle_range(symbol, "3", first, end)
                    daily, execution = await asyncio.gather(
                        self.cache.candles(self.client, symbol, "D", 500, now),
                        self.cache.candles(self.client, symbol, "240", 500, now),
                    )
                    daily = await self.store.candle_history(symbol, "D", daily)
                    execution = await self.store.candle_history(
                        symbol, "240", execution
                    )
                    for identity, trade in list(trades.items()):
                        if (
                            trade["status"] not in {"PENDING", "OPEN"}
                            or _next_price(trade) >= end
                        ):
                            continue
                        trades[identity] = advance(
                            trade, bars, end / 1000, daily=daily, execution=execution
                        )
                except (asyncio.CancelledError, NotLeader):
                    raise
                except Exception as exc:
                    errors.update(
                        {t["id"]: "Price/structure: " + str(exc) for t in active}
                    )
        needing_funding = [
            t
            for t in trades.values()
            if t.get("opened_at") is not None and not t.get("funding_complete")
        ]
        if needing_funding:
            start = min(_funding_cursor(t) for t in needing_funding)
            horizons = [
                min(
                    now,
                    (t["last_bar"] + INTERVAL_MS) / 1000,
                    t["closed_at"] if t.get("closed_at") is not None else now,
                )
                for t in needing_funding
                if t.get("last_bar") is not None
            ]
            end = min(max(horizons), start + MAX_FUNDING_SECONDS) if horizons else start
            if end >= start:
                try:
                    # This client returns only after complete, validated pagination.
                    rates = await self.client.funding_history(
                        symbol, int(start * 1000), int(end * 1000)
                    )
                    for trade in needing_funding:
                        trades[trade["id"]] = apply_funding(
                            trade, rates, end, covered_from=start, history_complete=True
                        )
                except (asyncio.CancelledError, NotLeader):
                    raise
                except Exception as exc:
                    errors.update(
                        {t["id"]: "Funding: " + str(exc) for t in needing_funding}
                    )
        for row in rows:
            await self._save(row, trades[row["id"]], now, errors.get(row["id"]))

    async def resolve(self, now):
        """Fair persisted work queue: at most 20 candidates/500 bars per symbol."""
        if not _finite(now) or now < 0:
            raise ValueError("Invalid observer resolution time")
        async with self._resolve_lock:
            rows = await self._claim(now)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["symbol"]].append(row)
            for symbol, selected in grouped.items():
                try:
                    await self._resolve_symbol(symbol, selected, now)
                except (asyncio.CancelledError, NotLeader):
                    raise
                except Exception as exc:
                    for row in selected:
                        await self._save(
                            row, row["trade"], now, "Observer: " + str(exc)
                        )
            return {"processed": len(rows), "symbols": len(grouped)}

    async def summary(self):
        def read(cur):
            cur.execute(
                f"""SELECT COUNT(*),
                COALESCE(SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN status='OPEN' THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN status='EXPIRED' THEN 1 ELSE 0 END),0),
                COALESCE(SUM(outcome_complete),0),
                COALESCE(SUM(CASE WHEN outcome_complete=1 AND net_r>1e-9 THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN outcome_complete=1 AND net_r< -1e-9 THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN outcome_complete=1 THEN net_r ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN done=0 THEN 1 ELSE 0 END),0)
                FROM {TABLE}"""
            )
            return cur.fetchone()

        (
            total,
            pending,
            opened,
            closed,
            expired,
            complete,
            wins,
            losses,
            net_r,
            issues,
            unfinished,
        ) = await asyncio.to_thread(self.store._run, read)
        return {
            "experiment": VERSION,
            "total": total,
            "pending": pending,
            "open": opened,
            "closed": closed,
            "expired": expired,
            "complete_closed": complete,
            "wins": wins,
            "losses": losses,
            "breakeven": complete - wins - losses,
            "wr": 100 * wins / complete if complete else None,
            "net_r": net_r,
            "mean_r": net_r / complete if complete else None,
            "incomplete_closed": closed - complete,
            "data_issues": issues,
            "unfinished": unfinished,
            "basis": "Capacity-rejected confirmed candidates; net R after modeled costs and estimated funding. Independent opportunities, not portfolio returns.",
        }
