"""Offline runtime integration; real engine/risk/Store/AI policy/simulation/service.

Only external exchange, AI HTTP, calendar, and Telegram boundaries are replaced.
No candidate backtest metrics, network access, production credentials or databases.
Regression failures here deliberately identify parent-owned runtime/storage defects.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib
import json
import math
from types import SimpleNamespace

import aiohttp
import pytest

from apex_bot import ai, cache, execution, service, storage
from apex_bot.config import Config
from apex_bot.evidence import context_fingerprint
from apex_bot.models import Candle, Instrument, Opportunity
from apex_bot.storage import NotLeader, Store, initial_state
from apex_bot.telegram import TelegramError
from .test_ai import FakeSession as AISession, review_response
from .test_engine import DAY, H4, daily_history, execution_history


SHIFT = 20_000 * DAY
DAILY = [replace(c, open_time=c.open_time + SHIFT * 1000) for c in daily_history()]
EXECUTION = [
    replace(c, open_time=c.open_time + SHIFT * 1000) for c in execution_history()
]
NOW = EXECUTION[-1].open_time / 1000 + H4
INSTRUMENT = Instrument("BTCUSDT", 0.01, 0.01, 10_000, 0.01, 5, 480, 5)


class OfflineClient:
    def __init__(
        self, session, base_url="https://api.bybit.com", api_key="", api_secret=""
    ):
        self.market, self.base_url = session.market, base_url
        self.calls = []

    async def instruments(self):
        return {"BTCUSDT": INSTRUMENT}

    async def candles(self, symbol, interval="D", limit=500, now_ms=None):
        self.calls.append(("candles", symbol, interval, limit, now_ms))
        bars = (
            self.market.daily
            if interval == "D"
            else self.market.execution if interval == "240" else self.market.intraday
        )
        return deepcopy(bars[-limit:])

    async def candle_range(self, symbol, interval, start_ms, end_ms):
        self.calls.append(("candle_range", symbol, interval, start_ms, end_ms))
        assert interval == "3"
        return deepcopy(
            [
                c
                for c in self.market.intraday
                if start_ms <= c.open_time and c.open_time + 180_000 <= end_ms
            ]
        )

    async def ticker(self, symbol):
        return {"symbol": symbol, **self.market.ticker}

    async def positions(self):
        self.calls.append(("positions",))
        return deepcopy(self.market.positions)

    async def open_orders(self):
        self.calls.append(("open_orders",))
        return deepcopy(list(self.market.orders.values()))

    async def equity(self):
        self.calls.append(("equity",))
        return self.market.equity

    async def account_info(self):
        self.calls.append(("account_info",))
        return {"marginMode": self.market.margin_mode, "unifiedMarginStatus": 6}

    async def position_info(self, symbol):
        return deepcopy(self.market.settings)

    async def risk_limits(self, symbol):
        return deepcopy(self.market.tiers)

    async def submit_limit(self, symbol, side, qty, price, stop, target, order_link_id):
        assert (
            order_link_id not in self.market.orders
        ), "An existing intent must not be resubmitted"
        self.market.submissions.append(
            (symbol, side, qty, price, stop, target, order_link_id)
        )
        order = {
            "symbol": symbol,
            "side": side,
            "orderId": "offline-" + order_link_id,
            "orderLinkId": order_link_id,
            "orderStatus": "New",
            "cumExecQty": "0",
            "reduceOnly": False,
            "positionIdx": 0,
            "leavesQty": str(qty),
            "qty": str(qty),
            "price": str(price),
            "orderType": "Limit",
            "timeInForce": "GTC",
            "tpslMode": "Full",
            "stopLoss": str(stop),
            "takeProfit": str(target),
        }
        self.market.orders[order_link_id] = order
        return {"orderId": order["orderId"], "orderLinkId": order_link_id}

    async def order(self, symbol, order_link_id):
        return deepcopy(self.market.orders.get(order_link_id))

    async def executions(self, symbol, start_ms=None):
        return []

    async def funding_history(self, symbol, start_ms=None, end_ms=None):
        self.calls.append(("funding_history", symbol, start_ms, end_ms))
        return deepcopy(
            [
                r
                for r in self.market.funding
                if start_ms <= int(r["fundingRateTimestamp"]) <= end_ms
            ]
        )


class OfflineCalendar:
    def __init__(self, session):
        self.market = session.market

    async def build(self, btc, now):
        assert btc == self.market.daily
        return {
            "as_of": now,
            "expires_at": now + 21_600,
            "data_complete": True,
            "event_blackout": self.market.blackout,
            "risk_state": "NORMAL",
            "long_multiplier": 1.0,
            "short_multiplier": 1.0,
            "reason": "Offline official-calendar fixture; no scheduled event",
            "sources": [],
        }


class OfflineTelegram:
    def __init__(
        self, session, token, chat_id, allowed_user_ids, service, *, bot_username=None
    ):
        self.service = service
        self.bot_username = bot_username
        self.sent, self.offsets = [], []
        self.failures = []

    async def send(self, text):
        if self.failures:
            raise self.failures.pop(0)
        self.sent.append(text)
        return str(len(self.sent))

    async def poll_once(self, offset):
        self.offsets.append(offset)
        await self.send(await self.service.snapshot("dashboard"))
        return offset + 1


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Construct the real Runtime in a running loop with a fresh leased file Store."""

    async def forbidden_network(*args, **kwargs):
        raise AssertionError("Runtime tests must not contact a network service")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden_network)

    @asynccontextmanager
    async def make(mode="shadow", *, ai_actions=None):
        # Do not stub a missing production module: an unfinished context integration
        # must be visible to the parent instead of being disguised by a green test.
        runtime = importlib.import_module("apex_bot.runtime")
        clock = SimpleNamespace(now=NOW)
        for module in (runtime, ai, cache, execution, service, storage):
            original = module.time
            monkeypatch.setattr(
                module,
                "time",
                SimpleNamespace(
                    time=lambda: clock.now,
                    monotonic=getattr(original, "monotonic", lambda: clock.now),
                ),
            )

        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.fromtimestamp(clock.now, tz or timezone.utc)

        monkeypatch.setattr(runtime, "datetime", FrozenDateTime)
        monkeypatch.setattr(runtime, "BybitClient", OfflineClient)
        monkeypatch.setattr(runtime, "ContextService", OfflineCalendar)
        monkeypatch.setattr(runtime, "TelegramController", OfflineTelegram)
        session = AISession(actions=ai_actions)
        session.market = SimpleNamespace(
            daily=deepcopy(DAILY),
            execution=deepcopy(EXECUTION),
            intraday=[],
            ticker={"ask1Price": "142.01", "bid1Price": "141.99", "fundingRate": "0"},
            positions=[],
            orders={},
            submissions=[],
            equity=10_000.0,
            margin_mode="ISOLATED_MARGIN",
            settings=[
                {
                    "symbol": "BTCUSDT",
                    "size": "0",
                    "positionIdx": 0,
                    "leverage": "3",
                    "riskId": 1,
                }
            ],
            tiers=[
                {
                    "symbol": "BTCUSDT",
                    "id": 1,
                    "riskLimitValue": "1000000",
                    "maintenanceMargin": ".005",
                    "initialMargin": ".01",
                    "mmDeduction": "0",
                    "maxLeverage": "5",
                }
            ],
            blackout=False,
            funding=[],
        )
        config = Config(
            mode=mode,
            universe_mode="static",
            sqlite_path=str(tmp_path / "runtime.sqlite"),
            symbols=("BTCUSDT",),
            bybit_url="https://api-testnet.bybit.com",
            telegram_token="1:offline",
            telegram_chat_id="-100987654321",
            telegram_username="OfflineApexFixtureBot",
            telegram_user_ids=frozenset({987654321}),
            openai_key="offline-fixture-key",
            openai_model="offline-model",
        )
        h = SimpleNamespace(
            config=config,
            clock=clock,
            session=session,
            market=session.market,
            store=Store(sqlite_path=config.sqlite_path),
            runtime=None,
        )

        async def start():
            h.runtime = runtime.Runtime(config, h.store, session)
            await h.runtime.initialize()  # Real initialize + lease + testnet reconciliation.
            assert await h.store.lease(ttl=100_000)

        async def restart():
            await h.runtime.cache.close()
            await h.store.close()
            h.store = Store(sqlite_path=config.sqlite_path)
            await start()

        async def tick(seconds):
            clock.now += seconds
            assert await h.store.lease(ttl=100_000)

        h.restart, h.tick = restart, tick
        await start()
        try:
            await h.runtime.refresh_context()
            yield h
        finally:
            await h.runtime.cache.close()
            await h.store.close()

    return make


async def reviews(runtime):
    worker = asyncio.create_task(runtime.review_worker())
    try:
        await asyncio.wait_for(runtime.ai_queue.join(), timeout=5)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


def ready(state):
    return next(
        rec
        for rec in state["opportunities"].values()
        if rec["opportunity"]["state"] == "READY"
    )


async def events(store, kind=None):
    def read(cur):
        if kind is None:
            cur.execute(
                "SELECT event_key,kind,payload FROM apex_events ORDER BY event_key"
            )
        else:
            cur.execute(
                "SELECT event_key,kind,payload FROM apex_events WHERE kind=? ORDER BY event_key",
                (kind,),
            )
        return [
            (key, kind, json.loads(payload)) for key, kind, payload in cur.fetchall()
        ]

    return await asyncio.to_thread(store._run, read)


def test_real_scan_blind_ai_pair_simulation_funding_dashboard_and_restart(harness):
    async def scenario():
        async with harness() as h:
            assert h.runtime.started and h.runtime.execution.mode == "shadow"
            assert h.runtime.telegram.bot_username == "OfflineApexFixtureBot"
            await h.runtime.scan()
            state = await h.store.read()
            assert {t["arm"] for t in state["trades"].values()} == {"baseline_shadow"}
            assert ready(state)["opportunity"]["reason"] == "SWING_BREAK"
            await reviews(h.runtime)
            state = await h.store.read()
            assert {t["arm"] for t in state["trades"].values()} == {
                "baseline_shadow",
                "ai_shadow",
            }
            assert all(t["status"] == "PENDING" for t in state["trades"].values())
            assert ready(state)["ai"]["verdict"] == "APPROVE"
            assert len(ready(state)["ai"]["reviewers"]) == 2
            assert len(state["reviews"]) == 1 and len(h.session.requests) == 2
            assert state["orders"] == {} and not h.market.submissions
            assert not any(
                call[0] in {"positions", "open_orders", "equity", "account_info"}
                for call in h.runtime.client.calls
            )
            await h.runtime.poll()
            assert (await h.store.read())["telegram_offset"] == 1
            assert "Mode: SHADOW" in h.runtime.telegram.sent[-1]
            assert "WR —" in h.runtime.telegram.sent[-1]

            trade = next(iter(state["trades"].values()))
            entry, t1, t2, qty, step = (
                trade[k] for k in ("limit", "target1", "target2", "qty", "qty_step")
            )
            h.market.intraday = [
                Candle(int(NOW * 1000), entry, entry + 1, entry - 1, entry),
                Candle(int((NOW + 180) * 1000), entry + 1, t2 + 1, entry - 0.5, t2),
            ]
            await h.tick(360)
            await h.runtime.simulate()
            h.market.funding = [
                {
                    "symbol": "BTCUSDT",
                    "fundingRate": ".0001",
                    "fundingRateTimestamp": str(int((NOW + 240) * 1000)),
                }
            ]
            await h.runtime.funding()
            state = await h.store.read()
            split = math.ceil(qty / 2 / step - 1e-9) * step
            gross = split * (t1 * 0.9997 - entry) + (qty - split) * (
                t2 * 0.9997 - entry
            )
            fees = (
                qty * entry + split * t1 * 0.9997 + (qty - split) * t2 * 0.9997
            ) * 0.0006
            expected_net = gross - fees - qty * entry * 0.0001
            for closed in state["trades"].values():
                assert closed["status"] == "CLOSED" and closed["exit_reason"] == "TP2"
                assert closed["net_pnl"] == pytest.approx(expected_net)
                assert (
                    closed["funding_complete"] is True
                    and len(closed["funding_ids"]) == 1
                )
                assert [fill["reason"] for fill in closed["fills"]] == [
                    "ENTRY",
                    "TP1",
                    "TP2",
                ]
            dashboard = await h.runtime.service.snapshot("dashboard")
            assert "Ready: 1" in dashboard and dashboard.count("WR 100.0%") == 2
            assert dashboard.count(f"Estimated net {expected_net:.2f} USDT") == 2
            assert dashboard.count("Funding pending on 0 closed trades") == 2
            assert "Exchange-confirmed: 0 closed" in dashboard
            persisted = deepcopy(state["trades"])
            event_keys = [row[0] for row in await events(h.store, "shadow_state")]
            await h.restart()
            await h.runtime.scan()
            await reviews(h.runtime)
            await h.runtime.simulate()
            await h.runtime.funding()
            assert (await h.store.read())["trades"] == persisted
            assert [
                row[0] for row in await events(h.store, "shadow_state")
            ] == event_keys
            assert (
                len(h.session.requests) == 2
            ), "Restart must use durable AI decisions, not replay paid calls"
            assert (await h.store.read())["telegram_offset"] == 1
            assert (
                dashboard.split("📊 Forward performance")[1]
                == (await h.runtime.service.snapshot("dashboard")).split(
                    "📊 Forward performance"
                )[1]
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("verdict", ["WAIT", "REJECT"])
def test_ai_nonapproval_does_not_remove_rules_baseline_or_create_ai_trade(
    harness, verdict
):
    async def scenario():
        responses = [lambda body: review_response(body, verdict=verdict)] * 2
        async with harness(ai_actions=responses) as h:
            await h.runtime.scan()
            await reviews(h.runtime)
            state = await h.store.read()
            assert {t["arm"] for t in state["trades"].values()} == {"baseline_shadow"}
            assert ready(state)["ai"]["verdict"] == verdict
            assert not h.market.submissions

    asyncio.run(scenario())


def test_testnet_startup_reconciles_then_records_one_intent_and_no_false_fill(harness):
    async def scenario():
        async with harness("testnet") as h:
            account = (await h.store.read())["account"]
            assert account["blockers"] == [] and account["equity"] == 10_000
            assert account["account"]["marginMode"] == "ISOLATED_MARGIN"
            assert h.runtime.execution.mode == "testnet"
            await h.runtime.scan()
            assert (
                not h.market.submissions
            )  # Baseline alone never creates an exchange intent.
            await reviews(h.runtime)
            state = await h.store.read()
            assert len(h.market.submissions) == len(state["orders"]) == 1
            assert next(iter(state["orders"].values()))["status"] == "ACKNOWLEDGED"
            await h.runtime.reconcile()
            reconciled = await h.store.read()
            assert reconciled["account"]["blockers"] == []
            assert next(iter(reconciled["orders"].values()))["status"] == "PENDING"
            await h.restart()
            await h.runtime.scan()
            await reviews(h.runtime)
            assert len(h.market.submissions) == 1
            assert len((await h.store.read())["orders"]) == 1
            assert "Mode: TESTNET" in await h.runtime.service.snapshot("dashboard")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "gate,reason",
    [
        ("stale_account", "Exchange account snapshot is stale"),
        ("paused", None),
        ("wide_spread", "SPREAD_TOO_WIDE"),
        ("position_cap", "POSITION_CAP"),
        ("heat_cap", "HEAT_CAP"),
        ("outbox", "Telegram delivery delayed"),
        ("blackout", "Official macro event blackout"),
        ("missing_context", "macro context"),
    ],
)
def test_runtime_gates_block_actual_submission_after_ai_approval(harness, gate, reason):
    async def scenario():
        async with harness("testnet") as h:
            if gate == "paused":
                await h.runtime.service.change_setting(
                    "paused", True, "telegram:-100987654321:987654321"
                )
            elif gate == "wide_spread":
                h.market.ticker.update(ask1Price="150", bid1Price="135")
            elif gate == "outbox":
                await h.tick(301)

            def configure(tx):
                if gate == "stale_account":
                    tx.state["account"]["as_of"] = h.clock.now - 31
                if gate == "outbox":
                    tx.state["account"]["as_of"] = h.clock.now
                if gate in {"position_cap", "heat_cap"}:
                    count = 6 if gate == "position_cap" else 1
                    for i in range(count):
                        tx.state["orders"][f"reserved-{i}"] = {
                            "id": f"reserved-{i}",
                            "symbol": f"OTHER{i}USDT",
                            "status": "PENDING",
                            "bucket": f"other-{i}",
                            "risk_cash": 1 if count == 6 else 400,
                            "notional": 100,
                            "children": {},
                        }
                if gate == "blackout":
                    tx.state["context"]["event_blackout"] = True
                if gate == "missing_context":
                    tx.state["context"]["data_complete"] = False

            await h.store.update(configure)
            await h.runtime.scan()
            await reviews(h.runtime)
            state = await h.store.read()
            assert ready(state)["ai"]["verdict"] == "APPROVE"
            assert not h.market.submissions
            assert all(
                order["symbol"] != "BTCUSDT" for order in state["orders"].values()
            )
            if reason:
                assert reason.lower() in ready(state)["decision"].lower()
            if gate == "paused":
                assert state["trades"] == {}
                assert "Paused" in await h.runtime.service.snapshot("dashboard")

    asyncio.run(scenario())


def test_pause_keeps_existing_shadow_stops_active(harness):
    async def scenario():
        async with harness() as h:
            await h.runtime.scan()
            await reviews(h.runtime)
            state = await h.store.read()
            trade = next(iter(state["trades"].values()))
            entry, stop = trade["limit"], trade["stop"]
            h.market.intraday = [
                Candle(int(NOW * 1000), entry, entry + 1, entry - 1, entry)
            ]
            await h.tick(180)
            await h.runtime.simulate()
            assert all(
                t["status"] == "OPEN" for t in (await h.store.read())["trades"].values()
            )
            await h.runtime.service.change_setting(
                "paused", True, "telegram:-100987654321:987654321"
            )
            h.market.intraday.append(
                Candle(int((NOW + 180) * 1000), entry, entry + 1, stop - 1, stop)
            )
            await h.tick(180)
            await h.runtime.simulate()
            state = await h.store.read()
            assert state["settings"]["paused"] is True
            assert all(
                t["status"] == "CLOSED" and t["exit_reason"] == "STOP"
                for t in state["trades"].values()
            )

    asyncio.run(scenario())


def test_outbox_retries_persists_delivery_and_deduplicates_event_replay(harness):
    async def scenario():
        async with harness() as h:
            await h.runtime.deliver()  # Clear real startup/context events.
            await h.store.update(
                lambda tx: tx.event(
                    "fixture-alert", "fixture", {"n": 1}, "Offline alert"
                )
            )
            h.runtime.telegram.failures = [TelegramError(retry_after=7)]
            await h.runtime.deliver()
            assert not await h.store.pending_notifications()
            assert (await h.store.outbox_health())["pending"] == 1
            await h.tick(7)
            pending = await h.store.pending_notifications()
            assert pending[0]["key"] == "fixture-alert" and pending[0]["attempts"] == 1
            await h.runtime.deliver()
            assert h.runtime.telegram.sent.count("Offline alert") == 1
            assert (await h.store.outbox_health())["pending"] == 0
            await h.restart()
            await h.store.update(
                lambda tx: tx.event(
                    "fixture-alert", "fixture", {"n": 1}, "Offline alert"
                )
            )
            await h.runtime.deliver()
            assert "Offline alert" not in h.runtime.telegram.sent
            assert (
                len([e for e in await events(h.store) if e[0] == "fixture-alert"]) == 1
            )

    asyncio.run(scenario())


def test_stale_account_is_visible_in_dashboard_not_presented_as_current(harness):
    async def scenario():
        async with harness("testnet") as h:
            await h.tick(61)
            positions = await h.runtime.service.snapshot("positions")
            assert "stale" in positions.lower()
            dashboard = await h.runtime.service.snapshot("dashboard")
            assert (
                "stale" in dashboard.lower()
            ), "Dashboard presents expired exchange equity without a freshness warning"

    asyncio.run(scenario())


def test_manual_deposits_change_capital_not_strategy_performance(harness):
    async def scenario():
        async with harness("testnet") as h:
            before = await h.runtime.service.snapshot("performance")
            h.market.equity += 500
            await h.runtime.reconcile()
            state = await h.store.read()
            assert state["account"]["equity"] == 10_500
            assert state["orders"] == {} and state["trades"] == {}
            assert await h.runtime.service.snapshot("performance") == before
            assert h.runtime._losses(state, 10_500, None) == {
                "daily_loss_pct": 0,
                "weekly_loss_pct": 0,
                "drawdown_pct": 0,
            }

    asyncio.run(scenario())


def test_rescan_does_not_reassess_existing_shadow_trade(harness, monkeypatch):
    async def scenario():
        async with harness() as h:
            runtime = importlib.import_module("apex_bot.runtime")
            original, assessments = runtime.assess, []

            def assess(*args, **kwargs):
                assessments.append((args, kwargs))
                return original(*args, **kwargs)

            monkeypatch.setattr(runtime, "assess", assess)
            await h.runtime.scan()
            await reviews(h.runtime)
            before = await h.store.read()
            count = len(assessments)
            assert count == 2
            await h.runtime.scan()
            await reviews(h.runtime)
            after = await h.store.read()
            assert after["trades"] == before["trades"]
            assert len(assessments) == count
            assert ready(after)["decision"]
            assert len(h.session.requests) == 2
            assert len(await events(h.store, "shadow_order")) == 2
            assert "SYMBOL_ALREADY_EXPOSED" not in ready(after)["decision"]
            assert ready(after)["last_risk_review"] == ready(before)["last_risk_review"]
            assert ready(after)["last_risk_review"]["assessment"]["rr_target1"] >= 1.5

    asyncio.run(scenario())


def test_paused_rescan_preserves_existing_candidate_decision(harness):
    async def scenario():
        async with harness() as h:
            await h.runtime.scan()
            await reviews(h.runtime)
            before = await h.store.read()
            await h.runtime.service.change_setting(
                "paused", True, "telegram:-100987654321:987654321"
            )
            await h.runtime.scan()
            await reviews(h.runtime)
            after = await h.store.read()
            assert ready(after)["decision"] == ready(before)["decision"]
            assert after["trades"] == before["trades"]

    asyncio.run(scenario())


def test_dashboard_capital_is_usdt_and_rejects_explicit_usd_snapshot(harness):
    async def scenario():
        async with harness("testnet") as h:
            dashboard = await h.runtime.service.snapshot("dashboard")
            assert "USDT coin equity" in dashboard
            assert (
                h.runtime.service._capital(await h.store.read(), h.clock.now)["equity"]
                == 10_000
            )
            await h.store.update(lambda tx: tx.state["account"].update(currency="USD"))
            capital = h.runtime.service._capital(await h.store.read(), h.clock.now)
            assert capital["equity"] is None and capital["error"]

    asyncio.run(scenario())


def test_fixed_strategy_reference_and_open_losses_ignore_manual_deposit(harness):
    async def scenario():
        async with harness("testnet") as h:
            await h.runtime._risk_snapshot(await h.store.read(), 10_000, None)

            def account_loss(tx):
                tx.state["orders"]["closed-fixture"] = {
                    "id": "closed-fixture",
                    "status": "CLOSED",
                    "net_pnl": -400,
                    "closed_at": h.clock.now - 1,
                }
                tx.state["account"].update(
                    equity=15_000,
                    positions=[
                        {"symbol": "BTCUSDT", "size": "1", "unrealisedPnl": "-600"}
                    ],
                )

            await h.store.update(account_loss)
            metrics = await h.runtime._risk_snapshot(await h.store.read(), 15_000, None)
            assert metrics["drawdown_pct"] == pytest.approx(10)
            assert metrics["daily_loss_pct"] == pytest.approx(1000 / 15_000 * 100)
            assert metrics["weekly_loss_pct"] == pytest.approx(1000 / 15_000 * 100)
            assert (await h.store.read())["risk_reference_equities"][
                "testnet"
            ] == 10_000

    asyncio.run(scenario())


def test_open_strategy_high_water_mark_survives_restart(harness):
    async def scenario():
        async with harness("testnet") as h:
            await h.runtime._risk_snapshot(await h.store.read(), 10_000, None)
            await h.store.update(
                lambda tx: tx.state["account"].update(
                    equity=11_000,
                    positions=[
                        {"symbol": "BTCUSDT", "size": "1", "unrealisedPnl": "1000"}
                    ],
                )
            )
            await h.runtime._risk_snapshot(await h.store.read(), 11_000, None)
            # The gain later returns to zero while the process is offline. The
            # persisted strategy peak must retain the observed open valuation.
            await h.restart()
            metrics = await h.runtime._risk_snapshot(await h.store.read(), 10_000, None)
            assert metrics["drawdown_pct"] == pytest.approx(
                1000 / 11_000 * 100
            ), "Observed unrealized strategy peak was lost across restart"

    asyncio.run(scenario())


def test_failed_health_attempt_does_not_refresh_last_success(harness):
    async def scenario():
        async with harness() as h:
            await h.runtime._health("scan")
            await h.tick(10)
            await h.runtime._health(
                "scan", ValueError("https://secret.invalid?token=private")
            )
            health = (await h.store.read())["health"]
            assert health["scan_at"] == NOW
            assert health["scan_attempt_at"] == NOW + 10
            assert h.runtime.last_loop["scan"] == NOW
            assert "secret.invalid" not in json.dumps(health)

    asyncio.run(scenario())


def test_all_symbol_failures_do_not_report_scan_success(harness, monkeypatch):
    async def scenario():
        async with harness() as h:
            await h.runtime.scan()
            before = (await h.store.read())["health"]["market_BTCUSDT_at"]
            await h.tick(10)

            async def failed_candles(*args, **kwargs):
                raise RuntimeError("offline market unavailable")

            monkeypatch.setattr(h.runtime.client, "candles", failed_candles)
            with pytest.raises(RuntimeError, match="scan"):
                await h.runtime.scan()
            state = await h.store.read()
            assert state["health"]["market_BTCUSDT_at"] == before
            assert state["health"]["market_BTCUSDT_attempt_at"] == h.clock.now

    asyncio.run(scenario())


def test_telegram_partial_progress_is_persisted_monotonically_on_failure(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            await h.store.update(lambda tx: tx.state.update(telegram_offset=3))
            for checkpoint in (7, 2):
                error = TelegramError("offline polling failure")
                error.next_offset = checkpoint

                async def failed_poll(offset):
                    raise error

                monkeypatch.setattr(h.runtime.telegram, "poll_once", failed_poll)
                with pytest.raises(TelegramError):
                    await h.runtime.poll()
                assert (await h.store.read())["telegram_offset"] == 7

    asyncio.run(scenario())


def test_renew_failure_sets_stopping(harness, monkeypatch):
    async def scenario():
        async with harness() as h:
            assert not h.runtime.stopping.is_set()

            async def unavailable_lease():
                raise TimeoutError("offline database renewal timeout")

            monkeypatch.setattr(h.store, "lease", unavailable_lease)
            with pytest.raises(TimeoutError, match="renewal timeout"):
                await h.runtime.renew()
            assert (
                h.runtime.stopping.is_set()
            ), "Renewal failure must stop all worker loops"

    asyncio.run(scenario())


def test_poll_lost_lease_never_calls_get_updates(harness):
    async def scenario():
        async with harness() as h:
            replacement = Store(sqlite_path=h.config.sqlite_path)
            try:
                await replacement.initialize()
                assert not await replacement.lease()
                before = (await h.store.read())["telegram_offset"]
                # Expire the real lease without the harness tick helper renewing it.
                h.clock.now += 100_001
                assert await replacement.lease()
                with pytest.raises(NotLeader):
                    await h.runtime.poll()
                assert (
                    h.runtime.telegram.offsets == []
                ), "Lost leader reached the getUpdates boundary"
                assert h.runtime.telegram.sent == []
                assert (await replacement.read())["telegram_offset"] == before
            finally:
                await replacement.close()

    asyncio.run(scenario())


def test_context_changed_after_saved_review_creates_no_trades(harness, monkeypatch):
    async def scenario():
        async with harness() as h:
            # Discover a real READY candidate without creating the baseline arm.
            await h.store.update(lambda tx: tx.state["settings"].update(paused=True))
            await h.runtime.scan()
            assert (await h.store.read())["trades"] == {}
            await h.store.update(lambda tx: tx.state["settings"].update(paused=False))
            original_consider = h.runtime.consider
            intercepted = []

            async def change_context_before_consider(op, review):
                state = await h.store.read()
                saved = state["opportunities"][op.id]["ai"]
                assert saved["verdict"] == review["verdict"] == "APPROVE"
                assert saved["context_fingerprint"] == context_fingerprint(
                    state, op.symbol, h.clock.now
                )
                original_context = deepcopy(state["context"])
                # Keep freshness, multipliers and blackout valid. Only the reviewed
                # evidence changes, so another risk gate cannot explain rejection.
                await h.store.update(
                    lambda tx: tx.state["context"].update(
                        reason="New macro evidence arrived after the review was saved"
                    )
                )
                changed = await h.store.read()
                assert saved["context_fingerprint"] != context_fingerprint(
                    changed, op.symbol, h.clock.now
                )
                await original_consider(op, review)
                intercepted.append((op, deepcopy(saved), original_context))

            monkeypatch.setattr(h.runtime, "consider", change_context_before_consider)
            await reviews(h.runtime)
            state = await h.store.read()
            assert len(intercepted) == 1 and len(h.session.requests) == 2
            assert "ai_error" not in state["health"]
            assert state["trades"] == {} and state["orders"] == {}
            assert not h.market.submissions
            assert await events(h.store, "shadow_order") == []

            # The same saved approval is otherwise executable with its reviewed
            # context, proving the negative assertion did not pass on bad sizing.
            op, saved, original_context = intercepted[0]
            await h.store.update(lambda tx: tx.state.update(context=original_context))
            await original_consider(op, saved)
            trades = (await h.store.read())["trades"]
            assert set(trades) == {"ai_shadow:" + op.id}

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", ["context", "candidate", "expired"])
def test_inflight_ai_approval_cannot_use_changed_or_expired_evidence(harness, changed):
    async def scenario():
        async with harness("testnet") as h:
            h.session.gate = asyncio.Event()
            await h.runtime.scan()
            worker = asyncio.create_task(h.runtime.review_worker())
            try:
                await asyncio.wait_for(h.session.entered.wait(), timeout=5)
                if changed == "expired":
                    op = ready(await h.store.read())["opportunity"]
                    await h.tick(op["expires_at"] - h.clock.now + 1)
                else:

                    def mutate(tx):
                        if changed == "context":
                            tx.state["context"]["event_blackout"] = True
                        else:
                            ready(tx.state)["opportunity"]["entry"] += 1

                    await h.store.update(mutate)
                h.session.gate.set()
                await asyncio.wait_for(h.runtime.ai_queue.join(), timeout=5)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            state = await h.store.read()
            assert ready(state)["ai"]["verdict"] == "WAIT"
            assert {t["arm"] for t in state["trades"].values()} == {"baseline_shadow"}
            assert not h.market.submissions

    asyncio.run(scenario())


def test_reconciliation_serializes_with_candidate_decisions(harness, monkeypatch):
    async def scenario():
        async with harness("testnet") as h:
            await h.runtime.scan()
            op = Opportunity(**ready(await h.store.read())["opportunity"])
            started, gate = asyncio.Event(), asyncio.Event()
            original = h.runtime.client.positions

            async def held_positions():
                started.set()
                await gate.wait()
                return await original()

            monkeypatch.setattr(h.runtime.client, "positions", held_positions)
            reconcile = asyncio.create_task(h.runtime.reconcile())
            await asyncio.wait_for(started.wait(), timeout=5)
            decision = asyncio.create_task(h.runtime.consider(op, None))
            try:
                for _ in range(3):
                    await asyncio.sleep(0)
                assert (
                    not decision.done()
                ), "Candidate decision ran against an account being reconciled"
            finally:
                gate.set()
                await asyncio.wait_for(asyncio.gather(reconcile, decision), timeout=5)

    asyncio.run(scenario())


def test_runtime_cannot_start_or_write_without_owning_real_store_lease(harness):
    async def scenario():
        async with harness() as h:
            contender = Store(sqlite_path=h.config.sqlite_path)
            try:
                other = type(h.runtime)(h.config, contender, h.session)
                with pytest.raises(RuntimeError, match="lease"):
                    await other.initialize()
                assert not other.started
                with pytest.raises(NotLeader):
                    await contender.update(
                        lambda tx: tx.state["settings"].update(paused=True)
                    )
                assert (await h.store.read())["settings"]["paused"] is False
            finally:
                await contender.close()

    asyncio.run(scenario())


def test_store_rolls_back_state_event_and_outbox_together(harness):
    async def scenario():
        async with harness() as h:
            before = await h.store.read()
            pending = await h.store.outbox_health()

            def bad_update(tx):
                tx.state["settings"]["paused"] = True
                tx.event(
                    "bad-json", "fixture", {"value": float("nan")}, "Do not deliver"
                )

            with pytest.raises(ValueError):
                await h.store.update(bad_update)
            assert await h.store.read() == before
            assert await h.store.outbox_health() == pending
            assert not any(row[0] == "bad-json" for row in await events(h.store))

    asyncio.run(scenario())


def test_restart_recovery_uses_full_range_instead_of_latest_1000_candles(harness):
    async def scenario():
        async with harness() as h:
            await h.runtime.scan()
            # More than 1000 closed bars arrive while the worker is down. The
            # first missed bar fills and stops; the latest page contains neither.
            trade = next(iter((await h.store.read())["trades"].values()))
            entry, stop = trade["limit"], trade["stop"]
            h.market.intraday = [
                Candle(int(NOW * 1000), entry, entry + 1, stop - 1, stop)
            ]
            h.market.intraday += [
                Candle(
                    int((NOW + n * 180) * 1000),
                    entry + 1,
                    entry + 2,
                    entry + 0.5,
                    entry + 1,
                )
                for n in range(1, 2000)
            ]
            await h.tick(2000 * 180)
            await h.restart()
            await h.runtime.simulate()
            actual = next(iter((await h.store.read())["trades"].values()))
            assert (
                actual["status"] == "CLOSED" and actual["exit_reason"] == "STOP"
            ), "Restart skipped the entry/stop bar before the latest 1000-bar page"
            assert any(call[0] == "candle_range" for call in h.runtime.client.calls)

    asyncio.run(scenario())


def test_postgres_sql_uses_native_parameters_server_clock_and_row_locks():
    """Dialect/transaction-path inspection only; this does not claim a PostgreSQL run."""
    store = Store(database_url="postgresql://offline.invalid/apex_test")
    statements, transaction = [], []

    class Cursor:
        def execute(self, sql, params=()):
            statements.append((sql, params))
            if sql.startswith("SELECT EXTRACT"):
                self.result = (NOW,)
            elif sql.startswith("SELECT owner,expires"):
                self.result = (store.owner, NOW + 90)
            elif sql.startswith("SELECT value FROM apex_state"):
                self.result = (json.dumps(initial_state()),)
            elif sql.startswith("SELECT COUNT(*)"):
                self.result = (0, None)
            else:
                self.result = None

        def fetchone(self):
            return self.result

        def fetchall(self):
            return []

        def close(self):
            pass

    class Connection:
        closed = False

        def cursor(self):
            return Cursor()

        def commit(self):
            transaction.append("commit")

        def rollback(self):
            transaction.append("rollback")

    connection = Connection()
    store.pool = SimpleNamespace(
        getconn=lambda: connection,
        putconn=lambda conn, close=False: None,
        closeall=lambda: None,
    )

    async def scenario():
        await store.initialize()
        assert await store.lease()
        payload = (
            "Literal %s, ?, quotes ' and SQL-looking input; DROP TABLE not executed"
        )
        await store.update(
            lambda tx: tx.event("sql-fixture", "fixture", {"text": payload}, payload)
        )
        await store.candle_history("BTCUSDT", "D", DAILY[-2:])
        await store.pending_notifications()
        await store.notification_result("sql-fixture", message_id="1")
        await store.outbox_health()
        await store.close()

    asyncio.run(scenario())
    sql = [statement for statement, _ in statements]
    assert all("?" not in statement for statement in sql)
    assert not any("BEGIN IMMEDIATE" in statement for statement in sql)
    assert "SELECT owner,expires FROM apex_lease WHERE id=1 FOR UPDATE" in sql
    assert "SELECT value FROM apex_state WHERE id=1 FOR UPDATE" in sql
    assert "SELECT EXTRACT(EPOCH FROM clock_timestamp())" in sql
    assert any("ON CONFLICT(event_key) DO NOTHING" in statement for statement in sql)
    candle_inserts = [
        (statement, params)
        for statement, params in statements
        if statement.startswith("INSERT INTO apex_candles")
    ]
    assert any(
        len(params) > 4 for _, params in candle_inserts
    ), "Candles should use a parameterized bulk insert"
    bound_candles = []
    for statement, params in candle_inserts:
        assert statement.count("%s") == len(params)
        assert "BTCUSDT" not in statement and '"open_time"' not in statement
        for offset in range(0, len(params), 4):
            symbol, interval, open_time, value = params[offset : offset + 4]
            bound_candles.append((symbol, interval, open_time, json.loads(value)))
    assert bound_candles == [
        ("BTCUSDT", "D", bar.open_time, asdict(bar)) for bar in DAILY[-2:]
    ]
    assert any("due<=%s" in statement and "LIMIT %s" in statement for statement in sql)
    assert all("DROP TABLE" not in statement for statement in sql)
    assert "rollback" not in transaction
