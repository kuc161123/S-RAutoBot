"""Offline integration regressions for persisted dynamic entry membership.

Production runtime/execution/selector/Store are exercised; only venue I/O and
deliberate commit races are replaced. Failures identify parent-owned defects.
"""

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apex_bot import execution, runtime
from apex_bot.config import Config
from apex_bot.models import Candle, Opportunity
from .test_execution import FakeBybit
from .test_runtime import DAY, INSTRUMENT, harness, ready


def configuration_policy(config=None):
    config = config or Config()
    return {
        "target": config.universe_size,
        "min_turnover_usdt": config.universe_min_turnover,
        "max_spread_bps": config.universe_max_spread_bps,
        "min_depth_usdt": config.universe_min_depth,
    }


def selection(now, symbols=("BTCUSDT",)):
    return {
        "mode": "dynamic",
        "as_of": now,
        "status": "ready",
        "target": 50,
        "active_symbols": list(symbols),
        "members": {symbol: {"status": "eligible"} for symbol in symbols},
        "blocked": {},
        "policy": {
            **configuration_policy(),
            "min_listing_days": 30,
            "depth_band_bps": 25,
            "max_book_age": 120,
            "max_snapshot_age": 120,
            "replacement_ratio": 1.5,
            "confirmation_observations": 2,
            "max_daily_replacements": 5,
        },
    }


async def dynamic_market(h, monkeypatch, symbols=None, *, refresh=True):
    symbols = symbols or ["BTCUSDT", *(f"COIN{i:02}USDT" for i in range(1, 60))]
    config = replace(h.config, universe_mode="dynamic", universe_size=50)
    h.runtime.config = h.runtime.service.config = config
    market = SimpleNamespace(
        metadata={
            symbol: {
                "symbol": symbol,
                "baseCoin": symbol[:-4],
                "quoteCoin": "USDT",
                "settleCoin": "USDT",
                "contractType": "LinearPerpetual",
                "status": "Trading",
                "launchTime": str(int((h.clock.now - 365 * DAY) * 1000)),
                "isPreListing": False,
            }
            for symbol in symbols
        },
        tickers=[
            {
                "symbol": symbol,
                "turnover24h": str(300_000_000 - i * 1_000_000),
                "bid1Price": "99.99",
                "ask1Price": "100.01",
            }
            for i, symbol in enumerate(symbols)
        ],
        failure=None,
        calls=[],
        book_calls=[],
        running=0,
        max_running=0,
    )

    async def instruments():
        market.calls.append("instruments")
        if market.failure == "instruments":
            raise RuntimeError("metadata unavailable")
        h.runtime.client.instrument_metadata = deepcopy(market.metadata)
        return {s: replace(INSTRUMENT, symbol=s) for s in market.metadata}

    async def tickers():
        market.calls.append("tickers")
        if market.failure == "transport":
            raise RuntimeError("ticker transport failed")
        rows = deepcopy(market.tickers)
        if market.failure == "incomplete":
            rows.pop()
        if market.failure == "empty":
            rows = []
        return {
            "list": rows,
            "as_of": h.clock.now - (121 if market.failure == "stale" else 0),
        }

    async def book(symbol):
        market.book_calls.append(symbol)
        if market.failure == "books":
            raise RuntimeError("book unavailable")
        market.running += 1
        market.max_running = max(market.max_running, market.running)
        try:
            await asyncio.sleep(0)
            return {
                "as_of": h.clock.now,
                "band_bps": 25,
                "spread_bps": 2,
                "bid_depth_usdt": 100_000,
                "ask_depth_usdt": 120_000,
            }
        finally:
            market.running -= 1

    monkeypatch.setattr(h.runtime.client, "instruments", instruments)
    monkeypatch.setattr(h.runtime.client, "tickers", tickers, raising=False)
    monkeypatch.setattr(h.runtime.client, "liquidity_book", book, raising=False)
    await h.store.update(
        lambda tx: tx.state.update(
            universe_mode="dynamic",
            universe_policy=configuration_policy(config),
            universe={},
        )
    )
    if refresh:
        await h.runtime.refresh_universe()
    return market


def test_fifty_fresh_members_are_selected_from_venue_and_reviewed_every_six_hours(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            market = await dynamic_market(h, monkeypatch)
            state = await h.store.read()
            universe = state["universe"]
            assert len(universe["active_symbols"]) == universe["target"] == 50
            assert universe["as_of"] == h.clock.now and universe["status"] == "ready"
            assert all(
                member["sample_count"] == 1 for member in universe["members"].values()
            )
            assert (
                h.runtime._entry_symbols(state, h.clock.now)
                == universe["active_symbols"]
            )
            assert len(market.book_calls) == 60 and market.max_running <= 4
            assert (
                len(h.config.symbols) == 1
            )  # Never use the manual seed as membership.
            calls = deepcopy(market.calls)
            await h.tick(6 * 3600 - 1)
            await h.runtime.refresh_universe()
            assert market.calls == calls
            assert (await h.store.read())["universe"] == universe
            await h.tick(1)
            await h.runtime.refresh_universe()
            refreshed = (await h.store.read())["universe"]
            assert refreshed["as_of"] == h.clock.now
            assert len(refreshed["active_symbols"]) == 50
            assert all(
                member["sample_count"] == 2 for member in refreshed["members"].values()
            )
            assert market.calls.count("tickers") == 2

    asyncio.run(scenario())


def test_book_shortlist_uses_sustained_turnover_score_instead_of_latest_spike(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            symbols = ["BTCUSDT", "ALPHAUSDT", "BETAUSDT", "GAMMAUSDT"]
            market = await dynamic_market(h, monkeypatch, symbols, refresh=False)
            h.runtime.config = replace(h.runtime.config, universe_size=1)
            historical = {
                "BTCUSDT": 50_000_000,
                "ALPHAUSDT": 100_000_000,
                "BETAUSDT": 25_000_000,
                "GAMMAUSDT": 24_000_000,
            }
            current = {
                "BTCUSDT": 50_000_000,
                "ALPHAUSDT": 20_000_000,
                "BETAUSDT": 150_000_000,
                "GAMMAUSDT": 140_000_000,
            }
            for ticker in market.tickers:
                ticker["turnover24h"] = str(current[ticker["symbol"]])
            previous = selection(h.clock.now - 6 * 3600)
            previous.update(
                target=1,
                last_observation_at=h.clock.now - 6 * 3600,
                snapshots={
                    symbol: [
                        {"as_of": h.clock.now - age * 3600, "turnover24h": value}
                        for age in (18, 12, 6)
                    ]
                    for symbol, value in historical.items()
                },
            )
            await h.store.update(
                lambda tx: tx.state.update(universe=deepcopy(previous))
            )
            await h.runtime.refresh_universe()
            assert set(market.book_calls) == {"ALPHAUSDT", "BTCUSDT", "BETAUSDT"}
            universe = (await h.store.read())["universe"]
            assert universe["eligibility"]["ALPHAUSDT"]["score"] == 100_000_000
            assert all(len(rows) == 4 for rows in universe["snapshots"].values())
            assert universe["challenger_streaks"]["ALPHAUSDT"]["count"] == 1

    asyncio.run(scenario())


def test_fresh_hard_filter_failures_commit_empty_membership_without_book_requests(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            market = await dynamic_market(h, monkeypatch)
            await h.tick(6 * 3600)
            previous_calls = list(market.book_calls)
            for ticker in market.tickers:
                ticker["turnover24h"] = "1"
            await h.runtime.refresh_universe()
            universe = (await h.store.read())["universe"]
            assert universe["active_symbols"] == []
            assert (
                universe["status"] == "underfilled" and universe["as_of"] == h.clock.now
            )
            assert market.book_calls == previous_calls
            assert len(universe["removed"]) == 50

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["transport", "instruments", "incomplete", "empty", "stale", "books"]
)
def test_failed_selection_snapshot_retains_all_committed_state(
    harness, monkeypatch, failure
):
    async def scenario():
        async with harness() as h:
            market = await dynamic_market(h, monkeypatch)
            await h.tick(6 * 3600)
            market.failure = failure
            before = await h.store.read()
            notifications = await h.store.pending_notifications()
            with pytest.raises((ValueError, RuntimeError)):
                await h.runtime.refresh_universe()
            assert await h.store.read() == before
            assert await h.store.pending_notifications() == notifications

    asyncio.run(scenario())


def test_ineligible_high_turnover_symbols_do_not_starve_fifty_qualified_members(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            young = [f"YOUNG{i:03}USDT" for i in range(150)]
            eligible = [f"LIQUID{i:02}USDT" for i in range(50)]
            market = await dynamic_market(
                h, monkeypatch, young + eligible, refresh=False
            )
            for symbol in young:
                market.metadata[symbol]["launchTime"] = str(
                    int((h.clock.now - DAY) * 1000)
                )
            await h.runtime.refresh_universe()
            universe = (await h.store.read())["universe"]
            assert set(universe["active_symbols"]) == set(eligible)
            assert (
                len(h.runtime._entry_symbols(await h.store.read(), h.clock.now)) == 50
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "age,expected", [(86399, ["BTCUSDT"]), (86400, []), (86401, [])]
)
def test_stale_membership_blocks_entry_at_twenty_four_hours(harness, age, expected):
    async def scenario():
        async with harness() as h:
            h.runtime.config = replace(h.config, universe_mode="dynamic")
            await h.store.update(
                lambda tx: tx.state.update(
                    universe_mode="dynamic", universe=selection(h.clock.now - age)
                )
            )
            assert (
                h.runtime._entry_symbols(await h.store.read(), h.clock.now) == expected
            )

    asyncio.run(scenario())


def test_scan_rechecks_membership_after_awaiting_market_data(harness, monkeypatch):
    async def scenario():
        async with harness() as h:
            h.runtime.config = replace(h.config, universe_mode="dynamic")
            await h.store.update(
                lambda tx: tx.state.update(
                    universe_mode="dynamic", universe=selection(h.clock.now)
                )
            )
            original = h.runtime.cache.candles
            retired = []

            async def candles(*args, **kwargs):
                if not retired:
                    retired.append(True)
                    await h.store.update(
                        lambda tx: tx.state.update(universe=selection(h.clock.now, ()))
                    )
                return await original(*args, **kwargs)

            monkeypatch.setattr(h.runtime.cache, "candles", candles)
            await h.runtime.scan()
            state = await h.store.read()
            assert retired and state["opportunities"] == {} and state["trades"] == {}
            assert h.runtime.ai_queue.empty()
            await h.store.update(
                lambda tx: tx.state.update(universe=selection(h.clock.now))
            )
            await h.runtime.scan()
            assert ready(await h.store.read())["opportunity"]["symbol"] == "BTCUSDT"

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["removed", "stale", "blocked"])
def test_shadow_membership_is_rechecked_inside_creation_transaction(
    harness, monkeypatch, change
):
    async def scenario():
        async with harness() as h:
            await h.store.update(lambda tx: tx.state["settings"].update(paused=True))
            await h.runtime.scan()
            op = Opportunity(**ready(await h.store.read())["opportunity"])
            h.runtime.config = replace(h.config, universe_mode="dynamic")
            await h.store.update(
                lambda tx: (
                    tx.state.update(
                        universe_mode="dynamic", universe=selection(h.clock.now)
                    ),
                    tx.state["settings"].update(paused=False),
                )
            )
            original = h.store.update
            intercepted = []

            async def update(callback, *args, **kwargs):
                if callback.__name__ == "create":

                    def changed(tx):
                        intercepted.append(True)
                        if change == "removed":
                            tx.state["universe"] = selection(h.clock.now, ())
                        elif change == "stale":
                            tx.state["universe"]["as_of"] = h.clock.now - DAY - 1
                        else:
                            tx.state["universe"]["members"]["BTCUSDT"][
                                "status"
                            ] = "blocked_pending_fresh"
                        return callback(tx)

                    return await original(changed, *args, **kwargs)
                return await original(callback, *args, **kwargs)

            monkeypatch.setattr(h.store, "update", update)
            await h.runtime.consider(op, None)
            assert intercepted and not (await h.store.read())["trades"]
            monkeypatch.setattr(h.store, "update", original)
            await h.store.update(
                lambda tx: tx.state.update(universe=selection(h.clock.now))
            )
            await h.runtime.consider(op, None)
            assert "baseline_shadow:" + op.id in (await h.store.read())["trades"]

    asyncio.run(scenario())


async def exchange_candidate(h):
    manager = h.runtime.execution
    client = FakeBybit(h.store)
    manager.client = client
    manager.instruments = {"BTCUSDT": INSTRUMENT}
    op = Opportunity(
        "universe-candidate",
        "BTCUSDT",
        "Buy",
        "2",
        "READY",
        100,
        90,
        120,
        140,
        92,
        99,
        h.clock.now - DAY,
        h.clock.now + 100,
        "trigger",
        {"risk_multiplier": 1, "zone_low": 95, "zone_high": 110},
    )
    sizing = dict(
        allowed=True,
        qty=2,
        qty_step=INSTRUMENT.qty_step,
        entry=100,
        stop=90,
        target1=120,
        target2=140,
        risk_cash=21,
        notional=200,
        risk_pct=0.25,
    )

    def save(tx):
        tx.state.update(
            universe_mode="dynamic",
            universe_policy=configuration_policy(),
            universe=selection(h.clock.now),
        )
        tx.state["settings"]["paused"] = False
        tx.state["opportunities"][op.id] = {
            "opportunity": op.to_dict(),
            "ai": {
                "verdict": "APPROVE",
                "evidence_hash": "a" * 64,
                "candidate_fingerprint": execution.candidate_fingerprint(op),
                "context_fingerprint": execution.context_fingerprint(
                    tx.state, op.symbol, h.clock.now
                ),
            },
        }

    await h.store.update(save)
    state = await h.store.read()
    assert manager._reservation_allowed(state, op, sizing, state["settings_version"])
    return manager, client, op, sizing, state["settings_version"]


@pytest.mark.parametrize("change", ["removed", "stale", "blocked"])
def test_exchange_membership_is_rechecked_inside_order_reservation(
    harness, monkeypatch, change
):
    async def scenario():
        async with harness("testnet") as h:
            manager, client, op, sizing, version = await exchange_candidate(h)
            original = h.store.update
            intercepted = []

            async def update(callback, *args, **kwargs):
                if callback.__name__ == "reserve":

                    def changed(tx):
                        intercepted.append(True)
                        if change == "removed":
                            tx.state["universe"] = selection(h.clock.now, ())
                        elif change == "stale":
                            tx.state["universe"]["as_of"] = h.clock.now - DAY - 1
                        else:
                            tx.state["universe"]["members"]["BTCUSDT"][
                                "status"
                            ] = "blocked_pending_fresh"
                        return callback(tx)

                    return await original(changed, *args, **kwargs)
                return await original(callback, *args, **kwargs)

            monkeypatch.setattr(h.store, "update", update)
            assert not await manager.submit(op, sizing, version)
            assert (
                intercepted
                and not (await h.store.read())["orders"]
                and not client.calls
            )
            monkeypatch.setattr(h.store, "update", original)
            await h.store.update(
                lambda tx: tx.state.update(universe=selection(h.clock.now))
            )
            assert await manager.submit(op, sizing, version)
            assert any(call[0] == "submit" for call in client.calls)

    asyncio.run(scenario())


@pytest.mark.parametrize("filled", [0, 1, 2])
@pytest.mark.parametrize("retirement", ["removed", "stale"])
def test_exchange_retirement_cancels_only_remainder_and_preserves_prior_fills(
    harness, filled, retirement
):
    async def scenario():
        async with harness("testnet") as h:
            manager, client, op, sizing, version = await exchange_candidate(h)
            assert await manager.submit(op, sizing, version)
            link = execution.client_id(op.id)
            await h.tick(1)
            if filled:
                client.fill(link, filled)
            await h.tick(1)

            def retire(tx):
                tx.state["universe"] = (
                    selection(h.clock.now, ())
                    if retirement == "removed"
                    else selection(h.clock.now - DAY - 1)
                )
                if retirement == "removed":
                    h.runtime._retire_pending(tx, set(), h.clock.now)

            await h.store.update(retire)
            await manager.reconcile(manager.instruments)
            order = (await h.store.read())["orders"][link]
            assert order["status"] == ("OPEN" if filled else "CANCELLED")
            assert order["filled_qty"] == order["remaining_qty"] == filled
            assert order["desired_stop"] == 90
            assert not any(call[0] == "reduce" for call in client.calls)
            assert any(call[0] == "cancel" for call in client.calls) == (filled < 2)
            assert "exit_request" not in order
            assert len(order["entry_fills"]) == (1 if filled else 0)

    asyncio.run(scenario())


@pytest.mark.parametrize("fill_before", [True, False])
def test_shadow_retirement_replays_pre_cutoff_fills_and_rejects_later_fills(
    harness, fill_before
):
    async def scenario():
        async with harness() as h:
            await h.runtime.scan()
            original = next(iter((await h.store.read())["trades"].values()))
            started = h.clock.now
            entry = original["limit"]
            h.market.intraday = [
                Candle(
                    int(started * 1000),
                    entry + 1,
                    entry + 2,
                    entry - 1 if fill_before else entry + 0.5,
                    entry + 1,
                ),
                Candle(
                    int((started + 180) * 1000),
                    entry + 1,
                    entry + 2,
                    entry - 1,
                    entry + 1,
                ),
            ]
            await h.tick(180)
            h.runtime.config = replace(h.config, universe_mode="dynamic")

            def retire(tx):
                tx.state.update(
                    universe_mode="dynamic", universe=selection(h.clock.now, ())
                )
                h.runtime._retire_pending(tx, set(), h.clock.now)

            await h.store.update(retire)
            pending = (await h.store.read())["trades"][original["id"]]
            assert (
                pending["status"] == "PENDING" and pending["expires_at"] == h.clock.now
            )
            await h.tick(180)
            await h.runtime.simulate()
            updated = (await h.store.read())["trades"][original["id"]]
            assert updated["status"] == ("OPEN" if fill_before else "EXPIRED")
            assert updated["original_expires_at"] == original["expires_at"]
            assert updated["stop"] == original["stop"]
            assert [f["reason"] for f in updated["fills"]] == (
                ["ENTRY"] if fill_before else []
            )
            if fill_before:
                assert updated["last_bar"] == int((started + 180) * 1000)
                assert updated["exit_reason"] is None

    asyncio.run(scenario())


def test_scan_manages_retired_positions_without_new_opportunities(harness, monkeypatch):
    async def scenario():
        async with harness("testnet") as h:
            manager, client, op, sizing, version = await exchange_candidate(h)
            assert await manager.submit(op, sizing, version)
            link = execution.client_id(op.id)
            await h.tick(1)
            client.fill(link, 2)
            await manager.reconcile(manager.instruments)
            h.runtime.config = replace(h.config, universe_mode="dynamic")
            await h.store.update(
                lambda tx: tx.state.update(universe=selection(h.clock.now, ()))
            )
            before = deepcopy((await h.store.read())["opportunities"])
            manage = AsyncMock(wraps=manager.manage_structure)
            monkeypatch.setattr(manager, "manage_structure", manage)
            await h.runtime.scan()
            manage.assert_awaited_once()
            assert manage.await_args.args[0] == "BTCUSDT"
            state = await h.store.read()
            assert state["opportunities"] == before
            assert state["orders"][link]["status"] == "OPEN"
            assert h.runtime.ai_queue.empty()
            assert not any(call[0] == "reduce" for call in client.calls)

    asyncio.run(scenario())


def test_dynamic_startup_defers_book_selection_and_reports_awaiting_state(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            await h.store.update(lambda tx: tx.state.pop("universe", None))
            config = replace(h.config, universe_mode="dynamic")
            worker = runtime.Runtime(config, h.store, h.session)
            heavy = AsyncMock(
                side_effect=AssertionError("book selection must wait for heartbeat")
            )
            monkeypatch.setattr(worker, "refresh_universe", heavy)
            try:
                await worker.initialize()
                heavy.assert_not_awaited()
                state = await h.store.read()
                assert state["universe_mode"] == "dynamic"
                assert worker._entry_symbols(state, h.clock.now) == []
                text = await worker.service.snapshot("dashboard")
                assert "Universe: 0/50 eligible symbols" in text
                detail = await worker.service.snapshot("status")
                assert "0/50 active" in detail and "Awaiting selection" in detail
                assert "Selection as of: unavailable" in detail
            finally:
                await worker.cache.close()

    asyncio.run(scenario())


def test_scan_rechecks_membership_when_persisting_analyzed_candidate(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            h.runtime.config = replace(h.config, universe_mode="dynamic")
            await h.store.update(
                lambda tx: tx.state.update(
                    universe_mode="dynamic", universe=selection(h.clock.now)
                )
            )
            original = h.store.update
            intercepted = []

            async def update(callback, *args, **kwargs):
                # Only scan's opportunity save closes over "op"; context/risk
                # writes use the same generic callback name and are untouched.
                if (
                    callback.__name__ == "save"
                    and "op" in callback.__code__.co_freevars
                ):

                    def changed(tx):
                        intercepted.append(True)
                        tx.state["universe"] = selection(h.clock.now, ())
                        return callback(tx)

                    return await original(changed, *args, **kwargs)
                return await original(callback, *args, **kwargs)

            monkeypatch.setattr(h.store, "update", update)
            await h.runtime.scan()
            state = await h.store.read()
            assert intercepted
            assert (
                not state["opportunities"]
                and not state["trades"]
                and not state["orders"]
            )
            assert h.runtime.ai_queue.empty()

    asyncio.run(scenario())


def test_membership_removed_during_ai_dispatch_records_wait_without_entry(harness):
    async def scenario():
        async with harness("testnet") as h:
            h.runtime.config = replace(h.config, universe_mode="dynamic")
            await h.store.update(
                lambda tx: tx.state.update(
                    universe_mode="dynamic", universe=selection(h.clock.now)
                )
            )
            h.session.gate = asyncio.Event()
            await h.runtime.scan()
            baseline = deepcopy((await h.store.read())["trades"])
            worker = asyncio.create_task(h.runtime.review_worker())
            try:
                await asyncio.wait_for(h.session.entered.wait(), timeout=5)
                await h.store.update(
                    lambda tx: tx.state.update(universe=selection(h.clock.now, ()))
                )
                h.session.gate.set()
                await asyncio.wait_for(h.runtime.ai_queue.join(), timeout=5)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            state = await h.store.read()
            assert ready(state)["ai"]["verdict"] == "WAIT"
            assert state["trades"] == baseline
            assert not state["orders"] and not h.market.submissions

    asyncio.run(scenario())


def test_refresh_marks_removed_ready_records_wait_and_retires_pending_atomically(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            market = await dynamic_market(h, monkeypatch, ["BTCUSDT", "ETHUSDT"])
            await h.runtime.scan()
            before = await h.store.read()
            bitcoin = [t for t in before["trades"].values() if t["symbol"] == "BTCUSDT"]
            assert bitcoin and all(t["status"] == "PENDING" for t in bitcoin)
            await h.tick(6 * 3600)
            market.tickers[0]["turnover24h"] = "1"
            await h.runtime.refresh_universe()
            state = await h.store.read()
            assert state["universe"]["active_symbols"] == ["ETHUSDT"]
            for rec in state["opportunities"].values():
                if rec["opportunity"]["symbol"] == "BTCUSDT":
                    assert rec["universe_blocked"] is True
                    if rec["opportunity"]["state"] in {"READY", "WAIT"}:
                        assert rec["decision"].startswith("WAIT:")
            for original in bitcoin:
                trade = state["trades"][original["id"]]
                assert trade["original_expires_at"] == original["expires_at"]
                assert trade["expires_at"] == min(original["expires_at"], h.clock.now)
                assert (
                    trade["status"] == "PENDING"
                )  # Replay must decide any historical fill.
                assert trade["stop"] == original["stop"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("universe_min_turnover", 25_000_000),
        ("universe_max_spread_bps", 5),
        ("universe_min_depth", 50_000),
    ],
)
def test_stricter_startup_policy_blocks_old_membership_before_first_refresh(
    harness, monkeypatch, field, value
):
    async def scenario():
        async with harness() as h:
            market = await dynamic_market(h, monkeypatch)
            old = deepcopy((await h.store.read())["universe"])
            h.runtime.config = replace(h.runtime.config, **{field: value})
            tickers_before = market.calls.count("tickers")
            # Model restart configuration loading without running the heavy
            # universe loop: the old committed selection is still fresh.
            await h.runtime.initialize()
            state = await h.store.read()
            assert state["universe"] == old
            assert market.calls.count("tickers") == tickers_before
            required = configuration_policy(h.runtime.config)
            assert all(state["universe_policy"][k] == v for k, v in required.items())
            assert h.runtime._entry_symbols(state, h.clock.now) == []
            # A failed refresh cannot silently restore the old policy's entry
            # permission, nor erase the history needed by the next refresh.
            market.failure = "transport"
            with pytest.raises(RuntimeError):
                await h.runtime.refresh_universe()
            assert (await h.store.read())["universe"] == old
            assert h.runtime._entry_symbols(await h.store.read(), h.clock.now) == []
            market.failure = None
            await h.runtime.refresh_universe()
            state = await h.store.read()
            assert all(state["universe"]["policy"][k] == v for k, v in required.items())
            assert len(h.runtime._entry_symbols(state, h.clock.now)) == 50

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_turnover_usdt", 25_000_000),
        ("max_spread_bps", 5),
        ("min_depth_usdt", 50_000),
        ("target", 40),
    ],
)
def test_execution_reservation_requires_current_desired_universe_policy(
    harness, field, value
):
    async def scenario():
        async with harness("testnet") as h:
            manager, client, op, sizing, version = await exchange_candidate(h)
            required = {**configuration_policy(), field: value}
            await h.store.update(lambda tx: tx.state.update(universe_policy=required))
            assert not manager._reservation_allowed(
                await h.store.read(), op, sizing, version
            )
            assert not await manager.submit(op, sizing, version)
            assert not client.calls and not (await h.store.read())["orders"]

            def refreshed(tx):
                tx.state["universe"]["policy"].update(required)
                tx.state["universe"]["target"] = required["target"]

            await h.store.update(refreshed)
            assert await manager.submit(op, sizing, version)

    asyncio.run(scenario())


def test_target_reduction_trims_weakest_members_and_keeps_turnover_history(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            market = await dynamic_market(h, monkeypatch)
            before = deepcopy((await h.store.read())["universe"])
            ranked = sorted(
                before["active_symbols"],
                key=lambda symbol: (-before["members"][symbol]["score"], symbol),
            )
            assert len(ranked) == 50
            survivor, retired = ranked[0], ranked[-1]
            # Persist both pending and already-open positions. Migration must
            # retire entry eligibility, never force an exit from owned inventory.
            await h.store.update(
                lambda tx: tx.state.update(
                    orders={
                        "open": {
                            "symbol": retired,
                            "status": "OPEN",
                            "desired_stop": 90,
                        },
                        "pending": {
                            "symbol": retired,
                            "status": "PENDING",
                            "desired_stop": 91,
                        },
                    },
                    trades={
                        "pending": {
                            "symbol": retired,
                            "status": "PENDING",
                            "created_at": h.clock.now - 180,
                            "expires_at": h.clock.now + 600,
                            "stop": 90,
                        },
                        "survives": {
                            "symbol": survivor,
                            "status": "PENDING",
                            "created_at": h.clock.now - 180,
                            "expires_at": h.clock.now + 600,
                            "stop": 90,
                        },
                        "open": {"symbol": retired, "status": "OPEN", "stop": 90},
                    },
                )
            )
            h.runtime.config = replace(h.runtime.config, universe_size=40)
            required = configuration_policy(h.runtime.config)
            await h.store.update(lambda tx: tx.state.update(universe_policy=required))
            assert h.runtime._entry_symbols(await h.store.read(), h.clock.now) == []
            # Fail before commit first; migration must not mutate persisted
            # membership/history in preparation for an unvalidated snapshot.
            market.failure = "books"
            with pytest.raises(RuntimeError):
                await h.runtime.refresh_universe()
            assert (await h.store.read())["universe"] == before
            market.failure = None
            await h.runtime.refresh_universe()
            state = await h.store.read()
            after = state["universe"]
            assert after["target"] == 40 and after["policy"]["target"] == 40
            assert after["active_symbols"] == ranked[:40]
            assert set(after["removed"]) == set(ranked[40:])
            assert after["snapshots"] == before["snapshots"]  # Same observation time.
            assert after["day_replacements"] == before["day_replacements"]
            assert len(h.runtime._entry_symbols(state, h.clock.now)) == 40
            assert state["trades"]["pending"]["expires_at"] == h.clock.now
            assert state["trades"]["pending"]["status"] == "PENDING"
            assert state["trades"]["survives"]["expires_at"] == h.clock.now + 600
            assert state["trades"]["open"] == {
                "symbol": retired,
                "status": "OPEN",
                "stop": 90,
            }
            assert state["orders"]["open"]["status"] == "OPEN"
            assert state["orders"]["pending"]["entry_retired_at"] == h.clock.now
            assert state["orders"]["open"]["desired_stop"] == 90

    asyncio.run(scenario())


def test_delisted_owned_symbol_remains_managed_after_metadata_loss_and_restart(
    harness, monkeypatch
):
    async def scenario():
        async with harness("testnet") as h:
            market = await dynamic_market(h, monkeypatch, ["BTCUSDT", "ETHUSDT"])
            manager, client, op, sizing, version = await exchange_candidate(h)
            assert await manager.submit(op, sizing, version)
            link = execution.client_id(op.id)
            await h.tick(1)
            client.fill(link, 1)
            await manager.reconcile(manager.instruments)
            assert (await h.store.read())["orders"][link]["status"] == "OPEN"
            assert (await h.store.read())["orders"][link]["instrument"] == asdict(
                INSTRUMENT
            )
            market.metadata.pop("BTCUSDT")
            market.tickers = [
                row for row in market.tickers if row["symbol"] != "BTCUSDT"
            ]
            await h.runtime.refresh_instruments()
            assert "BTCUSDT" not in h.runtime.instruments
            state = await h.store.read()
            assert "BTCUSDT" not in h.runtime._entry_symbols(state, h.clock.now)
            assert "BTCUSDT" not in h.runtime.service._universe_entries(
                state, h.clock.now
            )
            assert (
                "Fresh instrument rules unavailable: 1 members · entry wait"
                in await h.runtime.service.snapshot("universe")
            )
            # Keep the last universe snapshot fresh with BTC still in it: the
            # loss of current instrument metadata must itself cancel new fills.
            client.pos[0]["stopLoss"] = "0"
            client.calls.clear()
            await h.runtime.reconcile()
            order = (await h.store.read())["orders"][link]
            assert order["status"] == "OPEN" and order["remaining_qty"] == 1
            assert "BTCUSDT" not in manager.instruments
            assert manager.management_instruments["BTCUSDT"] == INSTRUMENT
            assert order["instrument"] == asdict(INSTRUMENT)
            assert any(call[0] == "protect" for call in client.calls)
            assert any(call[0] == "cancel" for call in client.calls)
            assert not any(call[0] == "reduce" for call in client.calls)
            assert float(client.pos[0]["stopLoss"]) == 90
            assert float(client.pos[0]["takeProfit"]) == 140
            assert float(client.pos[0]["size"]) == 1
            assert client.rows[link]["orderStatus"] == "Cancelled"
            assert float(client.rows[link]["leavesQty"]) == 0

            worker = runtime.Runtime(h.runtime.config, h.store, h.session)
            worker.client = h.runtime.client
            worker.execution.client = client
            client.pos[0]["stopLoss"] = "0"
            client.calls.clear()
            try:
                await worker.initialize()
                assert "BTCUSDT" not in worker.instruments
                assert "BTCUSDT" not in worker.execution.instruments
                assert worker.execution.management_instruments["BTCUSDT"] == INSTRUMENT
                recovered = (await h.store.read())["orders"][link]
                assert recovered["status"] == "OPEN"
                assert recovered["remaining_qty"] == recovered["filled_qty"] == 1
                assert recovered["desired_stop"] == 90
                assert any(call[0] == "protect" for call in client.calls)
                assert not any(call[0] == "reduce" for call in client.calls)
                assert float(client.pos[0]["stopLoss"]) == 90
                assert float(client.pos[0]["takeProfit"]) == 140
                assert float(client.pos[0]["size"]) == 1
            finally:
                await worker.cache.close()

    asyncio.run(scenario())


def test_saved_management_instrument_cannot_authorize_a_new_exchange_entry(harness):
    async def scenario():
        async with harness("testnet") as h:
            manager, client, op, sizing, version = await exchange_candidate(h)
            assert await manager.submit(op, sizing, version)
            old_link = execution.client_id(op.id)
            # Delisting cancels an unfilled owned order and reconstructs its
            # management rules. No open position or active order can explain
            # why the next candidate must be rejected.
            await manager.reconcile({})
            assert (await h.store.read())["orders"][old_link]["status"] == "CANCELLED"
            assert manager.management_instruments["BTCUSDT"] == INSTRUMENT
            assert manager.instruments == {}
            new_op = replace(op, id="new-candidate-after-delisting")

            def approve(tx):
                tx.state["opportunities"][new_op.id] = {
                    "opportunity": new_op.to_dict(),
                    "ai": {
                        "verdict": "APPROVE",
                        "evidence_hash": "a" * 64,
                        "candidate_fingerprint": execution.candidate_fingerprint(
                            new_op
                        ),
                        "context_fingerprint": execution.context_fingerprint(
                            tx.state, new_op.symbol, h.clock.now
                        ),
                    },
                }

            await h.store.update(approve)
            state = await h.store.read()
            assert (
                state["account"]["positions"] == state["account"]["open_orders"] == []
            )
            assert state["account"]["blockers"] == []
            assert not manager._reservation_allowed(state, new_op, sizing, version)
            client.calls.clear()
            assert not await manager.submit(new_op, sizing, version)
            assert client.calls == []
            assert set((await h.store.read())["orders"]) == {old_link}
            # Freshly listed metadata is the only changed condition in the
            # positive control; all candidate/account/risk evidence is identical.
            manager.instruments = {"BTCUSDT": INSTRUMENT}
            assert manager._reservation_allowed(state, new_op, sizing, version)
            assert await manager.submit(new_op, sizing, version)
            assert any(call[0] == "submit" for call in client.calls)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {**asdict(INSTRUMENT), "symbol": "ETHUSDT"},
        {**asdict(INSTRUMENT), "qty_step": 0},
        {**asdict(INSTRUMENT), "tick_size": "invalid"},
    ],
)
def test_missing_current_and_invalid_saved_instrument_blocks_management_without_guessing(
    harness, bad
):
    async def scenario():
        async with harness("testnet") as h:
            manager, client, op, sizing, version = await exchange_candidate(h)
            assert await manager.submit(op, sizing, version)
            link = execution.client_id(op.id)
            await h.tick(1)
            client.fill(link, 1)
            await manager.reconcile(manager.instruments)
            await h.store.update(
                lambda tx: tx.state["orders"][link].update(instrument=deepcopy(bad))
            )
            before = deepcopy((await h.store.read())["orders"][link])
            client.pos[0]["stopLoss"] = "0"
            client.calls.clear()
            cold = execution.ExecutionManager(client, h.store, "testnet")
            await cold.reconcile({})
            state = await h.store.read()
            assert state["account"]["blockers"]
            assert any("metadata" in reason for reason in state["account"]["blockers"])
            assert cold.instruments == cold.management_instruments == {}
            assert (
                state["orders"][link]["remaining_qty"] == before["remaining_qty"] == 1
            )
            assert state["orders"][link]["instrument"] == bad
            assert not any(
                call[0] in {"protect", "reduce", "submit"} for call in client.calls
            )

    asyncio.run(scenario())
