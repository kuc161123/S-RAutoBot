"""Shadow recovery isolation and worker exit semantics; all services are offline."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from apex_bot import __main__ as entry
from apex_bot import runtime as runtime_module
from apex_bot.config import Config
from apex_bot.models import Candle
from apex_bot.storage import NotLeader

from .test_runtime import INSTRUMENT, NOW, events, harness


async def two_symbols(h, *, close=False):
    await h.runtime.scan()
    trade = deepcopy(next(iter((await h.store.read())["trades"].values())))
    other = deepcopy(trade)
    other.update(id="baseline_shadow:offline-alt", symbol="AAAUSDT")
    h.runtime.config = replace(h.config, symbols=("AAAUSDT", "BTCUSDT"))
    h.runtime.instruments["AAAUSDT"] = replace(INSTRUMENT, symbol="AAAUSDT")
    await h.store.update(lambda tx: tx.state["trades"].update({other["id"]: other}))
    price = trade["limit"]
    h.market.intraday = [Candle(int(NOW * 1000), price, price + 1, price - 1, price)]
    if close:
        target = trade["target2"]
        h.market.intraday.append(
            Candle(int((NOW + 180) * 1000), price + 1, target + 1, price - 0.5, target)
        )
    await h.tick(360 if close else 180)
    return other["id"], trade["id"]


@pytest.mark.parametrize("failure_stage", ["fetch", "replay"])
def test_simulation_failure_keeps_other_symbol_moving_and_retries_cursor(
    harness, monkeypatch, failure_stage, caplog
):
    async def scenario():
        async with harness() as h:
            failed_id, healthy_id = await two_symbols(h)
            before = (await h.store.read())["trades"]
            fail = True
            fetched = []
            original_fetch, original_advance = (
                h.runtime.client.candle_range,
                runtime_module.advance,
            )

            async def fetch(symbol, interval, start, end):
                fetched.append((symbol, start, end))
                if fail and symbol == "AAAUSDT" and failure_stage == "fetch":
                    raise TimeoutError("private-provider-error-must-not-be-logged")
                return await original_fetch(symbol, interval, start, end)

            def advance(trade, *args, **kwargs):
                if fail and trade["symbol"] == "AAAUSDT" and failure_stage == "replay":
                    raise ValueError("private-provider-error-must-not-be-logged")
                return original_advance(trade, *args, **kwargs)

            monkeypatch.setattr(h.runtime.client, "candle_range", fetch)
            monkeypatch.setattr(runtime_module, "advance", advance)
            with pytest.raises(
                RuntimeError, match="simulation: 1 symbol updates failed"
            ):
                await h.runtime.simulate()
            state = await h.store.read()
            assert state["trades"][failed_id] == before[failed_id]
            assert state["trades"][healthy_id]["status"] == "OPEN"
            assert state["trades"][healthy_id]["last_bar"] == int(NOW * 1000)
            assert state["health"]["simulation_AAAUSDT_error"] in {
                "TimeoutError",
                "ValueError",
            }
            assert "simulation_BTCUSDT_error" not in state["health"]
            assert "private-provider-error" not in caplog.text
            assert any(
                payload["component"] == "simulation_AAAUSDT"
                for _, _, payload in await events(h.store, "health_issue")
            )
            healthy = deepcopy(state["trades"][healthy_id])
            fail = False
            await h.runtime.simulate()
            state = await h.store.read()
            assert state["trades"][failed_id]["status"] == "OPEN"
            assert state["trades"][healthy_id] == healthy
            assert "simulation_AAAUSDT_error" not in state["health"]
            attempts = [row for row in fetched if row[0] == "AAAUSDT"]
            assert len(attempts) == 2 and attempts[0] == attempts[1]
            assert h.market.submissions == []

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_stage", ["fetch", "settlement"])
def test_funding_failure_does_not_starve_other_symbol_or_double_count_recovery(
    harness, monkeypatch, failure_stage
):
    async def scenario():
        async with harness() as h:
            failed_id, healthy_id = await two_symbols(h, close=True)
            await h.runtime.simulate()
            before = (await h.store.read())["trades"]
            assert all(t["status"] == "CLOSED" for t in before.values())
            fail, fetched = True, []
            original_apply = runtime_module.apply_funding

            async def history(symbol, start, end):
                fetched.append((symbol, start, end))
                if fail and symbol == "AAAUSDT" and failure_stage == "fetch":
                    raise TimeoutError("offline funding unavailable")
                return [
                    {
                        "symbol": symbol,
                        "fundingRate": ".0001",
                        "fundingRateTimestamp": str(int((NOW + 240) * 1000)),
                    }
                ]

            def apply(trade, *args, **kwargs):
                if (
                    fail
                    and trade["symbol"] == "AAAUSDT"
                    and failure_stage == "settlement"
                ):
                    raise ValueError("offline settlement unavailable")
                return original_apply(trade, *args, **kwargs)

            monkeypatch.setattr(h.runtime.client, "funding_history", history)
            monkeypatch.setattr(runtime_module, "apply_funding", apply)
            with pytest.raises(RuntimeError, match="funding: 1 symbol updates failed"):
                await h.runtime.funding()
            state = await h.store.read()
            assert state["trades"][failed_id] == before[failed_id]
            healthy = deepcopy(state["trades"][healthy_id])
            assert healthy["funding_complete"] and len(healthy["funding_ids"]) == 1
            assert healthy["funding"] < 0
            assert "funding_AAAUSDT_error" in state["health"]
            fail = False
            await h.runtime.funding()
            state = await h.store.read()
            assert state["trades"][failed_id]["funding_complete"]
            assert len(state["trades"][failed_id]["funding_ids"]) == 1
            assert state["trades"][failed_id]["funding"] == healthy["funding"]
            assert state["trades"][healthy_id] == healthy
            assert "funding_AAAUSDT_error" not in state["health"]
            attempts = [row for row in fetched if row[0] == "AAAUSDT"]
            assert len(attempts) == 2 and attempts[0] == attempts[1]
            assert len([row for row in fetched if row[0] == "BTCUSDT"]) == 1
            await h.runtime.funding()
            assert (await h.store.read())["trades"] == state["trades"]

    asyncio.run(scenario())


@pytest.mark.parametrize("component", ["simulation", "funding"])
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, NotLeader])
def test_symbol_isolation_does_not_swallow_shutdown_or_lost_authority(
    harness, monkeypatch, component, error_type
):
    async def scenario():
        async with harness() as h:
            await two_symbols(h, close=component == "funding")
            if component == "funding":
                await h.runtime.simulate()
            before = (await h.store.read())["trades"]
            called = []

            async def unavailable(symbol, *args):
                called.append(symbol)
                raise error_type()

            monkeypatch.setattr(
                h.runtime.client,
                "candle_range" if component == "simulation" else "funding_history",
                unavailable,
            )
            with pytest.raises(error_type):
                await getattr(
                    h.runtime, "simulate" if component == "simulation" else component
                )()
            assert called == ["AAAUSDT"]
            assert (await h.store.read())["trades"] == before

    asyncio.run(scenario())


@pytest.mark.parametrize("lease_result", ["lost", "timeout", "stop"])
def test_worker_drains_and_distinguishes_fatal_lease_from_requested_stop(
    harness, monkeypatch, lease_result
):
    async def scenario():
        async with harness() as h:
            renewed = asyncio.Event()

            async def lease():
                renewed.set()
                if lease_result == "timeout":
                    raise TimeoutError("offline renewal timeout")
                return lease_result != "lost"

            async def idle():
                await asyncio.Event().wait()

            with monkeypatch.context() as patch:
                patch.setattr(h.store, "lease", lease)
                for name in (
                    "refresh_context",
                    "refresh_risk",
                    "refresh_instruments",
                    "refresh_universe",
                    "scan",
                    "simulate",
                    "funding",
                    "reconcile",
                    "poll",
                    "deliver",
                    "research",
                    "review_worker",
                ):
                    patch.setattr(h.runtime, name, idle)
                patch.setattr(h.runtime.references, "refresh", idle)
                cache_close, store_close = AsyncMock(), AsyncMock()
                patch.setattr(h.runtime.cache, "close", cache_close)
                patch.setattr(h.store, "close", store_close)
                task = asyncio.create_task(h.runtime.run())
                try:
                    await asyncio.wait_for(renewed.wait(), 2)
                    if lease_result == "stop":
                        h.runtime.stopping.set()  # Same event as the SIGTERM handler.
                        await asyncio.wait_for(task, 2)
                    else:
                        with pytest.raises(
                            RuntimeError, match="worker restart required"
                        ):
                            await asyncio.wait_for(task, 2)
                    assert all(t.done() for t in h.runtime.worker_tasks)
                    cache_close.assert_awaited_once()
                    store_close.assert_awaited_once()
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_requested_stop_during_lease_failure_is_not_a_fatal_restart(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:

            async def stopped_lease():
                h.runtime.stopping.set()
                raise TimeoutError("shutdown overlapped renewal")

            monkeypatch.setattr(h.store, "lease", stopped_lease)
            with pytest.raises(TimeoutError):
                await h.runtime.renew()
            assert not h.runtime._lease_failed

    asyncio.run(scenario())


@pytest.mark.parametrize("fatal", [False, True])
def test_entrypoint_exit_status_for_drained_worker(monkeypatch, fatal):
    async def served(config):
        if fatal:
            raise RuntimeError("Cloud lease failed; worker restart required")

    monkeypatch.setattr(entry.Config, "from_env", lambda: Config())
    monkeypatch.setattr(entry, "serve", served)
    monkeypatch.setattr(entry.sys, "argv", ["apex"])
    if fatal:
        with pytest.raises(SystemExit) as error:
            entry.main()
        assert error.value.code == 1
    else:
        assert entry.main() is None
