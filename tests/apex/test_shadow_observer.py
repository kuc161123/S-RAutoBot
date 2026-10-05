"""Observer isolation and conservative, restartable outcome accounting."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from apex_bot.engine import DAY, H4, VERSION as ENGINE_VERSION
from apex_bot.models import Candle, Instrument, Opportunity
from apex_bot.shadow_observer import ShadowObserver, TABLE, VERSION, MAX_PRICE_BARS
from apex_bot.storage import Store, NotLeader

NOW = 100 * DAY + H4
LOSSES = dict(daily_loss_pct=0, weekly_loss_pct=0, drawdown_pct=0)


def opportunity(identity="frozen", symbol="TESTUSDT", side="Buy"):
    long = side == "Buy"
    return Opportunity(
        identity,
        symbol,
        side,
        "1" if long else "1S",
        "READY",
        100,
        94 if long else 106,
        112 if long else 88,
        130 if long else 70,
        96 if long else 104,
        99 if long else 101,
        NOW - H4,
        NOW + 2 * H4,
        "SWING_BREAK",
        dict(
            engine_version=ENGINE_VERSION,
            evidence_id=identity,
            trigger_evidence_id="trigger",
            structural_valid=True,
            data_valid=True,
            terminal_status=None,
            trigger_kind="SWING_BREAK",
            trigger_closed_at=NOW,
            daily_closed_at=100 * DAY,
            execution_closed_at=NOW,
            as_of=NOW,
            zone_low=98,
            zone_high=102,
            entry_zone_low=98,
            entry_zone_high=102,
            trigger_price=100,
            risk_multiplier=1,
        ),
    )


class Client:
    def __init__(self):
        self.calls = []
        self.funding_calls = []
        self.fail_symbols = set()
        self.funding_failure = False
        self.gap = False
        self.stop = True
        self.rates = []

    async def candle_range(self, symbol, interval, start, end):
        self.calls.append((symbol, interval, start, end))
        if symbol in self.fail_symbols:
            raise RuntimeError("symbol temporarily unavailable")
        bars = [Candle(t, 100, 101, 99, 100, 10) for t in range(start, end, 180000)]
        if self.stop and len(bars) > 1:
            bars[1] = replace(bars[1], low=90, high=110)
        return bars[1:] if self.gap else bars

    async def funding_history(self, symbol, start, end):
        self.funding_calls.append((symbol, start, end))
        if self.funding_failure:
            raise RuntimeError("funding temporarily unavailable")
        return [
            r
            for r in self.rates
            if r["symbol"] == symbol and start <= int(r["fundingRateTimestamp"]) <= end
        ]


class Cache:
    async def candles(self, client, symbol, interval, limit, now):
        seconds = DAY if interval == "D" else H4
        origin = (NOW // seconds - 20) * seconds
        end = int(now // seconds) * seconds
        return [
            Candle(t * 1000, 100, 102, 98, 100, 10) for t in range(origin, end, seconds)
        ]


async def setup(path=":memory:"):
    store = Store(sqlite_path=path)
    await store.initialize()
    assert await store.lease()
    client = Client()
    observer = ShadowObserver(store, client, Cache())
    await observer.initialize()
    return store, client, observer


async def consider(observer, op=None, **kwargs):
    op = op or opportunity()
    args = dict(
        op=op,
        inst=Instrument(op.symbol, 0.01, 0.01, 100000, 0.01, 5),
        equity=10000,
        profile="cautious",
        risk_pct=None,
        losses=LOSSES,
        funding_8h=0,
        spread=0.01,
        now=NOW,
        rejection={"allowed": False, "reasons": ["SYMBOL_ALREADY_EXPOSED"]},
    )
    args.update(kwargs)
    return await observer.consider(**args)


async def records(store):
    def read(cur):
        cur.execute(f"SELECT id,payload,trade,error FROM {TABLE} ORDER BY id")
        return [
            dict(id=i, payload=json.loads(p), trade=json.loads(t), error=e)
            for i, p, t, e in cur.fetchall()
        ]

    return await asyncio.to_thread(store._run, read)


def test_insertion_and_resolve_never_touch_shared_ledgers_or_notifications():
    async def run():
        store, client, observer = await setup()
        before = await store.read()
        assert await consider(observer)
        captured = (await records(store))[0]["payload"]
        assert captured["bypass"] == "existing portfolio exposures only"
        assert captured["independent_assessment"]["allowed"]
        assert captured["original_price_r"] == pytest.approx(4.05 * 6)
        await observer.resolve(NOW + 360)
        row = (await records(store))[0]
        assert row["payload"] == captured
        assert (
            row["trade"]["arm"] == "baseline_shadow"
        )  # Stored only in observer table.
        assert row["trade"]["observer"] == VERSION
        assert await store.read() == before
        assert await store.pending_notifications() == []
        result = await observer.summary()
        assert (
            result["total"],
            result["closed"],
            result["complete_closed"],
            result["losses"],
        ) == (1, 1, 1, 1)
        assert result["net_r"] < -1  # Fees and adverse stop slippage included.
        assert result["net_r"] == pytest.approx(
            row["trade"]["net_pnl"] / captured["original_price_r"]
        )
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "reason",
    [
        "POSITION_CAP",
        "BUCKET_POSITION_CAP",
        "HEAT_CAP",
        "BUCKET_HEAT_CAP",
        "GROSS_NOTIONAL_CAP",
        "ROUNDED_RISK_CAP",
        "ROUNDED_BUCKET_HEAT_CAP",
    ],
)
def test_capacity_reasons_still_require_successful_independent_assessment(reason):
    async def run():
        store, _, observer = await setup()
        assert await consider(observer, rejection=[reason])
        bad = replace(opportunity("bad"), target1=102)
        assert not await consider(observer, bad, rejection=[reason])
        assert (await observer.summary())["total"] == 1
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "reason",
    [
        "RR_TARGET1",
        "ADVERSE_FUNDING",
        "SPREAD_UNKNOWN",
        "DAILY_LOSS_HALT",
        "EXPOSURES_UNKNOWN",
        "EXPOSURE_RISK_OR_NOTIONAL_UNKNOWN",
        "SINGLE_NOTIONAL_CAP",
        "QUANTITY_BELOW_MINIMUM",
        "UNKNOWN_REASON",
    ],
)
def test_mixed_noncapacity_rejection_is_never_bypassed(reason):
    async def run():
        store, _, observer = await setup()
        assert not await consider(observer, rejection=["POSITION_CAP", reason])
        assert (await observer.summary())["total"] == 0
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "changes",
    [
        {"funding_8h": None},
        {"spread": None},
        {"losses": dict(LOSSES, daily_loss_pct=2)},
        {"losses": {}},
        {"now": NOW + 100000},
        {"rejection": {"allowed": True, "reasons": ["POSITION_CAP"]}},
    ],
)
def test_original_noncapacity_inputs_are_preserved(changes):
    async def run():
        store, _, observer = await setup()
        assert not await consider(observer, **changes)
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("style", ["monitored_zone", "resting_limit"])
def test_offline_entry_experiments_cannot_enter_cloud_observer(style):
    async def run():
        store, _, observer = await setup()
        op = opportunity()
        op = replace(op, evidence=dict(op.evidence, entry_style=style))
        assert not await consider(observer, op)
        await store.close()

    asyncio.run(run())


def test_restart_and_concurrent_insert_are_exactly_once_per_candidate(tmp_path):
    async def run():
        path = str(tmp_path / "observer.sqlite")
        store, _, observer = await setup(path)
        results = await asyncio.gather(*(consider(observer) for _ in range(8)))
        assert sum(results) == 1
        captured = await records(store)
        await store.close()
        store, _, observer = await setup(path)
        assert not await consider(observer, equity=11000)
        assert await records(store) == captured
        # Different plans on the same symbol can coexist without a position cap.
        assert await consider(observer, opportunity("second"))
        assert (await observer.summary())["pending"] == 2
        await store.close()

    asyncio.run(run())


def test_missing_first_candle_retains_cursor_and_retries_without_fabricated_fill():
    async def run():
        store, client, observer = await setup()
        assert await consider(observer)
        client.gap = True
        await observer.resolve(NOW + 360)
        row = (await records(store))[0]
        assert row["trade"]["last_bar"] is None
        assert row["trade"]["status"] == "PENDING"
        assert row["trade"]["data_gap"]
        assert (await observer.summary())["complete_closed"] == 0
        client.gap = False
        await observer.resolve(NOW + 390)
        assert client.calls[0][2] == client.calls[1][2] == NOW * 1000
        assert (await observer.summary())["complete_closed"] == 1
        assert (await records(store))[0]["error"] is None
        await store.close()

    asyncio.run(run())


def test_funding_failure_preserves_price_progress_and_excludes_provisional_outcome():
    async def run():
        store, client, observer = await setup()
        assert await consider(observer)
        client.funding_failure = True
        await observer.resolve(NOW + 360)
        result = await observer.summary()
        assert (
            result["closed"],
            result["incomplete_closed"],
            result["complete_closed"],
        ) == (1, 1, 0)
        assert result["wr"] is None and result["net_r"] == 0
        client.funding_failure = False
        client.rates = [
            {
                "symbol": "TESTUSDT",
                "fundingRateTimestamp": str((NOW + 270) * 1000),
                "fundingRate": "0.001",
            }
        ]
        await observer.resolve(NOW + 390)
        trade = (await records(store))[0]["trade"]
        assert len(client.calls) == 1  # Closed prices are not replayed after restart.
        assert trade["funding"] == pytest.approx(-trade["qty"] * trade["entry"] * 0.001)
        assert trade["funding_complete"]
        assert (await observer.summary())["complete_closed"] == 1
        final = deepcopy(trade)
        await observer.resolve(NOW + 420)
        assert (await records(store))[0]["trade"] == final
        await store.close()

    asyncio.run(run())


def test_symbol_failure_does_not_block_other_symbol_outcomes():
    async def run():
        store, client, observer = await setup()
        assert await consider(observer, opportunity("a", "AAAUSDT"))
        assert await consider(observer, opportunity("b", "BBBUSDT"))
        client.fail_symbols.add("AAAUSDT")
        await observer.resolve(NOW + 360)
        result = await observer.summary()
        assert (
            result["pending"] == result["complete_closed"] == result["data_issues"] == 1
        )
        client.fail_symbols.clear()
        await observer.resolve(NOW + 390)
        assert (await observer.summary())["complete_closed"] == 2
        await store.close()

    asyncio.run(run())


def test_fair_batch_does_not_starve_candidates_after_first_twenty():
    async def run():
        store, client, observer = await setup()
        for i in range(25):
            assert await consider(observer, opportunity(f"{i:03}"))
        client.fail_symbols.add("TESTUSDT")
        first = await observer.resolve(NOW + 360)
        second = await observer.resolve(NOW + 360)
        third = await observer.resolve(NOW + 360)
        assert (first["processed"], second["processed"], third["processed"]) == (
            20,
            5,
            0,
        )
        assert (
            len(client.calls) == 2
        )  # Shared market reads per symbol, not per candidate.
        assert all(row["error"] for row in await records(store))
        await store.close()

    asyncio.run(run())


def test_recovery_price_request_is_bounded_and_resumes_exact_cursor():
    async def run():
        store, client, observer = await setup()
        client.stop = False
        assert await consider(observer)
        await observer.resolve(NOW + 3 * DAY)
        first = client.calls[-1]
        trade = (await records(store))[0]["trade"]
        assert first[3] - first[2] == MAX_PRICE_BARS * 180000
        assert trade["last_bar"] == first[3] - 180000
        await observer.resolve(NOW + 3 * DAY + 30)
        assert client.calls[-1][2] == first[3]
        await store.close()

    asyncio.run(run())


def test_empty_and_short_outcomes_have_correct_summary():
    async def run():
        store, _, observer = await setup()
        initial = await observer.summary()
        assert (
            initial["total"] == 0
            and initial["wr"] is None
            and initial["mean_r"] is None
        )
        assert await consider(observer, opportunity(side="Sell"))
        await observer.resolve(NOW + 360)
        result = await observer.summary()
        assert result["complete_closed"] == result["losses"] == 1
        await store.close()

    asyncio.run(run())


def test_fenced_worker_cannot_insert_or_resolve_after_lease_loss():
    async def run():
        store, client, observer = await setup()
        await store.lease(ttl=-1)
        with pytest.raises(NotLeader):
            await consider(observer)
        with pytest.raises(NotLeader):
            await observer.resolve(NOW)
        assert not client.calls
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_complete_winner_includes_partial_targets_and_fixed_price_r(side):
    async def run():
        store, client, observer = await setup()
        assert await consider(observer, opportunity(side=side))

        async def prices(symbol, interval, start, end):
            result = [
                Candle(start, 100, 101, 99, 100, 10),
                Candle(start + 180000, 100, 114, 99, 112, 10),
                Candle(start + 360000, 112, 131, 111, 130, 10),
            ]
            if side == "Sell":
                result = [
                    replace(
                        b,
                        open=200 - b.open,
                        high=200 - b.low,
                        low=200 - b.high,
                        close=200 - b.close,
                    )
                    for b in result
                ]
            return result

        client.candle_range = prices
        await observer.resolve(NOW + 540)
        row = (await records(store))[0]
        result = await observer.summary()
        assert result["wins"] == result["complete_closed"] == 1
        assert result["losses"] == 0 and result["wr"] == 100
        assert row["trade"]["tp1_done"]
        assert len(row["trade"]["fills"]) == 3
        assert result["net_r"] > 3
        assert row["trade"]["original_price_r"] == row["payload"]["original_price_r"]
        await store.close()

    asyncio.run(run())


def test_unfilled_expiry_is_not_counted_as_loss_or_complete_outcome():
    async def run():
        store, client, observer = await setup()
        assert await consider(observer)

        async def prices(symbol, interval, start, end):
            return [
                Candle(t, 110, 111, 109, 110, 10) for t in range(start, end, 180000)
            ]

        client.candle_range = prices
        await observer.resolve(NOW + DAY)
        result = await observer.summary()
        assert result["expired"] == 1
        assert result["closed"] == result["complete_closed"] == result["losses"] == 0
        assert result["wr"] is None and result["unfinished"] == 0
        assert client.funding_calls == []
        assert (await observer.resolve(NOW + DAY + 30))["processed"] == 0
        await store.close()

    asyncio.run(run())
