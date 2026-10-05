"""Offline differential checks of actual management transitions and prefixes."""

from bisect import bisect_right
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random

import pytest

from apex_bot import simulation
from apex_bot.engine import DAY, H4, confirmed_zigzag
from apex_bot.models import Candle, Opportunity
from apex_bot.replay_management import _ManagementCache

HOUR = 3600


def histories(prices, side="Buy"):
    prices = prices if side == "Buy" else [200 - p for p in prices]
    hourly = [Candle(i * HOUR * 1000, p, p + .2, p - .2, p, 1)
              for i, p in enumerate(prices)]

    def aggregate(n):
        return [Candle(
            hourly[i].open_time, hourly[i].open,
            max(b.high for b in hourly[i:i + n]),
            min(b.low for b in hourly[i:i + n]),
            hourly[i + n - 1].close, n,
        ) for i in range(0, len(hourly) - n + 1, n)]

    return hourly, aggregate(24), aggregate(4)


def new_trade(created, side="Buy", entry=100):
    sign = 1 if side == "Buy" else -1
    op = Opportunity(
        "management", "TESTUSDT", side, "1", "READY", entry,
        entry * (1 - sign * .10), entry * (1 + sign * .10),
        entry * (1 + sign * .25), entry * (1 - sign * .05), entry,
        created, created + 30 * DAY, "SWING_BREAK",
        dict(trigger_closed_at=created, zone_low=entry * .95, zone_high=entry * 1.05),
    )
    return simulation.create_trade(op, dict(
        allowed=True, entry=op.entry, stop=op.stop, target1=op.target1,
        target2=op.target2, qty=5, qty_step=1, risk_cash=entry * .5,
        notional=entry * 5,
    ), "acceptance_replay", created)


def compare_path(hourly, daily, execution, initial):
    cache = _ManagementCache(daily, execution)
    fast, slow = deepcopy(initial), deepcopy(initial)
    d_times = [b.open_time / 1000 + DAY for b in daily]
    e_times = [b.open_time / 1000 + H4 for b in execution]
    snapshots = []
    for bar in hourly:
        if bar.open_time / 1000 < initial["created_at"]:
            continue
        now = bar.open_time / 1000 + HOUR
        kwargs = dict(
            interval_ms=HOUR * 1000,
            daily=daily[:bisect_right(d_times, now)],
            execution=execution[:bisect_right(e_times, now)],
        )
        fast = cache.advance(fast, [bar], now, **kwargs)
        slow = simulation.advance(slow, [bar], now, **kwargs)
        assert fast == slow, now
        assert not fast.get("data_error") and not fast.get("data_gap")
        snapshots.append(fast)
    assert cache.pivot_hits > 0 and cache.pivot_builds == 1
    assert any(t["status"] == "OPEN" for t in snapshots)
    return snapshots


@pytest.mark.parametrize("side", ["Buy", "Sell"])
@pytest.mark.parametrize("outcome", ["trail", "daily", "timeout", "target", "open"])
def test_hourly_open_paths_match_every_field(side, outcome):
    prices = [100] * (22 * 24)
    if outcome == "trail":
        prices = [p for p in [100] * 14 + [100, 116, 112, 118, 114, 119, 115, 118, 111]
                  for _ in range(4)]
    elif outcome == "daily":
        prices = [100] * (3 * 24)
        prices[-1] = 94.5
    elif outcome == "target":
        prices = [100] * 60 + [112] * 4 + [126] * 4
    elif outcome == "open":
        prices[60:64] = [106] * 4
    hourly, daily, execution = histories(prices, side)
    snapshots = compare_path(hourly, daily, execution, new_trade(14 * H4, side))
    last = snapshots[-1]
    if outcome == "open":
        assert last["status"] == "OPEN" and last["favorable_4h_close"]
    else:
        assert last["status"] == "CLOSED"
        assert last["exit_reason"] == dict(
            trail="STOP", daily="DAILY_INVALIDATION", timeout="TIMEOUT", target="TP2"
        )[outcome]
    if outcome == "trail":
        assert last["tp1_done"] and last["last_trailing_pivot"]
        assert any(e["reason"] == "STRUCTURAL_TRAIL" for e in last["management_events"])
    if outcome == "timeout":
        assert any(e["reason"] == "TIME_REVIEW_10D" for e in last["management_events"])


@pytest.mark.parametrize("seed", range(5))
def test_precomputed_pivots_equal_every_closed_prefix_and_backwards_clock(seed):
    rng = random.Random(seed)
    execution = []
    price = 100
    for i in range(120):
        previous = price
        price += rng.choice([-4, -2, 0, 0, 2, 4])
        execution.append(Candle(i * H4 * 1000, previous,
                                max(previous, price) + 3,
                                min(previous, price) - 3, price, 1))
    cache = _ManagementCache([], execution)
    assert confirmed_zigzag(execution, H4, 120 * H4, atr_multiple=1)
    counts = list(range(121)) + list(reversed(range(121)))
    for count in counts:
        now = count * H4 + HOUR
        bars = execution[:count]
        cache._closed_bars(bars, H4 * 1000, now)
        pivots = cache._pivots(bars, H4, now, atr_multiple=1)
        assert pivots == confirmed_zigzag(bars, H4, now, atr_multiple=1)
        assert all(p.available_at <= now for p in pivots)
        pivots.clear()  # returned containers cannot corrupt future snapshots
    assert cache.pivot_builds == 1


def test_changed_future_and_interleaved_caches_cannot_change_earlier_trade():
    hourly, daily, execution = histories([100] * 72 + [116, 112, 118, 114] * 8)
    changed = [replace(b, high=b.high * 2, low=b.low / 2, close=b.close * 1.5)
               if b.open_time >= 72 * HOUR * 1000 else b for b in execution]
    a, b = _ManagementCache(daily, execution), _ManagementCache(daily, changed)
    original = dict(simulation.advance.__globals__)
    trade = new_trade(14 * H4)
    now = 64 * HOUR
    kwargs = dict(interval_ms=HOUR * 1000, daily=daily[:2], execution=execution[:16])
    expected = simulation.advance(trade, hourly[56:64], now, **kwargs)
    assert a.advance(trade, hourly[56:64], now, **kwargs) == expected
    assert b.advance(trade, hourly[56:64], now, **kwargs) == expected
    assert all(simulation.advance.__globals__[k] is v for k, v in original.items())
    assert a.advance.__globals__ is not b.advance.__globals__


@pytest.mark.parametrize("fault", [
    "mode", "legacy", "origin", "quantity", "clock", "interval", "price_gap",
    "daily_gap", "execution_gap", "short_warmup", "changed_prefix", "future",
    "nan", "unaligned", "bad_timestamp", "equal_bool_timestamp",
    "negative_timestamp", "missing_close",
])
def test_cache_keeps_original_trade_and_history_guards(fault):
    hourly, daily, execution = histories([100] * (6 * 24))
    cache = _ManagementCache(daily, execution)
    now = 3 * DAY
    kwargs = dict(interval_ms=HOUR * 1000, daily=daily[:3], execution=execution[:18])
    trade = cache.advance(new_trade(14 * H4), hourly[56:72], now, **kwargs)
    assert trade["status"] == "OPEN"
    now += HOUR
    bars = [hourly[72]]
    if fault == "mode":
        trade["management_mode"] = "price_only"
    elif fault == "legacy":
        trade["management_mode"] = None
    elif fault == "origin":
        trade["execution_origin"] = H4 * 1000
    elif fault == "quantity":
        trade["remaining"] = True
    elif fault == "clock":
        now -= 2 * HOUR
    elif fault == "interval":
        kwargs["interval_ms"] = 180000
    elif fault == "price_gap":
        bars = [hourly[73]]
        now += HOUR
    elif fault == "daily_gap":
        kwargs["daily"] = [daily[0], daily[2]]
    elif fault == "execution_gap":
        kwargs["execution"] = execution[:15] + execution[16:18]
    elif fault == "short_warmup":
        kwargs["execution"] = execution[5:18]
    elif fault == "changed_prefix":
        kwargs["execution"] = execution[:17] + [replace(execution[17], close=100.1)]
    elif fault == "future":
        kwargs["execution"] = execution
    elif fault in {"nan", "unaligned", "bad_timestamp", "equal_bool_timestamp", "negative_timestamp"}:
        change = dict(nan={"volume": float("nan")}, unaligned={"open_time": 1},
                      bad_timestamp={"open_time": True}, equal_bool_timestamp={"open_time": False},
                      negative_timestamp={"open_time": -1})
        kwargs["execution"] = [replace(execution[0], **change[fault])] + execution[1:18]
    elif fault == "missing_close":
        now = 4 * DAY
        bars = hourly[72:96]
    slow = simulation.advance(trade, bars, now, **kwargs)
    assert cache.advance(trade, bars, now, **kwargs) == slow
    if fault not in {"changed_prefix", "future"}:
        assert slow.get("data_error") or slow.get("data_gap")


@pytest.mark.parametrize("fault", ["ohlc", "gap", "timestamp"])
def test_bad_future_falls_back_without_contaminating_valid_earlier_prefix(fault):
    hourly, daily, execution = histories([100] * 96)
    if fault == "gap":
        execution.pop(20)
    else:
        execution[-1] = replace(execution[-1], **(
            {"high": 1} if fault == "ohlc" else {"open_time": "bad"}
        ))
    cache = _ManagementCache(daily, execution)
    trade = new_trade(14 * H4)
    kwargs = dict(interval_ms=HOUR * 1000, daily=daily[:3], execution=execution[:18])
    expected = simulation.advance(trade, hourly[56:72], 3 * DAY, **kwargs)
    assert expected["status"] == "OPEN"
    assert cache.advance(trade, hourly[56:72], 3 * DAY, **kwargs) == expected
    kwargs["execution"] = execution
    assert cache.advance(trade, hourly[56:96], 4 * DAY, **kwargs) == simulation.advance(
        trade, hourly[56:96], 4 * DAY, **kwargs
    )


def test_local_real_history_open_management_sample():
    """Bounded 48-hour sample; no downloads or full historical strategy run."""
    path = Path(__file__).resolve().parents[2] / "cache_ew_1h/BTCUSDT.parquet"
    if not path.exists():
        pytest.skip("optional local BTC cache unavailable")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from apex_bot.replay import _source

    daily, execution, hourly, start, _, _ = _source(path, 2, "2026-05-25T00:00:00Z")
    trade = new_trade(start, entry=hourly[1].open)
    # Wide levels keep a real OPEN path long enough to exercise many 4H closes.
    trade.update(stop=trade["limit"] * .5, target1=trade["limit"] * 1.5,
                 target2=trade["limit"] * 2, invalidation=trade["limit"] * .6)
    snapshots = compare_path(hourly, daily, execution, trade)
    assert sum(t["status"] == "OPEN" for t in snapshots) >= 24
