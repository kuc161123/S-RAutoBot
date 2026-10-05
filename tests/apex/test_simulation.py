"""Offline fixtures only: no client, store, network or shared test state."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from apex_bot.engine import DAY, H4
from apex_bot.models import Candle, Opportunity
from apex_bot.simulation import (
    FEE_RATE,
    INTERVAL_MS,
    SLIPPAGE,
    advance,
    apply_funding,
    comparison,
    create_trade,
    summary,
    MONITORED_HALF_SPREAD,
)

STEP = INTERVAL_MS // 1000


def new_trade(
    side="Buy",
    qty=5,
    qty_step=1,
    created=0,
    expiry=None,
    arm="baseline_shadow",
    identity="p1",
):
    long = side == "Buy"
    op = Opportunity(
        identity,
        "TESTUSDT",
        side,
        "1" if long else "1S",
        "READY",
        100,
        90 if long else 110,
        110 if long else 90,
        130 if long else 70,
        95 if long else 105,
        99 if long else 101,
        0,
        created + H4 if expiry is None else expiry,
        "trigger",
        dict(
            trigger_closed_at=0,
            zone_low=98 if long else 95,
            zone_high=105 if long else 102,
        ),
    )
    sizing = dict(
        allowed=True,
        entry=op.entry,
        stop=op.stop,
        target1=op.target1,
        target2=op.target2,
        qty=qty,
        qty_step=qty_step,
        risk_cash=qty * 10,
        notional=qty * 100,
    )
    return create_trade(op, sizing, arm, created)


def bar(i, open=100, high=101, low=99, close=100):
    return Candle(i * INTERVAL_MS, open, high, low, close, 1)


def monitored_trade(side="Buy", created=0, arm="monitored_shadow", **evidence):
    d = 1 if side == "Buy" else -1
    op = Opportunity(
        "watch",
        "TESTUSDT",
        side,
        "1" if d > 0 else "1S",
        "READY",
        100 * (1 + d * 0.0005),
        90 if d > 0 else 110,
        110 if d > 0 else 90,
        130 if d > 0 else 70,
        95 if d > 0 else 105,
        0,
        0,
        30 * DAY,
        "ZONE_ARRIVAL",
        dict(
            entry_style="monitored_zone",
            trigger_kind="ZONE_ARRIVAL",
            observation_at=created,
            zone_low=98,
            zone_high=102,
            **evidence,
        ),
    )
    return create_trade(
        op,
        dict(
            allowed=True,
            entry=op.entry,
            stop=op.stop,
            target1=op.target1,
            target2=op.target2,
            qty=5,
            qty_step=1,
            risk_cash=50,
            notional=500,
        ),
        arm,
        created,
    )


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_monitored_ioc_fills_next_open_with_costs_not_limit_or_later_touch(side):
    t = monitored_trade(side)
    d = 1 if side == "Buy" else -1
    expected = 100 * (1 + d * MONITORED_HALF_SPREAD) * (1 + d * SLIPPAGE)
    assert advance(t, [bar(0)], STEP - 1)["opened_at"] is None
    filled = advance(t, [bar(0)], STEP)
    assert filled["status"] == "OPEN"
    assert filled["entry"] == pytest.approx(expected)
    assert filled["opened_at"] == 0
    assert filled["fees"] == pytest.approx(expected * 5 * FEE_RATE)
    assert filled["entry_style"] == "monitored_zone"
    assert "next-open-ioc" in filled["cost_model"]
    # Replaying the same evidence is idempotent.
    assert advance(filled, [bar(0)], STEP) == filled


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_monitored_ioc_miss_cannot_fill_on_later_touch_or_revisit(side):
    d = 1 if side == "Buy" else -1
    # Opening quote violates the cap, but the bar later spans the desired entry.
    prices = [bar(0, open=100 + d, high=102, low=98, close=100), bar(1)]
    expired = advance(monitored_trade(side), prices, 2 * STEP)
    assert expired["status"] == "EXPIRED"
    assert expired["exit_reason"] == "IOC_PRICE_MISSED"
    assert expired["opened_at"] is None and not expired["fills"]
    assert advance(expired, [bar(2)], 3 * STEP) == expired


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_monitored_favorable_gap_outside_zone_does_not_fill(side):
    p = 97 if side == "Buy" else 103
    ended = advance(monitored_trade(side), [bar(0, p, p + 0.1, p - 0.1, p)], STEP)
    assert (ended["status"], ended["exit_reason"]) == ("EXPIRED", "IOC_OUTSIDE_ZONE")
    assert not ended["fills"]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_monitored_entry_bar_stop_wins_and_targets_get_no_credit(side):
    t = monitored_trade(side)
    ended = advance(t, [bar(0, high=135, low=85)], STEP)
    assert ended["status"] == "CLOSED" and ended["exit_reason"] == "STOP"
    assert [f["reason"] for f in ended["fills"]] == ["ENTRY", "STOP"]
    favorable = bar(0, high=135, low=99) if side == "Buy" else bar(0, high=101, low=65)
    still_open = advance(t, [favorable], STEP)
    assert still_open["status"] == "OPEN" and not still_open["tp1_done"]


def test_monitored_never_uses_partial_decision_bar_or_skips_missing_open():
    t = monitored_trade(created=1)
    first = advance(t, [bar(0)], STEP)
    assert first["opened_at"] is None
    filled = advance(first, [bar(1)], 2 * STEP)
    assert filled["opened_at"] == STEP
    gap = advance(t, [bar(2)], 3 * STEP)
    assert gap["opened_at"] is None and gap["data_gap"]


@pytest.mark.parametrize(
    "arm", ["baseline_shadow", "ai_shadow", "resting_shadow", "live"]
)
def test_monitored_create_requires_separate_book(arm):
    with pytest.raises(ValueError, match="monitored_shadow"):
        monitored_trade(arm=arm)


def test_monitored_create_does_not_accept_faked_confirmation():
    with pytest.raises(ValueError, match="monitored_shadow"):
        monitored_trade(trigger_closed_at=0)


def mirror(bars):
    return [
        replace(
            b,
            open=200 - b.open,
            high=200 - b.low,
            low=200 - b.high,
            close=200 - b.close,
        )
        for b in bars
    ]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_resting_pending_cancels_when_target_passes_before_entry(side):
    trade = new_trade(side=side, arm="resting_shadow")
    trade["entry_style"] = "resting_limit"
    prices = [bar(0, open=105, high=111, low=104, close=110)]
    if side == "Sell":
        prices = mirror(prices)
    ended = advance(trade, prices, STEP)
    assert ended["status"] == "EXPIRED"
    assert ended["exit_reason"] == "TARGET_BEFORE_ENTRY"
    assert ended["opened_at"] is None
    assert ended["fills"] == []


def aggregate(bars, seconds):
    size = seconds // STEP
    result = []
    for index in range(0, len(bars) - size + 1, size):
        group = bars[index : index + size]
        result.append(
            Candle(
                group[0].open_time,
                group[0].open,
                max(b.high for b in group),
                min(b.low for b in group),
                group[-1].close,
                sum(b.volume for b in group),
            )
        )
    return result


def histories(prices):
    bars = [
        bar(i, open=p, high=p + 0.2, low=p - 0.2, close=p) for i, p in enumerate(prices)
    ]
    return bars, aggregate(bars, DAY), aggregate(bars, H4)


def rate(seconds, value=0.001):
    return {
        "fundingRateTimestamp": str(int(seconds * 1000)),
        "fundingRate": str(value),
        "symbol": "TESTUSDT",
    }


def target_history():
    return [
        bar(0),
        bar(1, open=104, high=106, low=103, close=105),
        bar(2, open=109, high=111, low=108, close=110),
        bar(3, open=111, high=112, low=110, close=111),
        bar(4, open=128, high=131, low=127, close=130),
    ]


def test_first_missing_bar_never_silently_jumps_and_backfill_recovers():
    initial = new_trade(created=40)
    gap = advance(initial, [bar(2)], 3 * STEP)
    assert gap["status"] == "PENDING" and gap["last_bar"] is None
    assert gap["fees"] == 0 and gap["fills"] == []
    assert "180000" in gap["data_gap"]
    recovered = advance(gap, [bar(1), bar(2)], 3 * STEP)
    assert recovered["status"] == "OPEN" and "data_gap" not in recovered
    assert recovered["opened_at"] == 2 * STEP
    assert initial["last_bar"] is None  # inputs are immutable


def test_missing_middle_or_tail_freezes_at_last_resolved_bar():
    opened = advance(new_trade(), [bar(0)], STEP)
    gap = advance(opened, [bar(2, high=150)], 3 * STEP)
    assert gap["last_bar"] == 0 and gap["status"] == "OPEN"
    assert gap["gross_pnl"] == 0 and gap["data_gap"]
    tail = advance(opened, [], 2 * STEP)
    assert tail["data_gap"] and tail["last_bar"] == 0


def test_decision_bar_and_unclosed_bars_never_fill():
    initial = new_trade(created=40)
    before = advance(initial, [bar(0), bar(1)], 2 * STEP - 1)
    assert before["status"] == "PENDING" and before["last_bar"] is None
    after = advance(before, [bar(0), bar(1)], 2 * STEP)
    assert after["opened_at"] == 2 * STEP
    assert after["fills"][0]["time"] > initial["created_at"]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_entry_bar_target_credit_is_forbidden_and_stops_win(side):
    bars = [bar(0, high=140, low=99, close=100)]
    if side == "Sell":
        bars = mirror(bars)
    opened = advance(new_trade(side), bars, STEP)
    assert opened["status"] == "OPEN" and not opened["tp1_done"]
    assert opened["gross_pnl"] == 0 and opened["net_pnl"] == -100 * 5 * FEE_RATE
    tie = [bar(0, high=140, low=80, close=100)]
    stopped = advance(new_trade(side), tie if side == "Buy" else mirror(tie), STEP)
    assert stopped["status"] == "CLOSED" and stopped["exit_reason"] == "STOP"
    assert [f["reason"] for f in stopped["fills"]] == ["ENTRY", "STOP"]
    assert stopped["opened_at"] == stopped["closed_at"] == STEP


def test_expiry_requires_full_candle_window_and_prior_coverage():
    initial = new_trade(expiry=300)
    no_touch = bar(0, open=102, high=103, low=101, close=102)
    expired = advance(initial, [no_touch, bar(1)], 2 * STEP)
    assert expired["status"] == "EXPIRED" and expired["closed_at"] == 300
    assert expired["fees"] == 0
    gap = advance(initial, [], 2 * STEP)
    assert gap["status"] == "PENDING" and gap["data_gap"]
    # No full future candle fits this window, so evidence of a touch is never used.
    none_fit = advance(new_trade(created=40, expiry=300), [], 300)
    assert none_fit["status"] == "EXPIRED" and not none_fit.get("data_gap")
    at_boundary = advance(new_trade(expiry=STEP), [bar(0)], STEP)
    assert (
        at_boundary["status"] == "OPEN"
    )  # candle closing exactly at expiry is eligible


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_stop_gap_uses_worse_open_and_realized_costs(side):
    bars = [bar(0), bar(1, open=85, high=87, low=84, close=86)]
    if side == "Sell":
        bars = mirror(bars)
    stopped = advance(new_trade(side), bars, 2 * STEP)
    sign = 1 if side == "Buy" else -1
    exit_price = (85 if sign > 0 else 115) * (1 - sign * SLIPPAGE)
    gross = sign * (exit_price - 100) * 5
    fees = (100 * 5 + exit_price * 5) * FEE_RATE
    assert stopped["exit_reason"] == "STOP"
    assert stopped["fills"][-1]["price"] == pytest.approx(exit_price)
    assert stopped["gross_pnl"] == pytest.approx(gross)
    assert stopped["net_pnl"] == pytest.approx(gross - fees)


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_odd_lot_partial_and_targets_follow_cost_model(side):
    bars = target_history()
    if side == "Sell":
        bars = mirror(bars)
    closed = advance(new_trade(side), bars, 5 * STEP)
    assert closed["status"] == "CLOSED" and closed["exit_reason"] == "TP2"
    assert [f["qty"] for f in closed["fills"]] == [5, 3, 2]
    sign = 1 if side == "Buy" else -1
    t1, t2 = (110, 130) if sign > 0 else (90, 70)
    p1, p2 = t1 * (1 - sign * SLIPPAGE), t2 * (1 - sign * SLIPPAGE)
    gross = sign * ((p1 - 100) * 3 + (p2 - 100) * 2)
    assert closed["gross_pnl"] == pytest.approx(gross)
    assert closed["net_pnl"] == pytest.approx(
        gross - (500 + p1 * 3 + p2 * 2) * FEE_RATE
    )


def test_fractional_lots_use_decimal_rounding_not_binary_ceil():
    result = advance(new_trade(qty=0.3, qty_step=0.1), target_history(), 5 * STEP)
    assert [f["qty"] for f in result["fills"]] == [0.3, 0.2, 0.1]
    assert result["remaining"] == 0


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_single_lot_tp1_keeps_position_moves_stop_and_exits_later(side):
    bars = [bar(0), bar(1, open=109, high=111, low=99, close=110), bar(2)]
    if side == "Sell":
        bars = mirror(bars)
    tp1 = advance(new_trade(side, qty=1), bars[:2], 2 * STEP)
    assert tp1["status"] == "OPEN" and tp1["remaining"] == 1
    assert tp1["tp1_done"] and tp1["stop"] == 100
    assert len(tp1["fills"]) == 1 and tp1["gross_pnl"] == 0
    closed = advance(tp1, bars, 3 * STEP)
    assert closed["exit_reason"] == "STOP" and closed["net_pnl"] < 0


def test_prefix_future_append_restart_and_idempotency():
    bars = target_history()
    initial = new_trade()
    streaming = initial
    for n in range(1, len(bars) + 1):
        now = n * STEP
        batch = advance(initial, bars[:n], now)
        assert batch == advance(initial, bars, now)
        streaming = advance(streaming, bars[:n], now)
        assert batch == streaming
        assert advance(streaming, bars[:n], now) == streaming
    assert initial["status"] == "PENDING" and initial["fills"] == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0, -1])
def test_invalid_prices_cannot_produce_cashflows(bad):
    initial = new_trade()
    invalid = advance(initial, [replace(bar(0), close=bad)], STEP)
    assert invalid["data_error"] and invalid["fills"] == [] and invalid["fees"] == 0
    assert math.isfinite(invalid["net_pnl"])


def test_bad_order_duplicate_and_missing_lot_metadata_fail_closed():
    initial = new_trade()
    for bars in ([bar(0), bar(0)], [bar(1), bar(0)]):
        assert advance(initial, bars, 2 * STEP)["data_error"]
    broken = dict(initial, qty_step=0)
    assert advance(broken, [bar(0)], STEP)["data_error"]
    assert advance(initial, [bar(0)], math.nan)["data_error"]
    with pytest.raises(ValueError):
        new_trade(qty=0.31, qty_step=0.1)
    opened = advance(initial, [bar(0)], STEP)
    assert advance(opened, [bar(0)], STEP - 1)["data_error"]
    assert advance(new_trade(created=40), [], 39)["data_error"]


def test_funding_uses_actual_settlements_quantities_and_explicit_coverage():
    closed = advance(new_trade(), target_history(), 5 * STEP)
    rates = [rate(t) for t in (0, 179, 180, 270, 540, 700, 900, 999)]
    funded = apply_funding(closed, rates, 1200)
    assert funded["funding"] == pytest.approx(-0.9)
    assert [r["qty"] for r in funded["funding_events"]] == [5, 2, 2]
    assert not funded["funding_complete"]
    assert "estimate" in funded["funding_basis"]
    complete = apply_funding(funded, rates, 1200, covered_from=0, history_complete=True)
    assert complete["funding_complete"]
    assert complete["net_pnl"] == pytest.approx(closed["net_pnl"] - 0.9)
    assert (
        apply_funding(complete, rates, 1200, covered_from=0, history_complete=True)
        == complete
    )
    short = advance(new_trade("Sell"), mirror(target_history()), 5 * STEP)
    assert apply_funding(short, rates, 1200)["funding"] == pytest.approx(0.9)


def test_no_funding_before_entry_after_exit_or_beyond_resolved_history():
    pending = new_trade()
    assert apply_funding(pending, [rate(100)], 200) == pending
    opened = advance(pending, [bar(0), bar(1)], 2 * STEP)
    rates = [rate(180), rate(270), rate(540)]
    funded = apply_funding(opened, rates, 9000)
    assert funded["funding"] == pytest.approx(-0.5)
    assert not funded["funding_complete"]
    same_bar_stop = advance(pending, [bar(0, low=80)], STEP)
    assert apply_funding(same_bar_stop, [rate(0), rate(STEP)], STEP)["funding"] == 0


def test_coverage_is_range_not_page_or_latest_timestamp():
    closed = advance(new_trade(), target_history(), 5 * STEP)
    assert not apply_funding(closed, [], 9000)["funding_complete"]
    assert apply_funding(closed, [], 9000, history_complete=True)["funding_error"]
    partial = apply_funding(closed, [], 500, covered_from=180, history_complete=True)
    assert not partial["funding_complete"]
    hole = apply_funding(partial, [], 900, covered_from=501, history_complete=True)
    assert not hole["funding_complete"]
    complete = apply_funding(hole, [], 501, covered_from=500, history_complete=True)
    assert complete["funding_complete"]


def test_coverage_never_certifies_unresolved_future_position_quantities():
    bars = target_history()
    opened = advance(new_trade(), bars, 2 * STEP)
    early = apply_funding(
        opened, [rate(270), rate(700)], 900, covered_from=0, history_complete=True
    )
    assert early["funding_coverage"] == [[0, 360]]
    closed = advance(early, bars, 5 * STEP)
    still_missing = apply_funding(closed, [], 900)
    assert not still_missing["funding_complete"]
    complete = apply_funding(
        closed, [rate(270), rate(700)], 900, covered_from=0, history_complete=True
    )
    assert complete["funding_complete"]
    assert complete["funding"] == pytest.approx(-0.7)


@pytest.mark.parametrize("bad", ["nan", "inf", None])
def test_invalid_funding_is_atomic_and_never_complete(bad):
    closed = advance(new_trade(), target_history(), 5 * STEP)
    output = apply_funding(
        closed,
        [rate(270), dict(rate(540), fundingRate=bad)],
        900,
        covered_from=0,
        history_complete=True,
    )
    assert output["funding_error"] and not output["funding_complete"]
    assert output["funding"] == 0 and output["net_pnl"] == closed["net_pnl"]


def test_conflicting_settlements_are_not_silently_overwritten():
    closed = advance(new_trade(), target_history(), 5 * STEP)
    first = apply_funding(closed, [rate(270)], 900)
    conflict = apply_funding(first, [rate(270, 0.002)], 900)
    assert conflict["funding_error"] and conflict["funding"] == first["funding"]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_daily_invalidation_at_closed_day_with_mirrored_net_exit(side):
    prices = [100] * (3 * DAY // STEP)
    prices[-1] = 94.5
    bars, daily, execution = histories(prices)
    if side == "Sell":
        bars, daily, execution = mirror(bars), mirror(daily), mirror(execution)
    trade = new_trade(side, created=14 * H4)
    early = advance(trade, bars, 3 * DAY - 1, daily=daily, execution=execution)
    assert early["status"] == "OPEN"
    closed = advance(early, bars, 3 * DAY, daily=daily, execution=execution)
    assert (
        closed["exit_reason"] == "DAILY_INVALIDATION" and closed["closed_at"] == 3 * DAY
    )
    assert closed["net_pnl"] < 0 and closed["management_complete"]


def test_daily_invalidation_can_cancel_pending_before_first_eligible_bar():
    bars, daily, execution = histories([94.5] * (4 * DAY // STEP))
    initial = new_trade(created=3 * DAY - 40)
    output = advance(initial, bars, 3 * DAY + STEP, daily=daily, execution=execution)
    assert (
        output["status"] == "EXPIRED" and output["exit_reason"] == "DAILY_INVALIDATION"
    )
    assert output["opened_at"] is None and output["fees"] == 0


def test_entry_at_daily_close_takes_adverse_invalidation_without_target_credit():
    prices = [100] * (3 * DAY // STEP)
    prices[-1] = 94.5
    bars, daily, execution = histories(prices)
    initial = new_trade(created=3 * DAY - STEP)
    result = advance(initial, bars, 3 * DAY, daily=daily, execution=execution)
    assert result["opened_at"] == result["closed_at"] == 3 * DAY
    assert result["exit_reason"] == "DAILY_INVALIDATION" and result["net_pnl"] < 0


def test_ten_day_review_twenty_day_exit_only_without_closed_zone_progress():
    bars, daily, execution = histories([100] * (22 * DAY // STEP))
    initial = new_trade(created=14 * H4)
    review = advance(initial, bars, 12 * DAY, daily=daily, execution=execution)
    assert review["status"] == "OPEN" and review["daily_bars_held"] == 10
    assert review["time_review_due"]
    assert (
        len(
            [x for x in review["management_events"] if x["reason"] == "TIME_REVIEW_10D"]
        )
        == 1
    )
    exited = advance(review, bars, 22 * DAY, daily=daily, execution=execution)
    assert exited["exit_reason"] == "TIMEOUT" and exited["daily_bars_held"] == 20
    assert exited["net_pnl"] < 0  # flat price still has entry/exit fees and slippage
    prices = [100] * (22 * DAY // STEP)
    prices[15 * H4 // STEP : 16 * H4 // STEP] = [106] * (H4 // STEP)
    progressed, d, e = histories(prices)
    held = advance(initial, progressed, 22 * DAY, daily=d, execution=e)
    assert held["status"] == "OPEN" and held["favorable_4h_close"]
    assert not held["time_review_due"]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_closed_4h_structural_trail_ratchets_after_midpoint(side):
    four_hour_prices = [100] * 14 + [100, 116, 112, 118, 114, 119, 115, 118]
    prices = [price for price in four_hour_prices for _ in range(H4 // STEP)]
    bars, daily, execution = histories(prices)
    if side == "Sell":
        bars, daily, execution = mirror(bars), mirror(daily), mirror(execution)
    trade = new_trade(side, created=14 * H4)
    trade["target2"] = 125 if side == "Buy" else 75
    before = advance(trade, bars, 18 * H4 - 1, daily=daily, execution=execution)
    assert before["trailing_active"] and before["stop"] == 100
    confirmed = advance(before, bars, 18 * H4, daily=daily, execution=execution)
    sign = 1 if side == "Buy" else -1
    assert sign * (confirmed["stop"] - 100) > 0
    assert confirmed["last_trailing_pivot"]
    later = advance(confirmed, bars, 22 * H4, daily=daily, execution=execution)
    assert sign * (later["stop"] - confirmed["stop"]) >= 0
    stops = [event["stop"] for event in later["management_events"] if "stop" in event]
    assert all(sign * (b - a) > 0 for a, b in zip(stops, stops[1:]))


def test_management_gaps_mode_changes_and_warmup_are_explicit():
    bars, daily, execution = histories([100] * (4 * DAY // STEP))
    initial = new_trade(created=14 * H4)
    missing = advance(initial, bars, 3 * DAY, daily=daily, execution=execution[:17])
    assert missing["data_gap"] and missing["last_bar"] < 3 * DAY * 1000 - INTERVAL_MS
    warmup = advance(initial, bars, 3 * DAY, daily=daily, execution=execution[5:])
    assert warmup["data_error"]
    price_only = advance(initial, bars, 3 * DAY)
    assert not price_only["management_complete"] and price_only["management_warning"]
    changed = advance(price_only, bars, 4 * DAY, daily=daily, execution=execution)
    assert changed["data_error"] and changed["last_bar"] == price_only["last_bar"]


def test_matched_comparison_keeps_real_ai_latency_and_baseline_all_cohort():
    base = new_trade(created=160)
    ai = new_trade(created=200, arm="ai_shadow")
    extra = new_trade(created=160, identity="unselected")
    # Both share the source signal; only the baseline can use the next 3m candle.
    bars = [bar(1), bar(2, open=104, high=106, low=103, close=105)]
    base = advance(base, bars, 3 * STEP)
    ai = advance(ai, bars, 3 * STEP)
    assert base["status"] == "OPEN" and ai["status"] == "PENDING"
    report = comparison([base, ai, extra])
    assert base["counterfactual_id"] == ai["counterfactual_id"]
    assert report["baseline_all"]["total"] == 2 and report["ai_selected"]["total"] == 1
    assert report["baseline_unselected_count"] == 1
    match = report["matches"][0]
    assert match["decision_latency_seconds"] == 40 and not match["same_entry_window"]
    assert match["confounds"] == ["entry_latency"]
    assert report["pure_filter_effect"] is False
    assert report["comparison_basis"] == "selection_plus_latency_or_sizing"


def test_comparison_execution_parity_and_summary_uses_realized_net():
    base = advance(new_trade(), target_history(), 5 * STEP)
    ai = advance(new_trade(arm="ai_shadow"), target_history(), 5 * STEP)
    report = comparison([base, ai])
    assert report["paired_closed_count"] == 1 and report["matched_net_delta"] == 0
    assert report["matches"][0]["confounds"] == []
    assert report["comparison_basis"] == "selection_on_observed_candidates"
    invalid = dict(base, id="invalid", net_pnl=math.nan)
    pending = new_trade(identity="pending")
    result = summary([base, invalid, pending], "baseline_shadow")
    assert result["wins"] == 1 and result["wr"] == 100
    assert result["invalid_outcomes"] == 1 and result["net_pnl"] == base["net_pnl"]
    assert summary([], "baseline_shadow")["wr"] is None


def test_acceptance_replay_is_not_mislabeled_as_ai_selected():
    assert (
        summary([], "acceptance_replay")["cohort"]
        == "arm-specific simulated candidates"
    )
