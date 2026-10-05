"""Causal, in-memory tests for the independent SHADOW order proposal path."""

from dataclasses import asdict, replace
import json
import math

import pytest

from apex_bot import engine
from apex_bot.models import Candle, Opportunity
from apex_bot.zone_orders import ORDER_LIFETIME, analyze_zones

DAY, H4 = engine.DAY, engine.H4
SETUPS = ("1", "1S", "2", "2S", "2X", "2XS", "5S", "5L")


def candles(prices, seconds=DAY, start=0):
    return [
        Candle(int((start + i * seconds) * 1000), p, p + 0.2, p - 0.2, p, 10)
        for i, p in enumerate(prices)
    ]


def daily_history():
    prices = [100] * 16
    for endpoint in (95, 130, 110, 175, 150, 190, 120, 160, 140):
        start = prices[-1]
        prices.extend(start + (endpoint - start) * i / 8 for i in range(1, 9))
    return candles(prices)


def mirror(bars):
    return [
        replace(
            b,
            open=400 - b.open,
            high=400 - b.low,
            low=400 - b.high,
            close=400 - b.close,
        )
        for b in bars
    ]


def case(setup="1"):
    daily = daily_history()
    if setup in {"1S", "2S", "2XS", "5L"}:
        daily = mirror(daily)
    plan = next(
        p
        for p in engine._plans(daily, engine.confirmed_zigzag(daily))
        if p.setup == setup
    )
    daily = [b for b in daily if b.open_time / 1000 + DAY <= plan.created]
    execution = candles([daily[-1].close], H4, plan.created - H4)
    return plan, daily, execution


def select(ops, plan):
    return next(
        o for o in ops if o.setup == plan.setup and o.created_at == plan.created
    )


def proposal(plan, daily, execution, now=None, **kwargs):
    return select(
        analyze_zones(
            "TESTUSDT", daily, execution, plan.created if now is None else now, **kwargs
        ),
        plan,
    )


def touching_bar(bar, level):
    return replace(bar, low=min(bar.low, level), high=max(bar.high, level))


def assert_no_trigger(op):
    assert op.evidence["trigger_kind"] == "ZONE_LIMIT"
    for key in ("trigger_closed_at", "trigger_price", "trigger_evidence_id"):
        assert op.evidence[key] is None


@pytest.mark.parametrize("setup", SETUPS)
def test_all_setups_offer_at_actual_availability_with_original_levels(setup):
    plan, daily, execution = case(setup)
    op = proposal(plan, daily, execution)
    parent = select(engine.analyze("TESTUSDT", daily, execution, plan.created), plan)
    assert isinstance(op, Opportunity)
    assert (op.state, op.reason) == ("READY", "ZONE_LIMIT")
    assert op.side == ("Buy" if plan.direction > 0 else "Sell")
    assert op.entry == (plan.zone_low + plan.zone_high) / 2
    assert (op.stop, op.target1, op.target2, op.invalidation, op.confirmation) == (
        parent.stop,
        parent.target1,
        parent.target2,
        parent.invalidation,
        plan.confirmation,
    )
    assert op.expires_at == plan.created + 48 * 60 * 60
    assert op.created_at == op.evidence["order_armed_at"] == plan.created
    assert op.evidence["order_expires_at"] == op.expires_at
    assert op.evidence["parent_plan_id"] == parent.id
    assert op.evidence["plan_evidence_id"] == parent.id
    assert op.id == op.evidence["evidence_id"] != parent.id
    assert op.evidence["engine_version"] == engine.VERSION
    assert op.evidence["entry_style"] == "resting_limit"
    for key in ("anchors", "daily_evidence_id", "risk_multiplier", "stop_buffer_pct"):
        assert op.evidence[key] == parent.evidence[key]
    assert op.evidence["entry_zone_low"] == op.evidence["zone_low"] == plan.zone_low
    assert op.evidence["entry_zone_high"] == op.evidence["zone_high"] == plan.zone_high
    assert op.evidence["structural_valid"] is True
    assert op.evidence["data_valid"] is True
    assert op.evidence["terminal_status"] is None
    assert op.evidence["as_of"] == plan.created
    assert op.evidence["daily_closed_at"] == plan.created
    assert op.evidence["execution_closed_at"] == plan.created
    assert_no_trigger(op)
    json.dumps(op.to_dict(), allow_nan=False)

    # A historical pivot/zone is not yet an available plan. Neither future
    # daily data nor future 4H data may make it appear one second early.
    assert plan.anchors[-1].open_time / 1000 + DAY < plan.created
    before = analyze_zones("TESTUSDT", daily, execution, plan.created - 1)
    assert not any(o.setup == setup and o.created_at == plan.created for o in before)


@pytest.mark.parametrize("reflected", [False, True])
def test_every_daily_and_execution_prefix_ignores_future_appends(reflected):
    daily = daily_history()
    execution = candles([160] * (len(daily) * 6), H4)
    if reflected:
        daily, execution = mirror(daily), mirror(execution)
    for n in range(len(daily) + 1):
        now = n * DAY
        assert analyze_zones("TESTUSDT", daily[:n], execution[: 6 * n], now) == (
            analyze_zones("TESTUSDT", daily, execution, now)
        ), n
    # Check every 4H boundary through all plan availabilities, not only the
    # final fixture's prefix. Also check each incomplete bar just before close.
    for n in range(len(execution) + 1):
        now = n * H4
        expected = analyze_zones("TESTUSDT", daily, execution[:n], now)
        assert expected == analyze_zones("TESTUSDT", daily, execution, now), n
        assert analyze_zones("TESTUSDT", daily, execution[:n], now - 1) == (
            analyze_zones("TESTUSDT", daily, execution, now - 1)
        ), n


@pytest.mark.parametrize("setup", SETUPS)
def test_future_prices_do_not_reprice_or_reidentify_a_plan(setup):
    plan, daily, execution = case(setup)
    initial = proposal(plan, daily, execution)
    for offset in (H4, DAY, ORDER_LIFETIME, ORDER_LIFETIME + DAY):
        later_daily = daily + candles(
            [daily[-1].close] * int(offset // DAY), DAY, plan.created
        )
        later_execution = execution + candles(
            [daily[-1].close] * int(offset // H4), H4, plan.created
        )
        later = proposal(plan, later_daily, later_execution, plan.created + offset)
        assert (later.id, later.entry, later.stop, later.target1, later.target2) == (
            initial.id,
            initial.entry,
            initial.stop,
            initial.target1,
            initial.target2,
        )
        assert later.expires_at == initial.expires_at
        for key in (
            "plan_evidence_id",
            "parent_plan_id",
            "daily_evidence_id",
            "anchors",
            "order_armed_at",
        ):
            assert later.evidence[key] == initial.evidence[key]
        if offset >= ORDER_LIFETIME:
            assert (later.state, later.reason) == ("INVALID", "EXPIRED")
            assert later.evidence["terminal_at"] == initial.expires_at
        else:
            assert later.state == "READY"
        assert_no_trigger(later)


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_missing_then_fresh_data_never_resets_arming_or_deadline(setup):
    plan, daily, execution = case(setup)
    waiting = proposal(plan, daily, [])
    assert (waiting.state, waiting.reason) == ("WAIT", "DATA_STALE_OR_MISSING")
    assert waiting.evidence["data_valid"] is False
    delayed = execution + candles([daily[-1].close] * 3, H4, plan.created)
    ready = proposal(plan, daily, delayed, plan.created + 3 * H4)
    assert ready.state == "READY"
    assert ready.id == waiting.id
    assert ready.evidence["order_armed_at"] == plan.created
    assert ready.expires_at == waiting.expires_at == plan.created + ORDER_LIFETIME
    late_first_call = proposal(plan, daily, delayed, ready.expires_at)
    assert (late_first_call.state, late_first_call.reason) == ("INVALID", "EXPIRED")
    assert late_first_call.id == ready.id


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_expiry_has_exact_boundary_and_is_capped_by_engine_lifetime(setup, monkeypatch):
    plan, daily, execution = case(setup)
    monkeypatch.setattr(engine, "PLAN_LIFETIME", H4)
    initial = proposal(plan, daily, execution)
    assert initial.expires_at == plan.created + H4
    assert proposal(plan, daily, execution, initial.expires_at - 1).state == "READY"
    expired = proposal(plan, daily, execution, initial.expires_at)
    assert (expired.state, expired.reason) == ("INVALID", "EXPIRED")
    assert expired.id == initial.id
    assert expired.evidence["terminal_at"] == initial.expires_at


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("failure", ["stop", "target"])
def test_pre_availability_execution_cancels_before_any_offer(setup, failure):
    plan, daily, _ = case(setup)
    start = daily[plan.history_start].open_time / 1000
    execution = candles(
        [daily[-1].close] * int((plan.created - start) // H4), H4, start
    )
    initial = proposal(plan, daily, execution)
    assert initial.state == "READY"
    level = initial.stop if failure == "stop" else plan.target1
    execution[0] = touching_bar(execution[0], level)
    invalid = proposal(plan, daily, execution)
    expected = "HARD_STOP_BREACHED" if failure == "stop" else "TARGET_ALREADY_PASSED"
    assert (invalid.state, invalid.reason) == ("INVALID", expected)
    assert invalid.evidence["terminal_status"] == expected
    assert invalid.evidence["terminal_at"] == plan.created
    assert invalid.evidence["plan_history_start"] == start
    assert invalid.id == initial.id  # Execution history never chooses a new offer.
    assert invalid.evidence["structural_valid"] is (failure != "stop")
    assert_no_trigger(invalid)


@pytest.mark.parametrize("setup", ["2", "2S"])
def test_target_passed_on_daily_availability_bar_is_never_offered(setup):
    plan, daily, execution = case(setup)
    daily[-1] = touching_bar(daily[-1], plan.target1)
    op = proposal(plan, daily, execution)
    assert (op.state, op.reason) == ("INVALID", "TARGET_ALREADY_PASSED")
    assert op.evidence["terminal_at"] == plan.created
    assert_no_trigger(op)


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_only_complete_bars_since_plan_history_start_can_cancel(setup):
    plan, daily, _ = case(setup)
    start = daily[plan.history_start].open_time / 1000
    execution = candles(
        [daily[-1].close] * (1 + int((plan.created - start) // H4)), H4, start - H4
    )
    initial = proposal(plan, daily, execution)
    # Even a stop wick on the bar ending at history_start is outside history.
    execution[0] = touching_bar(execution[0], initial.stop)
    assert proposal(plan, daily, execution) == initial
    current = touching_bar(
        candles([daily[-1].close], H4, plan.created)[0], initial.stop
    )
    with_current = execution + [current]
    assert proposal(plan, daily, with_current) == initial
    almost = proposal(plan, daily, with_current, plan.created + H4 - 1)
    assert almost.state == "READY"
    closed = proposal(plan, daily, with_current, plan.created + H4)
    assert (closed.state, closed.reason) == ("INVALID", "HARD_STOP_BREACHED")
    assert closed.id == initial.id


@pytest.mark.parametrize("setup", SETUPS)
def test_daily_close_invalidation_uses_frozen_level(setup):
    plan, daily, execution = case(setup)
    initial = proposal(plan, daily, execution)
    bar = touching_bar(
        candles([daily[-1].close], DAY, plan.created)[0], plan.invalidation
    )
    daily += [replace(bar, close=plan.invalidation)]
    execution += candles([daily[-1].open] * 6, H4, plan.created)
    assert proposal(plan, daily, execution, plan.created + DAY - 1).state == "READY"
    invalid = proposal(plan, daily, execution, plan.created + DAY)
    assert (invalid.state, invalid.reason) == ("INVALID", "DAILY_INVALIDATION")
    assert invalid.evidence["structural_valid"] is False
    assert invalid.evidence["terminal_at"] == plan.created + DAY
    assert invalid.id == initial.id


@pytest.mark.parametrize("setup", SETUPS)
def test_structure_wicks_cancel_only_engine_structure_setups(setup):
    plan, daily, execution = case(setup)
    initial = proposal(plan, daily, execution)
    bar = touching_bar(
        candles([daily[-1].close], H4, plan.created)[0], plan.invalidation
    )
    op = proposal(plan, daily, execution + [bar], plan.created + H4)
    if setup in {"2X", "2XS", "5S", "5L"}:
        assert (op.state, op.reason) == ("INVALID", "STRUCTURE_INVALIDATED")
        assert op.evidence["structural_valid"] is False
    else:
        assert op.state == "READY"  # Ordinary invalidation needs a daily close.
    assert op.id == initial.id


@pytest.mark.parametrize("setup", ["1", "1S"])
@pytest.mark.parametrize("first", ["stop", "target"])
def test_first_terminal_event_survives_later_failure_and_expiry(setup, first):
    plan, daily, execution = case(setup)
    initial = proposal(plan, daily, execution)
    levels = [initial.stop, initial.target1]
    expected = "HARD_STOP_BREACHED"
    if first == "target":
        levels.reverse()
        expected = "TARGET_ALREADY_PASSED"
    extra = candles([daily[-1].close] * 2, H4, plan.created)
    execution += [touching_bar(b, level) for b, level in zip(extra, levels)]
    for now in (plan.created + H4, plan.created + 2 * H4, initial.expires_at + DAY):
        op = proposal(plan, daily, execution, now)
        assert (op.state, op.reason) == ("INVALID", expected)
        assert op.evidence["terminal_at"] == plan.created + H4
        assert op.id == initial.id
        assert op.entry == initial.entry
        assert_no_trigger(op)


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_stop_wins_ambiguous_bar_and_daily_invalidation_wins_tied_close(setup):
    plan, daily, execution = case(setup)
    initial = proposal(plan, daily, execution)
    bar = candles([daily[-1].close], H4, plan.created)[0]
    ambiguous = touching_bar(touching_bar(bar, initial.stop), initial.target1)
    assert proposal(plan, daily, execution + [ambiguous], plan.created + H4).reason == (
        "HARD_STOP_BREACHED"
    )
    day = touching_bar(
        candles([daily[-1].close], DAY, plan.created)[0], plan.invalidation
    )
    daily += [replace(day, close=plan.invalidation)]
    execution += candles([day.open] * 6, H4, plan.created)
    execution[-1] = touching_bar(execution[-1], initial.target1)
    op = proposal(plan, daily, execution, plan.created + DAY)
    assert op.reason == "DAILY_INVALIDATION"


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_expiry_precedes_cancel_on_bar_closing_at_deadline(setup):
    plan, daily, execution = case(setup)
    initial = proposal(plan, daily, execution)
    daily += candles([daily[-1].close] * 2, DAY, plan.created)
    execution += candles([daily[-1].close] * 12, H4, plan.created)
    execution[-1] = touching_bar(execution[-1], initial.stop)
    op = proposal(plan, daily, execution, initial.expires_at)
    assert (op.state, op.reason) == ("INVALID", "EXPIRED")
    assert op.evidence["terminal_at"] == initial.expires_at


@pytest.mark.parametrize("setup", ["1", "1S", "5S", "5L"])
def test_historical_and_new_zone_touches_never_claim_a_fill(setup):
    plan, daily, _ = case(setup)
    midpoint = (plan.zone_low + plan.zone_high) / 2
    execution = candles([midpoint] * 3, H4, plan.created - H4)
    for n in (1, 2, 3):
        op = proposal(plan, daily, execution, plan.created + (n - 1) * H4)
        assert op.state == "READY"
        assert op.entry == midpoint
        assert op.evidence["order_armed_at"] == plan.created
        assert op.expires_at == plan.created + ORDER_LIFETIME
        assert_no_trigger(op)


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_price_moving_away_and_returning_does_not_optimize_midpoint(setup):
    plan, daily, execution = case(setup)
    initial = proposal(plan, daily, execution)
    prices = [initial.entry, daily[-1].close, initial.entry]
    execution += candles(prices, H4, plan.created)
    for n in (1, 2, 3):
        op = proposal(plan, daily, execution, plan.created + n * H4)
        assert op.state == "READY"
        assert op.entry == initial.entry
        assert op.id == initial.id
        assert_no_trigger(op)


@pytest.mark.parametrize("setup", ["1", "1S"])
@pytest.mark.parametrize(
    "symbol,bucket,buffer",
    [
        ("BTCUSDT", "crypto", 0.01),
        ("ETHUSDT", "meme", 0.01),
        ("TESTUSDT", "crypto", 0.015),
        ("TESTUSDT", "B4", 0.025),
    ],
)
def test_original_symbol_bucket_stop_buffers(setup, symbol, bucket, buffer):
    plan, daily, execution = case(setup)
    op = select(
        analyze_zones(symbol, daily, execution, plan.created, bucket=bucket), plan
    )
    assert op.stop == plan.invalidation * (1 - plan.direction * buffer)
    assert op.evidence["stop_buffer_pct"] == buffer
    assert op.bucket == bucket


@pytest.mark.parametrize("setup", SETUPS)
def test_resting_is_tier_one_only_and_preserves_risk_multiplier(setup):
    plan, daily, execution = case(setup)
    op = proposal(plan, daily, execution, tier=2)
    assert (op.state, op.reason) == ("WAIT", "TIER_NOT_ELIGIBLE")
    if setup in {"2X", "2XS"}:
        assert op.evidence["risk_multiplier"] == 0.5
    else:
        assert op.evidence["risk_multiplier"] == 1
    assert op.tier == 2
    assert proposal(plan, daily, execution, tier=3).reason == "TIER_NOT_ELIGIBLE"


def test_stale_data_is_wait_and_does_not_hide_a_terminal_status():
    plan, daily, execution = case()
    initial = proposal(plan, daily, execution)
    stale = proposal(plan, daily, execution, plan.created + H4 + engine.DATA_GRACE + 1)
    assert (stale.state, stale.reason) == ("WAIT", "DATA_STALE_OR_MISSING")
    assert stale.evidence["data_valid"] is False
    assert stale.id == initial.id
    assert stale.evidence["terminal_status"] is None
    # Fresh execution cannot compensate for an absent complete daily close.
    execution += candles([daily[-1].close] * 7, H4, plan.created)
    daily_stale = proposal(plan, daily, execution, plan.created + 7 * H4)
    assert (daily_stale.state, daily_stale.reason) == ("WAIT", "DATA_STALE_OR_MISSING")
    execution[1] = touching_bar(execution[1], initial.stop)
    invalid = proposal(plan, daily, execution, plan.created + 7 * H4)
    assert invalid.reason == invalid.evidence["terminal_status"] == "HARD_STOP_BREACHED"
    assert invalid.evidence["data_valid"] is False


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_execution_must_cover_plan_availability_before_ready(setup):
    plan, daily, _ = case(setup)
    execution = candles([daily[-1].close], H4, plan.created - 2 * H4)
    waiting = proposal(plan, daily, execution)
    # The usual freshness allowance alone admits this bar, but it cannot
    # support the research risk contract's armed_at <= execution_closed_at.
    assert plan.created - waiting.evidence["execution_closed_at"] == H4
    assert (waiting.state, waiting.reason) == ("WAIT", "DATA_STALE_OR_MISSING")
    assert waiting.evidence["data_valid"] is False
    execution += candles([daily[-1].close], H4, plan.created - H4)
    ready = proposal(plan, daily, execution)
    assert ready.state == "READY"
    assert ready.id == waiting.id
    assert ready.evidence["order_armed_at"] == ready.evidence["execution_closed_at"]


def test_future_malformed_ohlc_is_ignored_and_bad_closed_data_fails_closed():
    plan, daily, execution = case()
    baseline = analyze_zones("TESTUSDT", daily, execution, plan.created)
    future = Candle(int(plan.created * 1000), math.nan, math.inf, -1, 0, -1)
    assert baseline == analyze_zones(
        "TESTUSDT", daily + [future], execution + [future], plan.created
    )
    for bad in (
        replace(daily[-1], close=math.nan),
        replace(daily[-1], low=0),
        replace(daily[-1], high=1),
        replace(daily[-1], volume=-1),
        replace(daily[-1], open_time=daily[-1].open_time - 1),
    ):
        assert (
            analyze_zones("TESTUSDT", daily[:-1] + [bad], execution, plan.created) == []
        )
    for bad in (
        execution + execution,
        [replace(execution[0], close=math.inf)],
        execution + candles([100], H4, plan.created + H4),
    ):
        assert analyze_zones("TESTUSDT", daily, bad, plan.created + 2 * H4) == []
    for bad in (daily[:20] + daily[21:], daily + [daily[-1]], list(reversed(daily))):
        assert analyze_zones("TESTUSDT", bad, execution, plan.created) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"now": math.nan},
        {"now": math.inf},
        {"now": -1},
        {"now": True},
        {"tier": True},
        {"tier": 1.5},
        {"symbol": " "},
        {"symbol": None},
        {"bucket": ""},
        {"bucket": None},
        {"daily": None},
        {"execution": None},
    ],
)
def test_invalid_public_arguments_return_no_candidates(kwargs):
    plan, daily, execution = case()
    args = dict(symbol="TESTUSDT", daily=daily, execution=execution, now=plan.created)
    args.update(kwargs)
    assert analyze_zones(**args) == []


def test_identity_includes_entry_style_and_is_independent_of_current_snapshot():
    plan, daily, execution = case()
    op = proposal(plan, daily, execution)
    keys = (
        "plan_evidence_id",
        "entry_style",
        "entry",
        "stop",
        "confirmation",
        "stop_buffer_pct",
        "order_armed_at",
        "order_expires_at",
        "plan_history_start",
    )
    frozen_order = {key: op.evidence[key] for key in keys}
    assert engine._id(frozen_order) == op.id
    frozen_order["entry_style"] = "confirmed"
    assert engine._id(frozen_order) != op.id
    later = proposal(plan, daily, execution, plan.created + 1)
    assert later.id == op.id
    assert later.evidence["as_of"] != op.evidence["as_of"]
    assert op == select(
        analyze_zones(" testusdt ", daily, execution, plan.created), plan
    )
    assert asdict(op)["evidence"]["trigger_evidence_id"] is None


@pytest.mark.parametrize("reflected", [False, True])
def test_opposing_ready_plans_wait_without_changing_ids_or_using_future(
    reflected, monkeypatch
):
    plan, daily, execution = case()
    # Supply overlapping validated-plan outputs to isolate the shared conflict
    # policy from the engine's pivot discovery. Each plan is independently READY.
    other = replace(
        plan,
        setup="1S",
        direction=-1,
        zone_low=170,
        zone_high=180,
        invalidation=200,
        target1=140,
        target2=120,
    )
    plans = [plan, other]
    if reflected:
        daily, execution = mirror(daily), mirror(execution)
        plans = [
            replace(
                p,
                setup="1S" if p.setup == "1" else "1",
                direction=-p.direction,
                zone_low=400 - p.zone_high,
                zone_high=400 - p.zone_low,
                invalidation=400 - p.invalidation,
                target1=400 - p.target1,
                target2=400 - p.target2,
            )
            for p in plans
        ]
    independent = []
    for candidate in plans:
        monkeypatch.setattr(engine, "_plans", lambda bars, pivots: [candidate])
        independent += analyze_zones("TESTUSDT", daily, execution, plan.created)
    assert {o.side for o in independent if o.state == "READY"} == {"Buy", "Sell"}
    monkeypatch.setattr(engine, "_plans", lambda bars, pivots: list(reversed(plans)))
    conflicted = analyze_zones("TESTUSDT", daily, execution, plan.created)
    assert conflicted == sorted(
        [
            replace(o, state="WAIT", reason="CONFLICTING_DIRECTIONS")
            for o in independent
        ],
        key=lambda o: (o.created_at, o.id),
    )
    future_daily = daily + candles([daily[-1].close], DAY, plan.created)
    future_execution = execution + candles([daily[-1].close] * 6, H4, plan.created)
    assert conflicted == analyze_zones(
        "TESTUSDT", future_daily, future_execution, plan.created
    )
    later = analyze_zones(
        "TESTUSDT", future_daily, future_execution, plan.created + DAY
    )
    assert [o.id for o in later] == [o.id for o in conflicted]
    assert all(o.reason == "CONFLICTING_DIRECTIONS" for o in later)
    assert all(o.evidence["terminal_status"] is None for o in later)
    # A terminal opponent must not keep blocking the surviving pending plan.
    first = independent[0]
    future_execution[1] = touching_bar(future_execution[1], first.target1)
    resolved = analyze_zones("TESTUSDT", daily, future_execution, plan.created + H4)
    assert (
        next(o for o in resolved if o.id == first.id).reason == "TARGET_ALREADY_PASSED"
    )
    surviving = next(o for o in resolved if o.id != first.id)
    assert (surviving.state, surviving.reason) == ("READY", "ZONE_LIMIT")


@pytest.mark.parametrize("setup", SETUPS)
def test_risk_contract_requires_research_opt_in_and_keeps_rr_hurdles(setup):
    from apex_bot.models import Instrument
    from apex_bot.risk import assess

    plan, daily, execution = case(setup)
    op = proposal(plan, daily, execution)
    instrument = Instrument("TESTUSDT", 0.01, 0.01, 10000, 0.01, 5)
    kwargs = dict(exposures=[], funding_rate_8h=0, spread_pct=0.01, now=plan.created)
    default = assess(op, instrument, 10000, **kwargs)
    assert default["allowed"] is False
    assert "RESTING_RESEARCH_ONLY" in default["reasons"]
    assert default["qty"] == default["risk_cash"] == 0
    research = assess(op, instrument, 10000, allow_resting=True, **kwargs)
    assert research["rr_target1"] is not None
    assert research["rr_blended"] is not None
    if setup == "2S":
        assert research["allowed"] is False
        assert set(research["reasons"]) == {"RR_BLENDED", "RR_ROUNDED_SPLIT"}
        assert research["qty"] == research["risk_cash"] == 0
    else:
        assert research["allowed"] is True, research
        assert research["reasons"] == []
