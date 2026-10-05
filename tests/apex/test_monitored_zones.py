"""Causal offline watches: frozen plans, bounded observations, no fake triggers."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from types import FunctionType, SimpleNamespace

import pytest

from apex_bot import engine, monitored_zones
from apex_bot.models import Candle, Opportunity
from apex_bot.monitored_zones import (
    ADVERSE_CAP_RATE,
    CONTROL_LIFETIME,
    PRIMARY_LIFETIME,
    QUOTE_MAX_AGE,
    analyze_monitored_zones,
    observe_bar,
    observe_quote,
)
from apex_bot.zone_orders import analyze_zones

DAY, H4, HOUR = engine.DAY, engine.H4, 3600
SETUPS = ("1", "1S", "2", "2S", "2X", "2XS", "5S", "5L")
FROZEN_KEYS = (
    "plan_evidence_id",
    "parent_plan_id",
    "evidence_id",
    "entry_style",
    "entry",
    "anchors",
    "daily_evidence_id",
    "stop_buffer_pct",
    "risk_multiplier",
    "zone_low",
    "zone_high",
    "entry_zone_low",
    "entry_zone_high",
    "invalidation",
    "target1",
    "target2",
    "monitored_started_at",
    "order_armed_at",
    "order_expires_at",
    "lifetime_seconds",
    "plan_history_start",
)


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
    return plan, daily, candles([daily[-1].close], H4, plan.created - H4)


def select(ops, plan):
    return next(
        o for o in ops if o.setup == plan.setup and o.created_at == plan.created
    )


def watch(plan, daily, execution, now=None, **kwargs):
    return select(
        analyze_monitored_zones(
            "TESTUSDT", daily, execution, plan.created if now is None else now, **kwargs
        ),
        plan,
    )


def extend(plan, daily, execution, offset):
    price = daily[-1].close
    return (
        daily + candles([price] * int(offset // DAY), DAY, plan.created),
        execution + candles([price] * int(offset // H4), H4, plan.created),
    )


def touching(bar, price):
    return replace(bar, low=min(bar.low, price), high=max(bar.high, price))


def assert_no_trigger(op):
    expected = "ZONE_ARRIVAL" if op.state == "READY" else "ZONE_WATCH"
    assert op.evidence["trigger_kind"] == expected
    for key in ("trigger_closed_at", "trigger_price", "trigger_evidence_id"):
        assert op.evidence[key] is None
    if op.state != "READY":
        for key in ("observation_at", "observation_price", "quote_limit"):
            assert op.evidence[key] is None
    json.dumps(op.to_dict(), allow_nan=False)


def assert_frozen(original, later):
    assert later.id == original.id
    for key in FROZEN_KEYS:
        assert later.evidence[key] == original.evidence[key], key
    for field in (
        "stop",
        "target1",
        "target2",
        "invalidation",
        "confirmation",
        "created_at",
        "expires_at",
    ):
        assert getattr(later, field) == getattr(original, field), field
    assert_no_trigger(later)


@pytest.mark.parametrize("setup", SETUPS)
def test_all_setups_watch_original_frozen_plan_without_confirmation(setup):
    plan, daily, execution = case(setup)
    op = watch(plan, daily, execution)
    parent = select(engine.analyze("TESTUSDT", daily, execution, plan.created), plan)
    resting = select(analyze_zones("TESTUSDT", daily, execution, plan.created), plan)
    assert isinstance(op, Opportunity)
    assert (op.state, op.reason) == ("WAIT", "AWAIT_ZONE")
    assert op.side == ("Buy" if plan.direction > 0 else "Sell")
    assert op.entry == op.evidence["entry"] == (plan.zone_low + plan.zone_high) / 2
    assert (op.stop, op.target1, op.target2, op.invalidation, op.confirmation) == (
        parent.stop,
        parent.target1,
        parent.target2,
        parent.invalidation,
        plan.confirmation,
    )
    assert op.created_at == op.evidence["monitored_started_at"] == plan.created
    assert PRIMARY_LIFETIME == engine.PLAN_LIFETIME == 30 * DAY
    assert CONTROL_LIFETIME == 2 * DAY
    assert op.expires_at == op.evidence["order_expires_at"] == plan.created + 30 * DAY
    assert op.evidence["lifetime_seconds"] == 30 * DAY
    assert op.evidence["parent_plan_id"] == op.evidence["plan_evidence_id"] == parent.id
    assert op.id == op.evidence["evidence_id"]
    assert len({op.id, parent.id, resting.id}) == 3
    assert op.evidence["entry_style"] == "monitored_zone"
    for key in ("anchors", "daily_evidence_id", "risk_multiplier", "stop_buffer_pct"):
        assert op.evidence[key] == parent.evidence[key]
    assert op.evidence["structural_valid"] is op.evidence["data_valid"] is True
    assert op.evidence["terminal_status"] is None
    assert_no_trigger(op)
    assert not any(
        o.setup == setup and o.created_at == plan.created
        for o in analyze_monitored_zones("TESTUSDT", daily, execution, plan.created - 1)
    )


@pytest.mark.parametrize("setup", SETUPS)
def test_valid_beyond_48h_fixed_30day_expiry_never_rearms_or_reprices(setup):
    plan, daily, execution = case(setup)
    initial = watch(plan, daily, execution)
    for offset in (3 * DAY, 29 * DAY, 30 * DAY, 31 * DAY):
        d, e = extend(plan, daily, execution, offset)
        later = watch(plan, d, e, plan.created + offset)
        assert_frozen(initial, later)
        assert later.entry == initial.entry
        if offset < PRIMARY_LIFETIME:
            assert (later.state, later.reason) == ("WAIT", "AWAIT_ZONE")
            arrived = observe_quote(
                later, later.entry, plan.created + offset, plan.created + offset
            )
            assert (arrived.state, arrived.reason) == ("READY", "ZONE_ARRIVAL")
            assert_frozen(initial, arrived)
        else:
            assert (later.state, later.reason) == ("INVALID", "EXPIRED")
            assert later.evidence["terminal_at"] == initial.expires_at


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_48h_control_changes_only_watch_identity_and_deadline(setup):
    plan, daily, execution = case(setup)
    primary = watch(plan, daily, execution)
    control = watch(plan, daily, execution, lifetime_seconds=CONTROL_LIFETIME)
    assert primary.id != control.id
    assert primary.evidence["plan_evidence_id"] == control.evidence["plan_evidence_id"]
    assert control == watch(
        plan, daily, execution, lifetime_seconds=float(CONTROL_LIFETIME)
    )
    resting = select(analyze_zones("TESTUSDT", daily, execution, plan.created), plan)
    assert control.id != resting.id  # Same lifetime, different entry logic.
    for field in (
        "entry",
        "stop",
        "target1",
        "target2",
        "invalidation",
        "confirmation",
    ):
        assert getattr(primary, field) == getattr(control, field)
    d, e = extend(plan, daily, execution, 3 * DAY)
    control_later = watch(
        plan, d, e, plan.created + 3 * DAY, lifetime_seconds=CONTROL_LIFETIME
    )
    assert_frozen(control, control_later)
    assert (control_later.state, control_later.reason) == ("INVALID", "EXPIRED")
    assert watch(plan, d, e, plan.created + 3 * DAY).reason == "AWAIT_ZONE"


def test_exact_lifetime_boundary_and_minimum_duration():
    plan, daily, execution = case()
    short = watch(plan, daily, execution, lifetime_seconds=1)
    assert short.expires_at == plan.created + 1
    assert (
        observe_quote(short, short.entry, plan.created + 0.5, plan.created + 0.5).state
        == "READY"
    )
    terminal = observe_quote(short, short.entry, plan.created + 1, plan.created + 1)
    assert (terminal.state, terminal.reason) == ("INVALID", "EXPIRED")
    assert terminal.evidence["terminal_at"] == short.expires_at
    assert_frozen(short, terminal)


@pytest.mark.parametrize(
    "lifetime", [0, 0.5, -1, 30 * DAY + 1, math.nan, math.inf, True, None, "172800"]
)
def test_invalid_lifetime_fails_closed(lifetime):
    plan, daily, execution = case()
    assert (
        analyze_monitored_zones(
            "TESTUSDT", daily, execution, plan.created, lifetime_seconds=lifetime
        )
        == []
    )


@pytest.mark.parametrize("reflected", [False, True])
def test_every_daily_and_4h_prefix_ignores_future_appends(reflected):
    daily = daily_history()
    execution = candles([160] * (len(daily) * 6), H4)
    if reflected:
        daily, execution = mirror(daily), mirror(execution)
    for n in range(len(daily) + 1):
        now = n * DAY
        assert analyze_monitored_zones(
            "TESTUSDT", daily[:n], execution[: 6 * n], now
        ) == (analyze_monitored_zones("TESTUSDT", daily, execution, now)), n
    for n in range(len(execution) + 1):
        for now in (n * H4, n * H4 - 1):
            assert analyze_monitored_zones("TESTUSDT", daily, execution[:n], now) == (
                analyze_monitored_zones("TESTUSDT", daily, execution, now)
            ), (n, now)


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("failure", ["stop", "target"])
def test_precreation_execution_obsoletes_plan_before_watch(setup, failure):
    plan, daily, _ = case(setup)
    start = daily[plan.history_start].open_time / 1000
    execution = candles(
        [daily[-1].close] * int((plan.created - start) // H4), H4, start
    )
    initial = watch(plan, daily, execution)
    assert initial.reason == "AWAIT_ZONE"
    execution[0] = touching(
        execution[0], initial.stop if failure == "stop" else initial.target1
    )
    invalid = watch(plan, daily, execution)
    expected = "HARD_STOP_BREACHED" if failure == "stop" else "TARGET_ALREADY_PASSED"
    assert (invalid.state, invalid.reason) == ("INVALID", expected)
    assert invalid.evidence["terminal_at"] == plan.created
    assert_frozen(initial, invalid)
    assert observe_quote(invalid, initial.entry, plan.created, plan.created) == invalid


@pytest.mark.parametrize("setup", SETUPS)
def test_daily_close_and_structure_cancellations_preserve_engine_rules(setup):
    plan, daily, execution = case(setup)
    initial = watch(plan, daily, execution)
    bar = touching(candles([daily[-1].close], H4, plan.created)[0], plan.invalidation)
    wick = watch(plan, daily, execution + [bar], plan.created + H4)
    if setup in {"2X", "2XS", "5S", "5L"}:
        assert wick.reason == "STRUCTURE_INVALIDATED"
    else:
        assert wick.reason == "AWAIT_ZONE"
    day = touching(candles([daily[-1].close], DAY, plan.created)[0], plan.invalidation)
    daily += [replace(day, close=plan.invalidation)]
    execution += candles([day.open] * 6, H4, plan.created)
    terminal = watch(plan, daily, execution, plan.created + DAY)
    assert (terminal.state, terminal.reason) == ("INVALID", "DAILY_INVALIDATION")
    assert terminal.evidence["structural_valid"] is False
    assert_frozen(initial, terminal)


@pytest.mark.parametrize("first", ["stop", "target"])
def test_first_chronological_terminal_survives_later_failures_and_staleness(first):
    plan, daily, execution = case()
    initial = watch(plan, daily, execution)
    levels = [initial.stop, initial.target1]
    if first == "target":
        levels.reverse()
    execution += [
        touching(b, p)
        for b, p in zip(candles([daily[-1].close] * 2, H4, plan.created), levels)
    ]
    expected = "HARD_STOP_BREACHED" if first == "stop" else "TARGET_ALREADY_PASSED"
    for now in (plan.created + H4, plan.created + DAY, initial.expires_at + DAY):
        terminal = watch(plan, daily, execution, now)
        assert (terminal.state, terminal.reason) == ("INVALID", expected)
        assert terminal.evidence["terminal_at"] == plan.created + H4
        assert_frozen(initial, terminal)


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_daily_wins_tied_close_and_expiry_wins_its_exact_boundary(setup):
    plan, daily, execution = case(setup)
    original = watch(plan, daily, execution)
    d, e = extend(plan, daily, execution, DAY)
    d[-1] = replace(touching(d[-1], plan.invalidation), close=plan.invalidation)
    e[-1] = touching(e[-1], original.target1)
    assert watch(plan, d, e, plan.created + DAY).reason == "DAILY_INVALIDATION"
    assert (
        watch(plan, d, e, plan.created + DAY, lifetime_seconds=DAY).reason == "EXPIRED"
    )


@pytest.mark.parametrize("setup", SETUPS)
def test_quote_arrival_uses_adverse_cap_no_synthetic_trigger_and_no_mutation(setup):
    plan, daily, execution = case(setup)
    initial = watch(plan, daily, execution)
    snapshot = deepcopy(initial)
    arrived = observe_quote(initial, initial.entry, plan.created, plan.created)
    cap = initial.entry * (1 + plan.direction * ADVERSE_CAP_RATE)
    assert (arrived.state, arrived.reason) == ("READY", "ZONE_ARRIVAL")
    assert arrived.entry == arrived.evidence["quote_limit"] == cap
    assert arrived.evidence["observation_price"] == initial.entry
    assert arrived.evidence["observation_at"] == plan.created
    assert arrived.evidence["entry"] == initial.entry
    assert_frozen(initial, arrived)
    assert initial == snapshot


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_strict_zone_and_cap_have_no_directional_tolerance_or_clamping(setup):
    plan, daily, execution = case(setup)
    initial = watch(plan, daily, execution)
    low, high = plan.zone_low, plan.zone_high
    for price in (low * 0.9999, high * 1.0001):
        outside = observe_quote(initial, price, plan.created, plan.created)
        assert (outside.state, outside.reason) == ("WAIT", "QUOTE_OUTSIDE_ZONE")
        assert_frozen(initial, outside)
    adverse_edge = high if plan.direction > 0 else low
    rejected = observe_quote(initial, adverse_edge, plan.created, plan.created)
    assert (rejected.state, rejected.reason) == ("WAIT", "QUOTE_LIMIT_OUTSIDE_ZONE")
    assert_frozen(initial, rejected)
    favorable_edge = low if plan.direction > 0 else high
    accepted = observe_quote(initial, favorable_edge, plan.created, plan.created)
    assert accepted.state == "READY"
    assert_frozen(initial, accepted)


@pytest.mark.parametrize(
    "price", [None, 0, -1, math.nan, math.inf, -math.inf, True, "100"]
)
def test_missing_or_nonpositive_nonfinite_quote_waits_without_bad_evidence(price):
    plan, daily, execution = case()
    initial = watch(plan, daily, execution)
    result = observe_quote(initial, price, plan.created, plan.created)
    assert (result.state, result.reason) == ("WAIT", "QUOTE_STALE_OR_MISSING")
    assert_frozen(initial, result)


def test_quote_age_includes_boundary_and_rejects_future_or_preplan():
    plan, daily, execution = case()
    initial = watch(plan, daily, execution)
    for at in (None, True, math.nan, math.inf, plan.created - 1, plan.created + 1):
        result = observe_quote(initial, initial.entry, at, plan.created)
        assert result.reason == "QUOTE_STALE_OR_MISSING"
        assert_frozen(initial, result)
    assert (
        observe_quote(
            initial, initial.entry, plan.created, plan.created + QUOTE_MAX_AGE
        ).state
        == "READY"
    )
    stale = observe_quote(
        initial, initial.entry, plan.created, plan.created + QUOTE_MAX_AGE + 0.01
    )
    assert stale.reason == "QUOTE_STALE_OR_MISSING"
    assert_frozen(initial, stale)


def test_repeat_quotes_keep_identity_clear_old_offers_and_do_not_invalidate_for_rr():
    plan, daily, execution = case("2S")
    initial = watch(plan, daily, execution)
    current = initial
    for price in (
        initial.entry,
        plan.zone_high * 1.0001,
        None,
        plan.zone_low * 1.001,
        initial.entry,
    ):
        current = observe_quote(current, price, plan.created, plan.created)
        assert_frozen(initial, current)
        assert current.evidence["terminal_status"] is None
        assert current.state in {"WAIT", "READY"}
    assert current.state == "READY"
    # This actual fixture fails the parent's net blended hurdle at midpoint;
    # even raw R:R below 2.5 near the adverse edge must not kill the watch.
    adverse = observe_quote(initial, plan.zone_low * 1.001, plan.created, plan.created)
    raw_rr = abs((adverse.target1 + adverse.target2) / 2 - adverse.entry) / abs(
        adverse.entry - adverse.stop
    )
    assert raw_rr < 2.5
    assert adverse.state == "READY"
    assert adverse.evidence["terminal_status"] is None


@pytest.mark.parametrize("setup", ["1", "1S"])
def test_expiry_is_terminal_for_missing_and_outside_quotes_and_never_resets(setup):
    plan, daily, execution = case(setup)
    initial = watch(plan, daily, execution)
    for price in (None, initial.entry, plan.zone_high * 2):
        expired = observe_quote(initial, price, initial.expires_at, initial.expires_at)
        assert (expired.state, expired.reason) == ("INVALID", "EXPIRED")
        assert expired.evidence["terminal_at"] == initial.expires_at
        assert_frozen(initial, expired)
        assert (
            observe_quote(expired, initial.entry, plan.created, plan.created) == expired
        )


@pytest.mark.parametrize("tier", [2, 3])
def test_tier_gate_cannot_be_overridden_by_quote(tier):
    plan, daily, execution = case()
    initial = watch(plan, daily, execution, tier=tier)
    assert initial.reason == "TIER_NOT_ELIGIBLE"
    observed = observe_quote(initial, initial.entry, plan.created, plan.created)
    assert (observed.state, observed.reason) == ("WAIT", "TIER_NOT_ELIGIBLE")
    assert_frozen(initial, observed)


def test_quote_never_refreshes_stale_missing_or_precreation_4h_evidence():
    plan, daily, execution = case()
    initial = watch(plan, daily, execution)
    for e in ([], candles([daily[-1].close], H4, plan.created - 2 * H4)):
        missing = watch(plan, daily, e)
        assert missing.reason == "DATA_STALE_OR_MISSING"
        result = observe_quote(missing, missing.entry, plan.created, plan.created)
        assert result.reason == "DATA_STALE_OR_MISSING"
    now = plan.created + H4 + engine.DATA_GRACE + 1
    stale = observe_quote(initial, initial.entry, now, now)
    assert (stale.state, stale.reason) == ("WAIT", "DATA_STALE_OR_MISSING")
    assert stale.evidence["data_valid"] is False
    assert_frozen(initial, stale)
    d, e = extend(plan, daily, execution, 3 * DAY)
    recovered = watch(plan, d, e, plan.created + 3 * DAY)
    assert_frozen(initial, recovered)
    assert recovered.reason == "AWAIT_ZONE"


@pytest.mark.parametrize("reflected", [False, True])
def test_opposing_eligible_watches_block_quotes_until_one_is_obsolete(
    reflected, monkeypatch
):
    plan, daily, execution = case()
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
        independent += analyze_monitored_zones(
            "TESTUSDT", daily, execution, plan.created
        )
    assert {o.side for o in independent if o.reason == "AWAIT_ZONE"} == {"Buy", "Sell"}
    monkeypatch.setattr(engine, "_plans", lambda bars, pivots: list(reversed(plans)))
    conflicted = analyze_monitored_zones("TESTUSDT", daily, execution, plan.created)
    assert conflicted == sorted(
        [replace(o, reason="CONFLICTING_DIRECTIONS") for o in independent],
        key=lambda o: (o.created_at, o.id),
    )
    for op in conflicted:
        assert (
            observe_quote(op, op.entry, plan.created, plan.created).reason
            == "CONFLICTING_DIRECTIONS"
        )
    first = independent[0]
    bar = touching(candles([daily[-1].close], H4, plan.created)[0], first.target1)
    resolved = analyze_monitored_zones(
        "TESTUSDT", daily, execution + [bar], plan.created + H4
    )
    assert (
        next(o for o in resolved if o.id == first.id).reason == "TARGET_ALREADY_PASSED"
    )
    survivor = next(o for o in resolved if o.id != first.id)
    assert survivor.reason == "AWAIT_ZONE"


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("failure", ["stop", "structure", "target"])
def test_hourly_cancellation_precedes_quote_and_persists_tombstone(setup, failure):
    plan, daily, execution = case(setup)
    initial = watch(plan, daily, execution)
    snapshot = deepcopy(initial)
    bar = candles([daily[-1].close], HOUR, plan.created)[0]
    level = {
        "stop": initial.stop,
        "structure": initial.invalidation,
        "target": initial.target1,
    }[failure]
    bar = touching(bar, level)
    now = plan.created + HOUR
    updated = observe_bar(initial, bar, now)
    assert initial == snapshot
    assert_frozen(initial, updated)
    if failure == "structure" and setup not in {"2X", "2XS", "5S", "5L"}:
        assert updated.reason == "AWAIT_ZONE"
    else:
        expected = {
            "stop": "HARD_STOP_BREACHED",
            "structure": "STRUCTURE_INVALIDATED",
            "target": "TARGET_ALREADY_PASSED",
        }[failure]
        assert (updated.state, updated.reason) == ("INVALID", expected)
        assert updated.evidence["terminal_at"] == now
        assert updated.evidence["structural_valid"] is (failure == "target")
        assert observe_quote(updated, initial.entry, now, now) == updated
        for later in (now + HOUR, initial.expires_at + DAY):
            assert observe_bar(updated, bar, later) == updated


@pytest.mark.parametrize("setup", ["1", "1S", "2X", "2XS"])
def test_hourly_stop_precedes_target_and_structure_in_ambiguous_bar(setup):
    plan, daily, execution = case(setup)
    initial = watch(plan, daily, execution)
    bar = candles([daily[-1].close], HOUR, plan.created)[0]
    ambiguous = touching(touching(bar, initial.stop), initial.target1)
    assert (
        observe_bar(initial, ambiguous, plan.created + HOUR).reason
        == "HARD_STOP_BREACHED"
    )
    if setup in {"2X", "2XS"}:
        structural = touching(touching(bar, initial.invalidation), initial.target1)
        assert (
            observe_bar(initial, structural, plan.created + HOUR).reason
            == "STRUCTURE_INVALIDATED"
        )


def test_hourly_observations_ignore_future_precreation_and_incomplete_bars():
    plan, daily, execution = case()
    initial = watch(plan, daily, execution)
    bar = touching(candles([daily[-1].close], HOUR, plan.created)[0], initial.stop)
    assert observe_bar(initial, bar, plan.created + HOUR - 1) == initial
    future_bad = replace(bar, high=math.inf, low=-1, close=math.nan)
    assert observe_bar(initial, future_bad, plan.created + HOUR - 1) == initial
    prior = replace(bar, open_time=int((plan.created - 2 * HOUR) * 1000))
    assert observe_bar(initial, prior, plan.created + HOUR) == initial
    at_creation = replace(bar, open_time=int((plan.created - HOUR) * 1000))
    terminal = observe_bar(initial, at_creation, plan.created)
    assert terminal.reason == "HARD_STOP_BREACHED"
    assert terminal.evidence["terminal_at"] == plan.created


def test_hourly_expiry_boundary_and_earlier_event_at_late_observation():
    plan, daily, execution = case()
    initial = watch(plan, daily, execution, lifetime_seconds=2 * HOUR)
    first = touching(candles([daily[-1].close], HOUR, plan.created)[0], initial.target1)
    late = observe_bar(initial, first, initial.expires_at + DAY)
    assert late.reason == "TARGET_ALREADY_PASSED"
    assert late.evidence["terminal_at"] == plan.created + HOUR
    at_expiry = replace(first, open_time=int((initial.expires_at - HOUR) * 1000))
    expired = observe_bar(initial, at_expiry, initial.expires_at)
    assert expired.reason == "EXPIRED"
    assert expired.evidence["terminal_at"] == initial.expires_at
    future = replace(first, open_time=int((initial.expires_at + DAY) * 1000))
    assert observe_bar(initial, future, initial.expires_at).reason == "EXPIRED"


def test_hourly_bar_never_refreshes_4h_data_or_reuses_quote_as_trigger():
    plan, daily, execution = case()
    initial = watch(plan, daily, execution)
    now = plan.created + HOUR
    bar = candles([initial.entry], HOUR, plan.created)[0]
    updated = observe_bar(initial, bar, now)
    assert (
        updated.evidence["execution_closed_at"]
        == initial.evidence["execution_closed_at"]
    )
    assert updated.evidence["daily_closed_at"] == initial.evidence["daily_closed_at"]
    assert updated.reason == "AWAIT_ZONE"
    # Parent's 0.01% spread: quote is close plus the long's half-spread.
    quote = bar.close * (1 + 0.0001 / 2)
    arrived = observe_quote(updated, quote, now, now)
    assert arrived.state == "READY"
    assert_frozen(initial, arrived)
    later_bar = replace(bar, open_time=bar.open_time + HOUR * 1000)
    cleared = observe_bar(arrived, later_bar, now + HOUR)
    assert cleared.reason == "AWAIT_ZONE"
    assert_frozen(initial, cleared)


@pytest.mark.parametrize("kind", ["nonfinite", "inconsistent", "unaligned", "missing"])
def test_malformed_closed_bar_blocks_quote_without_structural_invalidation(kind):
    plan, daily, execution = case()
    initial = watch(plan, daily, execution)
    bar = candles([initial.entry], HOUR, plan.created)[0]
    bar = {
        "nonfinite": replace(bar, close=math.nan),
        "inconsistent": replace(bar, high=1),
        "unaligned": replace(bar, open_time=bar.open_time + 1),
        "missing": None,
    }[kind]
    now = plan.created + HOUR + 1
    rejected = observe_bar(initial, bar, now)
    assert (rejected.state, rejected.reason) == ("WAIT", "INVALID_BAR")
    assert rejected.evidence["terminal_status"] is None
    assert observe_quote(rejected, initial.entry, now, now).reason == "INVALID_BAR"


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
def test_invalid_analyzer_inputs_fail_closed(kwargs):
    plan, daily, execution = case()
    args = dict(symbol="TESTUSDT", daily=daily, execution=execution, now=plan.created)
    args.update(kwargs)
    assert analyze_monitored_zones(**args) == []


def test_closed_history_validation_ignores_malformed_future_ohlc():
    plan, daily, execution = case()
    expected = analyze_monitored_zones("TESTUSDT", daily, execution, plan.created)
    future = Candle(int(plan.created * 1000), math.nan, math.inf, -1, 0, -1)
    assert expected == analyze_monitored_zones(
        "TESTUSDT", daily + [future], execution + [future], plan.created
    )
    for d, e in (
        (daily[:-1] + [replace(daily[-1], low=0)], execution),
        (daily[:20] + daily[21:], execution),
        (daily, execution + execution),
        (daily, [replace(execution[-1], close=math.nan)]),
    ):
        assert analyze_monitored_zones("TESTUSDT", d, e, plan.created) == []


def test_private_cache_globals_and_keyword_forwarding_are_compatible():
    plan, daily, execution = case()
    original = analyze_monitored_zones.__globals__
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return original["_evaluate"](*args, **kwargs)

    namespace = dict(
        original, engine=SimpleNamespace(**vars(original["engine"])), _evaluate=evaluate
    )
    cached = FunctionType(
        analyze_monitored_zones.__code__,
        namespace,
        analyze_monitored_zones.__name__,
        analyze_monitored_zones.__defaults__,
    )
    cached.__kwdefaults__ = deepcopy(analyze_monitored_zones.__kwdefaults__)
    for lifetime in (PRIMARY_LIFETIME, CONTROL_LIFETIME):
        assert cached(
            "TESTUSDT", daily, execution, plan.created, lifetime_seconds=lifetime
        ) == (
            analyze_monitored_zones(
                "TESTUSDT", daily, execution, plan.created, lifetime_seconds=lifetime
            )
        )
        assert calls[-1] == {"lifetime_seconds": float(lifetime)}
    assert monitored_zones.engine is engine
