"""Hand-calculated collateral checks and frozen long/short replay regressions."""

from copy import deepcopy
from dataclasses import replace
import inspect
import math

import pytest

from apex_bot import risk, simulation
from apex_bot.engine import DAY, H4
from apex_bot.leverage_research_accounting import (
    LeverageModel,
    account_snapshot,
    guard,
    pending_requirement,
)
from apex_bot.models import Instrument
from apex_bot.spot_research_accounting import ResearchModel

from .test_replay_management import HOUR, histories, new_trade as managed_trade
from .test_risk import NOW, monitored_opportunity
from .test_simulation import STEP, bar, monitored_trade, new_trade, target_history


FEE = 0.00055


def price_history(side):
    bars = target_history()
    if side == "Buy":
        return bars
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


def pending(side="Buy", **kwargs):
    return dict(new_trade(side, **kwargs), research_entry_fee_rate=FEE)


def conservation(account, initial=10000):
    assert account["wallet_balance"] == pytest.approx(initial + account["realized_net"])
    assert account["equity"] == pytest.approx(
        account["wallet_balance"] + account["unrealized"]
    )
    assert account["cash"] + account["open_margin"] == pytest.approx(
        account["wallet_balance"]
    )
    assert account["initial_margin"] == account["open_margin"]
    assert account["available_cash"] + account["reserved_pending"] + account[
        "reserved_close_fees"
    ] == pytest.approx(account["cash"])


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_leverage_does_not_change_risk_sizing_and_fee_clone_is_isolated(side):
    model = LeverageModel([], [])
    assert inspect.signature(model.assess) == inspect.signature(risk.assess)
    assert model.assess.__globals__ is not risk.assess.__globals__
    op = monitored_opportunity(side)
    results = [
        model.assess(
            op,
            Instrument("TESTUSDT", 0.01, 0.01, 100000, 0.01, 5, max_leverage=lev),
            10000,
            exposures=[],
            funding_rate_8h=0,
            spread_pct=0.01,
            now=NOW,
            allow_monitored=True,
        )
        for lev in (5, 10)
    ]
    assert results[0]["allowed"], results
    assert results[0] == results[1]
    # Unchanged cautious 0.25% budget, quantity determined by loss to stop+costs.
    unit_risk = (
        abs(results[0]["entry"] - results[0]["stop"]) + results[0]["entry"] * 0.0018
    )
    assert results[0]["qty"] == pytest.approx(math.floor(25 / unit_risk / 0.01) * 0.01)
    assert results[0]["risk_cash"] <= 25


@pytest.mark.parametrize("side", ["Buy", "Sell"])
@pytest.mark.parametrize("leverage", [5, 10])
def test_partial_exits_funding_and_balances_hand_calculated(side, leverage):
    model = LeverageModel([], [])
    path = price_history(side)
    sign = 1 if side == "Buy" else -1
    opened = model.advance(new_trade(side), path[:1], STEP)
    first = account_snapshot([opened], {"TESTUSDT": 100}, leverage)
    assert first["wallet_balance"] == pytest.approx(10000 - 0.275)
    assert first["open_margin"] == 500 / leverage
    assert first["reserved_close_fees"] == pytest.approx(500 * (1 + 1 / leverage) * FEE)
    assert first["unrealized"] == 0
    partial = model.advance(opened, path[:3], 3 * STEP)
    # Three of five lots close at T1 with side-aware 3bp adverse slippage.
    exit1 = (110 if sign == 1 else 90) * (1 - sign * 0.0003)
    pnl1 = sign * 3 * (exit1 - 100) - 0.275 - 3 * exit1 * FEE
    assert partial["remaining"] == 2
    assert partial["net_pnl"] == pytest.approx(pnl1)
    funded = simulation.apply_funding(
        partial,
        [
            {
                "fundingRateTimestamp": int(3 * STEP * 1000),
                "fundingRate": 0.0001,
                "symbol": "TESTUSDT",
            }
        ],
        3 * STEP,
    )
    # Settlement at a partial-exit timestamp charges only the remaining lots.
    assert funded["funding"] == pytest.approx(-sign * 0.02)
    mark = 111 if sign == 1 else 89
    current = account_snapshot([funded], {"TESTUSDT": mark}, leverage)
    assert current["wallet_balance"] == pytest.approx(10000 + pnl1 - sign * 0.02)
    assert current["open_margin"] == 200 / leverage
    assert current["gross_notional"] == 2 * mark
    assert current["unrealized"] == 22
    assert current["cash"] == pytest.approx(current["wallet_balance"] - 200 / leverage)
    closed = model.advance(funded, path, 5 * STEP)
    exit2 = (130 if sign == 1 else 70) * (1 - sign * 0.0003)
    expected = pnl1 - sign * 0.02 + sign * 2 * (exit2 - 100) - 2 * exit2 * FEE
    assert closed["net_pnl"] == pytest.approx(expected)
    finished = account_snapshot([closed], {}, leverage)
    assert finished["equity"] == pytest.approx(10000 + expected)
    assert finished["open_margin"] == finished["reserved_close_fees"] == 0
    assert finished["available_cash"] == finished["equity"]
    for snapshot in (first, current, finished):
        conservation(snapshot)


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_same_fills_equal_equity_and_half_initial_margin_at_ten_x(side):
    model = LeverageModel([], [])
    trade = model.advance(new_trade(side), price_history(side)[:3], 3 * STEP)
    snapshots = [account_snapshot([trade], {"TESTUSDT": 105}, lev) for lev in (5, 10)]
    five, ten = snapshots
    assert five["equity"] == ten["equity"]
    assert five["realized_net"] == ten["realized_net"]
    assert five["unrealized"] == ten["unrealized"]
    assert five["open_margin"] == 2 * ten["open_margin"]
    assert ten["available_cash"] - five["available_cash"] == pytest.approx(
        20 + 20 * FEE
    )
    # Unrealized losses/profits affect equity, never become spendable collateral.
    changed = account_snapshot([trade], {"TESTUSDT": 180}, 5)
    assert changed["available_cash"] == five["available_cash"]
    assert changed["equity"] != five["equity"]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
@pytest.mark.parametrize("leverage", [5, 10])
def test_pending_fees_reserved_once_and_release_on_fill_cancel_expire(side, leverage):
    initial = pending(side)
    before = deepcopy(initial)
    snap = account_snapshot([initial], {}, leverage)
    expected = 500 / leverage + 0.275 + 500 * (1 + 1 / leverage) * FEE
    assert pending_requirement(5, 100, leverage) == pytest.approx(expected)
    assert snap["reserved_pending"] == pytest.approx(expected)
    assert snap["equity"] == 10000
    opened = LeverageModel([], []).advance(initial, [bar(0)], STEP)
    filled = account_snapshot([opened], {"TESTUSDT": 100}, leverage)
    assert filled["available_cash"] == pytest.approx(snap["available_cash"])
    assert filled["reserved_pending"] == 0
    assert filled["equity"] == pytest.approx(10000 - 0.275)
    for status in ("CANCELLED", "EXPIRED"):
        released = account_snapshot([dict(initial, status=status)], {}, leverage)
        assert released["available_cash"] == 10000
        assert released["reserved_pending"] == 0
    assert initial == before


def test_shared_book_counts_pending_both_sides_partial_and_closed_without_netting_margin():
    model = LeverageModel([], [])
    opened = model.advance(monitored_trade("Sell"), [bar(0)], STEP)
    assert opened["opened_at"] == 0
    partial = model.advance(
        new_trade(identity="partial"), target_history()[:3], 3 * STEP
    )
    closed = model.advance(new_trade(identity="closed"), target_history(), 5 * STEP)
    waiting = pending("Sell", identity="pending")
    trades = [opened, partial, closed, waiting]
    before = deepcopy(trades)
    snap = account_snapshot(iter(trades), {"TESTUSDT": 111}, 5)
    assert snap["open_count"] == 2 and snap["pending_count"] == 1
    assert snap["open_margin"] == pytest.approx((5 * opened["entry"] + 200) / 5)
    assert snap["unrealized"] == pytest.approx(5 * (opened["entry"] - 111) + 22)
    assert snap["realized_net"] == pytest.approx(sum(t["net_pnl"] for t in trades))
    assert snap["gross_notional"] == 7 * 111
    conservation(snap)
    assert trades == before


@pytest.mark.parametrize("leverage", [5, 10])
def test_monitored_short_reserves_favorable_upper_zone_fill_not_sell_limit(leverage):
    pending_short = dict(monitored_trade("Sell"), research_entry_fee_rate=FEE)
    before = account_snapshot([pending_short], {}, leverage)
    expected = pending_requirement(
        5, pending_short["limit"], leverage, upper_fill_price=102
    )
    assert before["reserved_pending"] == pytest.approx(expected)
    # Short sells above its IOC minimum at an improved price, still inside zone.
    opened = LeverageModel([], []).advance(
        pending_short,
        [bar(0, open=101.9, high=102, low=101, close=101.5)],
        STEP,
    )
    assert opened["status"] == "OPEN" and opened["entry"] > pending_short["limit"]
    after = account_snapshot([opened], {"TESTUSDT": 101.5}, leverage)
    assert after["available_cash"] >= before["available_cash"]
    with pytest.raises(ValueError, match="short upper fill price"):
        account_snapshot([dict(pending_short, zone_high=None)], {}, leverage)


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_both_sides_same_bar_stop_losses_are_not_bounded_by_initial_margin(side):
    sign = 1 if side == "Buy" else -1
    b = bar(0, low=80) if side == "Buy" else bar(0, high=120)
    t = LeverageModel([], []).advance(new_trade(side), [b], STEP)
    assert t["status"] == "CLOSED" and t["exit_reason"] == "STOP"
    px = (90 if sign == 1 else 110) * (1 - sign * 0.0003)
    assert t["net_pnl"] == pytest.approx(sign * 5 * (px - 100) - 0.275 - 5 * px * FEE)
    # This model does not silently cap losses or pretend it simulated liquidation.
    assert "no-liquidation-engine" in t["cost_model"]
    account = account_snapshot([t], {}, 10, initial=10)
    assert account["available_cash"] < 0
    conservation(account, initial=10)


@pytest.mark.parametrize("side", ["Buy", "Sell"])
@pytest.mark.parametrize("outcome", ["timeout", "daily"])
def test_management_fee_clones_apply_to_long_short_non_price_exits(side, outcome):
    days = 22 if outcome == "timeout" else 3
    prices = [100] * (days * 24)
    if outcome == "daily":
        prices[-1] = 94.5
    bars, daily, execution = histories(prices, side)
    model = LeverageModel(daily, execution)
    trade = model.advance(
        managed_trade(14 * H4, side),
        bars,
        days * DAY,
        interval_ms=HOUR * 1000,
        daily=daily,
        execution=execution,
    )
    assert not trade.get("data_error") and not trade.get("data_gap")
    assert trade["exit_reason"] == (
        "TIMEOUT" if outcome == "timeout" else "DAILY_INVALIDATION"
    )
    assert trade["fees"] == pytest.approx(
        sum(f["qty"] * f["price"] * FEE for f in trade["fills"])
    )
    assert model._management.validation_hits > 0


def test_long_replay_finances_match_previous_frozen_perpetual_model():
    prior = ResearchModel("perpetual", [], []).advance(
        new_trade(), target_history(), 5 * STEP
    )
    current = LeverageModel([], []).advance(new_trade(), target_history(), 5 * STEP)
    for key in (
        "entry",
        "qty",
        "remaining",
        "gross_pnl",
        "fees",
        "net_pnl",
        "funding",
        "status",
        "opened_at",
        "closed_at",
        "stop",
        "target1",
        "target2",
    ):
        assert current[key] == prior[key]
    for old, new in zip(prior["fills"], current["fills"]):
        for key in ("time", "qty", "price", "reason", "fee", "gross_pnl"):
            assert old[key] == new[key]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_streaming_batch_idempotence_and_inputs_not_mutated(side):
    initial = new_trade(side)
    original = deepcopy(initial)
    bars = price_history(side)
    model = LeverageModel([], [])
    streamed = initial
    for count in range(1, 6):
        streamed = model.advance(streamed, bars[:count], count * STEP)
    batch = model.advance(initial, bars, 5 * STEP)
    assert streamed == batch
    assert model.advance(batch, bars, 5 * STEP) == batch
    assert initial == original


def test_separate_fee_instances_leave_production_and_prior_research_untouched():
    functions = [risk.assess, simulation.advance, simulation._manage, simulation._exit]
    before = [dict(f.__globals__) for f in functions]
    original = simulation.advance(new_trade("Sell"), price_history("Sell"), 5 * STEP)
    models = [LeverageModel([], [], fee_rate=f) for f in (0, FEE, 0.004)]
    results = [
        m.advance(new_trade("Sell"), price_history("Sell"), 5 * STEP) for m in models
    ]
    assert len({t["fees"] for t in results}) == 3
    assert results[0]["fees"] == 0
    assert (
        simulation.advance(new_trade("Sell"), price_history("Sell"), 5 * STEP)
        == original
    )
    for fn, saved in zip(functions, before):
        assert fn.__globals__.keys() == saved.keys()
        assert all(fn.__globals__[key] is value for key, value in saved.items())


@pytest.mark.parametrize("side", ["Buy", "Sell"])
@pytest.mark.parametrize("leverage", [5, 10])
@pytest.mark.parametrize("mmr", [0.005, 0.01, 0.025])
def test_guard_matches_runtime_rule_both_directions_and_mmr_sensitivity(
    side, leverage, mmr
):
    sign = 1 if side == "Buy" else -1
    buffer = 100 * (1 / leverage - mmr - 0.002)
    # On opposite sides of the exact runtime boundary, using a clear tiny delta.
    for fraction, allowed in ((0.499999, True), (0.500001, False)):
        result = guard(100, 100 - sign * buffer * fraction, side, leverage, mmr)
        assert result["allowed"] is allowed
        assert result["buffer_price"] == pytest.approx(buffer)
        assert result["estimated_liquidation_price"] == pytest.approx(
            100 - sign * buffer
        )
        assert result["liquidation_verified"] is False
        assert bool(result["reasons"]) is not allowed


def test_guard_exact_boundary_and_five_vs_ten_x_different_admissions():
    # Both quantities/risk remain unchanged; only collateral safety differs.
    assert guard(100, 94, "Buy", 5, 0.01)["allowed"]
    assert not guard(100, 94, "Buy", 10, 0.01)["allowed"]
    # Choose exactly representable stop/distance; mmr=.038 yields buffer=16.
    result = guard(100, 92, "Buy", 5, 0.038)
    assert result["buffer_price"] == 16
    assert result["allowed"]
    assert not guard(100, 99, "Buy", 10, 0.11)["allowed"]


@pytest.mark.parametrize(
    "changes",
    [
        dict(entry=0),
        dict(entry=math.inf),
        dict(stop=math.nan),
        dict(stop=100),
        dict(stop=110),
        dict(side="long"),
        dict(leverage=0),
        dict(leverage=0.5),
        dict(leverage=True),
        dict(leverage=math.inf),
        dict(maintenance_rate=None),
        dict(maintenance_rate=-0.01),
        dict(maintenance_rate=1),
        dict(maintenance_rate=math.nan),
    ],
)
def test_guard_bad_inputs_reject_without_inventing_tier(changes):
    args = dict(entry=100, stop=95, side="Buy", leverage=5, maintenance_rate=0.01)
    args.update(changes)
    result = guard(**args)
    assert not result["allowed"] and result["reasons"]
    assert result["buffer_price"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("side", "long"),
        ("qty", True),
        ("remaining", -1),
        ("remaining", 6),
        ("remaining", 0),
        ("entry", math.nan),
        ("entry", 0),
        ("fees", -1),
        ("net_pnl", 10),
        ("status", "CLOSED"),
        ("opened_at", None),
        ("research_market", "spot"),
        ("research_entry_fee_rate", None),
        ("funding_error", "missing"),
        ("data_gap", "missing"),
        ("research_leverage", 10),
    ],
)
def test_invalid_financial_state_rejects_valuation(field, value):
    trade = LeverageModel([], []).advance(new_trade(), [bar(0)], STEP)
    with pytest.raises(ValueError):
        account_snapshot([dict(trade, **{field: value})], {"TESTUSDT": 100}, 5)


@pytest.mark.parametrize("bad", [0, -1, 0.5, True, None, "5", math.nan, math.inf])
def test_invalid_leverage_rejects_account_and_requirement(bad):
    with pytest.raises(ValueError):
        account_snapshot([], {}, bad)
    with pytest.raises(ValueError):
        pending_requirement(5, 100, bad)


def test_duplicates_missing_quote_negative_cash_and_unfilled_cashflows():
    p = pending()
    with pytest.raises(ValueError, match="Duplicate"):
        account_snapshot([p, p], {}, 5)
    with pytest.raises(ValueError, match="cashflows"):
        account_snapshot([dict(p, net_pnl=1)], {}, 5)
    opened = LeverageModel([], []).advance(p, [bar(0)], STEP)
    with pytest.raises(ValueError, match="Missing current closed quote"):
        account_snapshot([opened], {}, 5)
    assert account_snapshot([p], {}, 5, initial=1)["available_cash"] < 0


def test_data_gap_cannot_be_valued_until_repaired_and_no_state_corruption():
    model = LeverageModel([], [])
    initial = new_trade("Sell")
    gapped = model.advance(initial, [bar(2)], 3 * STEP)
    assert gapped["data_gap"]
    with pytest.raises(ValueError, match="data_gap"):
        account_snapshot([gapped], {}, 5)
    fixed = model.advance(gapped, [bar(0), bar(1), bar(2)], 3 * STEP)
    assert fixed == model.advance(initial, [bar(0), bar(1), bar(2)], 3 * STEP)
    assert not fixed.get("data_gap")


@pytest.mark.parametrize("fee", [True, -1, 1, math.nan, math.inf, "0.001"])
def test_invalid_fee_rejects(fee):
    with pytest.raises(ValueError):
        LeverageModel([], [], fee)
    with pytest.raises(ValueError):
        pending_requirement(5, 100, 5, fee)
