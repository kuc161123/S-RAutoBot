"""Synthetic OFFLINE accounting checks; no production performance claims."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import inspect
import math

import pytest

from apex_bot import risk, simulation
from apex_bot.engine import DAY, H4
from apex_bot.models import Instrument
from apex_bot.replay_management import _ManagementCache
from apex_bot.spot_research_accounting import ResearchModel, account_snapshot

from .test_replay_management import HOUR, histories, new_trade as managed_trade
from .test_risk import NOW, opportunity
from .test_simulation import STEP, bar, monitored_trade, new_trade, target_history


@pytest.fixture(params=["spot", "perpetual"])
def model(request):
    return ResearchModel(request.param, [], [])


def snapshot(model, trades, price=100, initial=10000):
    return account_snapshot(trades, {"TESTUSDT": price}, initial, model.market)


def pending(model, **kwargs):
    return dict(new_trade(**kwargs), research_entry_fee_rate=model.entry_fee_rate)


def assert_conservation(account, initial=10000):
    assert account["equity"] == pytest.approx(initial + account["realized_net"] + account["unrealized"])
    assert account["equity"] == pytest.approx(account["cash"] + account["invested_value"])
    assert account["available_cash"] + account["reserved_pending"] == pytest.approx(account["cash"])


@pytest.mark.parametrize("market,fee", [("spot", .001), ("perpetual", .00055)])
def test_defaults_and_frozen_risk_cost_allowance(market, fee):
    model = ResearchModel(market, [], [])
    entry_rate = fee / (1 - fee) if market == "spot" else fee
    assert model.fee_rate == fee
    assert model.entry_fee_rate == entry_rate
    assert model.assess.__code__ is risk.assess.__code__
    assert model.assess.__globals__ is not risk.assess.__globals__
    assert inspect.signature(model.assess) == inspect.signature(risk.assess)
    base = model.assess.__globals__["BASE_COST_RATE"]
    assert isinstance(base, Decimal)
    assert float(base) == pytest.approx(entry_rate + fee + .0006)
    result = model.assess(
        opportunity(), Instrument("TESTUSDT", .01, .01, 100000, .01, 5),
        10000, exposures=[], funding_rate_8h=0, spread_pct=.01, now=NOW,
    )
    assert result["allowed"], result
    # Entry=100, stop=94, budget=25; .01% spread is added exactly once.
    unit_risk = 6 + 100 * (entry_rate + fee + .0006 + .0001)
    qty = math.floor(25 / unit_risk / .01) * .01
    assert result["qty"] == pytest.approx(qty)
    assert result["risk_cash"] == pytest.approx(qty * unit_risk)
    assert result["rr_target1"] == pytest.approx((12 - (unit_risk - 6)) / unit_risk)
    assert not model.assess(
        opportunity(), Instrument("TESTUSDT", .01, .01, 100000, .01, 5),
        10000, exposures=[], funding_rate_8h=0, spread_pct=.21, now=NOW,
    )["allowed"]


def test_flat_price_roundtrip_keeps_frozen_exit_slippage(model):
    # One net lot reaches T1 without selling, then stops at entry on next bar.
    bars = [bar(0), bar(1, open=109, high=111, low=99, close=110), bar(2)]
    closed = model.advance(new_trade(qty=1), bars, 3 * STEP)
    assert closed["exit_reason"] == "STOP" and closed["remaining"] == 0
    assert [fill["qty"] for fill in closed["fills"]] == [1, 1]
    # A flat reference-price roundtrip sells at 99.97 under frozen 3bp slippage.
    expected = 99.97 - 100 - 100 * model.entry_fee_rate - 99.97 * model.fee_rate
    assert closed["gross_pnl"] == pytest.approx(-.03)
    assert closed["net_pnl"] == pytest.approx(expected)
    account = account_snapshot([closed], {}, market=model.market)
    assert account["equity"] == pytest.approx(10000 + expected)
    assert account["cash"] == account["equity"]
    assert account["open_count"] == account["pending_count"] == 0
    assert_conservation(account)


def test_positive_price_roundtrip_and_partial_cash_are_hand_calculated(model):
    initial = new_trade()
    opened = model.advance(initial, target_history()[:1], STEP)
    bought = snapshot(model, [opened], price=100)
    assert bought["cash"] == pytest.approx(10000 - 500 - 500 * model.entry_fee_rate)
    assert bought["equity"] == pytest.approx(10000 - 500 * model.entry_fee_rate)
    assert bought["invested_value"] == 500
    partial = model.advance(opened, target_history()[:3], 3 * STEP)
    # Odd net lots sell 3 at 109.967, leaving 2 net base units.
    entry_fee = 500 * model.entry_fee_rate
    sale1 = 3 * 109.967
    pnl1 = sale1 - 300 - entry_fee - sale1 * model.fee_rate
    assert partial["remaining"] == 2
    assert partial["net_pnl"] == pytest.approx(pnl1)
    account = snapshot(model, [partial], price=111)
    assert account["cash"] == pytest.approx(10000 - 500 - entry_fee + sale1 * (1 - model.fee_rate))
    assert account["realized_net"] == pytest.approx(pnl1)
    assert account["unrealized"] == 22
    assert account["invested_value"] == 222
    assert account["equity"] == pytest.approx(10000 + pnl1 + 22)
    assert account["open_count"] == 1
    closed = model.advance(partial, target_history(), 5 * STEP)
    sale2 = 2 * 129.961
    pnl2 = sale1 + sale2 - 500 - entry_fee - (sale1 + sale2) * model.fee_rate
    assert closed["net_pnl"] == pytest.approx(pnl2)
    assert [fill["qty"] for fill in closed["fills"]] == [5, 3, 2]
    complete = account_snapshot([closed], {}, market=model.market)
    assert complete["cash"] == pytest.approx(10000 + pnl2)
    assert complete["unrealized"] == complete["invested_value"] == 0
    for state in (bought, account, complete):
        assert_conservation(state)


def test_same_bar_entry_and_stop_charges_both_fees(model):
    closed = model.advance(new_trade(), [bar(0, low=80)], STEP)
    assert closed["opened_at"] == closed["closed_at"] == STEP
    assert [fill["reason"] for fill in closed["fills"]] == ["ENTRY", "STOP"]
    expected_fees = 500 * model.entry_fee_rate + 5 * 89.973 * model.fee_rate
    assert closed["fees"] == pytest.approx(expected_fees)
    assert closed["net_pnl"] == pytest.approx(5 * (89.973 - 100) - expected_fees)
    account = account_snapshot([closed], {}, market=model.market)
    assert account["realized_net"] == closed["net_pnl"]
    assert account["reserved_pending"] == account["invested_value"] == 0
    assert_conservation(account)


def test_spot_withholds_base_and_never_sells_gross_purchase_quantity():
    model = ResearchModel("spot", [], [], fee_rate=.01)
    initial = new_trade(qty=.3, qty_step=.1)
    before = deepcopy(initial)
    closed = model.advance(initial, target_history(), 5 * STEP)
    entry, *sales = closed["fills"]
    assert entry["qty"] == .3
    assert entry["gross_purchase_qty"] == pytest.approx(.30303030303030303)
    assert entry["fee_base_qty"] == pytest.approx(.00303030303030303)
    assert entry["fee_asset"] == "base"
    assert entry["fee"] == pytest.approx(100 * entry["fee_base_qty"])
    assert entry["gross_purchase_qty"] - entry["fee_base_qty"] == pytest.approx(.3)
    assert [fill["qty"] for fill in sales] == [.2, .1]
    assert sum(fill["qty"] for fill in sales) == pytest.approx(.3)
    assert closed["remaining"] == 0
    assert initial == before
    assert "Fractional" in closed["research_assumptions"]
    assert "offline-research/spot" in closed["cost_model"]
    assert model.advance(closed, target_history(), 5 * STEP) == closed


def test_perpetual_entry_fee_is_quote_and_uses_full_principal():
    model = ResearchModel("perpetual", [], [])
    opened = model.advance(new_trade(), [bar(0)], STEP)
    entry = opened["fills"][0]
    assert entry["gross_purchase_qty"] == entry["qty"] == 5
    assert entry["fee_asset"] == "quote" and "fee_base_qty" not in entry
    assert entry["fee"] == pytest.approx(.275)
    account = snapshot(model, [opened], price=110)
    assert account["cash"] == pytest.approx(9499.725)
    assert account["equity"] == pytest.approx(10049.725)
    assert_conservation(account)


@pytest.mark.parametrize("market", ["spot", "perpetual"])
@pytest.mark.parametrize("fee", [0, .002])
def test_custom_fee_applies_to_risk_entry_and_both_partial_exits(market, fee):
    model = ResearchModel(market, [], [], fee_rate=fee)
    closed = model.advance(new_trade(), target_history(), 5 * STEP)
    entry_rate = fee / (1 - fee) if market == "spot" else fee
    assert [fill["fee"] for fill in closed["fills"]] == pytest.approx([
        500 * entry_rate, 3 * 109.967 * fee, 2 * 129.961 * fee,
    ])
    assert float(model.assess.__globals__["BASE_COST_RATE"]) == pytest.approx(entry_rate + fee + .0006)


@pytest.mark.parametrize("outcome", ["daily", "timeout", "same_bar_daily"])
def test_management_closures_charge_market_exit_fee(model, outcome):
    days = 22 if outcome == "timeout" else 3
    prices = [100] * (days * 24)
    if outcome != "timeout":
        prices[-1] = 94.5
    bars, daily, execution = histories(prices)
    actual = ResearchModel(model.market, daily, execution)
    created = 3 * DAY - HOUR if outcome == "same_bar_daily" else 14 * H4
    closed = actual.advance(
        managed_trade(created), bars, days * DAY,
        interval_ms=HOUR * 1000, daily=daily, execution=execution,
    )
    assert not closed.get("data_error") and not closed.get("data_gap")
    reason = "TIMEOUT" if outcome == "timeout" else "DAILY_INVALIDATION"
    assert closed["status"] == "CLOSED" and closed["exit_reason"] == reason
    exit_price = 99.97 if outcome == "timeout" else 94.47165
    expected_fees = 500 * model.entry_fee_rate + 5 * exit_price * model.fee_rate
    assert closed["fees"] == pytest.approx(expected_fees)
    assert closed["net_pnl"] == pytest.approx(5 * (exit_price - 100) - expected_fees)
    assert closed["fills"][-1]["fee"] == pytest.approx(5 * exit_price * model.fee_rate)
    if outcome == "same_bar_daily":
        assert closed["opened_at"] == closed["closed_at"] == 3 * DAY
    if outcome == "timeout":
        assert closed["daily_bars_held"] == 20
    assert isinstance(actual._management, _ManagementCache)
    assert actual._management.validation_hits > 0
    assert_conservation(account_snapshot([closed], {}, market=model.market))


def test_pending_reservations_fill_expiry_and_cancellation_release(model):
    initial = pending(model, expiry=STEP)
    expected = 500 * (1 + model.entry_fee_rate)
    reserved = snapshot(model, [initial])
    assert reserved["reserved_pending"] == pytest.approx(expected)
    assert reserved["cash"] == reserved["equity"] == 10000
    assert reserved["available_cash"] == pytest.approx(10000 - expected)
    assert reserved["pending_count"] == 1 and reserved["open_count"] == 0
    opened = model.advance(initial, [bar(0)], STEP)
    after_fill = snapshot(model, [opened])
    assert after_fill["reserved_pending"] == 0
    assert after_fill["available_cash"] == pytest.approx(reserved["available_cash"])
    expired = model.advance(initial, [bar(0, open=102, high=103, low=101, close=102)], STEP)
    assert expired["status"] == "EXPIRED" and expired["fees"] == 0
    cancelled = dict(initial, status="CANCELLED", closed_at=0)
    for terminal in (expired, cancelled):
        released = account_snapshot([terminal], {}, market=model.market)
        assert released["available_cash"] == released["equity"] == 10000
        assert released["reserved_pending"] == released["realized_net"] == 0
        assert released["pending_count"] == 0
        assert_conservation(released)


def test_runner_must_supply_pending_fee_rate_and_custom_rate_is_reserved():
    trade = new_trade()
    with pytest.raises(ValueError, match="entry fee rate"):
        account_snapshot([trade], {})
    model = ResearchModel("spot", [], [], fee_rate=.02)
    tagged = model.advance(trade, [], 0)
    assert tagged["research_entry_fee_rate"] == .02 / .98
    account = account_snapshot([tagged], {}, initial=500)
    assert account["reserved_pending"] == pytest.approx(500 / .98)
    assert account["available_cash"] < 0  # No leverage credit or zero clamp.
    assert_conservation(account, initial=500)


def test_equal_fee_principal_commitment_and_fee_only_cash_difference():
    spot, perp = (ResearchModel(market, [], [], fee_rate=.01) for market in ("spot", "perpetual"))
    states = []
    for model in (spot, perp):
        opened = model.advance(new_trade(), [bar(0)], STEP)
        account = snapshot(model, [opened], price=120)
        assert 10000 + account["realized_net"] - account["cash"] == pytest.approx(500)
        assert account["invested_value"] == 600 and account["unrealized"] == 100
        states.append(account)
    assert states[1]["cash"] - states[0]["cash"] == pytest.approx(500 / .99 - 505)


def test_portfolio_counts_open_partial_closed_pending_and_zero_time_entries(model):
    opened = model.advance(monitored_trade(), [bar(0)], STEP)
    assert opened["opened_at"] == 0  # Must not use truthiness to detect entry.
    partial = model.advance(new_trade(identity="partial"), target_history()[:3], 3 * STEP)
    closed = model.advance(new_trade(identity="closed"), target_history(), 5 * STEP)
    waiting = pending(model, identity="waiting")
    expired = dict(pending(model, identity="expired"), status="EXPIRED", closed_at=0)
    trades = [opened, partial, closed, waiting, expired]
    before = deepcopy(trades)
    quotes = {"TESTUSDT": 111}
    account = account_snapshot(iter(trades), quotes, initial=20000, market=model.market)
    expected_net = sum(t["net_pnl"] for t in (opened, partial, closed))
    expected_principal = opened["entry"] * 5 + 200
    expected_unrealized = (111 - opened["entry"]) * 5 + 22
    assert account["realized_net"] == pytest.approx(expected_net)
    assert account["cash"] == pytest.approx(20000 + expected_net - expected_principal)
    assert account["unrealized"] == pytest.approx(expected_unrealized)
    assert account["invested_value"] == 777
    assert account["open_count"] == 2 and account["pending_count"] == 1
    assert trades == before and quotes == {"TESTUSDT": 111}
    assert_conservation(account, initial=20000)


def test_snapshot_uses_explicit_quotes_and_does_not_subtract_future_exit_fees(model):
    opened = model.advance(new_trade(), [bar(0)], STEP)
    opened.update(mark_price=900, mark_at=0)
    assert snapshot(model, [opened], price=110)["unrealized"] == 50
    assert snapshot(model, [opened], price=90)["unrealized"] == -50
    with pytest.raises(ValueError, match="Missing current closed quote"):
        account_snapshot([opened], {}, market=model.market)


@pytest.mark.parametrize("bad", [None, True, "100", 0, -1, math.nan, math.inf, -math.inf])
def test_invalid_quotes_fail_closed(bad):
    model = ResearchModel("spot", [], [])
    opened = model.advance(new_trade(), [bar(0)], STEP)
    with pytest.raises(ValueError):
        account_snapshot([opened], {"TESTUSDT": bad})


@pytest.mark.parametrize("field,bad", [
    ("side", "Sell"), ("side", None), ("entry", 0), ("entry", math.inf),
    ("entry", "100"), ("qty", -1), ("qty", True), ("qty", math.nan),
    ("remaining", -1), ("remaining", 6), ("remaining", math.inf), ("remaining", 0),
    ("limit", 0), ("limit", math.nan), ("net_pnl", None), ("net_pnl", math.nan),
    ("fees", -1), ("fees", math.nan), ("gross_pnl", math.inf), ("funding", math.nan),
    ("net_pnl_before_funding", math.nan), ("opened_at", math.nan),
    ("opened_at", -1), ("opened_at", None), ("closed_at", math.inf),
    ("status", "UNKNOWN"), ("status", "PENDING"), ("status", "CLOSED"),
    ("research_entry_fee_rate", -1), ("research_entry_fee_rate", math.nan),
    ("research_market", "perpetual"), ("id", None), ("symbol", ""),
])
def test_invalid_trade_values_cannot_create_equity(field, bad):
    opened = ResearchModel("spot", [], []).advance(new_trade(), [bar(0)], STEP)
    opened[field] = bad
    with pytest.raises(ValueError):
        account_snapshot([opened], {"TESTUSDT": 100})


@pytest.mark.parametrize("status", ["PENDING", "EXPIRED", "CANCELLED"])
def test_unfilled_nonzero_costs_and_shorts_are_rejected(status):
    model = ResearchModel("spot", [], [])
    base = dict(pending(model), status=status)
    for field, value in (("net_pnl", 1), ("fees", .5), ("funding", -1), ("side", "Sell")):
        with pytest.raises(ValueError):
            account_snapshot([dict(base, **{field: value})], {})
    with pytest.raises(ValueError, match="long-only"):
        model.advance(new_trade("Sell"), [bar(0)], STEP)


def test_duplicates_missing_financial_fields_and_overflow_fail_closed():
    model = ResearchModel("spot", [], [])
    trade = pending(model)
    with pytest.raises(ValueError, match="Duplicate trade id"):
        account_snapshot([trade, deepcopy(trade)], {})
    for field in ("id", "qty", "remaining", "net_pnl", "limit"):
        broken = dict(trade)
        del broken[field]
        with pytest.raises(ValueError):
            account_snapshot([broken], {})
    with pytest.raises(ValueError, match="Nonfinite account total"):
        account_snapshot([dict(trade, qty=1e308, remaining=1e308)], {})
    with pytest.raises(ValueError, match="finite number"):
        account_snapshot([], {}, initial=10**1000)


@pytest.mark.parametrize("bad", ["linear", "SPOT", "", None, True])
def test_invalid_market(bad):
    with pytest.raises(ValueError):
        ResearchModel(bad, [], [])
    with pytest.raises(ValueError):
        account_snapshot([], {}, market=bad)


@pytest.mark.parametrize("bad", [-1, 1, 2, math.nan, math.inf, True, "0.001"])
def test_invalid_model_fee(bad):
    with pytest.raises(ValueError):
        ResearchModel("spot", [], [], fee_rate=bad)


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf, None, True, "10000"])
def test_invalid_initial_capital(bad):
    with pytest.raises(ValueError):
        account_snapshot([], {}, initial=bad)


def test_empty_account_and_no_input_mutation():
    assert account_snapshot([], {}) == dict(
        equity=10000, cash=10000, reserved_pending=0, available_cash=10000,
        unrealized=0, realized_net=0, invested_value=0, open_count=0, pending_count=0,
    )


def test_model_cannot_relabel_an_opened_trade_from_another_fee_schedule():
    low, high = (ResearchModel("spot", [], [], fee_rate=fee) for fee in (.001, .002))
    opened = low.advance(new_trade(), [bar(0)], STEP)
    with pytest.raises(ValueError, match="fee rate"):
        high.advance(opened, target_history(), 5 * STEP)
    original = simulation.advance(new_trade(), [bar(0)], STEP)
    with pytest.raises(ValueError, match="originate"):
        low.advance(original, target_history(), 5 * STEP)


def test_invalid_candles_preserve_frozen_failure_and_financial_state():
    model = ResearchModel("spot", [], [])
    failed = model.advance(new_trade(), [replace(bar(0), close=math.nan)], STEP)
    assert failed["data_error"] and not failed["fills"]
    assert failed["net_pnl"] == failed["fees"] == 0
    with pytest.raises(ValueError, match="data_error"):
        account_snapshot([failed], {})
    recovered = model.advance(failed, [bar(0)], STEP)
    assert "data_error" not in recovered and recovered["status"] == "OPEN"
    assert recovered == model.advance(new_trade(), [bar(0)], STEP)


@pytest.mark.parametrize("opened_first", [False, True])
def test_data_gap_blocks_valuation_until_backfill_repairs_history(model, opened_first):
    initial = new_trade()
    prior = model.advance(initial, [bar(0)], STEP) if opened_first else initial
    gapped = model.advance(prior, [bar(2)], 3 * STEP)
    assert gapped["data_gap"] and not gapped.get("data_error")
    assert gapped["status"] == ("OPEN" if opened_first else "PENDING")
    before = deepcopy(gapped)
    with pytest.raises(ValueError, match="data_gap"):
        snapshot(model, [gapped])
    bars = [bar(0), bar(1), bar(2)]
    recovered = model.advance(gapped, bars, 3 * STEP)
    assert "data_gap" not in recovered
    assert recovered == model.advance(initial, bars, 3 * STEP)
    assert_conservation(snapshot(model, [recovered]))
    assert gapped == before


def test_clones_leave_all_original_globals_and_functions_unchanged():
    functions = [risk.assess, simulation.advance, simulation._manage, simulation._exit]
    globals_before = [dict(fn.__globals__) for fn in functions]
    original = simulation.advance(new_trade(), target_history(), 5 * STEP)
    models = [ResearchModel("spot", [], []), ResearchModel("perpetual", [], []),
              ResearchModel("spot", [], [], fee_rate=.004)]
    results = [m.advance(new_trade(), target_history(), 5 * STEP) for m in models]
    assert len({t["fees"] for t in results}) == 3
    assert models[0].advance(new_trade(), target_history(), 5 * STEP) == results[0]
    assert simulation.advance(new_trade(), target_history(), 5 * STEP) == original
    for function, before in zip(functions, globals_before):
        assert function.__globals__.keys() == before.keys()
        assert all(function.__globals__[key] is value for key, value in before.items())
    assert [risk.assess, simulation.advance, simulation._manage, simulation._exit] == functions


def test_streamed_prefixes_and_inputs_match_batch_results(model):
    initial = new_trade()
    streaming = initial
    before = deepcopy(initial)
    for count in range(1, 6):
        now = count * STEP
        streaming = model.advance(streaming, target_history()[:count], now)
        assert streaming == model.advance(initial, target_history(), now)
        assert model.advance(streaming, target_history(), now) == streaming
        assert_conservation(snapshot(model, [streaming], price=target_history()[count - 1].close))
    assert initial == before
