from copy import deepcopy
import pytest
from apex_bot.accounting import loss_metrics
from apex_bot.storage import initial_state


NOW = 1728172800.0


def test_deposits_are_not_strategy_profit_or_loss():
    state = initial_state()
    state["risk_reference_equities"] = {"live": 10000}
    assert loss_metrics(state, 15000, None, NOW) == dict(
        daily_loss_pct=0, weekly_loss_pct=0, drawdown_pct=0
    )


def test_open_loss_is_included_conservatively_and_not_hidden_by_pending_orders():
    state = initial_state()
    state["account"] = {"positions": [{"size": "1", "unrealisedPnl": "-250"}]}
    state["risk_reference_equities"] = {"live": 10000}
    metrics = loss_metrics(state, 10000, None, NOW)
    assert metrics == dict(daily_loss_pct=2.5, weekly_loss_pct=2.5, drawdown_pct=2.5)
    state["account"]["positions"][0].pop("unrealisedPnl")
    with pytest.raises(ValueError):
        loss_metrics(state, 10000, None, NOW)


def test_shadow_books_are_isolated_and_stale_mark_fails_closed():
    state = initial_state()
    state["trades"] = {
        "one": {
            "arm": "baseline_shadow",
            "status": "OPEN",
            "side": "Buy",
            "entry": 100,
            "remaining": 2,
            "mark_price": 90,
            "mark_at": NOW,
            "net_pnl": -1,
        }
    }
    assert loss_metrics(state, 10000, "baseline_shadow", NOW)[
        "daily_loss_pct"
    ] == pytest.approx(0.21)
    assert loss_metrics(state, 10000, "ai_shadow", NOW)["daily_loss_pct"] == 0
    with pytest.raises(ValueError):
        loss_metrics(state, 10000, "baseline_shadow", NOW + 601)


def test_corrupt_closed_outcome_is_never_counted_as_zero():
    state = initial_state()
    state["orders"] = {"bad": {"status": "CLOSED", "closed_at": NOW}}
    with pytest.raises(ValueError):
        loss_metrics(state, 10000, None, NOW)


def test_persisted_peak_survives_restart_and_drawdown_recovers_without_deposit_profit():
    state = initial_state()
    state["risk_reference_equities"] = {"live": 10000}
    state["risk_circuits"] = {"live": {"strategy_peak": 11000}}
    assert loss_metrics(deepcopy(state), 20000, None, NOW)[
        "drawdown_pct"
    ] == pytest.approx(1000 / 11000 * 100)


@pytest.mark.parametrize(
    "field,value",
    [
        ("net_pnl", float("nan")),
        ("entry", float("nan")),
        ("remaining", float("nan")),
        ("remaining", -2),
        ("side", "unknown"),
    ],
)
def test_invalid_open_shadow_cashflow_or_position_cannot_clear_loss_gates(field, value):
    """Invalid valuation must fail closed, never reverse a loss or coerce NaN to 0."""
    state = initial_state()
    record = {
        "arm": "baseline_shadow",
        "status": "OPEN",
        "side": "Buy",
        "entry": 100,
        "remaining": 2,
        "mark_price": 90,
        "mark_at": NOW,
        "net_pnl": -1,
    }
    record[field] = value
    state["trades"]["invalid-open"] = record
    with pytest.raises(ValueError):
        loss_metrics(state, 10000, "baseline_shadow", NOW)


def test_nonfinite_live_open_funding_cannot_erase_confirmed_exchange_loss():
    state = initial_state()
    state["account"] = {"positions": [{"size": "1", "unrealisedPnl": "-250"}]}
    state["orders"]["open"] = {
        "status": "OPEN",
        "net_pnl_before_funding": -1,
        "funding": float("nan"),
    }
    with pytest.raises(ValueError):
        loss_metrics(state, 10000, None, NOW)


def test_observed_open_profit_returns_persistable_peak_without_counting_deposit_as_profit():
    state = initial_state()
    state["risk_reference_equities"] = {"live": 10000}
    state["account"] = {"positions": [{"size": "1", "unrealisedPnl": "1000"}]}
    details = loss_metrics(state, 21000, None, NOW, include_state=True)
    assert details["strategy_peak"] == 11000
    assert details["strategy_value"] == 11000
    state["risk_circuits"] = {"live": details}
    state["account"]["positions"][0]["unrealisedPnl"] = "0"
    assert loss_metrics(state, 20000, None, NOW)["drawdown_pct"] == pytest.approx(
        1000 / 11000 * 100
    )
