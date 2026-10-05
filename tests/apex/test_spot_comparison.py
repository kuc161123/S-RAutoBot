"""Offline portfolio regressions; real accounting, synthetic hourly archives."""

from copy import deepcopy
from concurrent.futures import Future
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from apex_bot import spot_comparison as comparison
from apex_bot.engine import DAY, H4
from apex_bot.spot_research_accounting import ResearchModel

# Reuse the existing replay's validated watch and hourly-source fixture.
from .test_replay import HOUR, START, source, watch


@pytest.fixture
def portfolio_source(source, tmp_path):
    import pandas as pd

    _, template = source
    serial = 0

    def build(symbols=("BTCUSDT",), *, days=1, runoff_days=0):
        nonlocal serial
        serial += 1
        root = tmp_path / f"portfolio-{serial}"
        root.mkdir()
        funding = root / "funding"
        funding.mkdir()
        periods = (19 + days + runoff_days) * 24
        frames = {}
        for symbol in symbols:
            frame = pd.concat([template.iloc[:24]] * (periods // 24), ignore_index=True)
            frame["start"] = pd.date_range("1970-01-01", periods=periods, freq="h")
            frames[symbol] = frame
            frame.to_parquet(root / f"{symbol}.parquet", index=False)
        finish = START + (days + runoff_days) * DAY
        book = SimpleNamespace(
            source=root,
            funding=funding,
            frames=frames,
            symbols=tuple(symbols),
            days=days,
            runoff_days=runoff_days,
            finish=finish,
        )
        for symbol in symbols:
            _write_funding(book, symbol)
        return book

    return build


def _save(book, symbol):
    book.frames[symbol].to_parquet(book.source / f"{symbol}.parquet", index=False)


def _write_funding(book, symbol, changes=None):
    import pandas as pd

    changes = changes or {}
    times = list(range(START * 1000, book.finish * 1000 + 1, 8 * HOUR * 1000))
    pd.DataFrame(
        {
            "ts_ms": times,
            "funding_rate": [changes.get(t // 1000, 0.0) for t in times],
        }
    ).to_parquet(book.funding / f"{symbol}.parquet", index=False)


def _run(book, market="spot", **kwargs):
    options = dict(
        funding_source=book.funding if market == "perpetual" else None,
        symbols=book.symbols,
        days=book.days,
        runoff_days=book.runoff_days,
        end=datetime.fromtimestamp(book.finish, timezone.utc).isoformat(),
        history_start="1970-01-01T00:00:00Z",
    )
    options.update(kwargs)
    return comparison.portfolio_replay(book.source, market, **options)


def _watch(symbol, identity, now, side="Buy", **changes):
    return replace(
        watch(identity, now, side),
        symbol=symbol,
        bucket="majors" if symbol in {"BTCUSDT", "ETHUSDT"} else "alts",
        **changes,
    )


def _analyze(monkeypatch, factory=None):
    if factory is None:
        factory = lambda symbol, now: [_watch(symbol, symbol, now)]

    def analyzer(symbol, daily, execution, now, **kwargs):
        # Every integration scenario also proves analyzer inputs are closed.
        assert all(bar.open_time / 1000 + DAY <= now for bar in daily)
        assert all(bar.open_time / 1000 + H4 <= now for bar in execution)
        return factory(symbol, now)

    monkeypatch.setattr(comparison, "analyze_monitored_zones", analyzer)


def _record_models(monkeypatch):
    """Observe real accounting/risk calls without changing their results."""
    assessments, advances = [], []

    def build(*args, **kwargs):
        model = ResearchModel(*args, **kwargs)
        assess, advance = model.assess, model.advance

        def recorded_assess(op, instrument, equity, **inputs):
            result = assess(op, instrument, equity, **inputs)
            assessments.append(
                dict(
                    id=op.id,
                    symbol=op.symbol,
                    equity=equity,
                    **deepcopy(inputs),
                    result=deepcopy(result),
                )
            )
            return result

        def recorded_advance(trade, bars, now, **inputs):
            result = advance(trade, bars, now, **inputs)
            advances.append(
                dict(now=now, before=deepcopy(trade), after=deepcopy(result))
            )
            return result

        model.assess = recorded_assess
        model.advance = recorded_advance
        return model

    monkeypatch.setattr(comparison, "ResearchModel", build)
    return assessments, advances


@pytest.mark.parametrize("market", ["spot", "perpetual"])
def test_simultaneous_symbols_share_equity_pending_cash_and_bucket_caps(
    portfolio_source,
    monkeypatch,
    market,
):
    book = portfolio_source(("SOLUSDT", "LINKUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT"))
    _analyze(monkeypatch)
    calls, _ = _record_models(monkeypatch)
    result = _run(book, market)
    accepted = result["accepted_evidence"]
    assert [a["opportunity"]["symbol"] for a in accepted] == [
        "BNBUSDT",
        "BTCUSDT",
        "ETHUSDT",
        "LINKUSDT",
    ]
    assert result["initial_equity"] == 10000
    assert result["maximum_positions_and_reservations"] == 4
    fee = 0.001 / 0.999 if market == "spot" else 0.00055
    reserved = 0.0
    for index, evidence in enumerate(accepted):
        account = evidence["account_before"]
        assert evidence["time"] == START
        assert account["equity"] == 10000
        assert account["pending_count"] == index
        assert account["reserved_pending"] == pytest.approx(reserved)
        assert account["available_cash"] == pytest.approx(10000 - reserved)
        reserved += evidence["sizing"]["qty"] * evidence["sizing"]["entry"] * (1 + fee)
    solar = next(c for c in calls if c["symbol"] == "SOLUSDT" and c["now"] == START)
    assert len(solar["exposures"]) == 4
    assert "BUCKET_POSITION_CAP" in solar["result"]["reasons"]
    assert result["unique_risk_rejections"]["BUCKET_POSITION_CAP"] == 1
    assert result["minimum_available_cash"] >= 0
    assert result["open"] == 4 and result["closed"] == 0
    assert result["wins"] == result["losses"] == 0


@pytest.mark.parametrize("market", ["spot", "perpetual"])
def test_alphabetical_plan_order_retries_after_exit_and_never_reuses_consumed_plan(
    portfolio_source,
    monkeypatch,
    market,
):
    book = portfolio_source()
    book.frames["BTCUSDT"].loc[19 * 24 + 1, "high"] = 131
    _save(book, "BTCUSDT")
    _analyze(
        monkeypatch,
        lambda s, n: [
            _watch(s, "z-retry", n, target1=140, target2=180),
            _watch(s, "a-first", n),
        ],
    )
    result = _run(book, market)
    assert [a["opportunity"]["id"] for a in result["accepted_evidence"]] == [
        "a-first",
        "z-retry",
    ]
    assert [a["time"] for a in result["accepted_evidence"]] == [START, START + 2 * HOUR]
    assert result["risk_rejections"]["SYMBOL_ALREADY_EXPOSED"] == 2
    assert result["trades"][0]["status"] == "CLOSED"
    assert result["accepted_evidence"][1]["account_before"]["equity"] > 10000


@pytest.mark.parametrize("market", ["spot", "perpetual"])
def test_ready_shorts_are_excluded_before_assessment(
    portfolio_source, monkeypatch, market
):
    book = portfolio_source()
    _analyze(monkeypatch, lambda s, n: [_watch(s, "short", n, "Sell")])
    calls, _ = _record_models(monkeypatch)
    result = _run(book, market)
    assert result["excluded_ready_shorts"] == 1
    assert not calls and not result["trades"]
    assert result["final_account"]["equity"] == 10000


@pytest.mark.parametrize("invalidate_short", [False, True])
def test_bearish_conflict_still_vetoes_longs_until_short_structure_fails(
    portfolio_source,
    monkeypatch,
    invalidate_short,
):
    book = portfolio_source()
    if invalidate_short:
        book.frames["BTCUSDT"].loc[19 * 24, "high"] = 107
        _save(book, "BTCUSDT")
    _analyze(
        monkeypatch,
        lambda s, n: [
            _watch(s, "long", n, reason="CONFLICTING_DIRECTIONS"),
            _watch(s, "short", n, "Sell", reason="CONFLICTING_DIRECTIONS"),
        ],
    )
    result = _run(book)
    if invalidate_short:
        assert result["accepted"] == 1
        assert result["trades"][0]["side"] == "Buy"
        assert result["trades"][0]["decision_at"] == START + HOUR
    else:
        assert result["accepted"] == 0 and result["trades"] == []


def test_first_scan_uses_previous_closed_quote_and_first_ioc_uses_next_open(
    portfolio_source,
    monkeypatch,
):
    book = portfolio_source()
    # Future first bar gaps above the cap, then trades through it. Neither the
    # future close nor the later touch may rescue this first-open IOC.
    frame = book.frames["BTCUSDT"]
    frame.loc[19 * 24, ["open", "high", "low", "close"]] = [102, 103, 99, 100]
    _save(book, "BTCUSDT")
    _analyze(monkeypatch)
    result = _run(book)
    evidence = result["accepted_evidence"][0]
    assert evidence["time"] == START
    assert evidence["opportunity"]["evidence"]["observation_price"] == pytest.approx(
        100 * 1.00005
    )
    trade = result["trades"][0]
    assert trade["status"] == "EXPIRED"
    assert trade["exit_reason"] == "IOC_PRICE_MISSED"
    assert trade["closed_at"] == START
    assert trade["opened_at"] is None and not trade["fills"]
    assert result["final_account"]["reserved_pending"] == 0
    assert result["final_account"]["cash"] == 10000


def test_first_hour_outside_zone_cannot_borrow_future_in_zone_close(
    portfolio_source, monkeypatch
):
    book = portfolio_source()
    book.frames["BTCUSDT"].loc[19 * 24 - 1, ["high", "close"]] = [103, 102]
    _save(book, "BTCUSDT")
    _analyze(monkeypatch)
    result = _run(book)
    assert result["accepted_evidence"][0]["time"] == START + HOUR
    assert result["trades"][0]["opened_at"] == START + HOUR


@pytest.mark.parametrize("market", ["spot", "perpetual"])
def test_entry_cutoff_is_exclusive_and_existing_holdings_are_managed_in_runoff(
    portfolio_source,
    monkeypatch,
    market,
):
    book = portfolio_source(days=1, runoff_days=1)
    cutoff = START + DAY
    # First opportunity appears on the last scan; another first appears exactly
    # at the cutoff. A target during runoff must still close the existing lot.
    book.frames["BTCUSDT"].loc[: 19 * 24 + 21, ["high", "close"]] = [103, 102]
    book.frames["BTCUSDT"].loc[20 * 24 + 1, "high"] = 131
    _save(book, "BTCUSDT")
    _analyze(
        monkeypatch,
        lambda s, n: [
            _watch(s, "late", n),
            *([_watch(s, "too-late", n)] if n >= cutoff else []),
        ],
    )
    result = _run(book, market)
    assert result["accepted"] == result["filled"] == result["closed"] == 1
    trade = result["trades"][0]
    assert trade["decision_at"] == trade["opened_at"] == cutoff - HOUR
    assert trade["closed_at"] == cutoff + 2 * HOUR
    assert trade["exit_reason"] == "TP2"
    assert all(
        t["decision_at"] < cutoff and t["opened_at"] < cutoff for t in result["trades"]
    )


def test_unfinished_holding_is_marked_without_forced_exit_or_win_classification(
    portfolio_source,
    monkeypatch,
):
    book = portfolio_source(days=1, runoff_days=1)
    _analyze(monkeypatch)
    result = _run(book)
    trade = result["trades"][0]
    assert trade["status"] == "OPEN" and trade["closed_at"] is None
    assert trade["last_bar"] / 1000 + HOUR == book.finish
    assert result["closed"] == result["wins"] == result["losses"] == 0
    assert result["win_rate_pct"] is None
    assert result["estimated_liquidation_equity"] < result["final_account"]["equity"]


def test_pending_entry_is_expired_when_its_first_available_open_is_at_cutoff(
    portfolio_source,
    monkeypatch,
):
    book = portfolio_source(days=1, runoff_days=1)
    cutoff = START + DAY
    _analyze(monkeypatch)
    calls = []

    def delayed_model(*args, **kwargs):
        model = ResearchModel(*args, **kwargs)

        def advance(trade, bars, now, **inputs):
            # Failure-path injection only: a hypothetical still-unresolved IOC
            # must be cancelled by the runner, never filled during runoff.
            calls.append(now)
            return deepcopy(trade)

        model.advance = advance
        return model

    monkeypatch.setattr(comparison, "ResearchModel", delayed_model)
    result = _run(book)
    assert result["filled"] == 0 and result["expired"] == 1
    assert result["trades"][0]["exit_reason"] == "ENTRY_CUTOFF"
    assert result["trades"][0]["closed_at"] == cutoff
    assert max(calls) == cutoff
    assert result["final_account"]["reserved_pending"] == 0
    assert result["final_account"]["available_cash"] == 10000


@pytest.mark.parametrize(
    "fault", ["origin", "historical_gap", "replay_gap", "duplicate", "nan"]
)
def test_incomplete_or_corrupt_hourly_history_fails_before_analysis(
    portfolio_source,
    monkeypatch,
    fault,
):
    book = portfolio_source()
    frame = book.frames["BTCUSDT"]
    if fault == "origin":
        frame = frame.iloc[1:]
    elif fault in {"historical_gap", "replay_gap"}:
        frame = frame.drop(7 * 24 if fault == "historical_gap" else 19 * 24 + 7)
    elif fault == "duplicate":
        frame.loc[1, "start"] = frame.loc[0, "start"]
    else:
        frame.loc[1, "close"] = float("nan")
    book.frames["BTCUSDT"] = frame
    _save(book, "BTCUSDT")
    monkeypatch.setattr(
        comparison,
        "analyze_monitored_zones",
        lambda *a, **k: pytest.fail("analysis of invalid history"),
    )
    with pytest.raises(ValueError):
        _run(book)


def test_first_scan_requires_a_preceding_completed_hour(portfolio_source, monkeypatch):
    book = portfolio_source()
    _analyze(monkeypatch)
    with pytest.raises(ValueError, match="missing comparison quotes"):
        _run(book, days=20)


@pytest.mark.parametrize("market", ["spot", "perpetual"])
def test_risk_uses_current_remaining_exposure_after_partial_exit(
    portfolio_source,
    monkeypatch,
    market,
):
    book = portfolio_source(("BTCUSDT", "ETHUSDT"))
    frame = book.frames["BTCUSDT"]
    frame.loc[19 * 24 + 1 :, ["open", "high", "low", "close"]] = [110, 111, 109, 110]
    frame.loc[19 * 24 + 1, "high"] = 113
    _save(book, "BTCUSDT")
    _analyze(
        monkeypatch,
        lambda s, n: [
            _watch(s, s + "-first", n),
            _watch(s, s + "-blocked", n),
        ],
    )
    calls, _ = _record_models(monkeypatch)
    result = _run(book, market)
    trade = next(t for t in result["trades"] if t["symbol"] == "BTCUSDT")
    assert trade["tp1_done"] and 0 < trade["remaining"] < trade["qty"]
    call = next(
        c for c in calls if c["symbol"] == "ETHUSDT" and c["now"] == START + 2 * HOUR
    )
    exposure = next(e for e in call["exposures"] if e["symbol"] == "BTCUSDT")
    assert exposure["notional"] == pytest.approx(110 * trade["remaining"])
    assert exposure["risk_cash"] == pytest.approx(
        trade["risk_cash"] * trade["remaining"] / trade["qty"]
    )


def test_funding_enters_equity_and_risk_at_settlement_never_early_or_twice(
    portfolio_source,
    monkeypatch,
):
    book = portfolio_source()
    settlement = START + 8 * HOUR
    rates = {
        START: 0.0001,
        settlement: 0.0002,
        START + 16 * HOUR: -0.0001,
        book.finish: 0.0003,
    }
    _write_funding(book, "BTCUSDT", rates)
    _analyze(
        monkeypatch, lambda s, n: [_watch(s, "a-first", n), _watch(s, "z-retry", n)]
    )
    calls, _ = _record_models(monkeypatch)
    result = _run(book, "perpetual")
    trade = result["trades"][0]
    events = trade["funding_events"]
    assert [e["timestamp_ms"] for e in events] == [
        settlement * 1000,
        (START + 16 * HOUR) * 1000,
        book.finish * 1000,
    ]
    principal = trade["entry"] * trade["qty"]
    assert trade["funding"] == pytest.approx(-principal * (0.0002 - 0.0001 + 0.0003))
    retry = {c["now"]: c for c in calls if c["id"] == "z-retry"}
    assert retry[settlement - HOUR]["funding_rate_8h"] == 0.0001
    assert retry[settlement]["funding_rate_8h"] == 0.0002
    assert retry[settlement]["equity"] == pytest.approx(
        retry[settlement - HOUR]["equity"] - principal * 0.0002
    )
    assert retry[settlement + HOUR]["equity"] == retry[settlement]["equity"]
    assert retry[START + 16 * HOUR]["equity"] == pytest.approx(
        retry[settlement]["equity"] + principal * 0.0001
    )
    assert result["funding_estimate"] == pytest.approx(trade["funding"])
    assert result["final_account"]["cash"] == pytest.approx(
        10000 - principal - trade["fees"] + trade["funding"]
    )


@pytest.mark.parametrize("offset_ms", [1, 60000])
def test_offset_settlement_is_charged_once_at_first_subsequent_scan_with_actual_timestamp(
    portfolio_source,
    monkeypatch,
    offset_ms,
):
    import pandas as pd

    book = portfolio_source()
    nominal = START + 8 * HOUR
    _write_funding(
        book, "BTCUSDT", {START: 0.0001, nominal: 0.0002, book.finish: 0.0003}
    )
    path = book.funding / "BTCUSDT.parquet"
    frame = pd.read_parquet(path)
    frame.loc[
        frame.ts_ms.isin([nominal * 1000, book.finish * 1000]), "ts_ms"
    ] += offset_ms
    frame.to_parquet(path, index=False)
    _analyze(
        monkeypatch, lambda s, n: [_watch(s, "a-first", n), _watch(s, "z-retry", n)]
    )
    calls, _ = _record_models(monkeypatch)
    result = _run(book, "perpetual")
    trade = result["trades"][0]
    retry = {c["now"]: c for c in calls if c["id"] == "z-retry"}
    principal = trade["entry"] * trade["qty"]
    assert retry[nominal]["funding_rate_8h"] == 0.0001
    assert retry[nominal]["equity"] == retry[nominal - HOUR]["equity"]
    assert retry[nominal + HOUR]["funding_rate_8h"] == 0.0002
    assert retry[nominal + HOUR]["equity"] == pytest.approx(
        retry[nominal]["equity"] - principal * 0.0002
    )
    assert retry[nominal + 2 * HOUR]["equity"] == retry[nominal + HOUR]["equity"]
    charged = [e for e in trade["funding_events"] if e["rate"] != 0]
    assert len(charged) == 1
    assert charged[0]["timestamp_ms"] == nominal * 1000 + offset_ms
    assert charged[0]["amount"] == pytest.approx(-principal * 0.0002)
    # The final nominal settlement is present for coverage, but its actual
    # timestamp lies after the replay horizon and must not debit the account.
    assert trade["funding"] == pytest.approx(-principal * 0.0002)


@pytest.mark.parametrize("exit_kind", ["partial", "full"])
def test_funding_settlement_uses_post_partial_qty_and_excludes_full_exit_tie(
    portfolio_source,
    monkeypatch,
    exit_kind,
):
    book = portfolio_source()
    settlement = START + 8 * HOUR
    frame = book.frames["BTCUSDT"]
    frame.loc[19 * 24 + 7 :, ["open", "high", "low", "close"]] = [110, 111, 109, 110]
    frame.loc[19 * 24 + 7, "high"] = 113 if exit_kind == "partial" else 131
    _save(book, "BTCUSDT")
    _write_funding(book, "BTCUSDT", {settlement: 0.0002})
    _analyze(monkeypatch)
    result = _run(book, "perpetual")
    trade = result["trades"][0]
    if exit_kind == "full":
        assert trade["closed_at"] == settlement
        assert not trade["funding_events"] and trade["funding"] == 0
    else:
        event = next(
            e for e in trade["funding_events"] if e["timestamp_ms"] == settlement * 1000
        )
        assert event["qty"] == trade["remaining"] < trade["qty"]
        assert event["amount"] == pytest.approx(
            -trade["remaining"] * trade["entry"] * 0.0002
        )


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "off_schedule", "fractional", "nan"]
)
def test_funding_schedule_rejects_missing_or_invalid_settlements(
    portfolio_source, fault
):
    import pandas as pd

    book = portfolio_source()
    path = book.funding / "BTCUSDT.parquet"
    frame = pd.read_parquet(path)
    if fault == "missing":
        frame = frame.drop(1)
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif fault == "off_schedule":
        frame.loc[1, "ts_ms"] += HOUR * 1000
    elif fault == "fractional":
        frame["ts_ms"] = frame["ts_ms"].astype(float)
        frame.loc[1, "ts_ms"] += 0.5
    else:
        frame.loc[1, "funding_rate"] = float("nan")
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="funding|Funding"):
        _run(book, "perpetual")


@pytest.mark.parametrize(
    "reference,boundary", [("daily_loss_pct", 20 * DAY), ("weekly_loss_pct", 25 * DAY)]
)
def test_day_and_iso_week_reference_uses_current_mark_before_first_hour_management(
    portfolio_source,
    monkeypatch,
    reference,
    boundary,
):
    book = portfolio_source(days=8)
    book.frames["BTCUSDT"].loc[boundary // HOUR - 1, "close"] = 99
    _save(book, "BTCUSDT")
    _analyze(
        monkeypatch, lambda s, n: [_watch(s, "a-first", n), _watch(s, "z-retry", n)]
    )
    calls, _ = _record_models(monkeypatch)
    _run(book)
    call = next(c for c in calls if c["id"] == "z-retry" and c["now"] == boundary)
    # A mark change at the first scan belongs in its reference. With no exits,
    # fees or funding this hour, the new period starts with zero loss.
    assert call[reference] == pytest.approx(0, abs=1e-12)


def test_higher_spot_costs_can_fail_rr_while_same_perpetual_plan_passes(
    portfolio_source,
    monkeypatch,
):
    book = portfolio_source()
    _analyze(
        monkeypatch,
        lambda s, n: [_watch(s, "marginal-rr", n, target1=109.7, target2=130)],
    )
    spot = _run(book)
    perpetual = _run(book, "perpetual")
    assert not spot["trades"]
    assert spot["risk_rejections"]["RR_TARGET1"] > 0
    assert perpetual["filled"] == 1


def test_future_adverse_funding_does_not_gate_entry_until_settlement(
    portfolio_source, monkeypatch
):
    book = portfolio_source()
    _write_funding(book, "BTCUSDT", {START + 8 * HOUR: 0.01})
    _analyze(
        monkeypatch, lambda s, n: [_watch(s, "a-first", n), _watch(s, "z-retry", n)]
    )
    calls, _ = _record_models(monkeypatch)
    result = _run(book, "perpetual")
    assert result["trades"][0]["decision_at"] == START
    retry = {c["now"]: c for c in calls if c["id"] == "z-retry"}
    assert "ADVERSE_FUNDING" not in retry[START + 7 * HOUR]["result"]["reasons"]
    assert "ADVERSE_FUNDING" in retry[START + 8 * HOUR]["result"]["reasons"]


@pytest.mark.parametrize(
    "settled_rate,reason",
    [
        (0.55, "DAILY_LOSS_HALT"),
        (1.1, "WEEKLY_LOSS_HALT"),
        (3.2, "DRAWDOWN_HALT"),
    ],
)
def test_real_funding_losses_activate_portfolio_circuits_and_management_continues(
    portfolio_source,
    monkeypatch,
    settled_rate,
    reason,
):
    book = portfolio_source(("BTCUSDT", "ETHUSDT"))
    settlement = START + 8 * HOUR
    _write_funding(book, "BTCUSDT", {settlement: settled_rate})
    _analyze(
        monkeypatch,
        lambda s, n: ([] if s == "ETHUSDT" and n < settlement else [_watch(s, s, n)]),
    )
    calls, _ = _record_models(monkeypatch)
    result = _run(book, "perpetual")
    rejected = next(
        c for c in calls if c["symbol"] == "ETHUSDT" and c["now"] == settlement
    )
    assert rejected["funding_rate_8h"] == 0  # ETH has no adverse-funding gate.
    assert reason in rejected["result"]["reasons"]
    assert result["filled"] == 1
    assert result["trades"][0]["last_bar"] / 1000 + HOUR == book.finish
    assert result["trades"][0]["status"] == "OPEN"


def test_eight_percent_drawdown_halves_risk_after_new_day_and_week_reset(
    portfolio_source,
    monkeypatch,
):
    book = portfolio_source(("BTCUSDT", "ETHUSDT"), days=8)
    monday = 25 * DAY
    _write_funding(book, "BTCUSDT", {START + 8 * HOUR: 2.2})
    _analyze(
        monkeypatch,
        lambda s, n: ([] if s == "ETHUSDT" and n < monday else [_watch(s, s, n)]),
    )
    calls, _ = _record_models(monkeypatch)
    result = _run(book, "perpetual")
    reduced = next(c for c in calls if c["symbol"] == "ETHUSDT" and c["now"] == monday)
    assert 8 <= reduced["drawdown_pct"] < 12
    assert reduced["daily_loss_pct"] == reduced["weekly_loss_pct"] == 0
    assert reduced["result"]["allowed"]
    assert reduced["result"]["risk_pct"] == 0.125
    assert reduced["result"]["risk_cash"] <= reduced["equity"] * 0.00125
    assert result["accepted"] == 2


def test_drawdown_tracks_hourly_marked_peak_even_when_daily_samples_miss_it(
    portfolio_source,
    monkeypatch,
):
    book = portfolio_source()
    book.frames["BTCUSDT"].loc[19 * 24 + 3, ["high", "close"]] = [110, 110]
    _save(book, "BTCUSDT")
    _analyze(monkeypatch)
    result = _run(book)
    trade = result["trades"][0]
    peak = 10000 - trade["fees"] + trade["qty"] * (110 - trade["entry"])
    expected = 100 * (1 - result["final_account"]["equity"] / peak)
    assert peak > max(row["equity"] for row in result["daily_equity"])
    assert result["max_drawdown_pct"] == pytest.approx(expected)


@pytest.mark.parametrize("market", ["spot", "perpetual"])
def test_cash_guard_counts_pending_fee_reservations_even_after_risk_allows(
    portfolio_source,
    monkeypatch,
    market,
):
    book = portfolio_source(("BTCUSDT", "ETHUSDT"))
    _analyze(monkeypatch)

    def model_factory(*args, **kwargs):
        model = ResearchModel(*args, **kwargs)
        assess = model.assess

        def oversized(op, instrument, equity, **inputs):
            # Deliberately bypass sizing ONLY to reach the independent cash
            # rejection path. All reservations, fills and fees remain real.
            result = assess(op, instrument, equity, **inputs)
            return dict(
                result, allowed=True, reasons=[], qty=60, notional=60 * result["entry"]
            )

        model.assess = oversized
        return model

    monkeypatch.setattr(comparison, "ResearchModel", model_factory)
    result = _run(book, market)
    assert result["accepted"] == result["filled"] == 1
    assert result["trades"][0]["symbol"] == "BTCUSDT"
    assert result["unique_risk_rejections"]["INSUFFICIENT_UNRESERVED_CASH"] == 1
    assert result["minimum_available_cash"] >= 0
    assert result["final_account"]["pending_count"] == 0


@pytest.mark.parametrize("flag", ["data_error", "data_gap"])
def test_simulation_failure_aborts_replay_instead_of_reporting_complete(
    portfolio_source,
    monkeypatch,
    flag,
):
    book = portfolio_source()
    _analyze(monkeypatch)

    def model_factory(*args, **kwargs):
        model = ResearchModel(*args, **kwargs)
        model.advance = lambda trade, *a, **k: dict(
            trade, **{flag: "synthetic failure"}
        )
        return model

    monkeypatch.setattr(comparison, "ResearchModel", model_factory)
    with pytest.raises(ValueError, match="simulation data issue"):
        _run(book)


@pytest.mark.parametrize(
    "market,changed", [("spot", "source"), ("perpetual", "funding")]
)
def test_end_checks_detect_candle_or_funding_source_change(
    portfolio_source,
    monkeypatch,
    market,
    changed,
):
    book = portfolio_source()
    _analyze(monkeypatch)
    path = (book.source if changed == "source" else book.funding) / "BTCUSDT.parquet"
    real_sha, calls = comparison._sha, 0

    def changed_sha(filename):
        nonlocal calls
        if Path(filename) == path:
            calls += 1
            if calls > 1:
                return "f" * 64
        return real_sha(filename)

    monkeypatch.setattr(comparison, "_sha", changed_sha)
    result = _run(book, market)
    assert calls == 2
    assert (
        result[
            "source_unchanged" if changed == "source" else "funding_source_unchanged"
        ]
        is False
    )


def _stub_cli(monkeypatch, tmp_path, result_change=None):
    """Real artifact/manifest path, completed local Futures, no child processes."""
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spot_comparison",
            "--spot-source",
            str(tmp_path / "spot"),
            "--perpetual-source",
            str(tmp_path / "perpetual"),
            "--funding-source",
            str(tmp_path / "funding"),
            "--output",
            str(output),
        ],
    )

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            future = Future()
            try:
                future.set_result(function(job))
            except Exception as exc:
                future.set_exception(exc)
            return future

    def row(job):
        result = dict(
            market=job["market"],
            complete=True,
            source_unchanged=True,
            funding_source_unchanged=True,
            trades=[],
            daily_equity=[dict(time=0, equity=10000)],
        )
        result.update(result_change or {})
        return result

    monkeypatch.setattr(comparison, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(comparison, "_run_job", row)
    return output


def test_manifest_hashes_module_that_supplies_the_frozen_symbol_universe(
    monkeypatch, tmp_path
):
    output = _stub_cli(monkeypatch, tmp_path)
    comparison.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert "monitored_comparison.py" in manifest["code_sha256"]
    assert manifest["code_sha256"]["monitored_comparison.py"] == comparison._sha(
        Path(comparison.__file__).with_name("monitored_comparison.py")
    )


@pytest.mark.parametrize("changed", ["engine.py", "SPOT_COMPARISON_PROTOCOL.txt"])
def test_cli_cannot_certify_complete_if_code_or_protocol_changes(
    monkeypatch, tmp_path, changed
):
    output = _stub_cli(monkeypatch, tmp_path)
    real_sha, calls = comparison._sha, 0

    def changed_sha(path):
        nonlocal calls
        if Path(path).name == changed:
            calls += 1
            if calls > 1:
                return "f" * 64
        return real_sha(path)

    monkeypatch.setattr(comparison, "_sha", changed_sha)
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 1
    assert calls == 2
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] is False


@pytest.mark.parametrize("flag", ["source_unchanged", "funding_source_unchanged"])
def test_cli_cannot_certify_complete_when_an_arm_reports_changed_source(
    monkeypatch, tmp_path, flag
):
    output = _stub_cli(monkeypatch, tmp_path, {flag: False})
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 1
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] is False
