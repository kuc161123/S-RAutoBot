"""Offline replay orchestration, chronology and cache-equivalence regressions."""

from dataclasses import replace
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from apex_bot.engine import DAY, H4, VERSION
from apex_bot.models import Candle, Opportunity

replay_module = importlib.import_module("apex_bot.replay")
HOUR = 3600
START = 19 * DAY


@pytest.fixture
def source(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        dict(
            start=pd.date_range("1970-01-01", periods=20 * 24, freq="h"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
    )
    path = tmp_path / "TESTUSDT.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def opportunity(identity, now, side="Buy", **changes):
    long = side == "Buy"
    expired = now >= START + 2 * H4
    op = Opportunity(
        identity,
        "TESTUSDT",
        side,
        "1" if long else "1S",
        "INVALID" if expired else "READY",
        100,
        94 if long else 106,
        112 if long else 88,
        130 if long else 70,
        96 if long else 104,
        99 if long else 101,
        START - H4,
        START + 2 * H4,
        "EXPIRED" if expired else "SWING_BREAK",
        dict(
            engine_version=VERSION,
            evidence_id=identity,
            trigger_evidence_id=identity + ":trigger",
            structural_valid=True,
            data_valid=True,
            terminal_status="EXPIRED" if expired else None,
            trigger_kind="SWING_BREAK",
            trigger_closed_at=START,
            daily_closed_at=now // DAY * DAY,
            execution_closed_at=now // H4 * H4,
            as_of=now,
            zone_low=98,
            zone_high=101,
            entry_zone_low=98,
            entry_zone_high=101,
            trigger_price=100,
            risk_multiplier=1,
        ),
    )
    return replace(op, **changes)


def watch(identity, now, side="Buy", **changes):
    op = opportunity(identity, now, side)
    return replace(
        op,
        state="WAIT",
        reason="AWAIT_ZONE",
        confirmation=0,
        created_at=START - 3 * DAY,
        expires_at=START + 27 * DAY,
        evidence=dict(
            op.evidence,
            entry_style="monitored_zone",
            trigger_kind="ZONE_WATCH",
            trigger_closed_at=None,
            trigger_price=None,
            trigger_evidence_id=None,
            terminal_status=None,
            plan_evidence_id=identity + ":plan",
            monitored_started_at=START - 3 * DAY,
            observation_at=None,
            observation_price=None,
            quote_limit=None,
        ),
        **changes
    )


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_monitored_waits_for_current_zone_quote_then_fills_next_open(
    source, monkeypatch, side
):
    monitor = importlib.import_module("apex_bot.monitored_zones")
    path, frame = source
    # No opportunity may use the low/high of these past bars to invent an entry.
    frame.loc[19 * 24 - 1 : 19 * 24, "close"] = 101.5 if side == "Buy" else 97.5
    frame.loc[19 * 24 - 1 : 19 * 24, "high" if side == "Buy" else "low"] = (
        102 if side == "Buy" else 97
    )
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(
        monitor,
        "analyze_monitored_zones",
        lambda symbol, d, e, now, **kw: [watch("wait-for-quote", now, side)],
    )
    result = replay_module.replay(path, 1, entry_style="monitored_zone")
    trade = result["trades"][0]
    assert result["complete"] and result["data_issues"] == 0
    assert trade["decision_at"] == START + 2 * HOUR
    assert trade["opened_at"] == trade["decision_at"]
    assert trade["fills"][0]["time"] == trade["opened_at"]
    assert len(result["trades"]) == 1
    op = result["plans"][0]["accepted_opportunity"]
    assert op["evidence"]["trigger_closed_at"] is None
    assert op["evidence"]["observation_at"] == trade["decision_at"]
    assert op["entry"] != 100  # risk assessed at the price cap, never a past touch


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_monitored_hourly_invalidation_persists_between_four_hour_snapshots(
    source, monkeypatch, side
):
    monitor = importlib.import_module("apex_bot.monitored_zones")
    path, frame = source
    # Quote is outside at start; the next closed hour touches the original stop.
    frame.loc[19 * 24 - 1, "close"] = 101.5 if side == "Buy" else 97.5
    frame.loc[19 * 24 - 1, "high" if side == "Buy" else "low"] = (
        102 if side == "Buy" else 97
    )
    frame.loc[19 * 24, "low" if side == "Buy" else "high"] = (
        93 if side == "Buy" else 107
    )
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(
        monitor,
        "analyze_monitored_zones",
        lambda symbol, d, e, now, **kw: [watch("invalid", now, side)],
    )
    result = replay_module.replay(path, 1, entry_style="monitored_zone")
    assert result["funnel"]["unique_accepted"] == 0
    assert result["plans"][0]["last_state"] == "INVALID"
    assert result["unique_plans_by_reason"]["HARD_STOP_BREACHED"] == 1


def test_monitored_ioc_miss_is_not_a_later_resting_fill(source, monkeypatch):
    monitor = importlib.import_module("apex_bot.monitored_zones")
    path, frame = source
    frame.loc[19 * 24, "open"] = 100.8
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(
        monitor,
        "analyze_monitored_zones",
        lambda symbol, d, e, now, **kw: [watch("miss", now)],
    )
    result = replay_module.replay(path, 1, entry_style="monitored_zone")
    assert result["funnel"]["unique_accepted"] == 1
    assert result["funnel"]["unique_filled"] == 0
    assert result["trades"][0]["exit_reason"] == "IOC_PRICE_MISSED"
    assert result["summary"]["expired"] == 1


def test_monitored_hourly_cancel_releases_conflict_without_waiting_for_4h_close(
    source, monkeypatch
):
    monitor = importlib.import_module("apex_bot.monitored_zones")
    path, frame = source
    frame.loc[19 * 24, "high"] = 103
    frame.to_parquet(path, index=False)

    def conflicted(symbol, d, e, now, **kw):
        return [
            replace(watch("valid-long", now), reason="CONFLICTING_DIRECTIONS"),
            replace(
                watch("failed-short", now, "Sell"),
                reason="CONFLICTING_DIRECTIONS",
                stop=102.7,
                invalidation=101.1,
            ),
        ]

    monkeypatch.setattr(monitor, "analyze_monitored_zones", conflicted)
    result = replay_module.replay(path, 1, entry_style="monitored_zone")
    records = {p["id"]: p for p in result["plans"]}
    assert records["failed-short"]["last_state"] == "INVALID"
    assert records["valid-long"]["accepted_at"] == START + HOUR


@pytest.mark.parametrize("kwargs", [{"monitor_days": 3}, {"scan_seconds": 60}])
def test_monitored_rejects_unregistered_policy_or_stale_subhourly_quotes(
    source, kwargs
):
    with pytest.raises(ValueError):
        replay_module.replay(source[0], 1, entry_style="monitored_zone", **kwargs)


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_retries_after_exposure_exit_and_never_recreates_consumed_plan(
    source, monkeypatch, side
):
    path, frame = source
    # The first trade fills after the decision, then closes on the next bar.
    frame.loc[19 * 24 + 1, "high" if side == "Buy" else "low"] = (
        131 if side == "Buy" else 69
    )
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(
        replay_module,
        "analyze",
        lambda symbol, d, e, now, **kw: [
            opportunity("first", now, side),
            opportunity("retry", now, side),
        ],
    )
    result = replay_module.replay(path, 1)
    by_id = {p["id"]: p for p in result["plans"]}
    assert by_id["first"]["attempts"] == 1
    assert by_id["retry"]["attempts"] == 3
    assert by_id["retry"]["accepted_at"] == START + 2 * HOUR
    assert by_id["retry"]["rejection_reasons"] == {"SYMBOL_ALREADY_EXPOSED": 2}
    assert result["funnel"]["accepted_after_retry"] == 1
    assert result["funnel"]["accepted_by_side"][side] == 2
    assert result["ready_candidates"] == 2
    assert result["unique_plans_by_state"] == {"READY": 2, "INVALID": 2}
    assert result["states_observed"]["INVALID"] == 32
    assert result["funnel"]["unique_plans"] == 2
    assert result["data_issues"] == 0
    assert result["trades"][0]["opened_at"] == START + HOUR
    assert len(result["trades"]) == 2
    json.dumps(result, allow_nan=False)


def test_unchanged_frozen_entry_retries_same_rr_failure_until_expiration(
    source, monkeypatch
):
    path, _ = source
    monkeypatch.setattr(
        replay_module,
        "analyze",
        lambda symbol, d, e, now, **kw: [
            opportunity("rr", now, target1=108, target2=114)
        ],
    )
    result = replay_module.replay(path, 1)
    assert result["funnel"]["risk_attempts"] == 8
    assert result["funnel"]["unique_attempted"] == 1
    assert result["funnel"]["unique_accepted"] == 0
    assert result["risk_rejections"]["RR_TARGET1"] == 8
    assert result["unique_plans_by_risk_rejection"]["RR_TARGET1"] == 1
    assert (
        max(e["time"] for e in result["events"] if e["kind"] == "RISK_REJECTED")
        < START + 2 * H4
    )


def test_wait_invalid_and_expired_ready_never_assessed(source, monkeypatch):
    path, _ = source
    monkeypatch.setattr(
        replay_module,
        "analyze",
        lambda symbol, d, e, now, **kw: [
            opportunity("wait", now, state="WAIT"),
            opportunity("invalid", now, state="INVALID"),
            opportunity("expired", now, expires_at=START),
        ],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("noneligible snapshot reached risk")

    monkeypatch.setattr(replay_module, "assess", forbidden)
    result = replay_module.replay(path, 1)
    assert result["funnel"]["risk_attempts"] == 0


def test_minute_retries_have_fresh_evidence_without_reusing_partial_hour(
    source, monkeypatch
):
    path, _ = source
    monkeypatch.setattr(
        replay_module,
        "analyze",
        lambda symbol, d, e, now, **kw: [opportunity("one", now)],
    )
    original = replay_module.assess

    def delayed(op, instrument, equity, **kwargs):
        assert op.evidence["as_of"] == kwargs["now"]
        if kwargs["now"] == START:
            return dict(allowed=False, reasons=["TEMPORARY"])
        return original(op, instrument, equity, **kwargs)

    monkeypatch.setattr(replay_module, "assess", delayed)
    result = replay_module.replay(path, 1, scan_seconds=60, max_scans=122)
    trade = result["trades"][0]
    assert trade["created_at"] == START + 60
    assert trade["entry_eligible_at"] == START + HOUR
    assert trade["opened_at"] == START + 2 * HOUR
    assert result["performance"]["engine_calls"] == 1
    assert result["performance"]["cache_hits"] == 121
    assert not result["complete"] and result["stop_reason"] == "MAX_SCANS"


def test_uses_full_fixed_origin_and_no_future_bars(source, monkeypatch):
    path, _ = source
    seen = []

    def analyze(symbol, d, e, now, **kwargs):
        assert all(b.open_time / 1000 + DAY <= now for b in d)
        assert all(b.open_time / 1000 + H4 <= now for b in e)
        seen.append((now, d[0].open_time, e[0].open_time))
        return []

    monkeypatch.setattr(replay_module, "analyze", analyze)
    result = replay_module.replay(path, 1)
    assert seen[0][0] == START  # scan at window start, not an hour late
    assert all(d == e == 0 for _, d, e in seen)
    assert result["data"]["initial_daily_bars"] == 19
    assert result["complete"] and result["replayed_through"] == result["end"]
    # Explicitly exceed the former 500 execution bar trim without changing
    # source origin; changing replay start must not change the evidence origin.
    pd = pytest.importorskip("pandas")
    frame = pd.read_parquet(path)
    extended = pd.concat(
        [frame.assign(start=frame.start + pd.Timedelta(days=20 * i)) for i in range(6)]
    )
    extended.to_parquet(path, index=False)
    result = replay_module.replay(path, 1, max_scans=1)
    assert result["data"]["initial_execution_bars"] > 500
    assert result["data"]["execution_origin"] == "1970-01-01T00:00:00+00:00"


def test_gap_selects_latest_segment_and_reports_missing_window_and_warmup(source):
    path, frame = source
    frame = frame.drop(18 * 24 + 1)
    frame.to_parquet(path, index=False)
    result = replay_module.replay(path, 5)
    assert result["start"] == "1970-01-19T02:00:00+00:00"
    assert result["end"] == "1970-01-21T00:00:00+00:00"
    assert result["data"]["source_gaps"][0]["missing_hours"] == 1
    assert result["data"]["window_truncated"]
    assert result["data"]["excluded_requested_hours"] == 74
    assert result["data"]["initial_daily_bars"] == 0
    assert result["data"]["warmup_scans"] == result["scans"]
    assert result["funnel"]["unique_plans"] == 0
    earlier = replay_module.replay(path, 1, end="1970-01-19T01:00:00Z", max_scans=1)
    assert earlier["data"]["segment_start"] == "1970-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="inside a source gap"):
        replay_module.replay(path, 1, end="1970-01-19T02:00:00Z")


@pytest.mark.parametrize(
    "bad", ["duplicate", "unaligned", "nan", "infinite", "ohlc", "empty"]
)
def test_bad_source_is_explicit_not_zero_trades(source, bad):
    path, frame = source
    if bad == "duplicate":
        frame.loc[1, "start"] = frame.loc[0, "start"]
    elif bad == "unaligned":
        pd = pytest.importorskip("pandas")
        frame.loc[0, "start"] += pd.Timedelta(minutes=1)
    elif bad in {"nan", "infinite"}:
        frame.loc[0, "close"] = float("nan" if bad == "nan" else "inf")
    elif bad == "ohlc":
        frame.loc[0, "low"] = 200
    else:
        frame = frame.iloc[:0]
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        replay_module.replay(path, 1)


def test_cached_and_uncached_replays_are_identical(source, monkeypatch):
    path, _ = source
    monkeypatch.setattr(
        replay_module,
        "analyze",
        lambda symbol, d, e, now, **kw: [
            opportunity("rr", now, target1=108, target2=114)
        ],
    )
    cached = replay_module.replay(path, 1)
    plain = replay_module.replay(path, 1, use_cache=False)
    assert cached.pop("performance")["engine_calls"] == 6
    assert plain.pop("performance")["engine_calls"] == 24
    assert cached == plain


@pytest.mark.parametrize(
    "entry_style",
    ["confirmed", "resting_limit", "monitored_zone_2", "monitored_zone_30"],
)
def test_real_engine_cache_matches_each_hour_and_stale_history(entry_style):
    # Same deterministic structure for both engines; future bars stay hidden.
    prices = [100] * 16
    for endpoint in (95, 130, 110, 165, 140, 180, 150):
        first = prices[-1]
        prices.extend(first + (endpoint - first) * i / 8 for i in range(1, 9))
    daily = [
        Candle(i * DAY * 1000, p, p + 0.2, p - 0.2, p, 10) for i, p in enumerate(prices)
    ]
    prices = [151] * 16 + [
        145,
        143,
        141,
        143,
        145,
        144,
        142,
        139,
        140,
        141,
        140,
        139,
        138,
        140,
        142,
    ]
    execution = [
        Candle((67 * DAY + i * H4) * 1000, p, p + 0.2, p - 0.2, p, 10)
        for i, p in enumerate(prices)
    ]
    analyzer = None
    analyzer_kwargs = {}
    if entry_style == "resting_limit":
        analyzer = importlib.import_module("apex_bot.zone_orders").analyze_zones
    elif entry_style.startswith("monitored_zone_"):
        analyzer = importlib.import_module(
            "apex_bot.monitored_zones"
        ).analyze_monitored_zones
        analyzer_kwargs = {
            "lifetime_seconds": int(entry_style.rsplit("_", 1)[-1]) * DAY
        }
    cached = replay_module._AnalysisCache(
        "TESTUSDT", daily, execution, analyzer=analyzer, analyzer_kwargs=analyzer_kwargs
    )
    plain = replay_module._AnalysisCache(
        "TESTUSDT", daily, execution, False, analyzer, analyzer_kwargs
    )
    observed = set()
    for now in range(67 * DAY, 74 * DAY, HOUR):
        result = cached.get(now)
        assert result == plain.get(now)
        observed.update(op.id for op in result)
        if result:
            result[0].evidence["anchors"].clear()  # must not poison later hits
    assert observed
    assert cached.hits > 0 and cached.calls < plain.calls
    if entry_style == "confirmed":
        assert cached.terminal_hits > 0 and cached.plan_hits > 0
    # Optimization never changes the shared production function namespace.
    assert "evaluate_plan" not in str(replay_module.analyze.__globals__["_evaluate"])


def test_cache_rechecks_expiry_between_bar_boundaries_and_backwards_time(monkeypatch):
    daily = [Candle(i * DAY * 1000, 100, 101, 99, 100) for i in range(19)]
    execution = [Candle(i * H4 * 1000, 100, 101, 99, 100) for i in range(19 * 6)]
    expiry = START + HOUR

    def analyze(symbol, d, e, now, **kwargs):
        return [
            opportunity(
                "x",
                now,
                expires_at=expiry,
                state="READY" if now < expiry else "INVALID",
            )
        ]

    monkeypatch.setattr(replay_module, "analyze", analyze)
    cached = replay_module._AnalysisCache("TESTUSDT", daily, execution)
    assert cached.get(START)[0].state == "READY"
    assert cached.get(START + 60)[0].state == "READY"
    assert cached.get(expiry)[0].state == "INVALID"
    assert cached.get(START)[0].state == "READY"
    assert cached.calls == 3


def test_cli_preserves_existing_report(source, monkeypatch, tmp_path):
    path, _ = source
    output = tmp_path / "old.json"
    output.write_text("old research")
    monkeypatch.setattr(
        "sys.argv", ["replay", str(path), "--days", "1", "--output", str(output)]
    )
    with pytest.raises(SystemExit):
        replay_module.main()
    assert output.read_text() == "old research"


def test_time_budget_marks_incomplete(source, monkeypatch):
    path, _ = source
    ticks = iter([0, 2, 3])
    monkeypatch.setattr(replay_module, "perf_counter", lambda: next(ticks))
    result = replay_module.replay(path, 1, max_seconds=1)
    assert not result["complete"] and result["stop_reason"] == "MAX_SECONDS"
    assert result["scans"] == 0


def test_resting_style_is_explicit_separate_book_and_risk_opt_in(source, monkeypatch):
    path, _ = source
    calls = []

    def confirmed(symbol, d, e, now, **kwargs):
        return [opportunity("same-structure", now)]

    def zones(symbol, d, e, now, **kwargs):
        op = opportunity("same-structure", now)
        return [
            replace(
                op,
                evidence=dict(
                    op.evidence,
                    entry_style="resting_limit",
                    trigger_kind="ZONE_LIMIT",
                    order_armed_at=op.created_at,
                ),
            )
        ]

    def risk(op, instrument, equity, **kwargs):
        resting = op.evidence.get("entry_style") == "resting_limit"
        assert kwargs.get("allow_resting", False) is resting
        if not resting:
            assert "allow_resting" not in kwargs
        calls.append((resting, equity, kwargs["exposures"]))
        return dict(
            allowed=True,
            reasons=[],
            entry=op.entry,
            stop=op.stop,
            target1=op.target1,
            target2=op.target2,
            qty=1,
            qty_step=instrument.qty_step,
            risk_cash=6,
            notional=100,
        )

    monkeypatch.setattr(replay_module, "analyze", confirmed)
    monkeypatch.setattr(replay_module, "assess", risk)
    monkeypatch.setitem(
        sys.modules, "apex_bot.zone_orders", SimpleNamespace(analyze_zones=zones)
    )
    normal = replay_module.replay(path, 1, max_scans=1)
    resting = replay_module.replay(path, 1, max_scans=1, entry_style="resting_limit")
    assert normal["entry_style"] == "confirmed" and normal["arm"] == "acceptance_replay"
    assert (
        resting["entry_style"] == "resting_limit" and resting["arm"] == "resting_shadow"
    )
    assert normal["trades"][0]["id"] != resting["trades"][0]["id"]
    assert calls == [(False, 10000, []), (True, 10000, [])]


def test_exact_scan_budget_can_complete_final_management(source):
    path, _ = source
    result = replay_module.replay(path, 1, max_scans=24)
    assert result["complete"] and result["replayed_through"] == result["end"]


@pytest.mark.parametrize("days", [366, 730, 1460])
def test_long_entry_windows_are_deliberately_supported(source, days):
    result = replay_module.replay(source[0], days, max_scans=1)
    assert result["entry_days"] == days
    assert result["window_days"] == days
    assert result["data"]["window_truncated"]


@pytest.mark.parametrize("kwargs", [
    {"days": True}, {"days": 0}, {"days": -1}, {"days": 1461}, {"days": 1.0},
    {"runoff_days": True}, {"runoff_days": -1}, {"runoff_days": 1461},
    {"runoff_days": 1.0}, {"runoff_days": None}, {"runoff_days": "1"},
    {"max_scans": True}, {"max_scans": 0}, {"max_scans": 1.5},
    {"max_seconds": True}, {"max_seconds": 0}, {"max_seconds": float("nan")},
    {"max_seconds": float("inf")}, {"use_cache": 1}, {"use_cache": None},
    {"scan_seconds": True}, {"scan_seconds": 0}, {"monitor_days": True},
    {"end": True}, {"end": 0}, {"end": 1000.0}, {"end": "bad date"},
    {"end": "NaT"}, {"end": "1970-01-20T00:01:00Z"},
    {"end": "1970-01-01T00:00:00Z"}, {"end": "1970-01-22T00:00:00Z"},
])
def test_strict_replay_window_and_budget_parameters(source, kwargs):
    with pytest.raises(ValueError):
        replay_module.replay(source[0], **dict({"days": 1}, **kwargs))


def test_default_ninety_day_result_matches_explicit_zero_runoff(source):
    default = replay_module.replay(source[0], max_scans=2)
    explicit = replay_module.replay(source[0], 90, runoff_days=0, max_scans=2)
    default.pop("performance")
    explicit.pop("performance")
    assert default == explicit
    assert default["entry_end"] == default["end"]
    assert default["entry_days"] == 90 and default["runoff_days"] == 0
    assert default["open_at_end"] is None  # incomplete prefix isn't final exposure


@pytest.mark.parametrize("outcome", ["closed", "open", "pending", "expired"])
def test_runoff_stops_new_decisions_but_manages_accepted_trades(
    source, monkeypatch, outcome
):
    path, frame = source
    cutoff = 19 * DAY
    decision = cutoff - H4
    if outcome == "closed":
        frame.loc[19 * 24 + 1, "high"] = 131
    frame.to_parquet(path, index=False)
    observations = []

    def analyze(symbol, daily, execution, now, **kwargs):
        observations.append(now)
        assert now < cutoff
        if now < decision:
            return []
        op = opportunity("runoff", now)
        return [replace(
            op, created_at=decision - H4,
            expires_at=cutoff - HOUR / 2 if outcome == "expired" else decision + 2 * H4,
            entry=98 if outcome in {"pending", "expired"} else 100,
            evidence=dict(op.evidence, trigger_closed_at=decision),
        )]

    monkeypatch.setattr(replay_module, "analyze", analyze)
    cached = replay_module.replay(path, 1, runoff_days=1, max_scans=24)
    plain = replay_module.replay(path, 1, runoff_days=1, max_scans=24, use_cache=False)
    cached.pop("performance")
    plain.pop("performance")
    assert cached == plain
    assert cached["complete"] and cached["scans"] == 24
    assert cached["start"] == "1970-01-19T00:00:00+00:00"
    assert cached["entry_end"] == "1970-01-20T00:00:00+00:00"
    assert cached["end"] == cached["replayed_through"] == "1970-01-21T00:00:00+00:00"
    assert cached["entry_cutoff"] == cutoff and cached["window_days"] == 2
    assert max(observations) < cutoff
    assert cached["funnel"]["unique_accepted"] == 1
    trade = cached["trades"][0]
    assert trade["created_at"] == decision
    assert trade["status"] == ("EXPIRED" if outcome == "pending" else outcome.upper())
    assert cached["open_at_end"] == (outcome == "open")
    assert cached["pending_at_end"] == 0
    assert all(t["opened_at"] is None or t["opened_at"] < cutoff for t in cached["trades"])
    if outcome == "open":
        assert trade["closed_at"] is None and trade["exit_reason"] is None
    if outcome == "pending":
        assert trade["closed_at"] == cutoff and trade["exit_reason"] == "ENTRY_CUTOFF"
        assert trade["opened_at"] is None and trade["fills"] == []
    if outcome == "closed":
        assert trade["closed_at"] == cutoff + 2 * HOUR
    if outcome == "expired":
        assert trade["exit_reason"] == "UNFILLED"
        assert trade["closed_at"] == cutoff - HOUR / 2
    assert cached["data_issues"] == 0


@pytest.mark.parametrize("entry_style", ["confirmed", "monitored_zone"])
def test_cutoff_uses_actual_fill_timestamp_including_last_entry_hour_ioc(
    source, monkeypatch, entry_style
):
    cutoff = START
    original = replay_module.assess

    def delayed(op, instrument, equity, **kwargs):
        if kwargs["now"] < cutoff - HOUR:
            return dict(allowed=False, reasons=["TEMPORARY"])
        return original(op, instrument, equity, **kwargs)

    def analyze(symbol, daily, execution, now, **kwargs):
        op = opportunity("late", now) if entry_style == "confirmed" else watch("late", now)
        if entry_style == "confirmed":
            op = replace(
                op, created_at=op.created_at - H4, expires_at=cutoff + H4,
                evidence=dict(op.evidence, trigger_closed_at=cutoff - H4),
            )
        return [op]

    if entry_style == "confirmed":
        monkeypatch.setattr(replay_module, "analyze", analyze)
    else:
        monitor = importlib.import_module("apex_bot.monitored_zones")
        monkeypatch.setattr(monitor, "analyze_monitored_zones", analyze)
    monkeypatch.setattr(replay_module, "assess", delayed)
    cached = replay_module.replay(source[0], 1, runoff_days=1, entry_style=entry_style)
    slow = replay_module.replay(source[0], 1, runoff_days=1, entry_style=entry_style, use_cache=False)
    cached.pop("performance")
    slow.pop("performance")
    assert cached == slow
    trade = cached["trades"][0]
    assert trade["created_at"] == cutoff - HOUR
    if entry_style == "confirmed":
        assert trade["status"] == "EXPIRED" and trade["exit_reason"] == "ENTRY_CUTOFF"
        assert trade["opened_at"] is None and trade["fills"] == []
    else:
        assert trade["status"] == "OPEN" and trade["opened_at"] == cutoff - HOUR


def test_truncated_window_does_not_shift_entry_cutoff_into_runoff(source, monkeypatch):
    path, frame = source
    frame = frame.drop(19 * 24)
    frame.to_parquet(path, index=False)

    def forbidden(*args, **kwargs):
        pytest.fail("entry scan during runoff")

    monkeypatch.setattr(replay_module, "analyze", forbidden)
    result = replay_module.replay(path, 1, runoff_days=1)
    assert result["data"]["window_truncated"]
    assert result["entry_cutoff"] == START
    assert result["scans"] == 0 and result["trades"] == []
    assert result["complete"]


def test_requested_long_window_uses_total_days_and_optional_source_end(source, monkeypatch):
    pd = pytest.importorskip("pandas")
    actual_source = replay_module._source
    calls = []

    def recording_source(path, days, end):
        calls.append((days, end))
        return actual_source(path, days, end)

    monkeypatch.setattr(replay_module, "_source", recording_source)
    end = "1970-01-20T00:00:00Z"
    result = replay_module.replay(source[0], 730, runoff_days=180, end=end, max_scans=1)
    assert calls == [(910, end)]
    assert result["data"]["requested_start"] == (pd.Timestamp(end) - pd.Timedelta(days=910)).isoformat()
    assert result["entry_end"] == (pd.Timestamp(end) - pd.Timedelta(days=180)).isoformat()
    assert result["entry_days"] == 730 and result["runoff_days"] == 180


def test_cli_passes_runoff_and_keeps_monitored_thirty_day_policy(source, monkeypatch, tmp_path):
    output = tmp_path / "runoff.json"
    monkeypatch.setattr(sys, "argv", [
        "replay", str(source[0]), "--days", "1", "--runoff-days", "1",
        "--entry-style", "monitored_zone", "--output", str(output),
    ])
    replay_module.main()
    report = json.loads(output.read_text())["results"][0]
    assert report["entry_days"] == report["runoff_days"] == 1
    assert report["monitor_days"] == 30 and report["entry_style"] == "monitored_zone"
    assert report["scans"] == 24 and report["complete"]


def test_local_real_monitored_entry_stays_open_through_runoff_with_cache_hits():
    """A bounded, naturally generated entry from the existing offline cache."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "cache_ew_1h/LINKUSDT.parquet"
    if not path.exists():
        pytest.skip("optional local LINK cache unavailable")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    kwargs = dict(
        days=1, runoff_days=1, end="2026-05-25T00:00:00Z",
        entry_style="monitored_zone", monitor_days=30,
    )
    cached = replay_module.replay(path, **kwargs)
    plain = replay_module.replay(path, use_cache=False, **kwargs)
    performance = cached.pop("performance")
    plain.pop("performance")
    assert cached == plain
    assert cached["complete"] and cached["data_issues"] == 0
    assert performance["management_pivot_hits"] >= 24
    assert performance["management_pivot_builds"] == 1
    assert cached["open_at_end"] >= 1 and cached["pending_at_end"] == 0
    assert all(
        t["opened_at"] is None or t["opened_at"] < cached["entry_cutoff"]
        for t in cached["trades"]
    )
