"""Offline selection contracts: causal scores, safe entry gates, persisted churn."""

from copy import deepcopy
import json

import pytest

from apex_bot.universe import (
    DAY,
    MAX_SAMPLES,
    SAMPLE_INTERVAL,
    STABLECOIN_BASES,
    entry_symbols,
    refresh_universe,
)


NOW = 20_000 * DAY
MILLION = 1_000_000


def market(turnovers, now=NOW):
    """Turnovers are expressed in millions for readable ranking examples."""
    metadata, tickers, books = {}, [], {}
    for symbol, turnover in turnovers.items():
        metadata[symbol] = {
            "symbol": symbol,
            "baseCoin": symbol[:-4],
            "quoteCoin": "USDT",
            "settleCoin": "USDT",
            "status": "Trading",
            "contractType": "LinearPerpetual",
            "launchTime": str((NOW - 100 * DAY) * 1000),
            "isPreListing": False,
        }
        tickers.append(
            {
                "symbol": symbol,
                "turnover24h": str(turnover * MILLION),
                "bid1Price": "99.99",
                "ask1Price": "100.01",
            }
        )
        books[symbol] = {
            "as_of": now,
            "bid_depth_usdt": 25_000,
            "ask_depth_usdt": 25_000,
            "band_bps": 25,
            "spread_bps": 2,
            "bid": 99.99,
            "ask": 100.01,
        }
    return metadata, tickers, books


def refresh(previous, turnovers, now=NOW, **kwargs):
    return refresh_universe(previous, *market(turnovers, now), now, **kwargs)


def restart(state):
    return json.loads(json.dumps(state, allow_nan=False))


def test_default_fills_best_50_immediately_and_exposes_ui_evidence_without_mutation():
    values = {f"C{i:03}USDT": 100 + i for i in range(64)}
    args = market(values)
    before = deepcopy(args)
    state = refresh_universe({}, *args, NOW)
    assert args == before
    assert state["status"] == "ready" and state["target"] == 50
    assert state["active_symbols"] == [f"C{i:03}USDT" for i in range(63, 13, -1)]
    assert state["added"] == state["active_symbols"] and state["removed"] == []
    assert entry_symbols(state, NOW) == state["active_symbols"]
    assert len(state["watchlist"]) == 14 and state["blocked"] == {}
    assert state["basis"]["cold_start"] is True
    assert state["policy"]["depth_band_bps"] == 25
    assert all(
        d["sample_count"] == 1 and d["score_basis"] == "available_24h_median"
        for d in state["members"].values()
    )
    assert restart(state) == state
    again = refresh({}, dict(reversed(list(values.items()))))
    assert again == state


def test_quality_shortfall_returns_fewer_than_target_without_fallback():
    state = refresh({}, {"AUSDT": 20, "BUSDT": 19})
    assert state["status"] == "underfilled"
    assert state["active_symbols"] == ["AUSDT"]
    assert state["blocked"]["BUSDT"] == ["low_or_invalid_turnover"]
    assert state["members"]["AUSDT"]["score"] == 20 * MILLION


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("status", "Settling", "not_trading"),
        ("contractType", "LinearFutures", "not_usdt_perpetual"),
        ("quoteCoin", "USDC", "not_usdt_perpetual"),
        ("settleCoin", "USDC", "not_usdt_perpetual"),
        ("launchTime", None, "listing_too_young_or_unknown"),
        ("launchTime", "nan", "listing_too_young_or_unknown"),
        (
            "launchTime",
            str((NOW - 30 * DAY + 1) * 1000),
            "listing_too_young_or_unknown",
        ),
        ("launchTime", str((NOW + DAY) * 1000), "listing_too_young_or_unknown"),
        ("isPreListing", True, "prelisting_or_unknown"),
        ("isPreListing", "false", "prelisting_or_unknown"),
        ("isPreListing", None, "prelisting_or_unknown"),
        ("baseCoin", None, "invalid_base_coin"),
        ("symbol", "OTHERUSDT", "metadata_symbol_mismatch"),
        ("symbolType", "xstocks", "non_crypto_instrument"),
        ("symbolType", "xStocks", "non_crypto_instrument"),
    ],
)
def test_instrument_filters(field, value, reason):
    metadata, tickers, books = market({"AUSDT": 100})
    metadata["AUSDT"][field] = value
    state = refresh_universe({}, metadata, tickers, books, NOW)
    assert state["active_symbols"] == []
    assert reason in state["blocked"]["AUSDT"]


@pytest.mark.parametrize(
    "symbol", ["aUSDT", "KUSDT", "ＡUSDT", "A-BUSDT", "AUSDT\n", "USDT"]
)
def test_symbols_are_exact_ascii_uppercase_with_a_nonempty_base(symbol):
    state = refresh({}, {symbol: 100})
    assert state["active_symbols"] == []
    assert "invalid_symbol" in state["blocked"][symbol]
    restart(state)


@pytest.mark.parametrize("base", sorted(STABLECOIN_BASES))
def test_stablecoin_bases_are_excluded(base):
    state = refresh({}, {base + "USDT": 100})
    assert state["active_symbols"] == []
    assert "stablecoin_base" in state["blocked"][base + "USDT"]


def test_inclusive_quality_boundaries_and_optional_symbol_type():
    metadata, tickers, books = market({"1000PEPEUSDT": 20})
    metadata["1000PEPEUSDT"].update(
        launchTime=str((NOW - 30 * DAY) * 1000), symbolType=None
    )
    tickers[0].update(bid1Price="9995", ask1Price="10005")
    books["1000PEPEUSDT"].update(as_of=NOW - 120, spread_bps=10)
    state = refresh_universe(
        {}, metadata, tickers, books, NOW, snapshot_as_of=NOW - 120
    )
    assert entry_symbols(state, NOW) == ["1000PEPEUSDT"]


def test_eligible_members_expose_validated_numeric_book_depths():
    metadata, tickers, books = market({"AUSDT": 100})
    books["AUSDT"].update(bid_depth_usdt="35000.25", ask_depth_usdt="48000.75")
    state = refresh_universe({}, metadata, tickers, books, NOW)
    for details in (state["members"]["AUSDT"], state["eligibility"]["AUSDT"]):
        assert details["bid_depth_usdt"] == 35000.25
        assert details["ask_depth_usdt"] == 48000.75
    restart(state)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("turnover24h", "nan", "low_or_invalid_turnover"),
        ("turnover24h", "inf", "low_or_invalid_turnover"),
        ("turnover24h", "0", "low_or_invalid_turnover"),
        ("turnover24h", "-1", "low_or_invalid_turnover"),
        ("turnover24h", True, "low_or_invalid_turnover"),
        ("turnover24h", "19999999", "low_or_invalid_turnover"),
        ("bid1Price", "nan", "invalid_bid_ask"),
        ("ask1Price", "inf", "invalid_bid_ask"),
        ("bid1Price", "0", "invalid_bid_ask"),
        ("ask1Price", "-1", "invalid_bid_ask"),
        ("bid1Price", "101", "invalid_bid_ask"),
        ("ask1Price", "101", "wide_spread"),
    ],
)
def test_ticker_filters_are_finite_and_fail_closed(field, value, reason):
    metadata, tickers, books = market({"AUSDT": 100})
    tickers[0][field] = value
    state = refresh_universe({}, metadata, tickers, books, NOW)
    assert state["active_symbols"] == []
    assert reason in state["blocked"]["AUSDT"]
    restart(state)


@pytest.mark.parametrize(
    "book_update,reason",
    [
        ({"bid_depth_usdt": 24_999}, "insufficient_depth"),
        ({"ask_depth_usdt": 24_999}, "insufficient_depth"),
        ({"spread_bps": 10.001}, "wide_book_spread"),
    ],
)
def test_fresh_book_failure_removes_incumbent_and_fills_without_margin(
    book_update, reason
):
    previous = refresh({}, {"AUSDT": 100, "BUSDT": 20}, target=1)
    metadata, tickers, books = market({"AUSDT": 100, "BUSDT": 20}, NOW + 1)
    books["AUSDT"].update(book_update)
    state = refresh_universe(previous, metadata, tickers, books, NOW + 1, target=1)
    assert state["active_symbols"] == ["BUSDT"]
    assert state["added"] == ["BUSDT"] and state["removed"] == ["AUSDT"]
    assert reason in state["blocked"]["AUSDT"]
    assert state["day_replacements"] == 0


@pytest.mark.parametrize(
    "bad_book,reason",
    [
        (None, "book_missing"),
        ({"as_of": NOW - 121}, "book_stale_or_invalid_time"),
        ({"as_of": NOW + 1}, "book_stale_or_invalid_time"),
        ({"as_of": "nan"}, "book_stale_or_invalid_time"),
        (
            {"as_of": NOW, "bid_depth_usdt": "inf", "ask_depth_usdt": 25000},
            "book_invalid_depth",
        ),
        (
            {
                "as_of": NOW,
                "bid_depth_usdt": 25000,
                "ask_depth_usdt": 25000,
                "band_bps": 50,
            },
            "book_invalid_band",
        ),
        (
            {
                "as_of": NOW,
                "bid_depth_usdt": 25000,
                "ask_depth_usdt": 25000,
                "spread_bps": "nan",
            },
            "book_invalid_spread",
        ),
    ],
)
def test_unavailable_books_reserve_slots_and_block_entries_until_recovered(
    bad_book, reason
):
    previous = refresh({}, {"AUSDT": 100}, target=1)
    metadata, tickers, books = market({"AUSDT": 100, "BUSDT": 500})
    books["AUSDT"] = bad_book
    state = refresh_universe(previous, metadata, tickers, books, NOW, target=1)
    assert state["active_symbols"] == ["AUSDT"]
    assert state["status"] == "degraded" and not state["added"] and not state["removed"]
    assert state["members"]["AUSDT"]["status"] == "blocked_pending_fresh"
    assert state["members"]["AUSDT"]["bid_depth_usdt"] is None
    assert state["members"]["AUSDT"]["ask_depth_usdt"] is None
    assert reason in state["blocked"]["AUSDT"]
    assert entry_symbols(state, NOW) == [] and not state["challenger_streaks"]
    recovered = refresh(state, {"AUSDT": 100, "BUSDT": 500}, NOW + 1, target=1)
    assert entry_symbols(recovered, NOW + 1) == ["AUSDT"]


def test_partial_book_scan_accepts_available_candidates_and_keeps_every_incumbent():
    values = {f"C{i:03}USDT": 30 + i for i in range(60)}
    previous = refresh({}, values)
    metadata, tickers, books = market(values, NOW + SAMPLE_INTERVAL)
    books = {s: book for s, book in books.items() if s.endswith("0USDT")}
    state = refresh_universe(previous, metadata, tickers, books, NOW + SAMPLE_INTERVAL)
    assert state["active_symbols"] == previous["active_symbols"]
    assert state["removed"] == []
    assert entry_symbols(state, NOW + SAMPLE_INTERVAL) == [
        s for s in previous["active_symbols"] if s in books
    ]
    empty_boot = refresh_universe({}, metadata, tickers, {}, NOW + SAMPLE_INTERVAL)
    assert empty_boot["status"] == "underfilled" and empty_boot["active_symbols"] == []


@pytest.mark.parametrize(
    "failure",
    [
        "empty_metadata",
        "empty_tickers",
        "missing_metadata",
        "missing_ticker",
        "duplicate",
        "stale_snapshot",
        "future_snapshot",
        "stale_row",
        "future_row",
        "older_than_state",
        "invalid_metadata",
        "invalid_ticker",
        "incomplete_metadata",
        "incomplete_ticker",
    ],
)
def test_failed_global_snapshots_raise_and_leave_previous_and_its_as_of_untouched(
    failure,
):
    previous = refresh({}, {"AUSDT": 100, "BUSDT": 80}, target=2)
    before = deepcopy(previous)
    metadata, tickers, books = market({"AUSDT": 100, "BUSDT": 80}, NOW + 1)
    at = NOW + 1
    if failure == "empty_metadata":
        metadata = {}
    elif failure == "empty_tickers":
        tickers = []
    elif failure == "missing_metadata":
        del metadata["AUSDT"]
    elif failure == "missing_ticker":
        tickers.pop()
    elif failure == "duplicate":
        tickers.append(dict(tickers[0]))
    elif failure == "stale_snapshot":
        at = NOW - 120
    elif failure == "future_snapshot":
        at = NOW + 2
    elif failure == "stale_row":
        tickers[0]["as_of"] = NOW - 120
    elif failure == "future_row":
        tickers[0]["as_of"] = NOW + 2
    elif failure == "older_than_state":
        at = NOW - 1
    elif failure == "invalid_metadata":
        metadata["AUSDT"] = None
    elif failure == "invalid_ticker":
        tickers.append(None)
    elif failure == "incomplete_metadata":
        del metadata["AUSDT"]["status"]
    elif failure == "incomplete_ticker":
        del tickers[0]["bid1Price"]
    with pytest.raises(ValueError):
        refresh_universe(
            previous, metadata, tickers, books, NOW + 1, target=2, snapshot_as_of=at
        )
    assert previous == before and previous["as_of"] == NOW


def test_delisting_and_turnover_failure_revoke_entries_immediately_without_an_observation():
    previous = refresh({}, {"AUSDT": 100, "BUSDT": 80, "CUSDT": 30}, target=2)
    state = refresh(previous, {"BUSDT": 19, "CUSDT": 30}, NOW + 1, target=2)
    assert state["active_symbols"] == ["CUSDT"]
    assert state["removed"] == ["AUSDT", "BUSDT"]
    assert state["blocked"]["AUSDT"] == ["not_listed"]
    assert state["day_replacements"] == 0
    assert entry_symbols(state, NOW + 1) == ["CUSDT"]


def test_median_ranking_ignores_future_expired_and_duplicate_history():
    previous = {
        "snapshots": {
            "AUSDT": [
                {"as_of": NOW - 7 * DAY, "turnover24h": 1},
                {"as_of": NOW - 2 * SAMPLE_INTERVAL, "turnover24h": 100 * MILLION},
                {"as_of": NOW - SAMPLE_INTERVAL, "turnover24h": 100 * MILLION},
                {"as_of": NOW - SAMPLE_INTERVAL, "turnover24h": 1},
            ],
            "BUSDT": [
                {"as_of": NOW - 2 * SAMPLE_INTERVAL, "turnover24h": 20 * MILLION},
                {"as_of": NOW - SAMPLE_INTERVAL, "turnover24h": 20 * MILLION},
                {"as_of": NOW + SAMPLE_INTERVAL, "turnover24h": 1000 * MILLION},
            ],
        }
    }
    state = refresh(previous, {"AUSDT": 20, "BUSDT": 1000}, target=1)
    assert state["active_symbols"] == ["AUSDT"]
    assert state["members"]["AUSDT"]["score"] == 100 * MILLION
    assert state["eligibility"]["BUSDT"]["score"] == 20 * MILLION
    assert state["members"]["AUSDT"]["sample_count"] == 3
    assert len(state["snapshots"]["BUSDT"]) == 3


def test_history_keeps_all_600_candidates_without_a_128_symbol_truncation():
    values = {f"C{i:03}USDT": 1 + i for i in range(600)}
    state = refresh({}, values)
    assert len(state["snapshots"]) == 600
    assert state["snapshots"]["C000USDT"][0]["turnover24h"] == MILLION


def test_history_rolls_at_seven_days_and_28_samples_and_never_fabricates_missed_samples():
    state = {}
    for step in range(35):
        state = refresh(
            restart(state), {"AUSDT": 20 + step}, NOW + step * SAMPLE_INTERVAL, target=1
        )
    samples = state["snapshots"]["AUSDT"]
    assert len(samples) == MAX_SAMPLES
    assert samples[0]["as_of"] == NOW + 7 * SAMPLE_INTERVAL
    assert state["members"]["AUSDT"]["score_basis"] == "7d_median"
    assert not state["basis"]["cold_start"]
    late = refresh(state, {"AUSDT": 80}, NOW + 20 * DAY, target=1)
    assert len(late["snapshots"]["AUSDT"]) == 1
    assert late["members"]["AUSDT"]["score"] == 80 * MILLION


def test_two_six_hour_confirmations_survive_restart_and_duplicate_refreshes():
    state = refresh({}, {"AUSDT": 100}, target=1)
    one = NOW + SAMPLE_INTERVAL
    state = refresh(state, {"AUSDT": 100, "BUSDT": 150}, one, target=1)
    assert state["active_symbols"] == ["AUSDT"]
    assert state["challenger_streaks"]["BUSDT"]["count"] == 1
    for now in (one, one + 1, one + SAMPLE_INTERVAL - 1):
        state = refresh(restart(state), {"AUSDT": 100, "BUSDT": 150}, now, target=1)
        assert state["challenger_streaks"]["BUSDT"]["count"] == 1
        assert len(state["snapshots"]["BUSDT"]) == 1
        assert state["day_replacements"] == 0
    state = refresh(
        restart(state), {"AUSDT": 100, "BUSDT": 150}, one + SAMPLE_INTERVAL, target=1
    )
    assert state["active_symbols"] == ["BUSDT"]
    assert state["added"] == ["BUSDT"] and state["removed"] == ["AUSDT"]
    assert state["day_replacements"] == 1 and state["challenger_streaks"] == {}
    same = refresh(
        restart(state), {"AUSDT": 100, "BUSDT": 150}, one + SAMPLE_INTERVAL, target=1
    )
    assert same["day_replacements"] == 1 and not same["added"] and not same["removed"]


def test_crossing_a_six_hour_bucket_by_one_second_does_not_create_an_observation():
    start = NOW + SAMPLE_INTERVAL - 1
    state = refresh({}, {"AUSDT": 100}, start, target=1)
    state = refresh(state, {"AUSDT": 100, "BUSDT": 150}, start + 1, target=1)
    assert state["challenger_streaks"] == {}
    assert len(state["snapshots"]["AUSDT"]) == 1
    state = refresh(
        state, {"AUSDT": 100, "BUSDT": 150}, start + SAMPLE_INTERVAL, target=1
    )
    assert state["challenger_streaks"]["BUSDT"]["count"] == 1
    assert state["active_symbols"] == ["AUSDT"]


def test_less_than_1_5_margin_never_replaces_and_a_failed_observation_resets_streak():
    state = refresh({}, {"AUSDT": 100}, target=1)
    for step in range(1, 4):
        state = refresh(
            state, {"AUSDT": 100, "BUSDT": 149}, NOW + step * SAMPLE_INTERVAL, target=1
        )
    assert state["active_symbols"] == ["AUSDT"] and not state["challenger_streaks"]
    state = refresh({}, {"AUSDT": 100}, target=1)
    state = refresh(
        state, {"AUSDT": 100, "BUSDT": 150}, NOW + SAMPLE_INTERVAL, target=1
    )
    metadata, tickers, books = market(
        {"AUSDT": 100, "BUSDT": 150}, NOW + SAMPLE_INTERVAL + 1
    )
    del books["BUSDT"]
    state = refresh_universe(
        state, metadata, tickers, books, NOW + SAMPLE_INTERVAL + 1, target=1
    )
    assert not state["challenger_streaks"]
    state = refresh(
        state, {"AUSDT": 100, "BUSDT": 150}, NOW + 2 * SAMPLE_INTERVAL, target=1
    )
    assert state["challenger_streaks"]["BUSDT"]["count"] == 1
    assert state["active_symbols"] == ["AUSDT"]


def test_daily_cap_persists_across_restart_resets_at_utc_midnight_and_exempts_safety_fills():
    incumbents = {f"A{i}USDT": 100 for i in range(7)}
    values = dict(incumbents, **{f"B{i}USDT": 200 for i in range(7)})
    state = refresh({}, incumbents, target=7)
    for step in (1, 2):
        state = refresh(restart(state), values, NOW + step * SAMPLE_INTERVAL, target=7)
    assert state["day_replacements"] == 5
    assert state["active_symbols"] == [f"B{i}USDT" for i in range(5)] + [
        "A0USDT",
        "A1USDT",
    ]
    for step in (2, 3):
        state = refresh(restart(state), values, NOW + step * SAMPLE_INTERVAL, target=7)
        assert state["day_replacements"] == 5 and len(state["active_symbols"]) == 7
    assert [
        row["symbol"]
        for row in state["watchlist"]
        if row["reason"] == "daily_replacement_limit"
    ] == ["B5USDT", "B6USDT"]
    unsafe = dict(values, A0USDT=19)
    safety = refresh(restart(state), unsafe, NOW + 3 * SAMPLE_INTERVAL + 1, target=7)
    assert (
        "A0USDT" not in safety["active_symbols"]
        and "B5USDT" in safety["active_symbols"]
    )
    assert safety["day_replacements"] == 5
    next_day = refresh(restart(state), values, NOW + DAY, target=7)
    assert next_day["active_symbols"] == [f"B{i}USDT" for i in range(7)]
    assert next_day["day_replacements"] == 2
    assert next_day["last_rotation_day"] != state["last_rotation_day"]


def test_ties_have_a_stable_symbol_order_for_selection_and_victim_pairing():
    state = refresh({}, {"BUSDT": 100, "AUSDT": 100}, target=2)
    assert state["active_symbols"] == ["AUSDT", "BUSDT"]
    values = {"DUSDT": 150, "CUSDT": 150, "BUSDT": 100, "AUSDT": 100}
    state = refresh(state, values, NOW + SAMPLE_INTERVAL, target=2)
    assert state["challenger_streaks"]["CUSDT"]["incumbent"] == "BUSDT"
    assert state["challenger_streaks"]["DUSDT"]["incumbent"] == "AUSDT"
    state = refresh(state, values, NOW + 2 * SAMPLE_INTERVAL, target=2)
    assert state["active_symbols"] == ["CUSDT", "DUSDT"]


def test_confirmations_cannot_transfer_to_a_different_incumbent():
    state = refresh({}, {"AUSDT": 100, "BUSDT": 110}, target=2)
    state = refresh(
        state,
        {"AUSDT": 200, "BUSDT": 110, "CUSDT": 180},
        NOW + SAMPLE_INTERVAL,
        target=2,
    )
    assert state["challenger_streaks"]["CUSDT"]["incumbent"] == "BUSDT"
    state = refresh(
        state,
        {"AUSDT": 20, "BUSDT": 110, "CUSDT": 180},
        NOW + 2 * SAMPLE_INTERVAL,
        target=2,
    )
    assert state["challenger_streaks"]["CUSDT"]["incumbent"] == "AUSDT"
    assert state["challenger_streaks"]["CUSDT"]["count"] == 1
    assert set(state["active_symbols"]) == {"AUSDT", "BUSDT"}


def test_entry_gate_expires_closed_at_explicit_age_and_rejects_future_state():
    state = refresh({}, {"AUSDT": 100})
    assert entry_symbols(state, NOW + DAY - 0.001) == ["AUSDT"]
    assert entry_symbols(state, NOW + DAY) == []
    assert entry_symbols(state, NOW + DAY + 0.001) == []
    assert entry_symbols(state, NOW - 1) == []
    assert entry_symbols(state, NOW + 121, max_age=120) == []
    assert entry_symbols(state, NOW + 120, max_age=120) == []
    assert entry_symbols(state, NOW, max_age=0) == []
    assert entry_symbols(state, float("nan")) == []
    assert entry_symbols({}, NOW) == []
    assert entry_symbols(dict(state, status="failed"), NOW) == []
    assert entry_symbols(dict(state, members={}), NOW) == []


@pytest.mark.parametrize(
    "patch",
    [
        {"as_of": float("nan")},
        {"as_of": float("inf")},
        {"as_of": -1},
        {"as_of": NOW + 1},
        {"as_of": None},
        {"status": []},
        {"status": {}},
        {"active_symbols": {}},
        {"members": []},
    ],
)
def test_malformed_entry_state_fails_closed(patch):
    state = refresh({}, {"AUSDT": 100})
    assert entry_symbols(dict(state, **patch), NOW) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": 0},
        {"target": True},
        {"target": 2.5},
        {"replacement_ratio": 1},
        {"max_daily_replacements": -1},
        {"confirmation_observations": 0},
        {"max_spread_bps": float("nan")},
        {"min_depth_usdt": float("inf")},
    ],
)
def test_invalid_policy_raises(kwargs):
    with pytest.raises(ValueError):
        refresh({}, {"AUSDT": 100}, **kwargs)


def test_extreme_finite_turnovers_still_produce_json_safe_medians():
    state = refresh({}, {"AUSDT": 1e302}, target=1)
    state = refresh(state, {"AUSDT": 1e302}, NOW + SAMPLE_INTERVAL, target=1)
    assert state["members"]["AUSDT"]["score"] == 1e308
    restart(state)
