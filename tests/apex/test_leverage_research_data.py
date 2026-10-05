"""Causal membership, immutable sources and explicit funding coverage failures."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apex_bot import leverage_research_data as data


START = pd.Timestamp("2024-01-01", tz="UTC")
WARM = START - pd.Timedelta(days=120)
CUTOFF = START + pd.Timedelta(days=2)
END = START + pd.Timedelta(days=4)


def _daily(score=100):
    return pd.DataFrame(
        {
            "date": pd.date_range(WARM, END, freq="D", inclusive="left"),
            "turnover": float(score),
        }
    )


def _hourly():
    return pd.DataFrame(
        {
            "start": pd.date_range(WARM, END, freq="h", inclusive="left"),
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 101.0,
            "volume": 10.0,
            "turnover": 1000.0,
        }
    )


def _funding():
    return pd.DataFrame(
        {
            "ts": pd.date_range(START - pd.Timedelta(days=1), END, freq="8h"),
            "rate": 0.0001,
        }
    )


def _save(root, folder, symbol, frame):
    path = root / folder / (symbol + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _symbol(root, symbol="AAAUSDT", score=100, hourly=None, funding=None):
    paths = [
        _save(root, "research/data/daily", symbol, _daily(score)),
        _save(root, "cache_ew_1h", symbol, _hourly() if hourly is None else hourly),
        _save(
            root,
            "research/data/funding",
            symbol,
            _funding() if funding is None else funding,
        ),
    ]
    return paths


def _prepare(root, output, **kwargs):
    return data.prepare(
        root, output, start=START, entry_cutoff=CUTOFF, end=END, **kwargs
    )


def test_ranking_uses_only_seven_completed_days_and_no_future_outcomes():
    daily = _daily()
    recent = (daily.date >= START - pd.Timedelta(days=7)) & (daily.date < START)
    daily.loc[recent, "turnover"] = [10, 20, 30, 40, 50, 60, 10000]
    daily.loc[daily.date >= START, "turnover"] = np.inf
    result = data.rank_history(daily, "AAAUSDT", start=START)
    assert result["score"] == 40
    assert result["ranking_start"] == "2023-12-25T00:00:00+00:00"
    changed = daily.copy()
    changed.loc[changed.date >= START, "turnover"] = -1e99
    assert data.rank_history(changed, "AAAUSDT", start=START) == result


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite", "zero"])
def test_ranking_requires_complete_valid_past_warmup(problem):
    frame = _daily()
    if problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "duplicate":
        frame = pd.concat([frame.iloc[:1], frame], ignore_index=True)
    else:
        frame.loc[0, "turnover"] = np.nan if problem == "nonfinite" else 0
    with pytest.raises(ValueError):
        data.rank_history(frame, "AAAUSDT", start=START)


def test_hourly_derived_daily_excludes_partial_days_and_duplicates():
    hourly = _hourly().iloc[1:]
    daily = data._hourly_daily(hourly, START)
    assert len(daily) == 119
    assert daily.date.iloc[0] == WARM + pd.Timedelta(days=1)
    assert daily.turnover.eq(24000).all()
    with pytest.raises(ValueError, match="Duplicate"):
        data._hourly_daily(pd.concat([hourly.iloc[:1], hourly]), START)


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "bad_high", "bad_low", "nan", "volume"]
)
def test_hourly_rejects_gaps_and_invalid_prices(problem):
    hourly = _hourly()
    if problem == "missing":
        hourly = hourly.drop(index=len(hourly) - 2)
    elif problem == "duplicate":
        hourly = pd.concat([hourly.iloc[:1], hourly])
    elif problem == "bad_high":
        hourly.loc[0, "high"] = 99
    elif problem == "bad_low":
        hourly.loc[0, "low"] = 102
    elif problem == "nan":
        hourly.loc[0, "close"] = np.nan
    else:
        hourly.loc[0, "volume"] = -1
    with pytest.raises(ValueError):
        data.validate_hourly(hourly, history_start=WARM, end=END)


def test_hourly_endpoint_is_exclusive_and_source_unchanged():
    frame = _hourly()
    after = frame.iloc[-1:].copy()
    after["start"] = END
    after["close"] = np.inf
    source = pd.concat([frame, after], ignore_index=True)
    before = source.copy(deep=True)
    result = data.validate_hourly(source, history_start=WARM, end=END)
    assert len(result) == len(frame)
    pd.testing.assert_frame_equal(source, before)


def test_funding_supports_both_native_schemas_without_timestamp_rounding():
    frame = _funding()
    frame.loc[3, "ts"] += pd.Timedelta(milliseconds=13)
    # The subsequent interval is slightly less than 8h and is retained exactly.
    frame.loc[2, "ts"] += pd.Timedelta(milliseconds=13)
    frame.loc[1, "ts"] += pd.Timedelta(milliseconds=13)
    native = data.validate_funding(frame, start=START, end=END)
    alternate = pd.DataFrame(
        {
            "ts_ms": frame.ts.astype("int64") // 1_000_000,
            "funding_rate": frame.rate,
        }
    )
    converted = data.validate_funding(alternate, start=START, end=END)
    pd.testing.assert_frame_equal(native, converted)
    assert native.ts.iloc[2] == START + pd.Timedelta(milliseconds=13)


def test_funding_interval_uses_previous_settlement_never_next():
    frame = (
        pd.concat(
            [
                _funding(),
                pd.DataFrame({"ts": [START - pd.Timedelta(hours=2)], "rate": [0.0002]}),
            ]
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )
    result = data.validate_funding(frame, start=START, end=END)
    at_start_minus_two = result.loc[result.ts == START - pd.Timedelta(hours=2)].iloc[0]
    assert at_start_minus_two.past_interval_hours == 6
    # Changing a later settlement cannot alter this row's observed interval.
    frame.loc[frame.ts == START, "ts"] += pd.Timedelta(hours=1)
    updated = data.validate_funding(frame, start=START, end=END)
    pd.testing.assert_series_equal(
        updated.loc[updated.ts == at_start_minus_two.ts].iloc[0], at_start_minus_two
    )


@pytest.mark.parametrize(
    "problem", ["gap", "duplicate", "nan", "short_start", "short_end"]
)
def test_funding_fails_closed_on_known_coverage_errors(problem):
    frame = _funding()
    if problem == "gap":
        frame = frame.drop(index=5)
    elif problem == "duplicate":
        frame = pd.concat([frame.iloc[5:6], frame], ignore_index=True)
    elif problem == "nan":
        frame.loc[5, "rate"] = np.nan
    elif problem == "short_start":
        frame = frame[frame.ts >= START]
    else:
        frame = frame.iloc[:-2]
    with pytest.raises(ValueError):
        data.validate_funding(frame, start=START, end=END)


def test_funding_comparison_reports_timestamp_and_rate_mismatches():
    primary = data.validate_funding(_funding(), start=START, end=END)
    secondary = primary.copy()
    secondary.loc[2, "rate"] = 0.0002
    secondary = secondary.drop(index=3)
    result = data.compare_funding(primary, secondary)
    assert result["rate_mismatches"] == 1
    assert result["timestamps_only_primary"] == 1
    assert result["timestamps_only_secondary"] == 0


def test_preparation_retains_rank_order_and_raw_history_hashes(tmp_path):
    source = tmp_path / "source"
    raw = _symbol(source, "BBBUSDT", 100) + _symbol(source, "AAAUSDT", 100)
    original = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in raw
    }
    output = tmp_path / "prepared"
    result = _prepare(source, output, count=2)
    assert result["complete"]
    assert result["symbols"] == ["AAAUSDT", "BBBUSDT"]
    assert result["start"] == int(START.timestamp())
    assert result["entry_cutoff"] == int(CUTOFF.timestamp())
    assert result["end"] == int(END.timestamp())
    assert not result["selection_audit"]["future_coverage_used_for_selection"]
    assert not result["selection_audit"]["current_trading_status_used_for_selection"]
    assert all(Path(path).is_absolute() for path in result["data_paths"].values())
    assert result["data_paths"]["AAAUSDT"] == str(
        source / "cache_ew_1h/AAAUSDT.parquet"
    )
    assert result["hashes"] == original
    assert [p.name for p in output.iterdir()] == ["manifest.json"]
    for path, digest in original.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert json.loads((output / "manifest.json").read_text()) == result
    with pytest.raises(ValueError, match="new directory"):
        _prepare(source, output, count=2)


def test_future_coverage_failure_blocks_selected_symbol_instead_of_replacing_it(
    tmp_path,
):
    source = tmp_path / "source"
    _symbol(source, "AAAUSDT", 100, hourly=_hourly().drop(index=2900))
    _symbol(source, "BBBUSDT", 90)
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert result["symbols"] == ["AAAUSDT"]
    assert not result["complete"]
    assert result["data_paths"] == {}
    assert "AAAUSDT" in result["blockers"][0]
    assert result["selection_audit"]["remaining_candidates"][0]["symbol"] == "BBBUSDT"


def test_same_symbol_fallback_is_allowed_without_splicing_or_replacing(tmp_path):
    source = tmp_path / "source"
    _symbol(source, "AAAUSDT", hourly=_hourly().iloc[2880:])
    fallback = _save(source, "cache_data/1h", "AAAUSDT", _hourly())
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert result["complete"]
    assert result["symbols"] == ["AAAUSDT"]
    assert result["data_paths"]["AAAUSDT"] == str(fallback)
    assert len(result["coverage_audit"][0]["hourly_attempts"]) == 1


def test_missing_funding_blocks_without_replacement(tmp_path):
    source = tmp_path / "source"
    raw = _symbol(source, "AAAUSDT", 100)
    raw[-1].unlink()
    _symbol(source, "BBBUSDT", 90)
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert result["symbols"] == ["AAAUSDT"]
    assert not result["complete"]
    assert "funding" in result["blockers"][0]


def test_current_status_is_not_a_future_survivorship_filter(tmp_path):
    source = tmp_path / "source"
    _symbol(source)
    metadata = pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "baseCoin": "AAA",
                "contractType": "LinearPerpetual",
                "quoteCoin": "USDT",
                "settleCoin": "USDT",
                "status": "Closed",
                "leverageFilter": {"maxLeverage": "2"},
            }
        ]
    )
    path = source / "research/data/instruments_linear.parquet"
    metadata.to_parquet(path, index=False)
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert result["complete"]
    assert result["symbols"] == ["AAAUSDT"]


@pytest.mark.parametrize(
    "symbol", ["BTCUSDT-26JUN26", "USDCUSDT", "USD1USDT", "abcUSDT"]
)
def test_cash_like_bases_and_dated_contracts_are_not_ranked(symbol):
    assert not data._eligible_symbol(symbol, {})


def test_verified_nonstandard_funding_transitions_have_explicit_epoch_allowlist(
    tmp_path,
):
    source = tmp_path / "source"
    funding = _funding()
    # 8 -> 6 -> 2 -> 8h preserves the actual settlement sequence.
    added = pd.DataFrame({"ts": [START + pd.Timedelta(hours=6)], "rate": [0.0002]})
    funding = pd.concat([funding, added]).sort_values("ts").reset_index(drop=True)
    _symbol(source, funding=funding)
    alternate = pd.DataFrame(
        {
            "ts_ms": funding.ts.astype("int64") // 1_000_000,
            "funding_rate": funding.rate,
        }
    )
    _save(source, "funding_cache", "AAAUSDT", alternate)
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert result["complete"]
    assert result["funding_transition_timestamps"]["AAAUSDT"] == [
        int((START + pd.Timedelta(hours=6)).timestamp() * 1000)
    ]
    assert result["coverage_audit"][0]["funding_crosscheck"]["rate_mismatches"] == 0


def test_unverified_nonstandard_funding_transition_blocks(tmp_path):
    source = tmp_path / "source"
    funding = (
        pd.concat(
            [
                _funding(),
                pd.DataFrame({"ts": [START + pd.Timedelta(hours=6)], "rate": [0.0002]}),
            ]
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )
    _symbol(source, funding=funding)
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert not result["complete"]
    assert "matching second capture" in result["blockers"][0]
    assert result["funding_transition_timestamps"]["AAAUSDT"] == []


def test_disagreeing_funding_captures_block(tmp_path):
    source = tmp_path / "source"
    _symbol(source)
    secondary = _funding()
    secondary.loc[5, "rate"] += 0.0001
    _save(source, "funding_cache", "AAAUSDT", secondary)
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert not result["complete"]
    assert "funding captures disagree" in result["blockers"][0]


def test_raw_history_origin_is_preserved_instead_of_truncated_to_minimum(tmp_path):
    source = tmp_path / "source"
    hourly = _hourly()
    prefix = hourly.iloc[:24].copy()
    prefix["start"] -= pd.Timedelta(days=1)
    raw = pd.concat([prefix, hourly], ignore_index=True)
    _symbol(source, hourly=raw)
    result = _prepare(source, tmp_path / "prepared", count=1)
    assert result["complete"]
    assert (
        result["coverage_audit"][0]["contiguous_history_start"]
        == (WARM - pd.Timedelta(days=1)).isoformat()
    )
    assert len(pd.read_parquet(result["data_paths"]["AAAUSDT"])) == len(raw)


def test_rejects_less_than_120_warmup_days(tmp_path):
    with pytest.raises(ValueError, match="120"):
        _prepare(tmp_path / "source", tmp_path / "out", count=1, warmup_days=119)
