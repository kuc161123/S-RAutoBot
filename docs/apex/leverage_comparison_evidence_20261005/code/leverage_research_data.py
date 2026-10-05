"""Prepare fixed-universe Bybit research data from local public-data caches.

Rank using only completed pre-test daily turnover. Freeze membership before
checking future coverage; missing data blocks the run, never replaces a symbol.
No network, credentials, account access, raw-cache writes or trading operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .universe import STABLECOIN_BASES

START = "2023-11-27T00:00:00Z"
ENTRY_CUTOFF = "2025-11-26T00:00:00Z"
END = "2026-05-25T00:00:00Z"
DISCOVERY_HOURLY = ("cache_ew_1h", "cache_3yr_1h")
HOURLY_SOURCES = DISCOVERY_HOURLY + ("cache_data/1h",)
FUNDING_SOURCES = ("research/data/funding", "funding_cache")
OHLCV = ["open", "high", "low", "close", "volume", "turnover"]
LIMITATIONS = [
    "Fixed top 50 within available cached histories, not a complete historical exchange census; renamed/delisted instruments absent from all caches cannot be recovered.",
    "Median turnover over seven completed UTC days is a historical liquidity proxy, not the live rolling universe with spread/depth/hysteresis filters.",
    "No historical order books, mark-price candles, risk tiers or point-in-time instrument filters are supplied; this data cannot establish exact exchange liquidations.",
    "Instrument metadata is a present-day cache; current Trading status, leverage limits and future candle coverage are not selection filters.",
    "Funding intervals may change. Actual settlement timestamps are preserved; past_interval_hours is backward-looking, not a forecast of the next interval.",
    "Maximum funding gap <=8h and agreement between local caches are consistency checks, not independent proof of a complete historical settlement schedule.",
    "Current instrument fundingInterval must not be applied retroactively to historical funding.",
]


def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Invalid date")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dates(values):
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _eligible_symbol(symbol, metadata):
    if re.fullmatch(r"[A-Z0-9]+USDT", symbol, flags=re.ASCII) is None:
        return False
    base = metadata.get("baseCoin") or symbol[:-4]
    if base in STABLECOIN_BASES:
        return False
    # These describe the contract, not future listing status or trading results.
    return not metadata or (
        metadata.get("contractType") == "LinearPerpetual"
        and metadata.get("quoteCoin") == "USDT"
        and metadata.get("settleCoin") == "USDT"
    )


def rank_history(frame, symbol, *, start=START, warmup_days=120, ranking_days=7):
    """Score a daily history without consulting the test window or its outcomes."""
    start = _utc(start)
    if start != start.floor("D"):
        raise ValueError("Ranking start must be a UTC midnight")
    _positive_integer(warmup_days, "warmup_days")
    _positive_integer(ranking_days, "ranking_days")
    if ranking_days > warmup_days:
        raise ValueError("Ranking window exceeds warmup")
    dates = _dates(frame["date"])
    warmup = start - pd.Timedelta(days=warmup_days)
    daily = frame.loc[(dates >= warmup) & (dates < start)].copy()
    daily["date"] = dates[(dates >= warmup) & (dates < start)]
    daily = daily.sort_values("date")
    expected = pd.date_range(warmup, start, freq="D", inclusive="left")
    if not pd.DatetimeIndex(daily.date).equals(expected):
        raise ValueError("Incomplete/duplicate pre-origin daily warmup")
    values = pd.to_numeric(daily.turnover, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid pre-origin daily turnover")
    recent = values[-ranking_days:]
    return {
        "symbol": symbol,
        "score": float(np.median(recent)),
        "median_daily_turnover_7d": float(np.median(recent)),
        "mean_daily_turnover": float(np.mean(recent)),
        "ranking_start": (start - pd.Timedelta(days=ranking_days)).isoformat(),
        "ranking_end_exclusive": start.isoformat(),
        "warmup_days": warmup_days,
        "ranking_days": ranking_days,
    }


def _hourly_daily(frame, start):
    """Recover only complete historical UTC days, with no duplicate summing."""
    dates = _dates(frame.start)
    prior = frame.loc[dates < start, ["turnover"]].copy()
    prior.index = dates[dates < start]
    if prior.index.has_duplicates:
        raise ValueError("Duplicate pre-origin hourly turnover")
    if not prior.index.equals(prior.index.floor("h")):
        raise ValueError("Off-hour pre-origin turnover")
    prior = prior.sort_index()
    groups = prior.turnover.resample("D")
    daily = pd.DataFrame({"turnover": groups.sum(), "count": groups.count()})
    return daily.loc[daily["count"] == 24, ["turnover"]].reset_index(names="date")


def validate_hourly(frame, *, history_start, end):
    """Require every real hourly bar; never fill gaps or repair bad prices."""
    start, end = _utc(history_start), _utc(end)
    dates = _dates(frame.start)
    keep = (dates >= start) & (dates < end)
    result = frame.loc[keep, OHLCV].copy()
    result.insert(0, "start", dates[keep])
    result = result.sort_values("start").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    observed = pd.DatetimeIndex(result.start)
    if not observed.equals(expected):
        missing = expected.difference(observed)
        raise ValueError(
            f"Incomplete hourly coverage: missing={len(missing)}, "
            f"first={list(missing[:3])}, duplicates={observed.has_duplicates}"
        )
    values = result[OHLCV].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values[:, :4] <= 0).any()
        or (values[:, 4:] < 0).any()
        or (values[:, 1] < values[:, [0, 2, 3]].max(axis=1)).any()
        or (values[:, 2] > values[:, [0, 1, 3]].min(axis=1)).any()
    ):
        raise ValueError("Invalid hourly OHLCV/turnover")
    return result


def validate_funding(frame, *, start, end):
    """Preserve actual settlements and calculate interval using only the past.

    Include two observations preceding entry to establish a backward interval.
    Funding at the exclusive price endpoint is retained for settlement handling.
    """
    start, end = _utc(start), _utc(end)
    if {"ts", "rate"} <= set(frame.columns):
        dates, rates = _dates(frame.ts), frame.rate.to_numpy(float)
    elif {"ts_ms", "funding_rate"} <= set(frame.columns):
        dates = pd.DatetimeIndex(pd.to_datetime(frame.ts_ms, unit="ms", utc=True))
        rates = frame.funding_rate.to_numpy(float)
    else:
        raise ValueError("Unknown funding schema")
    result = pd.DataFrame({"ts": dates, "rate": rates}).sort_values("ts")
    if result.empty or result.ts.isna().any():
        raise ValueError("Empty/invalid funding history")
    prior = result.loc[result.ts < start].tail(2)
    kept = result.loc[(result.ts >= start) & (result.ts <= end)]
    result = pd.concat([prior, kept]).reset_index(drop=True)
    if (
        len(prior) < 2
        or result.ts.duplicated().any()
        or not np.isfinite(result.rate.to_numpy(float)).all()
    ):
        raise ValueError("Incomplete/duplicate/nonfinite funding history")
    intervals = result.ts.diff().dt.total_seconds() / 3600.0
    if (
        (intervals.dropna() <= 0).any()
        or (intervals.dropna() > 8).any()
        or (start - prior.ts.iloc[-1]) > pd.Timedelta(hours=8)
        or (end - result.ts.iloc[-1]) > pd.Timedelta(hours=8)
    ):
        raise ValueError("Funding coverage gap exceeds eight hours")
    # A 5/6/7-hour transition is not silently shifted to another timestamp.
    result["past_interval_hours"] = intervals
    return result


def compare_funding(primary, secondary):
    """Compare a second local capture; do not claim independent authenticity."""
    left = primary.set_index("ts").rate
    right = secondary.set_index("ts").rate
    common = left.index.intersection(right.index)
    differences = (left.loc[common] - right.loc[common]).abs()
    return {
        "compared_rows": len(common),
        "timestamps_only_primary": len(left.index.difference(right.index)),
        "timestamps_only_secondary": len(right.index.difference(left.index)),
        "rate_mismatches": int((differences > 1e-12).sum()),
        "max_rate_difference": float(differences.max()) if len(common) else None,
    }


def prepare(
    root,
    output,
    *,
    start=START,
    entry_cutoff=ENTRY_CUTOFF,
    end=END,
    count=50,
    warmup_days=120,
):
    """Write a new manifest referencing raw caches, or a blocked audit manifest.

    Source histories are not clipped. The runner retains each symbol's actual
    contiguous history origin, while every selected series must contain at
    least the declared common 120-day warmup and the entire evaluation window.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(
            "Output must be a new directory; existing evidence is preserved"
        )
    start, entry_cutoff, end = map(_utc, (start, entry_cutoff, end))
    _positive_integer(count, "count")
    _positive_integer(warmup_days, "warmup_days")
    if warmup_days < 120:
        raise ValueError("At least 120 warmup days are required")
    if not start < entry_cutoff < end or any(
        stamp != stamp.floor("D") for stamp in (start, entry_cutoff, end)
    ):
        raise ValueError("Windows must be increasing complete UTC days")
    history_start = start - pd.Timedelta(days=warmup_days)
    hashes = {}

    def read(path):
        path = path.resolve()
        hashes.setdefault(str(path), _digest(path))
        return pd.read_parquet(path)

    metadata_path = root / "research/data/instruments_linear.parquet"
    metadata = (
        {r["symbol"]: r for r in read(metadata_path).to_dict("records")}
        if metadata_path.exists()
        else {}
    )
    daily_paths = {p.stem: p for p in (root / "research/data/daily").glob("*.parquet")}
    hourly_paths = {
        folder: {p.stem: p for p in (root / folder).glob("*.parquet")}
        for folder in HOURLY_SOURCES
    }
    candidates = set(daily_paths)
    for folder in DISCOVERY_HOURLY:
        candidates.update(hourly_paths[folder])
    ranked, excluded = [], []
    for symbol in sorted(candidates):
        if not _eligible_symbol(symbol, metadata.get(symbol, {})):
            continue
        path = daily_paths.get(symbol)
        try:
            if path is not None:
                daily = read(path)
            else:
                path = next(
                    hourly_paths[folder][symbol]
                    for folder in DISCOVERY_HOURLY
                    if symbol in hourly_paths[folder]
                )
                daily = _hourly_daily(read(path), start)
            item = rank_history(daily, symbol, start=start, warmup_days=warmup_days)
            ranked.append(dict(item, ranking_path=str(path.resolve())))
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append({"symbol": symbol, "reason": str(exc), "path": str(path)})
    ranked.sort(key=lambda item: (-item["score"], item["symbol"]))
    # Fixed now. No later validation failure may change this membership.
    selected = ranked[:count]
    blockers = []
    if len(selected) < count:
        blockers.append(f"Only {len(selected)} eligible histories; requested {count}")
    frames, funding_frames, coverage, transitions = {}, {}, [], {}
    for selected_item in selected:
        symbol = selected_item["symbol"]
        item = {"symbol": symbol, "hourly_attempts": [], "funding_attempts": []}
        for folder in HOURLY_SOURCES:
            path = hourly_paths[folder].get(symbol)
            if path is None:
                continue
            try:
                source = read(path)
                frame = validate_hourly(source, history_start=history_start, end=end)
                frames[symbol] = frame
                item["hourly_path"] = str(path.resolve())
                item["hourly_rows"] = len(frame)
                timestamps = _dates(source.start).sort_values()
                prior = timestamps[timestamps < end]
                gap_indices = np.flatnonzero(np.diff(prior.asi8) > 3600 * 10**9) + 1
                contiguous_start = (
                    prior[gap_indices[-1]] if len(gap_indices) else prior[0]
                )
                item["raw_history_start"] = prior[0].isoformat()
                item["contiguous_history_start"] = contiguous_start.isoformat()
                break
            except (ValueError, KeyError, TypeError) as exc:
                item["hourly_attempts"].append({"path": str(path), "error": str(exc)})
        if symbol not in frames:
            blockers.append(f"{symbol}: selected symbol lacks complete hourly coverage")
        captures = []
        for folder in FUNDING_SOURCES:
            path = root / folder / (symbol + ".parquet")
            if not path.exists():
                continue
            try:
                frame = validate_funding(read(path), start=start, end=end)
                captures.append((path, frame))
            except (ValueError, KeyError, TypeError) as exc:
                item["funding_attempts"].append({"path": str(path), "error": str(exc)})
        if not captures:
            blockers.append(f"{symbol}: selected symbol lacks usable funding coverage")
        else:
            path, frame = captures[0]
            funding_frames[symbol] = frame
            item["funding_path"] = str(path.resolve())
            item["funding_rows"] = len(frame)
            item["interval_hours_counts"] = {
                str(k): int(v)
                for k, v in frame.past_interval_hours.value_counts()
                .sort_index()
                .items()
            }
            crosschecked = False
            if len(captures) > 1:
                check = compare_funding(frame, captures[1][1])
                item["funding_crosscheck"] = check
                mismatch = any(
                    check[key]
                    for key in (
                        "rate_mismatches",
                        "timestamps_only_primary",
                        "timestamps_only_secondary",
                    )
                )
                if mismatch:
                    blockers.append(f"{symbol}: local funding captures disagree")
                crosschecked = not mismatch
            else:
                item["funding_crosscheck"] = {"available": False}
            exceptional = frame.loc[
                frame.past_interval_hours.notna()
                & ~frame.past_interval_hours.isin([1, 2, 4, 8])
            ]
            transitions[symbol] = []
            if not exceptional.empty:
                if not crosschecked:
                    blockers.append(
                        f"{symbol}: nonstandard funding transition lacks matching second capture"
                    )
                else:
                    transitions[symbol] = [
                        int(ts.timestamp() * 1000) for ts in exceptional.ts
                    ]
            item["funding_transition_timestamps"] = transitions[symbol]
        coverage.append(item)
    changed = [path for path, digest in hashes.items() if _digest(path) != digest]
    if changed:
        blockers.append("Source changed during preparation: " + ", ".join(changed))
    manifest = {
        "schema_version": 1,
        "venue": "Bybit",
        "market": "USDT linear perpetual",
        "symbols": [row["symbol"] for row in selected],
        "start": int(start.timestamp()),
        "entry_cutoff": int(entry_cutoff.timestamp()),
        "end": int(end.timestamp()),
        "history_start": int(history_start.timestamp()),
        "history_start_semantics": "Minimum guaranteed shared warmup; raw per-symbol contiguous origins are preserved in coverage_audit",
        "entry_days": (entry_cutoff - start).days,
        "runoff_days": (end - entry_cutoff).days,
        "data_paths": {},
        "funding_paths": {},
        "funding_transition_timestamps": transitions,
        "hashes": hashes,
        "selection_audit": {
            "method": "Descending median USDT turnover over seven completed UTC days; alphabetical symbol ties",
            "ranking_start": (start - pd.Timedelta(days=7)).isoformat(),
            "ranking_end_exclusive": start.isoformat(),
            "warmup_days": warmup_days,
            "future_coverage_used_for_selection": False,
            "current_trading_status_used_for_selection": False,
            "candidate_count": len(candidates),
            "eligible_count": len(ranked),
            "requested_count": count,
            "selected": selected,
            "remaining_candidates": ranked[count:],
            "excluded_before_ranking": excluded,
            "discovery_directories": ["research/data/daily", *DISCOVERY_HOURLY],
            "hourly_fallback_directories": list(HOURLY_SOURCES),
        },
        "coverage_audit": coverage,
        "limitations": LIMITATIONS,
        "blockers": blockers,
        "complete": not blockers,
    }
    output.mkdir(parents=True)
    if not blockers:
        manifest["data_paths"] = {row["symbol"]: row["hourly_path"] for row in coverage}
        manifest["funding_paths"] = {
            row["symbol"]: row["funding_path"] for row in coverage
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default=START)
    parser.add_argument("--entry-cutoff", default=ENTRY_CUTOFF)
    parser.add_argument("--end", default=END)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--warmup-days", type=int, default=120)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve() / "manifest.json"),
                "complete": result["complete"],
                "symbols": result["symbols"],
                "blockers": result["blockers"],
            },
            indent=2,
        )
    )
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
