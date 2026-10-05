"""Choose a common complete warmup AFTER documented gaps, before any outcomes.

Recheck each public archive checksum. Never interpolate missing candles, change
the predeclared entry/runoff period or overwrite the original acquisition audit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .monitored_comparison import SYMBOLS
from .spot_research_data import (
    START,
    END,
    parse_month,
    validate_archive,
    validate_series,
)

ENTRY_START = pd.Timestamp("2023-11-27", tz="UTC")


def common_origin(frames):
    origin = START
    gaps = []
    for (market, symbol), frame in frames.items():
        if market == "funding":
            continue
        prior = frame.loc[(frame.start >= START) & (frame.start < END)].copy()
        if prior.empty or prior.start.duplicated().any():
            raise ValueError("Empty or duplicated candle history")
        origin = max(origin, prior.start.iloc[0].ceil("D"))
        gap_rows = prior.loc[prior.start.diff() > pd.Timedelta(hours=1)]
        for index, row in gap_rows.iterrows():
            previous = prior.loc[:index].iloc[-2].start
            item = {
                "market": market,
                "symbol": symbol,
                "missing_from": (previous + pd.Timedelta(hours=1)).isoformat(),
                "resumes_at": row.start.isoformat(),
            }
            gaps.append(item)
            if row.start >= ENTRY_START:
                raise ValueError(f"Gap reaches fixed evaluation period: {item}")
            origin = max(origin, row.start.ceil("D"))
    if ENTRY_START - origin < pd.Timedelta(days=120):
        raise ValueError("Fewer than 120 complete common warmup days")
    return origin, gaps


def normalize(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Normalized output must be a new directory")
    frames, archives = {}, []
    months = pd.date_range(START, END, freq="MS").strftime("%Y-%m")
    for market in ("spot", "perpetual", "funding"):
        for symbol in SYMBOLS:
            pieces = []
            for month in months:
                kind = "fundingRate" if market == "funding" else "1h"
                name = f"{symbol}-{kind}-{month}.zip"
                path = source / "raw" / market / symbol / name
                payload, digest = validate_archive(
                    path.read_bytes(),
                    path.with_suffix(".zip.CHECKSUM").read_bytes(),
                    name,
                )
                pieces.append(parse_month(payload, market, month))
                archives.append({"path": str(path), "sha256": digest})
            key = "ts_ms" if market == "funding" else "start"
            frames[market, symbol] = (
                pd.concat(pieces).sort_values(key).reset_index(drop=True)
            )
    origin, gaps = common_origin(frames)
    validated = {
        key: validate_series(frame, key[0], start=origin)
        for key, frame in frames.items()
    }
    output.mkdir(parents=True)
    result = {
        "venue": "Binance",
        "markets": ["spot", "perpetual", "funding"],
        "history_origin": origin.isoformat(),
        "entry_start": ENTRY_START.isoformat(),
        "end": END.isoformat(),
        "warmup_days": (ENTRY_START - origin).days,
        "gaps_excluded_from_warmup": gaps,
        "archives": archives,
        "files": [],
        "complete": True,
    }
    for (market, symbol), frame in validated.items():
        dest = output / market / (symbol + ".parquet")
        dest.parent.mkdir(exist_ok=True)
        frame.to_parquet(dest, index=False)
        result["files"].append(
            {
                "market": market,
                "symbol": symbol,
                "path": str(dest),
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "rows": len(frame),
            }
        )
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k not in {"archives", "files"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize(args.source, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
