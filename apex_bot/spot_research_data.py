"""Download only public Binance research archives, never account data.

Every ZIP must match its official SHA256 sidecar. Preserve raw archives and
provenance; normalize into new files without patching holes or prior caches.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile

import pandas as pd

from .monitored_comparison import SYMBOLS

BASE = "https://data.binance.vision/data/"
START = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2026-05-25", tz="UTC")
MAX_BYTES = 8_000_000


def _get(url):
    if not url.startswith(BASE):
        raise ValueError("Only the official public data archive is allowed")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError("Archive exceeds research size limit")
                return data
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)


def validate_archive(blob, checksum, expected_name):
    fields = checksum.decode("ascii").strip().split()
    if len(fields) != 2 or fields[1].lstrip("*") != expected_name:
        raise ValueError("Checksum filename mismatch")
    actual = hashlib.sha256(blob).hexdigest()
    if fields[0].lower() != actual:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = archive.infolist()
        if (
            len(infos) != 1
            or infos[0].filename != expected_name.removesuffix(".zip") + ".csv"
            or infos[0].file_size > MAX_BYTES
            or infos[0].is_dir()
        ):
            raise ValueError("Unexpected archive content")
        return archive.read(infos[0]), actual


def parse_month(payload, market, month):
    if market == "funding":
        frame = pd.read_csv(io.BytesIO(payload))
        if list(frame.columns) != [
            "calc_time",
            "funding_interval_hours",
            "last_funding_rate",
        ]:
            raise ValueError("Unexpected funding schema")
        if not frame.funding_interval_hours.isin([1, 2, 4, 8]).all():
            raise ValueError("Unregistered funding interval")
        result = frame.rename(
            columns={"calc_time": "ts_ms", "last_funding_rate": "funding_rate"}
        )[["ts_ms", "funding_rate", "funding_interval_hours"]]
        timestamps = pd.to_datetime(result.ts_ms, unit="ms", utc=True)
    else:
        if market not in {"spot", "perpetual"}:
            raise ValueError("Unknown archive market")
        frame = pd.read_csv(
            io.BytesIO(payload), header=0 if market == "perpetual" else None
        )
        if frame.shape[1] != 12:
            raise ValueError("Unexpected candle schema")
        unit = "us" if market == "spot" and month >= "2025-01" else "ms"
        timestamps = pd.to_datetime(frame.iloc[:, 0], unit=unit, utc=True)
        result = pd.DataFrame({"start": timestamps})
        for index, name in enumerate(("open", "high", "low", "close", "volume"), 1):
            result[name] = pd.to_numeric(frame.iloc[:, index], errors="raise")
    if result.empty or not timestamps.dt.strftime("%Y-%m").eq(month).all():
        raise ValueError("Archive timestamp outside named month")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Unordered/duplicate archive timestamps")
    return result


def validate_series(frame, market, *, start=None, end=None):
    start = START if start is None else pd.Timestamp(start)
    end = END if end is None else pd.Timestamp(end)
    if market == "funding":
        scheduled = frame.ts_ms // 28800000 * 28800000
        result = frame.loc[
            (scheduled >= start.value // 1_000_000)
            & (scheduled <= end.value // 1_000_000)
        ].copy()
        if (
            "funding_interval_hours" in result
            and not result.funding_interval_hours.eq(8).all()
        ):
            raise ValueError(
                "Retained funding interval differs from paired eight-hour protocol"
            )
        if (result.ts_ms - result.ts_ms // 28800000 * 28800000).gt(60000).any():
            raise ValueError("Unexpectedly delayed settlement timestamp")
        expected = pd.date_range(start, end, freq="8h")
        observed = pd.DatetimeIndex(
            pd.to_datetime(result.ts_ms // 28800000 * 28800000, unit="ms", utc=True)
        )
        values = result[["funding_rate"]]
    else:
        result = frame.loc[(frame.start >= start) & (frame.start < end)].copy()
        expected = pd.date_range(start, end, freq="h", inclusive="left")
        observed = pd.DatetimeIndex(result.start)
        values = result[["open", "high", "low", "close", "volume"]]
        valid = (
            result[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & result.volume.ge(0)
            & result.low.le(result[["open", "close"]].min(axis=1))
            & result.high.ge(result[["open", "close"]].max(axis=1))
        )
        if not valid.all():
            raise ValueError("Invalid OHLCV")
    if (
        not (values.notna() & values.ne(float("inf")) & values.ne(-float("inf")))
        .all()
        .all()
    ):
        raise ValueError("Nonfinite archive values")
    if not observed.equals(expected):
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        raise ValueError(
            f"Noncontiguous {market} history: missing={len(missing)} first={list(missing[:3])}, extra={len(extra)}, duplicates={observed.has_duplicates}"
        )
    return result.reset_index(drop=True)


def _fetch(job):
    market, symbol, month, output = job
    if market == "funding":
        suffix = (
            f"futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
        )
    else:
        category = "spot" if market == "spot" else "futures/um"
        suffix = f"{category}/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    url = BASE + suffix
    checksum = _get(url + ".CHECKSUM")
    blob = _get(url)
    name = suffix.rsplit("/", 1)[-1]
    payload, digest = validate_archive(blob, checksum, name)
    parsed = parse_month(payload, market, month)
    raw = Path(output) / "raw" / market / symbol
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / name).open("xb") as stream:
        stream.write(blob)
    with (raw / (name + ".CHECKSUM")).open("xb") as stream:
        stream.write(checksum)
    return (
        market,
        symbol,
        parsed,
        {
            "market": market,
            "symbol": symbol,
            "month": month,
            "url": url,
            "sha256": digest,
            "rows": len(parsed),
            "bytes": len(blob),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new; preserve earlier evidence")
    args.output.mkdir(parents=True)
    report = {
        "venue": "Binance",
        "source": "official public monthly archives",
        "start": START.isoformat(),
        "end": END.isoformat(),
        "symbols": list(SYMBOLS),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archives": [],
        "errors": [],
        "files": [],
        "complete": False,
        "source_docs": "https://github.com/binance/binance-public-data",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    months = pd.date_range(START, END, freq="MS").strftime("%Y-%m")
    groups = {(m, s): [] for m in ("spot", "perpetual", "funding") for s in SYMBOLS}
    jobs = [(m, s, month, str(args.output)) for m, s in groups for month in months]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, job): job[:3] for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                market, symbol, parsed, evidence = future.result()
                groups[market, symbol].append(parsed)
                report["archives"].append(evidence)
            except Exception as exc:
                item = {"job": futures[future], "error": f"{type(exc).__name__}: {exc}"}
                report["errors"].append(item)
                print(json.dumps(item), flush=True)
            if count % 30 == 0:
                print(
                    f"Archives {count}/{len(jobs)}, errors {len(report['errors'])}",
                    flush=True,
                )
    if not report["errors"]:
        for (market, symbol), frames in groups.items():
            try:
                key = "ts_ms" if market == "funding" else "start"
                frame = validate_series(pd.concat(frames).sort_values(key), market)
                dest = args.output / market / (symbol + ".parquet")
                dest.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(dest, index=False)
                report["files"].append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "path": str(dest),
                        "rows": len(frame),
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                    }
                )
            except Exception as exc:
                report["errors"].append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    report["complete"] = not report["errors"] and len(report["files"]) == 15
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"complete": report["complete"], "errors": report["errors"]}),
        flush=True,
    )
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
