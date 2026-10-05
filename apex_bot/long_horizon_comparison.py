"""Fixed offline 730-day entry / 180-day runoff protocol; no external I/O.

Only reads the specified local candle/funding caches. Creates a new evidence
directory, never overwrites earlier research or connects to the trading runtime.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

from .monitored_comparison import SYMBOLS, metrics
from .replay import replay
from .simulation import apply_funding

END = "2026-05-25T00:00:00Z"
ARMS = ("confirmed", "monitored_zone")


def funding_estimates(result, path):
    """Rate-cache coverage checks do not certify historical settlement marks."""
    import pandas as pd

    frame = pd.read_parquet(path)
    rows = frame[["ts_ms", "funding_rate"]].sort_values("ts_ms")
    if rows.ts_ms.duplicated().any() or not all(
        rows.funding_rate.map(lambda x: float("-inf") < x < float("inf"))
    ):
        raise ValueError("Invalid cached funding records")
    rates = [
        {
            "fundingRateTimestamp": int(t),
            "fundingRate": float(r),
            "symbol": result["symbol"],
        }
        for t, r in rows.itertuples(index=False, name=None)
    ]
    start = datetime.fromisoformat(result["start"]).timestamp()
    end = datetime.fromisoformat(result["end"]).timestamp()
    expected = set(
        range(int(start // 28800 + 1) * 28800000, int(end * 1000) + 1, 28800000)
    )
    observed = {
        r["fundingRateTimestamp"]
        for r in rates
        if start * 1000 < r["fundingRateTimestamp"] <= end * 1000
    }
    estimates = []
    for trade in result["trades"]:
        if trade.get("opened_at") is None:
            continue
        updated = apply_funding(trade, rates, end, history_complete=False)
        if updated.get("funding_error"):
            raise ValueError(updated["funding_error"])
        estimates.append(
            {
                "id": trade["id"],
                "status": trade["status"],
                "funding_estimate": updated["funding"],
                "net_with_funding_estimate": updated["net_pnl"],
                "settlements": len(updated["funding_events"]),
                "funding_complete": False,
            }
        )
    return {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_8h_records": len(expected),
        "observed_records": len(observed),
        "matches_8h_schedule": expected == observed,
        "basis": "Cached settled rate x remaining entry notional; estimate, not settlement mark. Entry/exit timestamp ties excluded. Does not change sizing or replay decisions.",
        "closed_funding_estimate": sum(
            t["funding_estimate"] for t in estimates if t["status"] == "CLOSED"
        ),
        "trades": estimates,
    }


def _run(task):
    source, funding, arm, symbol = task
    result = replay(
        Path(source) / (symbol + ".parquet"),
        730,
        end=END,
        runoff_days=180,
        entry_style=arm,
        monitor_days=30,
    )
    estimates = funding_estimates(result, Path(funding) / (symbol + ".parquet"))
    return arm, symbol, result, estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("cache_ew_1h"))
    parser.add_argument("--funding", type=Path, default=Path("funding_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new")
    protocol = (
        Path(__file__).resolve().parents[1] / "docs/apex/LONG_HORIZON_PROTOCOL.txt"
    )
    names = (
        "engine.py",
        "zone_orders.py",
        "monitored_zones.py",
        "risk.py",
        "models.py",
        "simulation.py",
        "replay.py",
        "replay_management.py",
        "monitored_comparison.py",
        "long_horizon_comparison.py",
    )
    hashes = lambda: {
        n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
        for n in names
    }
    report = {
        "purpose": "Long-horizon offline comparison; no live or portfolio-return claim",
        "entry_days": 730,
        "runoff_days": 180,
        "end": END,
        "symbols": SYMBOLS,
        "arms": ARMS,
        "code_sha256": hashes(),
        "protocol": protocol.read_text(),
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "rows": [],
        "errors": [],
    }
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    jobs = [
        (str(args.source.resolve()), str(args.funding.resolve()), arm, symbol)
        for arm in ARMS
        for symbol in SYMBOLS
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                arm, symbol, result, funding = future.result()
                name = f"{arm}_{symbol}.json.gz"
                compressed = gzip.compress(
                    json.dumps(result, sort_keys=True, allow_nan=False).encode(),
                    mtime=0,
                )
                with (args.output / name).open("xb") as stream:
                    stream.write(compressed)
                closed = [t for t in result["trades"] if t["status"] == "CLOSED"]
                holds = [(t["closed_at"] - t["opened_at"]) / 86400 for t in closed]
                row = {
                    "arm": arm,
                    "symbol": symbol,
                    "metrics": metrics(result),
                    "entry_end": result["entry_end"],
                    "entry_cutoff": result["entry_cutoff"],
                    "data": result["data"],
                    "source_sha256": result["source_sha256"],
                    "evidence_file": name,
                    "evidence_sha256": hashlib.sha256(compressed).hexdigest(),
                    "trades": result["trades"],
                    "performance": result["performance"],
                    "funding": funding,
                    "closed_hold_days_median": median(holds) if holds else None,
                    "closed_hold_days_max": max(holds) if holds else None,
                }
                report["rows"].append(row)
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "symbol": symbol,
                            "metrics": row["metrics"],
                            "seconds": result["performance"]["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                item = {"job": futures[future], "error": f"{type(exc).__name__}: {exc}"}
                report["errors"].append(item)
                print(json.dumps(item), flush=True)
    report["rows"].sort(key=lambda r: (r["arm"], r["symbol"]))
    report["code_unchanged"] = hashes() == report["code_sha256"]
    report["complete"] = (
        len(report["rows"]) == len(jobs)
        and not report["errors"]
        and report["code_unchanged"]
        and all(
            r["metrics"]["complete"]
            and not r["metrics"]["window_truncated"]
            and not r["metrics"]["data_issues"]
            for r in report["rows"]
        )
    )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Complete={report['complete']}: {args.output / 'report.json'}", flush=True)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
