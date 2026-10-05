"""Run the fixed offline protocol in docs/apex/MONITORED_ZONE_PROTOCOL.txt.

No credentials, network, source writes, parameter optimization or exchange
submission. Outputs require a new directory. Each result retains full evidence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from .replay import replay

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT")
WINDOWS = {
    "primary": "2026-05-25T00:00:00Z",
    "later": "2026-09-18T00:00:00Z",
}
ARMS = {
    "confirmed": ("confirmed", 30),
    "resting_48h": ("resting_limit", 30),
    "monitored_48h": ("monitored_zone", 2),
    "monitored_30d": ("monitored_zone", 30),
}


def _run(task):
    source, window, arm, symbol = task
    style, age = ARMS[arm]
    result = replay(
        Path(source) / (symbol + ".parquet"),
        90,
        end=WINDOWS[window],
        entry_style=style,
        monitor_days=age,
        max_seconds=300,
    )
    return window, arm, symbol, result


def metrics(result):
    valid = [
        t
        for t in result["trades"]
        if t["status"] == "CLOSED" and not t.get("data_gap") and not t.get("data_error")
    ]
    wins = sum(t["net_pnl_before_funding"] > 1e-9 for t in valid)
    losses = sum(t["net_pnl_before_funding"] < -1e-9 for t in valid)
    return dict(
        complete=result["complete"],
        window_truncated=result["data"]["window_truncated"],
        ready=result["funnel"]["unique_ready"],
        accepted=result["funnel"]["unique_accepted"],
        filled=result["funnel"]["unique_filled"],
        filled_by_side=result["funnel"]["filled_by_side"],
        open=result["summary"]["open"],
        pending=result["summary"]["pending"],
        expired=result["summary"]["expired"],
        closed=len(valid),
        wins=wins,
        losses=losses,
        breakeven=len(valid) - wins - losses,
        wr_before_funding=100 * wins / len(valid) if valid else None,
        closed_net_before_funding=sum(t["net_pnl_before_funding"] for t in valid),
        closed_sum_r_before_funding=sum(
            t["net_pnl_before_funding"] / t["risk_cash"] for t in valid
        ),
        data_issues=result["data_issues"],
        unique_rejection_reasons=result["unique_plans_by_risk_rejection"],
        plan_reasons=result["unique_plans_by_reason"],
        warmup_scans=result["data"]["warmup_scans"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("cache_ew_1h"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must not already exist")
    protocol = (
        Path(__file__).resolve().parents[1] / "docs/apex/MONITORED_ZONE_PROTOCOL.txt"
    )
    args.output.mkdir(parents=True)
    jobs = [
        (str(args.source.resolve()), w, a, s)
        for w in WINDOWS
        for a in ARMS
        for s in SYMBOLS
    ]
    report = dict(
        purpose="Fixed historical monitored-zone comparison, no live orders or profitable-edge claim",
        experiment_version="monitored-zone-v1",
        code_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "engine.py",
                "zone_orders.py",
                "monitored_zones.py",
                "risk.py",
                "simulation.py",
                "replay.py",
                "monitored_comparison.py",
            )
        },
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        protocol=protocol.read_text(),
        started_at=datetime.now(timezone.utc).isoformat(),
        windows=WINDOWS,
        arms=ARMS,
        rows=[],
        errors=[],
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                window, arm, symbol, result = future.result()
                filename = f"{window}_{arm}_{symbol}.json.gz"
                payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
                compressed = gzip.compress(payload, mtime=0)
                with (args.output / filename).open("xb") as stream:
                    stream.write(compressed)
                row = dict(
                    window=window,
                    arm=arm,
                    symbol=symbol,
                    source_sha256=result["source_sha256"],
                    data=result["data"],
                    metrics=metrics(result),
                    evidence_file=filename,
                    evidence_sha256=hashlib.sha256(compressed).hexdigest(),
                    trades=result["trades"],
                    performance=result["performance"],
                )
                report["rows"].append(row)
                print(
                    json.dumps(
                        dict(window=window, arm=arm, symbol=symbol, **row["metrics"])
                    ),
                    flush=True,
                )
            except Exception as exc:
                report["errors"].append(
                    dict(job=job, error=f"{type(exc).__name__}: {exc}")
                )
                print(json.dumps(report["errors"][-1]), flush=True)
    report["rows"].sort(key=lambda r: (r["window"], r["arm"], r["symbol"]))
    report["complete"] = (
        len(report["rows"]) == len(jobs)
        and not report["errors"]
        and all(
            r["metrics"]["complete"] and not r["metrics"]["window_truncated"]
            for r in report["rows"]
        )
    )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Complete={report['complete']}; {args.output/'report.json'}", flush=True)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
