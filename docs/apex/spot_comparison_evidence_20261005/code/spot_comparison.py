"""Offline, cash-constrained spot/perpetual comparison. Never imports runtime.

Run only against validated hourly archives. No network, account credentials,
exchange orders, historical-data edits, or strategy parameter optimization.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from .engine import DAY
from .models import Instrument
from .monitored_comparison import SYMBOLS
from .monitored_zones import analyze_monitored_zones, observe_bar, observe_quote
from .replay import HOUR, _AnalysisCache, _iso, _source
from .simulation import apply_funding, create_trade
from .spot_research_accounting import ResearchModel, account_snapshot

END = "2026-05-25T00:00:00Z"
HISTORY_START = "2022-01-01T00:00:00Z"
INITIAL = 10000.0


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _funding(path, start, end):
    import pandas as pd

    frame = pd.read_parquet(path).sort_values("ts_ms")
    times = [int(t) for t in frame.ts_ms]
    rates = [float(r) for r in frame.funding_rate]
    if (
        len(set(times)) != len(times)
        or any(t != actual for t, actual in zip(times, frame.ts_ms))
        or any(not math.isfinite(r) for r in rates)
    ):
        raise ValueError("Invalid funding cache")
    expected = set(range(int(start // 28800) * 28800000, int(end * 1000) + 1, 28800000))
    scheduled = [t // 28800000 * 28800000 for t in times]
    if len(set(scheduled)) != len(scheduled) or any(
        t - s > 60000 for t, s in zip(times, scheduled)
    ):
        raise ValueError("Duplicate or unexpectedly delayed funding settlement")
    observed = {t for t in scheduled if start // 28800 * 28800000 <= t <= end * 1000}
    if observed != expected:
        raise ValueError(
            "Funding cache does not cover the expected eight-hour schedule"
        )
    if not times or times[0] > start * 1000:
        raise ValueError("No already-settled funding rate at replay start")
    return (
        times,
        rates,
        {
            "path": str(path),
            "sha256": _sha(path),
            "expected_settlements": len(expected),
            "observed_settlements": len(observed),
            "amount_basis": "remaining entry notional, not historical settlement mark",
        },
    )


def _observations(cache, terminals, bar, now):
    """Identical hourly watch cancellation/conflict ordering to replay.py."""
    observed = []
    for watch in cache.get(now):
        if watch.id in terminals:
            old = terminals[watch.id]
            watch = replace(old, evidence=dict(old.evidence, as_of=now))
        else:
            watch = observe_bar(watch, bar, now, interval_seconds=HOUR)
            if watch.state == "INVALID":
                terminals[watch.id] = watch
        observed.append(watch)
    if any(w.reason == "CONFLICTING_DIRECTIONS" for w in observed):
        sides = {
            w.side
            for w in observed
            if w.state != "INVALID"
            and w.reason in {"AWAIT_ZONE", "CONFLICTING_DIRECTIONS"}
        }
        if len(sides) <= 1:
            observed = [
                (
                    replace(w, reason="AWAIT_ZONE")
                    if w.state != "INVALID" and w.reason == "CONFLICTING_DIRECTIONS"
                    else w
                )
                for w in observed
            ]
    return [
        (
            observe_quote(
                w, bar.close * (1 + (1 if w.side == "Buy" else -1) * 0.00005), now, now
            )
            if w.state != "INVALID"
            else w
        )
        for w in observed
    ]


def _exposures(trades, quotes):
    result = []
    for trade in trades:
        if trade["status"] not in {"OPEN", "PENDING"}:
            continue
        remaining = trade["remaining"]
        mark = quotes[trade["symbol"]] if trade["status"] == "OPEN" else trade["limit"]
        result.append(
            {
                "symbol": trade["symbol"],
                "bucket": trade["bucket"],
                # Retain original risk per remaining lot: no optimistic release
                # from an uncertain intrahour trailing stop or correlated hedge.
                "risk_cash": trade["risk_cash"] * remaining / trade["qty"],
                "notional": mark * remaining,
            }
        )
    return result


def portfolio_replay(
    source,
    market,
    *,
    funding_source=None,
    symbols=SYMBOLS,
    days=730,
    runoff_days=180,
    end=END,
    history_start=HISTORY_START,
    progress=False,
):
    if market not in {"spot", "perpetual"}:
        raise ValueError("Unknown market")
    if (
        type(days) is not int
        or days < 1
        or type(runoff_days) is not int
        or runoff_days < 0
    ):
        raise ValueError("Invalid entry/runoff days")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Empty/duplicate universe")
    started = perf_counter()
    histories, caches, models, prices, metadata, funding = {}, {}, {}, {}, {}, {}
    terminals = {s: {} for s in symbols}
    start = finish = None
    expected_origin = datetime.fromisoformat(
        history_start.replace("Z", "+00:00")
    ).timestamp()
    for symbol in sorted(symbols):
        path = Path(source) / (symbol + ".parquet")
        daily, execution, hourly, begin, stop, data = _source(
            path, days + runoff_days, end
        )
        if data["window_truncated"] or begin != (start if start is not None else begin):
            raise ValueError("Incomplete or unaligned comparison window")
        origin = datetime.fromisoformat(data["segment_start"]).timestamp()
        if origin != expected_origin or not daily or not execution:
            raise ValueError(f"{symbol}: history gap or wrong fixed history origin")
        start, finish = begin, stop
        cache = _AnalysisCache(
            symbol,
            daily,
            execution,
            True,
            analyze_monitored_zones,
            {"lifetime_seconds": 30 * DAY},
        )
        caches[symbol] = cache
        histories[symbol] = (daily, execution)
        models[symbol] = ResearchModel(market, daily, execution)
        prices[symbol] = {b.open_time // 1000 + HOUR: b for b in hourly}
        expected_hours = set(range(int(start), int(finish) + 1, HOUR))
        if not expected_hours.issubset(prices[symbol]):
            raise ValueError(f"{symbol}: missing comparison quotes")
        d, e = cache.histories(start)
        metadata[symbol] = dict(
            data,
            path=str(path),
            sha256=_sha(path),
            initial_daily_bars=len(d),
            initial_execution_bars=len(e),
        )
        if market == "perpetual":
            if funding_source is None:
                raise ValueError("Perpetual comparison requires funding history")
            funding[symbol] = _funding(
                Path(funding_source) / (symbol + ".parquet"), start, finish
            )
    cutoff = finish - runoff_days * DAY
    trades, consumed, accepted, ready = {}, set(), [], set()
    rejection, rejected_ids, plan_reasons = Counter(), {}, Counter()
    daily_curve, yearly, events = [], {}, []
    peak = INITIAL
    max_dd = max_utilization = 0.0
    min_cash = INITIAL
    max_open = 0
    cash_utilization_sum = 0.0
    day_key = week_key = None
    day_base = week_base = INITIAL
    previous_equity = INITIAL
    short_ready = set()
    last_progress = -1
    passive_qty = {}
    passive_peak = INITIAL
    passive_max_dd = 0.0
    if market == "spot":
        fee = models[sorted(symbols)[0]].fee_rate
        for symbol in symbols:
            entry = prices[symbol][int(start) + HOUR].open * 1.00005 * 1.0003
            passive_qty[symbol] = INITIAL / len(symbols) / entry * (1 - fee)

    def snapshot(quotes):
        return account_snapshot(list(trades.values()), quotes, INITIAL, market)

    for now in range(int(start), int(finish) + 1, HOUR):
        prior_hour_equity = previous_equity
        date = datetime.fromtimestamp(now, timezone.utc)
        today, week = date.date().isoformat(), date.isocalendar()[:2]
        quotes = {s: prices[s][now].close for s in symbols}
        boundary_equity = snapshot(quotes)["equity"]
        if today != day_key:
            day_key, day_base = today, boundary_equity
        if week != week_key:
            week_key, week_base = week, boundary_equity
        # A separately labeled fully invested comparator, not equal-risk alpha.
        # First entry is the next hourly open, unknown at the decision boundary.
        if passive_qty and now > start:
            passive_equity = sum(passive_qty[s] * quotes[s] for s in symbols)
            passive_peak = max(passive_peak, passive_equity)
            passive_max_dd = max(
                passive_max_dd, 100 * (1 - passive_equity / passive_peak)
            )
        for key, old in list(trades.items()):
            if old["status"] not in {"PENDING", "OPEN"}:
                continue
            symbol = old["symbol"]
            bar = prices[symbol][now]
            if old["status"] == "PENDING" and bar.open_time / 1000 >= cutoff:
                updated = dict(
                    old, status="EXPIRED", closed_at=cutoff, exit_reason="ENTRY_CUTOFF"
                )
            else:
                d, e = caches[symbol].histories(now)
                updated = models[symbol].advance(
                    old, [bar], now, interval_ms=HOUR * 1000, daily=d, execution=e
                )
            if updated.get("data_gap") or updated.get("data_error"):
                raise ValueError(f"{symbol}: simulation data issue {updated}")
            if market == "perpetual" and updated.get("opened_at") is not None:
                times, rates, _ = funding[symbol]
                first = bisect_right(times, (now - HOUR) * 1000)
                last = bisect_right(times, now * 1000)
                # A rate enters accounting only when its settlement is known.
                if first < last:
                    updated = apply_funding(
                        updated,
                        [
                            {
                                "fundingRateTimestamp": times[index],
                                "fundingRate": rates[index],
                                "symbol": symbol,
                            }
                            for index in range(first, last)
                        ],
                        now,
                    )
                    if updated.get("funding_error"):
                        raise ValueError(updated["funding_error"])
            trades[key] = updated
            if updated["status"] != old["status"]:
                events.append(
                    {
                        "time": now,
                        "id": key,
                        "status": updated["status"],
                        "reason": updated.get("exit_reason"),
                    }
                )
        account = snapshot(quotes)
        peak = max(peak, account["equity"])
        dd = max(0, 100 * (1 - account["equity"] / peak))
        max_dd = max(max_dd, dd)
        if now < cutoff:
            candidates = []
            for symbol in sorted(symbols):
                ops = _observations(
                    caches[symbol], terminals[symbol], prices[symbol][now], now
                )
                for op in ops:
                    plan_reasons[op.reason] += 1
                    if op.state != "READY":
                        continue
                    if op.side != "Buy":
                        short_ready.add(op.id)
                        continue
                    ready.add(op.id)
                    if op.id not in consumed and now < op.expires_at:
                        candidates.append(op)
            for op in sorted(candidates, key=lambda o: (o.symbol, o.id)):
                model = models[op.symbol]
                account = snapshot(quotes)
                rate = 0.0
                if market == "perpetual":
                    times, rates, _ = funding[op.symbol]
                    rate = rates[bisect_right(times, now * 1000) - 1]
                sizing = model.assess(
                    op,
                    Instrument(op.symbol, 0.0001, 0.0001, 1000000, 0.0001, 5),
                    account["equity"],
                    exposures=_exposures(trades.values(), quotes),
                    daily_loss_pct=max(0, 100 * (1 - account["equity"] / day_base)),
                    weekly_loss_pct=max(0, 100 * (1 - account["equity"] / week_base)),
                    drawdown_pct=dd,
                    funding_rate_8h=rate,
                    spread_pct=0.01,
                    now=now,
                    allow_monitored=True,
                )
                if sizing["allowed"]:
                    required = (
                        sizing["qty"] * sizing["entry"] * (1 + model.entry_fee_rate)
                    )
                    if required > account["available_cash"] + 1e-9:
                        sizing = dict(
                            sizing,
                            allowed=False,
                            reasons=["INSUFFICIENT_UNRESERVED_CASH"],
                        )
                if not sizing["allowed"]:
                    for reason in set(sizing["reasons"]):
                        rejection[reason] += 1
                        rejected_ids.setdefault(reason, set()).add(op.id)
                    continue
                trade = create_trade(op, sizing, "monitored_shadow", now)
                trade.update(
                    interval_ms=HOUR * 1000,
                    entry_eligible_at=now,
                    research_market=market,
                    research_entry_fee_rate=model.entry_fee_rate,
                )
                trades[trade["id"]] = trade
                consumed.add(op.id)
                accepted.append(
                    {
                        "opportunity": op.to_dict(),
                        "sizing": sizing,
                        "time": now,
                        "account_before": account,
                    }
                )
                if snapshot(quotes)["available_cash"] < -1e-7:
                    raise AssertionError("Cash over-reserved")
        account = snapshot(quotes)
        utilization = (
            (account["equity"] - account["available_cash"]) / account["equity"]
            if account["equity"] > 0
            else 1
        )
        cash_utilization_sum += utilization
        max_utilization = max(max_utilization, utilization)
        min_cash = min(min_cash, account["available_cash"])
        max_open = max(max_open, account["open_count"] + account["pending_count"])
        previous_equity = account["equity"]
        year = yearly.setdefault(
            str(date.year),
            {"start_equity": prior_hour_equity, "end_equity": account["equity"]},
        )
        year["end_equity"] = account["equity"]
        if now % DAY == 0 or now in {int(start), int(finish)}:
            daily_curve.append(
                dict(time=now, utc=_iso(now), drawdown_pct=dd, **account)
            )
        progress_day = (now - start) // DAY
        if progress and progress_day % 30 == 0 and progress_day != last_progress:
            last_progress = progress_day
            print(
                json.dumps(
                    {
                        "market": market,
                        "through": _iso(now),
                        "accepted": len(accepted),
                        "seconds": round(perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )

    all_trades = list(trades.values())
    filled = [t for t in all_trades if t.get("opened_at") is not None]
    closed = [t for t in filled if t["status"] == "CLOSED"]
    opened = [t for t in filled if t["status"] == "OPEN"]
    final = snapshot(quotes)
    wins = sum(t["net_pnl"] > 1e-9 for t in closed)
    losses = sum(t["net_pnl"] < -1e-9 for t in closed)
    holds = [(t["closed_at"] - t["opened_at"]) / DAY for t in closed]
    fee = models[sorted(symbols)[0]].fee_rate
    liquidation_cost = sum(
        t["remaining"] * quotes[t["symbol"]] * (0.0003 + fee * (1 - 0.0003))
        for t in opened
    )
    per_symbol = {}
    for symbol in sorted(symbols):
        selected = [t for t in closed if t["symbol"] == symbol]
        per_symbol[symbol] = {
            "closed": len(selected),
            "wins": sum(t["net_pnl"] > 1e-9 for t in selected),
            "net": sum(t["net_pnl"] for t in selected),
            "open": sum(t["symbol"] == symbol for t in opened),
        }
    return {
        "market": market,
        "complete": True,
        "initial_equity": INITIAL,
        "start": _iso(start),
        "entry_cutoff": _iso(cutoff),
        "end": _iso(finish),
        "replayed_through": _iso(now),
        "source": metadata,
        "funding_source": {s: v[2] for s, v in funding.items()},
        "fee_rate": fee,
        "profile": "cautious",
        "symbols": sorted(symbols),
        "ready_longs": len(ready),
        "excluded_ready_shorts": len(short_ready),
        "accepted": len(accepted),
        "filled": len(filled),
        "closed": len(closed),
        "open": len(opened),
        "expired": sum(t["status"] == "EXPIRED" for t in all_trades),
        "wins": wins,
        "losses": losses,
        "breakeven": len(closed) - wins - losses,
        "win_rate_pct": 100 * wins / len(closed) if closed else None,
        "closed_net": sum(t["net_pnl"] for t in closed),
        "total_fees": sum(t["fees"] for t in filled),
        "funding_estimate": sum(t["funding"] for t in filled),
        "net_equity_change": final["equity"] - INITIAL,
        "net_return_pct": 100 * (final["equity"] / INITIAL - 1),
        "final_account": final,
        "estimated_liquidation_equity": final["equity"] - liquidation_cost,
        "max_drawdown_pct": max_dd,
        "minimum_available_cash": min_cash,
        "maximum_positions_and_reservations": max_open,
        "average_cash_utilization_pct": 100
        * cash_utilization_sum
        / (int((finish - start) / HOUR) + 1),
        "max_cash_utilization_pct": 100 * max_utilization,
        "holding_days_median": median(holds) if holds else None,
        "holding_days_max": max(holds) if holds else None,
        "per_symbol": per_symbol,
        "calendar_equity": yearly,
        "risk_rejections": dict(rejection),
        "unique_risk_rejections": {k: len(v) for k, v in rejected_ids.items()},
        "plan_reason_observations": dict(plan_reasons),
        "trades": all_trades,
        "accepted_evidence": accepted,
        "daily_equity": daily_curve,
        "events": events,
        "source_unchanged": all(
            _sha(v["path"]) == v["sha256"] for v in metadata.values()
        ),
        "funding_source_unchanged": all(
            _sha(v[2]["path"]) == v[2]["sha256"] for v in funding.values()
        ),
        "passive_equal_weight_spot": (
            {
                "marked_final_equity": sum(passive_qty[s] * quotes[s] for s in symbols),
                "liquidation_final_equity": sum(
                    passive_qty[s] * quotes[s] for s in symbols
                )
                * (1 - 0.0003)
                * (1 - fee),
                "max_marked_drawdown_pct": passive_max_dd,
                "note": "Fully invested without strategy stops; descriptive, not exposure/risk matched",
            }
            if passive_qty
            else None
        ),
        "elapsed_seconds": perf_counter() - started,
    }


def _run_job(job):
    return portfolio_replay(**job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spot-source", type=Path, required=True)
    parser.add_argument("--perpetual-source", type=Path, required=True)
    parser.add_argument("--funding-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-start", default=HISTORY_START)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    root = Path(__file__).resolve().parents[1]
    protocol = root / "docs/apex/SPOT_COMPARISON_PROTOCOL.txt"
    names = (
        "engine.py",
        "risk.py",
        "models.py",
        "simulation.py",
        "zone_orders.py",
        "monitored_zones.py",
        "replay.py",
        "replay_management.py",
        "spot_research_accounting.py",
        "spot_comparison.py",
        "spot_research_data.py",
        "monitored_comparison.py",
        "spot_research_normalize.py",
    )
    hashes = lambda: {n: _sha(Path(__file__).with_name(n)) for n in names}
    report = {
        "purpose": "Offline shared-account long-only spot/perpetual comparison",
        "protocol": protocol.read_text(),
        "protocol_sha256": _sha(protocol),
        "code_sha256": hashes(),
        "input_paths": {
            "spot": str(args.spot_source.resolve()),
            "perpetual": str(args.perpetual_source.resolve()),
            "funding": str(args.funding_source.resolve()),
            "history_start": args.history_start,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "rows": [],
        "errors": [],
    }
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    data_manifest = args.spot_source.parent / "manifest.json"
    if data_manifest.is_file():
        (args.output / "dataset_manifest.json").write_bytes(data_manifest.read_bytes())
        report["dataset_manifest_sha256"] = _sha(data_manifest)
    jobs = [
        dict(
            source=str(args.spot_source),
            market="spot",
            progress=True,
            history_start=args.history_start,
        ),
        dict(
            source=str(args.perpetual_source),
            market="perpetual",
            funding_source=str(args.funding_source),
            progress=True,
            history_start=args.history_start,
        ),
    ]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run_job, job): job["market"] for job in jobs}
        for future in as_completed(futures):
            market = futures[future]
            try:
                result = future.result()
                payload = gzip.compress(
                    json.dumps(result, sort_keys=True, allow_nan=False).encode(),
                    mtime=0,
                )
                name = market + ".json.gz"
                (args.output / name).write_bytes(payload)
                trade_fields = (
                    "id",
                    "symbol",
                    "side",
                    "setup",
                    "status",
                    "decision_at",
                    "opened_at",
                    "closed_at",
                    "entry",
                    "qty",
                    "remaining",
                    "original_stop",
                    "target1",
                    "target2",
                    "exit_reason",
                    "gross_pnl",
                    "fees",
                    "funding",
                    "net_pnl",
                    "risk_cash",
                )
                with (args.output / (market + "_trades.csv")).open(
                    "x", newline=""
                ) as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=trade_fields, extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(result["trades"])
                with (args.output / (market + "_daily_equity.csv")).open(
                    "x", newline=""
                ) as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=list(result["daily_equity"][0])
                    )
                    writer.writeheader()
                    writer.writerows(result["daily_equity"])
                row = {
                    k: v
                    for k, v in result.items()
                    if k
                    not in {"trades", "accepted_evidence", "daily_equity", "events"}
                }
                row.update(
                    evidence_file=name,
                    evidence_sha256=hashlib.sha256(payload).hexdigest(),
                )
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
            except Exception as exc:
                error = {"market": market, "error": f"{type(exc).__name__}: {exc}"}
                report["errors"].append(error)
                print(json.dumps(error), flush=True)
    report["code_unchanged"] = hashes() == report["code_sha256"]
    report["protocol_unchanged"] = _sha(protocol) == report["protocol_sha256"]
    report["complete"] = (
        len(report["rows"]) == 2
        and not report["errors"]
        and report["code_unchanged"]
        and report["protocol_unchanged"]
        and all(
            r["complete"] and r["source_unchanged"] and r["funding_source_unchanged"]
            for r in report["rows"]
        )
    )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
