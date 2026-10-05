"""Offline fixed-risk 5x/10x isolated-margin sensitivity; no exchange client.

Market observations are reusable, but every arm owns its chronological cash,
position sizing, reservations and risk halts. Historical mark prices and tiers
are unavailable: liquidation outputs are stress flags, never certified fills.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from .engine import DAY
from .models import Instrument, Opportunity
from .replay import HOUR, _AnalysisCache, _iso, _source
from .simulation import apply_funding, create_trade
from .spot_comparison import _exposures, _sha

INITIAL = 10000.0


def input_snapshot(manifest, tapes_dir):
    """Certify every worker uses the same preselected data and signal tapes."""
    result = {}
    for symbol in manifest["symbols"]:
        result[symbol] = {}
        for kind, path in (
            ("hourly", manifest["data_paths"][symbol]),
            ("funding", manifest["funding_paths"][symbol]),
            ("tape", Path(tapes_dir) / (symbol + ".json.gz")),
        ):
            digest = _sha(path)
            expected = manifest.get("source_sha256", {}).get(symbol, {}).get(kind)
            if expected is not None and digest != expected:
                raise ValueError(f"{symbol}: {kind} changed since universe selection")
            result[symbol][kind] = {"path": str(Path(path).resolve()), "sha256": digest}
    return result


def scenario_matches_inputs(result, snapshot):
    return all(
        result.get(field, {}).get(symbol, {}).get("sha256") == sources[kind]["sha256"]
        for symbol, sources in snapshot.items()
        for field, kind in (
            ("source", "hourly"),
            ("funding_source", "funding"),
            ("tape_source", "tape"),
        )
    )


def load_funding(path, start, end, verified_transitions=()):
    """Validate local observations without claiming exchange completeness.

    Only *past* observed intervals normalize the latest settled rate for the risk
    gate. An interval compatible with 1/2/4/8 hours cannot prove no missing rates
    when the exchange changes schedules; retain that limitation in the evidence.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    if {"ts", "rate"}.issubset(frame.columns):
        times = pd.to_datetime(frame.ts, utc=True).astype("int64") // 1000000
        rates = frame.rate
    else:
        times, rates = frame.ts_ms, frame.funding_rate
    values = sorted(zip(times, rates))
    if any(
        not math.isfinite(float(t))
        or int(t) != t
        or t % (HOUR * 1000)
        or not math.isfinite(float(r))
        for t, r in values
    ):
        raise ValueError("Invalid or unaligned Bybit funding records")
    if len({t for t, _ in values}) != len(values):
        raise ValueError("Duplicate funding timestamps")
    relevant = [
        (int(t), float(r))
        for t, r in values
        if (start - DAY) * 1000 <= t <= (end + DAY) * 1000
    ]
    if not relevant or relevant[0][0] > start * 1000 or relevant[-1][0] < end * 1000:
        raise ValueError("Funding does not cover replay boundaries")
    intervals = [(b[0] - a[0]) // (HOUR * 1000) for a, b in zip(relevant, relevant[1:])]
    exceptions = set(verified_transitions)
    if any(
        i not in (1, 2, 4, 8) and record[0] not in exceptions
        for record, i in zip(relevant[1:], intervals)
    ):
        raise ValueError("Unsupported interval or missing funding settlements")
    times, rates = map(list, zip(*relevant))
    normalized = [rates[0]] + [r * 8 / i for r, i in zip(rates[1:], intervals)]
    return (
        times,
        rates,
        normalized,
        {
            "path": str(Path(path).resolve()),
            "sha256": _sha(path),
            "interval_counts_hours": dict(Counter(intervals)),
            "records": len(relevant),
            "crosschecked_transition_timestamps": sorted(exceptions),
            "complete_exchange_schedule_verified": False,
            "basis": "remaining entry notional; inferred historical intervals",
        },
    )


def stress_observation(old, updated, bar, leverage, maintenance_rate):
    """Conservative range flag; a stop may have exited before the extreme.

    Use entry margin without favorable realized P&L, and include negative funding
    as an additional haircut even when the wallet could pay it. Last-price OHLC
    is not mark-price OHLC and this is explicitly not a liquidation simulator.
    """
    if updated.get("opened_at") is None:
        return None
    active = old if old.get("opened_at") is not None else updated
    entry, qty = active["entry"], active["qty"]
    remaining = active["remaining"] if active["status"] == "OPEN" else qty
    if remaining <= 0:
        return None
    sign = 1 if active["side"] == "Buy" else -1
    distance = entry * (1 / leverage - maintenance_rate - 0.002)
    distance += min(0, updated.get("funding", 0)) / remaining
    threshold = entry - sign * distance
    extreme = bar.low if sign > 0 else bar.high
    if sign * (extreme - threshold) > 0:
        return None
    return {
        "time": bar.open_time / 1000 + HOUR,
        "trade_id": active["id"],
        "symbol": active["symbol"],
        "threshold": threshold,
        "adverse_extreme": extreme,
        "gap_open_beyond_threshold": sign * (bar.open - threshold) <= 0,
        "note": "Range stress flag; unverified mark price / intrahour ordering",
    }


def run_scenario(manifest, tapes_dir, leverage, maintenance_rate, *, progress=False):
    from .leverage_research_accounting import (
        LeverageModel,
        account_snapshot,
        guard as liquidation_guard,
        pending_requirement,
    )
    from .leverage_research_tapes import load_tape

    started = perf_counter()
    symbols = sorted(manifest["symbols"])
    start, cutoff, end = (manifest[k] for k in ("start", "entry_cutoff", "end"))
    if not symbols or len(symbols) != len(set(symbols)) or not start < cutoff <= end:
        raise ValueError("Invalid universe or replay window")
    days = (end - start) / DAY
    if int(days) != days:
        raise ValueError("Whole UTC days required")
    caches, models, prices, tapes, funding, sources = {}, {}, {}, {}, {}, {}
    tape_sources = {}
    for symbol in symbols:
        path = Path(manifest["data_paths"][symbol])
        d, e, hourly, begin, finish, meta = _source(path, int(days), _iso(end))
        if meta["window_truncated"] or begin != start or finish != end:
            raise ValueError(f"{symbol}: incomplete source")
        caches[symbol] = _AnalysisCache(symbol, d, e, False)
        if len(caches[symbol].histories(start)[0]) < 120:
            raise ValueError(f"{symbol}: insufficient warmup")
        models[symbol] = LeverageModel(d, e)
        prices[symbol] = {b.open_time // 1000 + HOUR: b for b in hourly}
        if any(t not in prices[symbol] for t in range(start, end + 1, HOUR)):
            raise ValueError(f"{symbol}: missing quote")
        tape_path = Path(tapes_dir) / (symbol + ".json.gz")
        tape_sources[symbol] = {"path": str(tape_path), "sha256": _sha(tape_path)}
        tapes[symbol] = load_tape(tape_path, path, start, end, cutoff)
        funding[symbol] = load_funding(
            manifest["funding_paths"][symbol],
            start,
            end,
            manifest.get("funding_transition_timestamps", {}).get(symbol, ()),
        )
        sources[symbol] = dict(meta, path=str(path), sha256=_sha(path))

    trades, consumed, accepted = {}, set(), []
    rejections, rejected_ids, stress = Counter(), {}, []
    equity_curve, yearly = [], {}
    peak = previous_equity = day_base = week_base = INITIAL
    max_dd = max_margin = margin_sum = 0.0
    min_available, max_open, max_notional = INITIAL, 0, 0.0
    day_key = week_key = None

    def snapshot(quotes):
        return account_snapshot(list(trades.values()), quotes, leverage, INITIAL)

    for now in range(start, end + 1, HOUR):
        date = datetime.fromtimestamp(now, timezone.utc)
        quotes = {s: prices[s][now].close for s in symbols}
        boundary = snapshot(quotes)["equity"]
        if day_key != date.date():
            day_key, day_base = date.date(), boundary
        if week_key != date.isocalendar()[:2]:
            week_key, week_base = date.isocalendar()[:2], boundary
        for key, old in list(trades.items()):
            if old["status"] not in {"OPEN", "PENDING"}:
                continue
            symbol = old["symbol"]
            bar = prices[symbol][now]
            if old["status"] == "PENDING" and bar.open_time / 1000 >= cutoff:
                updated = dict(
                    old, status="EXPIRED", closed_at=cutoff, exit_reason="ENTRY_CUTOFF"
                )
            else:
                daily, execution = caches[symbol].histories(now)
                updated = models[symbol].advance(
                    old,
                    [bar],
                    now,
                    interval_ms=HOUR * 1000,
                    daily=daily,
                    execution=execution,
                )
            if updated.get("data_gap") or updated.get("data_error"):
                raise ValueError(f"Simulation failed: {updated}")
            if updated.get("opened_at") is not None:
                times, rates, _, _meta = funding[symbol]
                lo, hi = (bisect_right(times, t * 1000) for t in (now - HOUR, now))
                if lo < hi:
                    updated = apply_funding(
                        updated,
                        [
                            {
                                "fundingRateTimestamp": times[i],
                                "fundingRate": rates[i],
                                "symbol": symbol,
                            }
                            for i in range(lo, hi)
                        ],
                        now,
                    )
                    if updated.get("funding_error"):
                        raise ValueError(updated["funding_error"])
                flag = stress_observation(old, updated, bar, leverage, maintenance_rate)
                if flag:
                    stress.append(flag)
            trades[key] = updated
        account = snapshot(quotes)
        peak = max(peak, account["equity"])
        dd = max(0, 100 * (1 - account["equity"] / peak))
        max_dd = max(max_dd, dd)
        if now < cutoff:
            candidates = [
                Opportunity(**op)
                for s in symbols
                for op in tapes[s]["ready_by_time"].get(str(now), [])
                if op["id"] not in consumed and now < op["expires_at"]
            ]
            for op in sorted(candidates, key=lambda o: (o.symbol, o.id)):
                account = snapshot(quotes)
                times, _, normalized, _meta = funding[op.symbol]
                rate = normalized[bisect_right(times, now * 1000) - 1]
                model = models[op.symbol]
                sizing = model.assess(
                    op,
                    Instrument(
                        op.symbol,
                        0.0001,
                        0.0001,
                        1000000,
                        0.0001,
                        5,
                        max_leverage=leverage,
                    ),
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
                    guard = liquidation_guard(
                        sizing["entry"],
                        sizing["stop"],
                        op.side,
                        leverage,
                        maintenance_rate,
                    )
                    if not guard["allowed"]:
                        sizing = dict(sizing, allowed=False, reasons=guard["reasons"])
                if sizing["allowed"]:
                    required = pending_requirement(
                        sizing["qty"],
                        sizing["entry"],
                        leverage,
                        model.fee_rate,
                        upper_fill_price=(
                            op.evidence["zone_high"] if op.side == "Sell" else None
                        ),
                    )
                    if required > account["available_cash"] + 1e-9:
                        sizing = dict(
                            sizing,
                            allowed=False,
                            reasons=["INSUFFICIENT_UNRESERVED_MARGIN"],
                        )
                if not sizing["allowed"]:
                    for reason in set(sizing["reasons"]):
                        rejections[reason] += 1
                        rejected_ids.setdefault(reason, set()).add(op.id)
                    continue
                trade = create_trade(op, sizing, "monitored_shadow", now)
                trade.update(
                    interval_ms=HOUR * 1000,
                    entry_eligible_at=now,
                    research_market="perpetual",
                    research_leverage=leverage,
                    research_entry_fee_rate=model.fee_rate,
                )
                trades[trade["id"]] = trade
                consumed.add(op.id)
                accepted.append(
                    {
                        "time": now,
                        "opportunity": op.to_dict(),
                        "sizing": sizing,
                        "margin_guard": guard,
                        "account_before": account,
                    }
                )
                if snapshot(quotes)["available_cash"] < -1e-7:
                    raise AssertionError("Margin over-reserved")
        account = snapshot(quotes)
        margin = (
            account["reserved_pending"]
            + account["open_margin"]
            + account["reserved_close_fees"]
        )
        max_margin = max(max_margin, margin)
        margin_sum += margin
        min_available = min(min_available, account["available_cash"])
        max_open = max(max_open, account["open_count"] + account["pending_count"])
        max_notional = max(
            max_notional,
            sum(x["notional"] for x in _exposures(trades.values(), quotes)),
        )
        year = yearly.setdefault(str(date.year), {"start_equity": previous_equity})
        year["end_equity"] = account["equity"]
        previous_equity = account["equity"]
        if now % DAY == 0 or now in (start, end):
            equity_curve.append(
                dict(time=now, utc=_iso(now), drawdown_pct=dd, **account)
            )
        if progress and (now - start) % (90 * DAY) == 0:
            print(
                json.dumps(
                    {
                        "leverage": leverage,
                        "mmr": maintenance_rate,
                        "through": _iso(now),
                        "accepted": len(accepted),
                        "elapsed_s": round(perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )

    all_trades = list(trades.values())
    filled = [t for t in all_trades if t.get("opened_at") is not None]
    closed = [t for t in filled if t["status"] == "CLOSED"]
    opened = [t for t in filled if t["status"] == "OPEN"]
    holds = [(t["closed_at"] - t["opened_at"]) / DAY for t in closed]

    def stats(selected):
        return {
            "closed": len(selected),
            "wins": sum(t["net_pnl"] > 1e-9 for t in selected),
            "losses": sum(t["net_pnl"] < -1e-9 for t in selected),
            "net": sum(t["net_pnl"] for t in selected),
        }

    result = dict(
        leverage=leverage,
        maintenance_rate_assumption=maintenance_rate,
        complete=True,
        initial_equity=INITIAL,
        start=_iso(start),
        entry_cutoff=_iso(cutoff),
        end=_iso(end),
        symbols=symbols,
        profile="cautious",
        sides=["Buy", "Sell"],
        accepted=len(accepted),
        filled=len(filled),
        open=len(opened),
        expired=sum(t["status"] == "EXPIRED" for t in all_trades),
        **stats(closed),
        total_fees=sum(t["fees"] for t in filled),
        funding_estimate=sum(t["funding"] for t in filled),
        net_equity_change=account["equity"] - INITIAL,
        net_return_pct=100 * (account["equity"] / INITIAL - 1),
        final_account=account,
        max_drawdown_pct=max_dd,
        max_reserved_margin_usdt=max_margin,
        average_reserved_margin_usdt=margin_sum / ((end - start) // HOUR + 1),
        minimum_available_cash=min_available,
        max_gross_notional_usdt=max_notional,
        maximum_positions_and_reservations=max_open,
        holding_days_median=median(holds) if holds else None,
        holding_days_max=max(holds) if holds else None,
        risk_rejections=dict(rejections),
        unique_risk_rejections={k: len(v) for k, v in rejected_ids.items()},
        liquidation_stress_flagged_trades=len({s["trade_id"] for s in stress}),
        liquidation_stress_events=stress,
        actual_liquidations="Not observable: no historical mark prices / risk tiers",
        per_side={
            s: stats([t for t in closed if t["side"] == s]) for s in ("Buy", "Sell")
        },
        per_symbol={s: stats([t for t in closed if t["symbol"] == s]) for s in symbols},
        calendar_equity=yearly,
        trades=all_trades,
        daily_equity=equity_curve,
        accepted_evidence=accepted,
        source=sources,
        funding_source={s: f[3] for s, f in funding.items()},
        tape_source=tape_sources,
        tape_source_unchanged=all(
            _sha(v["path"]) == v["sha256"] for v in tape_sources.values()
        ),
        source_unchanged=all(_sha(v["path"]) == v["sha256"] for v in sources.values()),
        funding_source_unchanged=all(
            _sha(v[3]["path"]) == v[3]["sha256"] for v in funding.values()
        ),
        elapsed_seconds=perf_counter() - started,
    )
    result["win_rate_pct"] = 100 * result["wins"] / len(closed) if closed else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tapes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    manifest = json.loads(args.manifest.read_text())
    inputs = input_snapshot(manifest, args.tapes)
    selection_audit = args.manifest.with_name("selection_audit.json")
    if _sha(selection_audit) != manifest["selection_audit_sha256"]:
        raise ValueError("Universe-selection audit changed")
    protocol = Path(__file__).parents[1] / "docs/apex/LEVERAGE_COMPARISON_PROTOCOL.txt"
    # Freeze the actual offline dependencies. Independent cloud/UI fixes do
    # not alter these observations or accounting and must not invalidate them.
    names = (
        "leverage_comparison.py",
        "leverage_research_accounting.py",
        "leverage_research_tapes.py",
        "leverage_research_data.py",
        "spot_comparison.py",
        "spot_research_accounting.py",
        "monitored_comparison.py",
        "replay.py",
        "replay_management.py",
        "models.py",
        "engine.py",
        "risk.py",
        "simulation.py",
        "zone_orders.py",
        "monitored_zones.py",
    )
    code = [Path(__file__).with_name(n) for n in names]
    hashes = {p.name: _sha(p) for p in code}
    report = {
        "protocol": protocol.read_text(),
        "protocol_sha256": _sha(protocol),
        "code_sha256": hashes,
        "manifest_sha256": _sha(args.manifest),
        "input_snapshot": inputs,
        "selection_audit_sha256": _sha(selection_audit),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "rows": [],
        "errors": [],
    }
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "protocol.txt").write_text(report["protocol"])
    (args.output / "selection_audit.json").write_bytes(selection_audit.read_bytes())
    (args.output / "code").mkdir()
    for p in code:
        (args.output / "code" / p.name).write_bytes(p.read_bytes())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_scenario, manifest, args.tapes, lev, mmr, progress=True): (
                lev,
                mmr,
            )
            for mmr in (0.005, 0.01, 0.025)
            for lev in (5, 10)
        }
        for future in as_completed(futures):
            lev, mmr = futures[future]
            try:
                result = future.result()
                result["matches_parent_inputs"] = scenario_matches_inputs(
                    result, inputs
                )
                name = f"{lev}x_mmr{mmr:g}"
                data = gzip.compress(
                    json.dumps(result, allow_nan=False).encode(), mtime=0
                )
                path = args.output / (name + ".json.gz")
                path.write_bytes(data)
                for kind in ("trades", "daily_equity"):
                    rows = result[kind]
                    if rows:
                        fields = sorted(set().union(*(r.keys() for r in rows)))
                        with (args.output / (name + "_" + kind + ".csv")).open(
                            "x"
                        ) as stream:
                            writer = csv.DictWriter(stream, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(rows)
                row = {
                    k: v
                    for k, v in result.items()
                    if k
                    not in {
                        "trades",
                        "daily_equity",
                        "accepted_evidence",
                        "source",
                        "funding_source",
                        "tape_source",
                        "liquidation_stress_events",
                    }
                }
                row.update(evidence_file=path.name, evidence_sha256=_sha(path))
                report["rows"].append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "leverage",
                                "maintenance_rate_assumption",
                                "net_return_pct",
                                "closed",
                            )
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                report["errors"].append(
                    {
                        "leverage": lev,
                        "mmr": mmr,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(json.dumps(report["errors"][-1]), flush=True)
    report["code_unchanged"] = hashes == {p.name: _sha(p) for p in code}
    report["protocol_unchanged"] = _sha(protocol) == report["protocol_sha256"]
    report["manifest_unchanged"] = _sha(args.manifest) == report["manifest_sha256"]
    report["all_inputs_unchanged"] = all(
        _sha(item["path"]) == item["sha256"]
        for sources in inputs.values()
        for item in sources.values()
    )
    report["selection_audit_unchanged"] = (
        _sha(selection_audit) == report["selection_audit_sha256"]
    )
    report["complete"] = (
        len(report["rows"]) == 6
        and not report["errors"]
        and all(
            report[k]
            for k in (
                "code_unchanged",
                "protocol_unchanged",
                "manifest_unchanged",
                "all_inputs_unchanged",
                "selection_audit_unchanged",
            )
        )
        and all(
            r["source_unchanged"]
            and r["funding_source_unchanged"]
            and r["tape_source_unchanged"]
            and r["matches_parent_inputs"]
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
