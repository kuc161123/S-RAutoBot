"""Independent saved-fill audit. No imports from apex_bot or its accounting.

Run with the research virtualenv:
  python /private/tmp/reconcile_leverage.py EVIDENCE_DIRECTORY
  python /private/tmp/reconcile_leverage.py --self-test

Only reads evidence/public caches; writes accounting_audit.json after success.
IOC fills are stamped at the next bar open but become visible to the hourly
replay at that bar's CLOSE. This distinction matters for pending reservations.
Actual exchange liquidation/mark-price completeness is expressly not audited.
"""

from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math

HOUR = 3600
DAY = 86400
FEE = 0.00055
PRICE_CACHE = {}
FUNDING_CACHE = {}
HASH_CACHE = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, label, *, tolerance=1e-7):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), label)
    require(math.isfinite(actual) and math.isfinite(expected), label)
    require(
        math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance),
        f"{label}: {actual!r} != independently calculated {expected!r}",
    )


def stamp(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def sha(path):
    path = Path(path)
    stat = path.stat()
    identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if identity not in HASH_CACHE:
        HASH_CACHE[identity] = hashlib.sha256(path.read_bytes()).hexdigest()
    return HASH_CACHE[identity]


def read_prices(meta):
    import pandas as pd

    require(sha(meta["path"]) == meta["sha256"], "Price source hash changed")
    key = (meta["path"], meta["sha256"])
    if key not in PRICE_CACHE:
        frame = pd.read_parquet(meta["path"])
        starts = pd.to_datetime(frame.start, utc=True).astype("int64") // 1000000000
        rows = [
            (int(t) + HOUR, tuple(float(x) for x in prices))
            for t, prices in zip(
                starts,
                frame[["open", "high", "low", "close"]].itertuples(
                    index=False, name=None
                ),
            )
        ]
        require(len({t for t, _ in rows}) == len(rows), "Duplicate price candles")
        PRICE_CACHE[key] = dict(rows)
    return PRICE_CACHE[key]


def read_funding(meta):
    import pandas as pd

    require(sha(meta["path"]) == meta["sha256"], "Funding source hash changed")
    key = (meta["path"], meta["sha256"])
    if key not in FUNDING_CACHE:
        frame = pd.read_parquet(meta["path"])
        if {"ts", "rate"}.issubset(frame.columns):
            times = pd.to_datetime(frame.ts, utc=True).astype("int64") // 1000000
            rates = frame.rate
        else:
            times, rates = frame.ts_ms, frame.funding_rate
        rows = sorted((int(t), float(r)) for t, r in zip(times, rates))
        require(len({t for t, _ in rows}) == len(rows), "Duplicate funding times")
        require(
            all(math.isfinite(r) and t % (HOUR * 1000) == 0 for t, r in rows),
            "Nonfinite or nonhourly funding source",
        )
        FUNDING_CACHE[key] = rows
    return FUNDING_CACHE[key]


def reserve(trade, leverage):
    # Independent arithmetic: margin plus entry fee plus estimated close fee.
    price = trade["zone_high"] if trade["side"] == "Sell" else trade["limit"]
    require(price >= trade["limit"], "Short collateral bound below sell limit")
    notional = trade["qty"] * price
    return notional / leverage + notional * FEE + notional * (1 + 1 / leverage) * FEE


class Ledger:
    def __init__(self, initial, leverage, prices):
        self.initial = self.wallet = initial
        self.leverage = leverage
        self.prices = prices
        self.pending = {}
        self.positions = {}
        self.funding = defaultdict(float)

    def quote(self, symbol, now):
        return self.prices[symbol][now][3]

    def snapshot(self, now):
        margin = close_fees = unrealized = gross = 0.0
        for t, qty in self.positions.values():
            entry, mark = t["entry"], self.quote(t["symbol"], now)
            margin += qty * entry / self.leverage
            close_fees += qty * entry * (1 + 1 / self.leverage) * FEE
            unrealized += (1 if t["side"] == "Buy" else -1) * qty * (mark - entry)
            gross += qty * mark
        reserved = sum(reserve(t, self.leverage) for t in self.pending.values())
        return dict(
            equity=self.wallet + unrealized,
            wallet_balance=self.wallet,
            cash=self.wallet - margin,
            available_cash=self.wallet - margin - close_fees - reserved,
            initial_margin=margin,
            open_margin=margin,
            reserved_close_fees=close_fees,
            reserved_pending=reserved,
            realized_net=self.wallet - self.initial,
            unrealized=unrealized,
            gross_notional=gross,
            open_count=len(self.positions),
            pending_count=len(self.pending),
        )

    def accept(self, trade):
        require(
            trade["id"] not in self.pending and trade["id"] not in self.positions,
            "Duplicate accepted ID",
        )
        self.pending[trade["id"]] = trade

    def event(self, kind, trade, payload):
        key = trade["id"]
        if kind == "ENTRY":
            require(key in self.pending, "Entry lacks prior reservation")
            del self.pending[key]
            self.positions[key] = (trade, payload["qty"])
            self.wallet -= payload["qty"] * payload["price"] * FEE
        elif kind == "EXIT":
            require(key in self.positions, "Exit lacks open position")
            previous, remaining = self.positions[key]
            sign = 1 if trade["side"] == "Buy" else -1
            self.wallet += sign * payload["qty"] * (payload["price"] - trade["entry"])
            self.wallet -= payload["qty"] * payload["price"] * FEE
            remaining -= payload["qty"]
            require(remaining >= -1e-7, "Exit exceeds remaining quantity")
            if abs(remaining) < 1e-7:
                del self.positions[key]
            else:
                self.positions[key] = (previous, remaining)
        elif kind == "FUNDING":
            self.wallet += payload
            self.funding[key] += payload
        elif kind == "EXPIRE":
            require(key in self.pending, "Expiry lacks reservation")
            del self.pending[key]
        else:
            raise AssertionError(kind)


def compare_snapshot(saved, rebuilt, label):
    for key, value in rebuilt.items():
        if key == "initial_margin" and key not in saved:
            continue
        equal(saved[key], value, f"{label}.{key}")


def funding_for_trade(t, rows, end):
    if t["opened_at"] is None:
        return []
    sign = 1 if t["side"] == "Buy" else -1
    result = []
    for ms, rate in rows:
        when = ms / 1000
        if not t["opened_at"] < when <= end:
            continue
        if t["closed_at"] is not None and when >= t["closed_at"]:
            continue
        remaining = Decimal(str(t["qty"])) - sum(
            (
                Decimal(str(f["qty"]))
                for f in t["fills"]
                if f["reason"] != "ENTRY" and f["time"] <= when
            ),
            Decimal(0),
        )
        require(remaining >= 0, "Negative quantity at funding settlement")
        result.append(
            dict(
                timestamp_ms=ms,
                rate=rate,
                qty=float(remaining),
                amount=-sign * float(remaining) * t["entry"] * rate,
            )
        )
    return result


def audit_trade(t, result, prices, funding, events):
    start, cutoff, end = (stamp(result[k]) for k in ("start", "entry_cutoff", "end"))
    require(start <= t["created_at"] < cutoff, "Decision outside entry window")
    require(t["decision_at"] == t["created_at"], "Decision timestamp mismatch")
    require(t["side"] in ("Buy", "Sell"), "Unsupported side")
    require(t["entry_style"] == "monitored_zone", "Unexpected entry style")
    require(t["research_leverage"] == result["leverage"], "Trade leverage mismatch")
    equal(t["research_entry_fee_rate"], FEE, "Fee schedule")
    require(
        not any(t.get(k) for k in ("data_error", "data_gap", "funding_error")),
        "Unresolved trade data errors",
    )
    sign = 1 if t["side"] == "Buy" else -1
    qty = gross = paid = 0.0
    require(
        [f["time"] for f in t["fills"]] == sorted(f["time"] for f in t["fills"]),
        "Unsorted fill times",
    )
    if t["opened_at"] is None:
        require(
            not t["fills"] and t["status"] in ("PENDING", "EXPIRED", "CANCELLED"),
            "Unfilled status/fills mismatch",
        )
        if t["status"] != "PENDING":
            # Failed next-open IOC is observed after that hour has closed.
            visible = t["closed_at"] + (
                HOUR if t["exit_reason"].startswith("IOC_") else 0
            )
            visible = int(math.ceil(visible / HOUR) * HOUR)
            events[visible].append((0, "EXPIRE", t, None))
    else:
        require(t["created_at"] <= t["opened_at"] < cutoff, "Invalid entry chronology")
        require(t["fills"] and t["fills"][0]["reason"] == "ENTRY", "Missing entry fill")
        entry_visible = t["opened_at"] + HOUR
        for i, f in enumerate(t["fills"]):
            require(
                t["opened_at"] <= f["time"] <= end and f["qty"] > 0 and f["price"] > 0,
                "Invalid fill amount/time",
            )
            fee = f["qty"] * f["price"] * FEE
            equal(f["fee"], fee, "Fill fee")
            paid += fee
            if f["reason"] == "ENTRY":
                require(i == 0, "Multiple entries in one trade")
                equal(f["time"], t["opened_at"], "Entry time")
                equal(f["price"], t["entry"], "Entry price")
                equal(f["qty"], t["qty"], "Entry quantity")
                opened_bar = prices[t["symbol"]][int(entry_visible)]
                expected = opened_bar[0] * (1 + sign * 0.00005) * (1 + sign * 0.0003)
                equal(f["price"], expected, "Observed next-open IOC fill")
                require(sign * (expected - t["limit"]) <= 1e-9, "IOC cap exceeded")
                require(
                    t["zone_low"] <= expected <= t["zone_high"],
                    "Fill outside frozen zone",
                )
                equal(f["gross_pnl"], 0, "Entry gross P&L")
                qty += f["qty"]
                events[int(entry_visible)].append((1, "ENTRY", t, f))
            else:
                qty -= f["qty"]
                pnl = sign * f["qty"] * (f["price"] - t["entry"])
                equal(f["gross_pnl"], pnl, "Signed exit gross P&L")
                gross += pnl
                require(
                    qty >= -1e-7 and f["time"] >= entry_visible, "Premature/excess exit"
                )
                events[int(f["time"])].append((2, "EXIT", t, f))
        equal(qty, t["remaining"], "Remaining quantity")
        if t["status"] == "CLOSED":
            equal(qty, 0, "Closed remaining quantity")
            equal(t["closed_at"], t["fills"][-1]["time"], "Closure time")
        else:
            require(t["status"] == "OPEN" and qty > 0, "Filled status mismatch")
    expected_funding = funding_for_trade(t, funding[t["symbol"]], end)
    actual_events = t.get("funding_events", [])
    require(
        len(actual_events) == len(expected_funding), "Funding settlement count mismatch"
    )
    for saved, calculated in zip(actual_events, expected_funding):
        for key in calculated:
            equal(saved[key], calculated[key], "Funding event " + key)
        when = int(calculated["timestamp_ms"] // 1000)
        events[when].append((3, "FUNDING", t, calculated["amount"]))
    total_funding = sum(f["amount"] for f in expected_funding)
    for key, expected in (
        ("gross_pnl", gross),
        ("fees", paid),
        ("funding", total_funding),
        ("net_pnl_before_funding", gross - paid),
        ("net_pnl", gross - paid + total_funding),
    ):
        equal(t[key], expected, t["id"] + "." + key)
    return dict(
        gross=gross, fees=paid, funding=total_funding, net=gross - paid + total_funding
    )


def check_acceptance(record, trade, ledger, now, mmr, dd, day_base, week_base, funding):
    op, sizing, ev = (
        record["opportunity"],
        record["sizing"],
        record["opportunity"]["evidence"],
    )
    acct = ledger.snapshot(now)
    compare_snapshot(record["account_before"], acct, "Accepted account before")
    require(
        sizing["allowed"] is True and not sizing["reasons"],
        "Accepted rejected assessment",
    )
    require(op["id"] == trade["opportunity_id"], "Opportunity identity mismatch")
    require(
        op["state"] == "READY"
        and op["side"] == trade["side"]
        and op["symbol"] == trade["symbol"],
        "Opportunity side/symbol/state mismatch",
    )
    require(
        op["created_at"]
        <= ev["observation_at"]
        <= ev["as_of"]
        <= now
        < op["expires_at"],
        "Observation after decision or expired",
    )
    require(now - ev["observation_at"] <= 60, "Stale observation")
    require(
        ev["daily_closed_at"] <= now and ev["execution_closed_at"] <= now,
        "Future higher timeframe input",
    )
    require(
        ev["entry_style"] == "monitored_zone" and ev["trigger_kind"] == "ZONE_ARRIVAL",
        "Unexpected observation style",
    )
    for key, trade_key in (
        ("qty", "qty"),
        ("entry", "limit"),
        ("stop", "original_stop"),
        ("target1", "target1"),
        ("target2", "target2"),
        ("risk_cash", "risk_cash"),
    ):
        equal(sizing[key], trade[trade_key], "Approved trade " + key)
    e, q, stop = sizing["entry"], sizing["qty"], sizing["stop"]
    sign = 1 if trade["side"] == "Buy" else -1
    require(sign * (e - stop) > 0, "Stop is not adverse")
    buffer = e * (1 / ledger.leverage - mmr - 0.002)
    require(buffer >= 2 * abs(e - stop), "Entry failed liquidation buffer rule")
    g = record["margin_guard"]
    require(
        g["allowed"] is True and not g["reasons"], "Incorrect stored guard approval"
    )
    for key, value in (
        ("buffer_price", buffer),
        ("stop_distance", abs(e - stop)),
        ("estimated_liquidation_price", e - sign * buffer),
        ("maintenance_rate", mmr),
    ):
        equal(g[key], value, "Guard " + key)
    require(g["liquidation_verified"] is False, "Estimated guard mislabeled verified")
    rows = funding[op["symbol"]]
    idx = bisect_right([t for t, _ in rows], now * 1000) - 1
    require(idx >= 1, "No past funding interval at decision")
    rate = rows[idx][1] * 8 / ((rows[idx][0] - rows[idx - 1][0]) / (HOUR * 1000))
    require(sign * rate <= 0.0003, "Adverse funding accepted")
    unit = abs(e - stop) + e * (0.0018 + max(sign * rate, 0))
    fraction = min(
        ev.get("risk_multiplier", 1), 0.5 if op["setup"] in ("2X", "2XS") else 1
    )
    pct = 0.25 * fraction * (0.5 if op["tier"] == 2 else 1) * (0.5 if dd >= 8 else 1)
    equal(sizing["risk_pct"], pct, "Fixed risk percentage")
    budget = acct["equity"] * pct / 100
    equal(sizing["risk_cash"], q * unit, "Stop+cost sized risk")
    require(
        q * unit <= budget + 1e-7 and (q + 0.0001) * unit > budget - 1e-7,
        "Quantity not floor-sized to budget",
    )
    equal(sizing["notional"], q * e, "Notional is not leverage-scaled")
    require(acct["equity"] > 0 and dd < 12, "Drawdown or equity halt ignored")
    require(max(0, 100 * (1 - acct["equity"] / day_base)) < 2, "Daily halt ignored")
    require(max(0, 100 * (1 - acct["equity"] / week_base)) < 4, "Weekly halt ignored")
    existing = [
        (t, rem, ledger.quote(t["symbol"], now)) for t, rem in ledger.positions.values()
    ]
    existing += [(t, t["qty"], t["limit"]) for t in ledger.pending.values()]
    require(len(existing) < 6, "Position cap exceeded")
    require(
        all(t["symbol"] != trade["symbol"] for t, _, _ in existing),
        "Duplicate symbol exposure",
    )
    bucket = [
        (t, rem, px)
        for t, rem, px in existing
        if t["bucket"].lower() == trade["bucket"].lower()
    ]
    require(len(bucket) < 2, "Bucket position cap exceeded")
    heat = sum(t["risk_cash"] * rem / t["qty"] for t, rem, _ in existing)
    bucket_heat = sum(t["risk_cash"] * rem / t["qty"] for t, rem, _ in bucket)
    require(
        heat + budget <= acct["equity"] * 0.04 + 1e-7, "Portfolio heat cap exceeded"
    )
    require(
        bucket_heat + budget <= acct["equity"] * 0.02 + 1e-7, "Bucket heat cap exceeded"
    )
    require(q * e <= acct["equity"] * 0.25 + 1e-7, "Single notional cap exceeded")
    require(
        sum(rem * px for _, rem, px in existing) + q * e <= acct["equity"] + 1e-7,
        "Gross notional cap exceeded",
    )
    require(
        reserve(trade, ledger.leverage) <= acct["available_cash"] + 1e-7,
        "Insufficient margin at acceptance",
    )


def audit_scenario(r):
    start, cutoff, end = (stamp(r[k]) for k in ("start", "entry_cutoff", "end"))
    require(
        r["complete"] and r["source_unchanged"] and r["funding_source_unchanged"],
        "Incomplete scenario",
    )
    require(r["profile"] == "cautious", "Unexpected risk profile")
    prices = {s: read_prices(v) for s, v in r["source"].items()}
    funding = {s: read_funding(v) for s, v in r["funding_source"].items()}
    require(
        set(prices) == set(funding) == set(r["symbols"]), "Source universe mismatch"
    )
    require(
        all(
            all(t in bars for t in range(start, end + 1, HOUR))
            for bars in prices.values()
        ),
        "Missing hourly price source",
    )
    trades = {t["id"]: t for t in r["trades"]}
    require(len(trades) == len(r["trades"]), "Duplicate trade ID")
    events = defaultdict(list)
    calculated = {
        key: audit_trade(t, r, prices, funding, events) for key, t in trades.items()
    }
    decisions = defaultdict(list)
    accepted = r["accepted_evidence"]
    require(len(accepted) == len(trades) == r["accepted"], "Acceptance count mismatch")
    require(
        accepted
        == sorted(
            accepted,
            key=lambda x: (
                x["time"],
                x["opportunity"]["symbol"],
                x["opportunity"]["id"],
            ),
        ),
        "Nondeterministic candidate ordering",
    )
    consumed = set()
    for row in accepted:
        key = "monitored_shadow:" + row["opportunity"]["id"]
        require(
            key in trades and key not in consumed, "Duplicate/missing accepted plan"
        )
        require(
            row["time"] == trades[key]["created_at"], "Accepted decision time mismatch"
        )
        consumed.add(key)
        decisions[int(row["time"])].append((row, trades[key]))
    require(
        set(events).issubset(range(start, end + 1, HOUR)), "Event outside replay clock"
    )
    ledger = Ledger(r["initial_equity"], r["leverage"], prices)
    daily = {int(x["time"]): x for x in r["daily_equity"]}
    require(len(daily) == len(r["daily_equity"]), "Duplicate daily equity rows")
    expected_daily = {
        t for t in range(start, end + 1, HOUR) if t % DAY == 0 or t in (start, end)
    }
    require(set(daily) == expected_daily, "Daily curve coverage mismatch")
    peak = prior_equity = day_base = week_base = r["initial_equity"]
    max_dd = max_margin = margin_sum = max_notional = 0.0
    min_cash, max_count = r["initial_equity"], 0
    day_key = week_key = None
    yearly, stress = {}, []
    for now in range(start, end + 1, HOUR):
        date = datetime.fromtimestamp(now, timezone.utc)
        boundary = ledger.snapshot(now)["equity"]
        if day_key != date.date():
            day_key, day_base = date.date(), boundary
        if week_key != date.isocalendar()[:2]:
            week_key, week_base = date.isocalendar()[:2], boundary
        old_positions = dict(ledger.positions)
        for _, kind, trade, payload in sorted(events[now], key=lambda x: x[0]):
            if kind == "ENTRY":
                old_positions[trade["id"]] = (trade, payload["qty"])
            ledger.event(kind, trade, payload)
        # Reconstruct the documented last-price stress range, not liquidation.
        for t, qty in old_positions.values():
            sign = 1 if t["side"] == "Buy" else -1
            distance = t["entry"] * (
                1 / r["leverage"] - r["maintenance_rate_assumption"] - 0.002
            )
            distance += min(0, ledger.funding[t["id"]]) / qty
            threshold = t["entry"] - sign * distance
            b = prices[t["symbol"]][now]
            extreme = b[2] if sign == 1 else b[1]
            if sign * (extreme - threshold) <= 0:
                stress.append(
                    (now, t["id"], threshold, extreme, sign * (b[0] - threshold) <= 0)
                )
        account = ledger.snapshot(now)
        peak = max(peak, account["equity"])
        dd = max(0, 100 * (1 - account["equity"] / peak))
        max_dd = max(max_dd, dd)
        for row, trade in decisions[now]:
            check_acceptance(
                row,
                trade,
                ledger,
                now,
                r["maintenance_rate_assumption"],
                dd,
                day_base,
                week_base,
                funding,
            )
            ledger.accept(trade)
        account = ledger.snapshot(now)
        margin = (
            account["open_margin"]
            + account["reserved_close_fees"]
            + account["reserved_pending"]
        )
        max_margin = max(max_margin, margin)
        margin_sum += margin
        min_cash = min(min_cash, account["available_cash"])
        max_count = max(max_count, account["open_count"] + account["pending_count"])
        gross = account["gross_notional"] + sum(
            t["qty"] * t["limit"] for t in ledger.pending.values()
        )
        max_notional = max(max_notional, gross)
        yr = yearly.setdefault(str(date.year), dict(start_equity=prior_equity))
        yr["end_equity"] = account["equity"]
        prior_equity = account["equity"]
        if now in daily:
            compare_snapshot(daily[now], account, "Daily " + str(now))
            equal(daily[now]["drawdown_pct"], dd, "Daily drawdown")
    compare_snapshot(r["final_account"], account, "Final account")
    for key, value in (
        ("max_drawdown_pct", max_dd),
        ("max_reserved_margin_usdt", max_margin),
        ("average_reserved_margin_usdt", margin_sum / ((end - start) // HOUR + 1)),
        ("minimum_available_cash", min_cash),
        ("maximum_positions_and_reservations", max_count),
        ("max_gross_notional_usdt", max_notional),
        ("net_equity_change", account["equity"] - r["initial_equity"]),
        ("net_return_pct", 100 * (account["equity"] / r["initial_equity"] - 1)),
    ):
        equal(r[key], value, key)
    for year, values in yearly.items():
        for key, value in values.items():
            equal(r["calendar_equity"][year][key], value, "Year " + year + " " + key)
    saved_stress = sorted(
        r["liquidation_stress_events"], key=lambda x: (x["time"], x["trade_id"])
    )
    stress.sort(key=lambda x: x[:2])
    require(len(saved_stress) == len(stress), "Stress event count mismatch")
    for row, (when, identity, threshold, extreme, gap) in zip(saved_stress, stress):
        require(
            row["time"] == when and row["trade_id"] == identity,
            "Stress event identity mismatch",
        )
        equal(row["threshold"], threshold, "Stress threshold")
        equal(row["adverse_extreme"], extreme, "Stress extreme")
        require(row["gap_open_beyond_threshold"] == gap, "Stress gap flag mismatch")
    require(
        r["liquidation_stress_flagged_trades"] == len({x[1] for x in stress}),
        "Stress unique count mismatch",
    )
    counts = dict(
        filled=sum(t["opened_at"] is not None for t in trades.values()),
        open=sum(t["status"] == "OPEN" for t in trades.values()),
        closed=sum(t["status"] == "CLOSED" for t in trades.values()),
        expired=sum(t["status"] == "EXPIRED" for t in trades.values()),
    )
    for key, value in counts.items():
        equal(r[key], value, key)
    closed = [t for t in trades.values() if t["status"] == "CLOSED"]
    wins = sum(calculated[t["id"]]["net"] > 1e-9 for t in closed)
    losses = sum(calculated[t["id"]]["net"] < -1e-9 for t in closed)
    equal(r["wins"], wins, "wins")
    equal(r["losses"], losses, "losses")
    equal(r["net"], sum(calculated[t["id"]]["net"] for t in closed), "Closed net")
    if closed:
        equal(r["win_rate_pct"], 100 * wins / len(closed), "Win rate")
    else:
        require(r["win_rate_pct"] is None, "Zero-trade WR must be unavailable")
    per_side = {}
    for side in ("Buy", "Sell"):
        selected = [t for t in trades.values() if t["side"] == side]
        per_side[side] = {
            k: sum(calculated[t["id"]][k] for t in selected)
            for k in ("gross", "fees", "funding", "net")
        }
    for grouping, keys in (("per_side", ("Buy", "Sell")), ("per_symbol", r["symbols"])):
        for key in keys:
            selected = [
                t
                for t in closed
                if t["side" if grouping == "per_side" else "symbol"] == key
            ]
            values = dict(
                closed=len(selected),
                wins=sum(calculated[t["id"]]["net"] > 1e-9 for t in selected),
                losses=sum(calculated[t["id"]]["net"] < -1e-9 for t in selected),
                net=sum(calculated[t["id"]]["net"] for t in selected),
            )
            for field, value in values.items():
                equal(
                    r[grouping][key][field], value, grouping + " " + key + " " + field
                )
    equal(r["total_fees"], sum(v["fees"] for v in calculated.values()), "Total fees")
    equal(
        r["funding_estimate"],
        sum(v["funding"] for v in calculated.values()),
        "Total funding",
    )
    return dict(
        leverage=r["leverage"],
        maintenance_rate=r["maintenance_rate_assumption"],
        complete=True,
        independent_final_account=account,
        per_side_cashflows=per_side,
        counts=counts,
        daily_snapshots_checked=len(daily),
        hourly_snapshots_checked=(end - start) // HOUR + 1,
        accepted_cap_guard_checks=len(accepted),
        stress_flags=len(stress),
        free_cash_shortfall=min_cash < -1e-7,
        minimum_available_cash=min_cash,
        actual_liquidation_verified=False,
    )


def self_test():
    # Synthetic hand records prove ledger math, short signs, and disclosure time.
    prices = {
        "X": {
            0: (100, 100, 100, 100),
            HOUR: (100, 101, 99, 100),
            2 * HOUR: (95, 96, 94, 95),
        }
    }
    ledger = Ledger(1000, 5, prices)
    trade = dict(
        id="x", symbol="X", side="Sell", qty=5, limit=99.95, zone_high=102, entry=100
    )
    ledger.accept(trade)
    pending = ledger.snapshot(0)
    equal(pending["equity"], 1000, "Pending equity")
    equal(pending["reserved_pending"], 102 + 510 * FEE + 612 * FEE, "Upper-zone margin")
    ledger.event("ENTRY", trade, dict(qty=5, price=100))
    equal(ledger.snapshot(HOUR)["equity"], 999.725, "Entry fee")
    ledger.event("EXIT", trade, dict(qty=3, price=90))
    ledger.event("FUNDING", trade, 0.02)
    account = ledger.snapshot(2 * HOUR)
    equal(
        account["wallet_balance"],
        1000 - 0.275 + 30 - 0.1485 + 0.02,
        "Short partial wallet",
    )
    equal(account["open_margin"], 40, "Remaining margin")
    equal(account["unrealized"], 10, "Short unrealized")
    equal(account["reserved_close_fees"], 0.132, "Close fee reserve")
    corrupt = dict(account, equity=account["equity"] + 1)
    try:
        compare_snapshot(corrupt, account, "Tampering")
    except AssertionError:
        pass
    else:
        raise AssertionError("Audit failed to detect corrupted equity")
    ft = dict(
        opened_at=0,
        closed_at=3 * HOUR,
        side="Sell",
        qty=5,
        entry=100,
        fills=[
            dict(reason="ENTRY", time=0, qty=5),
            dict(reason="TP1", time=2 * HOUR, qty=3),
        ],
    )
    funding = funding_for_trade(
        ft,
        [
            (0, 0.0001),
            (HOUR * 1000, 0.0001),
            (2 * HOUR * 1000, 0.0001),
            (3 * HOUR * 1000, 0.0001),
        ],
        4 * HOUR,
    )
    require(len(funding) == 2, "Funding entry/exit ties not excluded")
    equal(funding[0]["amount"], 0.05, "Short funding before partial")
    equal(funding[1]["amount"], 0.02, "Short funding at partial")
    print("Independent audit self-test passed")


def verify_parent_inputs(root, report, scenario):
    """Independently match the parent's frozen universe and every worker file."""
    manifest = json.loads((root / "manifest.json").read_text())
    parent = report["input_snapshot"]
    require(
        set(manifest["symbols"]) == set(parent) == set(scenario["symbols"]),
        "Parent/worker universe mismatch",
    )
    require(scenario["matches_parent_inputs"] is True, "Worker rejected parent inputs")
    require(scenario["tape_source_unchanged"] is True, "Worker tape changed")
    for field, kind in (
        ("source", "hourly"),
        ("funding_source", "funding"),
        ("tape_source", "tape"),
    ):
        require(set(scenario[field]) == set(parent), "Worker source universe mismatch")
        for symbol, sources in parent.items():
            planned, used = sources[kind], scenario[field][symbol]
            require(
                used["sha256"] == planned["sha256"], "Parent/worker digest mismatch"
            )
            require(
                Path(used["path"]).resolve() == Path(planned["path"]).resolve(),
                "Parent/worker path mismatch",
            )
            require(sha(planned["path"]) == planned["sha256"], "Parent input changed")
            expected = manifest.get("source_sha256", {}).get(symbol, {}).get(kind)
            if expected is not None:
                require(
                    expected == planned["sha256"], "Universe-selection source changed"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.directory is None:
        parser.error("Evidence directory required")
    root = args.directory
    report = json.loads((root / "report.json").read_text())
    require(report["complete"] and not report["errors"], "Incomplete experiment")
    for key in (
        "code_unchanged",
        "protocol_unchanged",
        "manifest_unchanged",
        "all_inputs_unchanged",
        "selection_audit_unchanged",
    ):
        require(report[key] is True, "Changed experiment inputs: " + key)
    require(
        sha(root / "protocol.txt") == report["protocol_sha256"],
        "Frozen protocol hash mismatch",
    )
    for name, digest in report["code_sha256"].items():
        require(
            sha(root / "code" / name) == digest, "Frozen code hash mismatch: " + name
        )
    require(
        sha(root / "selection_audit.json") == report["selection_audit_sha256"],
        "Frozen universe-selection audit hash mismatch",
    )
    require(len(report["rows"]) == 6, "Expected six predefined scenarios")
    require(
        {(r["leverage"], r["maintenance_rate_assumption"]) for r in report["rows"]}
        == {(lev, mm) for lev in (5, 10) for mm in (0.005, 0.01, 0.025)},
        "Scenario set mismatch",
    )
    audit = dict(complete=True, independent_of_bot_accounting=True, arms=[])
    for row in report["rows"]:
        path = root / row["evidence_file"]
        require(sha(path) == row["evidence_sha256"], "Scenario evidence hash mismatch")
        r = json.loads(gzip.decompress(path.read_bytes()))
        verify_parent_inputs(root, report, r)
        for key, value in row.items():
            if key in r:
                require(value == r[key], "Summary/evidence mismatch: " + key)
        audit["arms"].append(audit_scenario(r))
        print(json.dumps(audit["arms"][-1], allow_nan=False), flush=True)
    audit["limitations"] = [
        "Checks saved executions/accounting, not strategy validity or expected profit.",
        "Reconciles all observed funding rows; source completeness remains unverified.",
        "Reproduces last-price stress flags; cannot establish actual liquidations.",
    ]
    audit["parent_worker_inputs_verified"] = True
    output = args.output or root / "accounting_audit.json"
    with output.open("x") as handle:
        handle.write(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
