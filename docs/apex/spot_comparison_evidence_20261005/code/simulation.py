"""Causal OHLC shadow accounting: no contacts, storage, orders or cash transfers.

Engineering assumptions versus charter 7.3:
- Only FULL candles opening at/after the actual decision and closing no later
  than expiry can fill. Never skip the first expected candle. Fill time is the
  candle CLOSE, when a touch becomes known, at the limit without favorable gap
  improvement. Partial decision/expiry candles cannot fill. Entry-bar stops
  apply, but entry-bar targets never receive credit.
- Existing stops win OHLC ties; gap stops use the worse open. A TP1 stop change
  or structural ratchet is effective next price candle. One lot stays intact at
  T1; odd multi-lot TP1 rounds UP. These lot rules come from the charter.
- With daily/execution supplied, daily invalidation exits at that close.
  Without a favorable closed 4H zone break, 10 subsequent daily closes flag
  review and 20 exit. The partial entry day counts at its subsequent close.
  Beyond the T1/T2 midpoint, newly confirmed ATR(14), 1-ATR 4H higher lows/lower
  highs trail by 0.1% beyond the pivot, ratcheting only. This numeric pivot rule
  and buffer are engineering choices, not charter precision or count/CCR proof.
- Omitted structure inputs mean price-only simulation, not full charter
  management. Adding management after a price-only fill requires fresh replay.
- Assumed fees: 6bp per fill; exit slippage: 3bp. Gross already includes adverse
  slippage. Net is REALIZED gross minus fill fees plus estimated funding.
- Funding uses settled historical rates times remaining ENTRY notional, not
  settlement mark notional: always an estimate. Completeness requires explicit
  covered_from/covered_until and history_complete=True from a range-complete
  paginated client. A timestamp or a nonempty page cannot certify coverage.
- Baseline-all versus AI-selected is observational selection, with latency and
  sizing confounds when execution differs. Shared counterfactual IDs do not make
  different decision times a pure filter experiment. Never move an earlier
  persisted baseline decision later to manufacture execution parity.

Integration: advance(..., daily=daily_bars, execution=closed_4h_bars).
Supply contiguous management series covering new boundaries and a fixed
execution history origin with at least 14 prior 4H bars for ATR warmup.
apply_funding(..., covered_from=start,
history_complete=True) is allowed only after successful complete pagination.
summary retains its old keys; comparison adds matched-counterfactual metrics.
Times are seconds except Candle.open_time, last_bar and interval_ms.
"""

from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import math

from .engine import DAY, H4, confirmed_zigzag
from .models import Candle, Opportunity

FEE_RATE = 0.0006
SLIPPAGE = 0.0003
TRAIL_BUFFER = 0.001
INTERVAL_MS = 180_000
COST_MODEL = "limit-close-known/fee6bps/exit-slippage3bps-v2"
MONITORED_COST_MODEL = (
    "next-open-ioc/half-spread0.5bps/entry-exit-slippage3bps/fee6bps-v1"
)
MONITORED_HALF_SPREAD = 0.00005


def _finite(value):
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _number(value, positive=False):
    if not _finite(value) or (value <= 0 if positive else value < 0):
        raise ValueError("Nonfinite, negative or zero simulation value")
    return value


def _ceil_open(timestamp, interval_ms):
    return (
        int(
            (Decimal(str(timestamp)) * 1000 / interval_ms).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        * interval_ms
    )


def _closed_bars(bars, interval_ms, now):
    result = []
    for bar in bars:
        if (
            not isinstance(bar, Candle)
            or type(bar.open_time) is not int
            or bar.open_time < 0
        ):
            raise ValueError("Invalid candle timestamp")
        if bar.open_time / 1000 + interval_ms / 1000 > now:
            continue
        if bar.open_time % interval_ms:
            raise ValueError("Unaligned candle")
        for value in (bar.open, bar.high, bar.low, bar.close):
            _number(value, True)
        _number(bar.volume)
        if (
            not bar.low
            <= min(bar.open, bar.close)
            <= max(bar.open, bar.close)
            <= bar.high
        ):
            raise ValueError("Inconsistent OHLC")
        if result and bar.open_time <= result[-1].open_time:
            raise ValueError("Unordered or duplicate candle")
        result.append(bar)
    return result


def _units(qty, step):
    units = Decimal(str(qty)) / Decimal(str(step))
    if units != units.to_integral_value():
        raise ValueError("Quantity is not a whole number of lots")
    return int(units)


def create_trade(op: Opportunity, sizing: dict, arm: str, now: float) -> dict:
    _number(now)
    if (
        not isinstance(arm, str)
        or not arm
        or sizing.get("allowed") is not True
        or op.state != "READY"
    ):
        raise ValueError("A READY opportunity and approved sizing are required")
    if op.side not in ("Buy", "Sell") or not isinstance(op.evidence, dict):
        raise ValueError("Invalid opportunity")
    _number(op.created_at)
    _number(op.expires_at)
    resting = op.evidence.get("entry_style") == "resting_limit"
    monitored = op.evidence.get("entry_style") == "monitored_zone"
    if resting and (
        arm != "resting_shadow" or op.evidence.get("trigger_kind") != "ZONE_LIMIT"
    ):
        raise ValueError("Resting limits require the separate resting_shadow book")
    if monitored and (
        arm != "monitored_shadow"
        or op.evidence.get("trigger_kind") != "ZONE_ARRIVAL"
        or any(
            op.evidence.get(k) is not None
            for k in ("trigger_closed_at", "trigger_price", "trigger_evidence_id")
        )
    ):
        raise ValueError("Monitored zones require the separate monitored_shadow book")
    trigger = _number(
        op.evidence.get(
            "observation_at"
            if monitored
            else "order_armed_at" if resting else "trigger_closed_at"
        )
    )
    if not op.created_at <= trigger <= now < op.expires_at:
        raise ValueError("Decision precedes evidence or follows expiry")
    for key in (
        "entry",
        "stop",
        "target1",
        "target2",
        "qty",
        "qty_step",
        "risk_cash",
        "notional",
    ):
        _number(sizing.get(key), True)
    for key in ("zone_low", "zone_high"):
        _number(op.evidence.get(key), True)
    _number(op.invalidation, True)
    if op.evidence["zone_low"] > op.evidence["zone_high"]:
        raise ValueError("Invalid zone")
    sign = 1 if op.side == "Buy" else -1
    if not all(
        sign * (a - b) > 0
        for a, b in (
            (sizing["entry"], sizing["stop"]),
            (sizing["target1"], sizing["entry"]),
            (sizing["target2"], sizing["target1"]),
        )
    ):
        raise ValueError("Invalid price order")
    _units(sizing["qty"], sizing["qty_step"])
    _number(
        sizing["qty"] * max(sizing[k] for k in ("entry", "stop", "target1", "target2")),
        True,
    )
    trade = {
        "id": f"{arm}:{op.id}",
        "opportunity_id": op.id,
        "counterfactual_id": op.id,
        "arm": arm,
        "symbol": op.symbol,
        "side": op.side,
        "setup": op.setup,
        "entry_style": (
            "monitored_zone"
            if monitored
            else "resting_limit" if resting else "confirmed"
        ),
        "bucket": op.bucket,
        "status": "PENDING",
        "created_at": now,
        "decision_at": now,
        "signal_available_at": trigger,
        "entry_eligible_at": _ceil_open(now, INTERVAL_MS) / 1000,
        "expires_at": op.expires_at,
        "limit": sizing["entry"],
        "qty": sizing["qty"],
        "remaining": sizing["qty"],
        "qty_step": sizing["qty_step"],
        "stop": sizing["stop"],
        "original_stop": sizing["stop"],
        "target1": sizing["target1"],
        "target2": sizing["target2"],
        "invalidation": op.invalidation,
        "zone_low": op.evidence["zone_low"],
        "zone_high": op.evidence["zone_high"],
        "risk_cash": sizing["risk_cash"],
        "notional": sizing["notional"],
        "entry": None,
        "opened_at": None,
        "closed_at": None,
        "exit_reason": None,
        "gross_pnl": 0.0,
        "fees": 0.0,
        "funding": 0.0,
        "funding_ids": [],
        "funding_events": [],
        "funding_coverage": [],
        "funding_complete": False,
        "funding_basis": "settled rate × remaining entry notional (estimate)",
        "tp1_done": False,
        "last_bar": None,
        "interval_ms": INTERVAL_MS,
        "fills": [],
        "cost_model": MONITORED_COST_MODEL if monitored else COST_MODEL,
        "management_mode": None,
        "management_events": [],
        "daily_bars_held": 0,
        "time_review_due": False,
        "favorable_4h_close": False,
        "trailing_active": False,
        "trail_activated_at": None,
        "execution_origin": None,
    }
    return _totals(trade)


def _totals(trade):
    trade["net_pnl_before_funding"] = trade["gross_pnl"] - trade["fees"]
    trade["net_pnl"] = trade["net_pnl_before_funding"] + trade["funding"]
    if not all(_finite(trade[k]) for k in ("gross_pnl", "fees", "funding", "net_pnl")):
        raise ValueError("Nonfinite simulated cashflow")
    return trade


def _exit(trade, qty, price, timestamp, reason):
    sign = 1 if trade["side"] == "Buy" else -1
    price *= 1 - sign * SLIPPAGE
    qty = min(qty, trade["remaining"])
    if qty <= 0 or timestamp < trade["opened_at"]:
        raise ValueError("Invalid exit quantity or chronology")
    gross = sign * (price - trade["entry"]) * qty
    fee = price * qty * FEE_RATE
    trade["gross_pnl"] += gross
    trade["fees"] += fee
    trade["remaining"] = float(Decimal(str(trade["remaining"])) - Decimal(str(qty)))
    trade["fills"].append(
        {
            "time": timestamp,
            "qty": qty,
            "price": price,
            "reason": reason,
            "fee": fee,
            "gross_pnl": gross,
        }
    )
    if trade["remaining"] == 0:
        trade.update(status="CLOSED", closed_at=timestamp, exit_reason=reason)


def _ratchet(trade, stop, timestamp, reason):
    sign = 1 if trade["side"] == "Buy" else -1
    if sign * (stop - trade["stop"]) > 0:
        trade["stop"] = stop
        trade["management_events"].append(
            {
                "time": timestamp,
                "reason": reason,
                "stop": stop,
                "effective": "next_price_bar",
            }
        )


def _manage(trade, daily, execution, pivots, timestamp):
    """Called after this price bar; a new stop cannot use its earlier extrema."""
    if trade["status"] == "PENDING":
        sign = 1 if trade["side"] == "Buy" else -1
        if daily is not None and sign * (daily.close - trade["invalidation"]) <= 0:
            trade.update(
                status="EXPIRED", closed_at=timestamp, exit_reason="DAILY_INVALIDATION"
            )
        return
    if trade["status"] != "OPEN" or timestamp < trade["opened_at"]:
        return
    sign = 1 if trade["side"] == "Buy" else -1
    if execution is not None and timestamp > trade["opened_at"]:
        edge = trade["zone_high"] if sign > 0 else trade["zone_low"]
        if sign * (execution.close - edge) > 0:
            trade["favorable_4h_close"] = True
            trade["time_review_due"] = False
    if daily is not None:
        if sign * (daily.close - trade["invalidation"]) <= 0:
            _exit(
                trade, trade["remaining"], daily.close, timestamp, "DAILY_INVALIDATION"
            )
            return
        if timestamp == trade["opened_at"]:
            return
        trade["daily_bars_held"] += 1
        if not trade["favorable_4h_close"]:
            if trade["daily_bars_held"] >= 20:
                _exit(trade, trade["remaining"], daily.close, timestamp, "TIMEOUT")
                return
            if trade["daily_bars_held"] >= 10 and not trade["time_review_due"]:
                trade["time_review_due"] = True
                trade["management_events"].append(
                    {"time": timestamp, "reason": "TIME_REVIEW_10D"}
                )
    if execution is not None and trade["trailing_active"]:
        kind = "low" if sign > 0 else "high"
        known = [p for p in pivots if p.kind == kind and p.available_at <= timestamp]
        if len(known) >= 2:
            pivot, previous = known[-1], known[-2]
            if (
                pivot.available_at == timestamp
                and pivot.available_at >= trade["trail_activated_at"]
                and pivot.open_time / 1000 >= trade["opened_at"]
                and sign * (pivot.price - previous.price) > 0
            ):
                stop = pivot.price * (1 - sign * TRAIL_BUFFER)
                if sign * (execution.close - stop) > 0:
                    _ratchet(trade, stop, timestamp, "STRUCTURAL_TRAIL")
                    trade["last_trailing_pivot"] = pivot.id


def _management_inputs(trade, daily, execution, now):
    enabled = daily is not None and execution is not None
    requested = "closed_structure" if enabled else "price_only"
    old = trade.get("management_mode")
    if trade["status"] == "OPEN" and old not in (None, requested):
        raise ValueError("Management mode changed after entry; replay required")
    if trade["status"] == "OPEN" and old is None and enabled:
        raise ValueError("Legacy open trade lacks management history; replay required")
    trade["management_mode"] = requested
    trade["management_complete"] = enabled
    if not enabled:
        trade["management_warning"] = (
            "Daily invalidation, 10/20-day review/exit and structural trail unavailable"
        )
        return {}, {}, []
    d = _closed_bars(daily, DAY * 1000, now)
    e = _closed_bars(execution, H4 * 1000, now)
    for bars, interval in ((d, DAY), (e, H4)):
        if any(
            b.open_time - a.open_time != interval * 1000 for a, b in zip(bars, bars[1:])
        ):
            raise ValueError("Management history gap; replay required")
    origin = e[0].open_time if e else None
    if origin is None or origin / 1000 > trade["created_at"] - 14 * H4:
        raise ValueError("At least 14 prior 4H bars are required for structural warmup")
    if (
        trade.get("execution_origin") is not None
        and origin != trade["execution_origin"]
    ):
        raise ValueError("Execution ATR history origin changed; replay required")
    if origin is not None:
        trade["execution_origin"] = origin
    trade.pop("management_warning", None)
    return (
        {b.open_time // 1000 + DAY: b for b in d},
        {b.open_time // 1000 + H4: b for b in e},
        confirmed_zigzag(e, H4, now, atr_multiple=1),
    )


def _validate_trade(trade):
    if trade["side"] not in ("Buy", "Sell"):
        raise ValueError("Invalid trade side")
    for key in ("gross_pnl", "funding"):
        if not _finite(trade[key]):
            raise ValueError("Invalid trade numeric field")
    _number(trade["created_at"])
    _number(trade["expires_at"])
    for key in (
        "limit",
        "qty",
        "qty_step",
        "stop",
        "target1",
        "target2",
        "invalidation",
    ):
        _number(trade[key], True)
    _number(trade["remaining"], True)
    _number(trade["fees"])
    _units(trade["qty"], trade["qty_step"])
    _units(trade["remaining"], trade["qty_step"])
    if trade["remaining"] > trade["qty"] or trade["created_at"] >= trade["expires_at"]:
        raise ValueError("Invalid remaining quantity or expiry")
    if trade["status"] == "OPEN":
        _number(trade["entry"], True)
        _number(trade["opened_at"])
        if trade.get("last_bar") is None:
            raise ValueError("Open trade without replay cursor")
    for fill in trade["fills"]:
        _number(fill["price"], True)
        _number(fill["qty"], True)
        _number(fill["time"])


def advance(
    trade: dict,
    bars: list[Candle],
    now: float,
    interval_ms=INTERVAL_MS,
    *,
    daily: list[Candle] | None = None,
    execution: list[Candle] | None = None,
) -> dict:
    """Replay fully evidenced consecutive candles. Backfill before a gap resolves.

    Bad data returns data_error with unchanged financial state, never invented
    PnL. Existing OPEN cursors cannot retroactively adopt structural management.
    """
    result = copy.deepcopy(trade)
    if result.get("status") not in {"PENDING", "OPEN"}:
        return result
    try:
        _number(now)
        if type(interval_ms) is not int or interval_ms <= 0 or H4 * 1000 % interval_ms:
            raise ValueError("Price interval must divide 4H exactly")
        _validate_trade(result)
        if now < result["created_at"] or (
            result.get("last_bar") is not None
            and now < (result["last_bar"] + interval_ms) / 1000
        ):
            raise ValueError("Replay time precedes decision or resolved cursor")
        if (
            result.get("last_bar") is not None
            and result.get("interval_ms", interval_ms) != interval_ms
        ):
            raise ValueError("Price interval changed during replay")
        result["interval_ms"] = interval_ms
        result.setdefault("management_events", [])
        result.setdefault("daily_bars_held", 0)
        result.setdefault("favorable_4h_close", False)
        result.setdefault("time_review_due", False)
        result.setdefault("trailing_active", False)
        price_bars = _closed_bars(bars, interval_ms, now)
        d, e, pivots = _management_inputs(result, daily, execution, now)
        expected = (
            _ceil_open(result["created_at"], interval_ms)
            if result["last_bar"] is None
            else result["last_bar"] + interval_ms
        )
        result["entry_eligible_at"] = (
            _ceil_open(result["created_at"], interval_ms) / 1000
        )
        sign = 1 if result["side"] == "Buy" else -1
        result.pop("data_error", None)
        result.pop("data_gap", None)
        for bar in price_bars:
            if bar.open_time < expected:
                continue
            if (
                result["status"] == "PENDING"
                and expected / 1000 + interval_ms / 1000 > result["expires_at"]
            ):
                break
            if bar.open_time != expected:
                result["data_gap"] = (
                    f"Missing price candle at {expected}; backfill required"
                )
                break
            timestamp = (bar.open_time + interval_ms) / 1000
            if result["management_mode"] == "closed_structure":
                # A decision just before midnight can have its first eligible
                # price candle AFTER a newly available daily invalidation.
                prior = (
                    (result["last_bar"] + interval_ms) / 1000
                    if result["last_bar"] is not None
                    else result["created_at"]
                )
                next_day = (math.floor(prior / DAY) + 1) * DAY
                if next_day <= bar.open_time / 1000:
                    if next_day not in d:
                        result["data_gap"] = (
                            f"Missing closed management candle at {next_day}; backfill required"
                        )
                        break
                    _manage(result, d[next_day], None, [], next_day)
                    if result["status"] == "EXPIRED":
                        break
                if (
                    timestamp % DAY == 0
                    and timestamp not in d
                    or timestamp % H4 == 0
                    and timestamp not in e
                ):
                    result["data_gap"] = (
                        f"Missing closed management candle at {timestamp}; backfill required"
                    )
                    break
            result["last_bar"] = bar.open_time
            expected += interval_ms
            entry_bar = result["status"] == "PENDING"
            if entry_bar:
                monitored = result.get("entry_style") == "monitored_zone"
                fill_price = result["limit"]
                fill_time = timestamp
                if monitored:
                    # An IOC is resolved at the first eligible OPEN, never by
                    # a later excursion through the price cap. Risk was sized
                    # at the cap; this observed fill can only be as good/better.
                    fill_price = (
                        bar.open
                        * (1 + sign * MONITORED_HALF_SPREAD)
                        * (1 + sign * SLIPPAGE)
                    )
                    fill_time = bar.open_time / 1000
                    reason = None
                    if sign * (fill_price - result["limit"]) > 1e-10:
                        reason = "IOC_PRICE_MISSED"
                    elif not result["zone_low"] <= fill_price <= result["zone_high"]:
                        reason = "IOC_OUTSIDE_ZONE"
                    elif sign * (bar.open - result["invalidation"]) <= 0:
                        reason = "IOC_INVALIDATED"
                    elif sign * (bar.open - result["target1"]) >= 0:
                        reason = "IOC_TARGET_PASSED"
                    if reason:
                        result.update(
                            status="EXPIRED", closed_at=fill_time, exit_reason=reason
                        )
                        break
                touched = monitored or (
                    bar.low <= result["limit"]
                    if sign > 0
                    else bar.high >= result["limit"]
                )
                if (
                    result.get("entry_style") == "resting_limit"
                    and not touched
                    and (
                        bar.high >= result["target1"]
                        if sign > 0
                        else bar.low <= result["target1"]
                    )
                ):
                    result.update(
                        status="EXPIRED",
                        closed_at=timestamp,
                        exit_reason="TARGET_BEFORE_ENTRY",
                    )
                    break
                if not touched:
                    _manage(
                        result, d.get(timestamp), e.get(timestamp), pivots, timestamp
                    )
                    if result["status"] == "EXPIRED":
                        break
                    continue
                result.update(
                    status="OPEN",
                    entry=fill_price,
                    opened_at=fill_time,
                    entry_window_start=bar.open_time / 1000,
                )
                fee = result["entry"] * result["qty"] * FEE_RATE
                result["fees"] += fee
                result["fills"].append(
                    {
                        "time": fill_time,
                        "qty": result["qty"],
                        "price": result["entry"],
                        "reason": "ENTRY",
                        "fee": fee,
                        "gross_pnl": 0.0,
                    }
                )
            stop_hit = (
                bar.low <= result["stop"] if sign > 0 else bar.high >= result["stop"]
            )
            if stop_hit:
                price = (
                    min(bar.open, result["stop"])
                    if sign > 0
                    else max(bar.open, result["stop"])
                )
                _exit(result, result["remaining"], price, timestamp, "STOP")
                break
            if entry_bar:
                _manage(result, d.get(timestamp), e.get(timestamp), pivots, timestamp)
                if result["status"] == "CLOSED":
                    break
                continue
            t1_hit = (
                bar.high >= result["target1"]
                if sign > 0
                else bar.low <= result["target1"]
            )
            if not result["tp1_done"] and t1_hit:
                lots = _units(result["qty"], result["qty_step"])
                if lots > 1:
                    partial = float(
                        Decimal((lots + 1) // 2) * Decimal(str(result["qty_step"]))
                    )
                    _exit(result, partial, result["target1"], timestamp, "TP1")
                result["tp1_done"] = True
                _ratchet(result, result["entry"], timestamp, "TP1_TO_ENTRY")
            t2_hit = (
                bar.high >= result["target2"]
                if sign > 0
                else bar.low <= result["target2"]
            )
            if result["tp1_done"] and t2_hit:
                _exit(result, result["remaining"], result["target2"], timestamp, "TP2")
                break
            midpoint = (result["target1"] + result["target2"]) / 2
            extreme = bar.high if sign > 0 else bar.low
            if (
                result["tp1_done"]
                and not result["trailing_active"]
                and sign * (extreme - midpoint) > 0
            ):
                result["trailing_active"] = True
                result["trail_activated_at"] = timestamp
            _manage(result, d.get(timestamp), e.get(timestamp), pivots, timestamp)
            if result["status"] == "CLOSED":
                break
        if (
            result["status"] == "PENDING"
            and now >= result["expires_at"]
            and not result.get("data_gap")
        ):
            if expected / 1000 + interval_ms / 1000 > result["expires_at"]:
                result.update(
                    status="EXPIRED",
                    closed_at=result["expires_at"],
                    exit_reason="UNFILLED",
                )
        if result["status"] in {"OPEN", "PENDING"} and not result.get("data_gap"):
            due = expected / 1000 + interval_ms / 1000
            if due <= now and (
                result["status"] == "OPEN" or due <= result["expires_at"]
            ):
                result["data_gap"] = (
                    f"Missing price candle at {expected}; backfill required"
                )
        return _totals(result)
    except (ValueError, TypeError, KeyError, OverflowError, InvalidOperation) as exc:
        failed = copy.deepcopy(trade)
        failed["data_error"] = str(exc)
        return failed


def apply_funding(
    trade: dict,
    rates: list[dict],
    covered_until: float,
    *,
    covered_from: float | None = None,
    history_complete: bool = False,
) -> dict:
    """Apply known settlements, certifying coverage only with a range guarantee.

    Inclusive retrieval ranges are not inferred from returned settlements.
    Empty complete ranges are valid. Entry/exit timestamp events are excluded
    rather than assuming intrabar ordering. Funding amounts remain estimates.
    """
    result = copy.deepcopy(trade)
    if result.get("opened_at") is None:
        return result
    try:
        _number(covered_until)
        if history_complete and (
            covered_from is None
            or not _finite(covered_from)
            or not 0 <= covered_from <= covered_until
        ):
            raise ValueError(
                "Complete funding history requires an explicit valid range"
            )
        _number(result["entry"], True)
        _number(result["opened_at"])
        if result.get("last_bar") is None:
            raise ValueError("Funding requires a resolved price history cursor")
        horizon = (result["last_bar"] + result.get("interval_ms", INTERVAL_MS)) / 1000
        end = min(
            covered_until,
            horizon,
            result["closed_at"] if result["closed_at"] is not None else horizon,
        )
        events = {
            str(event["timestamp_ms"]): event
            for event in result.get("funding_events", [])
        }
        if result.get("funding_ids") and not events:
            raise ValueError(
                "Legacy funding totals lack event evidence; replay required"
            )
        parsed = {}
        for rate in rates:
            stamp_decimal = Decimal(str(rate["fundingRateTimestamp"]))
            if (
                not stamp_decimal.is_finite()
                or stamp_decimal != stamp_decimal.to_integral_value()
                or stamp_decimal < 0
            ):
                raise ValueError("Invalid funding timestamp")
            stamp = int(stamp_decimal)
            value = float(rate["fundingRate"])
            if not _finite(value) or isinstance(rate["fundingRate"], bool):
                raise ValueError("Nonfinite funding rate")
            if rate.get("symbol", result["symbol"]) != result["symbol"]:
                raise ValueError("Funding symbol mismatch")
            if stamp in parsed and parsed[stamp] != value:
                raise ValueError("Conflicting funding records")
            parsed[stamp] = value
        for stamp, value in sorted(parsed.items()):
            ts, key = stamp / 1000, str(stamp)
            if not result["opened_at"] < ts <= end or (
                result["closed_at"] is not None and ts >= result["closed_at"]
            ):
                continue
            if key in events and events[key]["rate"] != value:
                raise ValueError(
                    "Historical funding rate changed; reconcile explicitly"
                )
            qty = Decimal(str(result["qty"])) - sum(
                (
                    Decimal(str(f["qty"]))
                    for f in result["fills"]
                    if f["reason"] != "ENTRY" and f["time"] <= ts
                ),
                Decimal(0),
            )
            if not 0 <= qty <= Decimal(str(result["qty"])):
                raise ValueError("Invalid quantity at settlement")
            sign = 1 if result["side"] == "Buy" else -1
            amount = -sign * float(qty) * result["entry"] * value
            if not _finite(amount):
                raise ValueError("Nonfinite funding cashflow")
            events[key] = {
                "timestamp_ms": stamp,
                "rate": value,
                "qty": float(qty),
                "amount": amount,
            }
        result["funding_events"] = sorted(
            events.values(), key=lambda event: event["timestamp_ms"]
        )
        result["funding_ids"] = [
            str(event["timestamp_ms"]) for event in result["funding_events"]
        ]
        result["funding"] = sum(event["amount"] for event in result["funding_events"])
        coverage = list(result.get("funding_coverage", []))
        if history_complete and covered_from <= end:
            # Do not certify a range whose cashflows were intentionally ignored
            # because the price replay has not established future position size.
            coverage.append([covered_from, end])
        merged = []
        for start, finish in sorted(coverage):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], finish)
            else:
                merged.append([start, finish])
        result["funding_coverage"] = merged
        result["funding_complete"] = bool(
            result["status"] == "CLOSED"
            and any(
                start <= result["opened_at"] and finish >= result["closed_at"]
                for start, finish in merged
            )
        )
        result["funding_basis"] = "settled rate × remaining entry notional (estimate)"
        result.pop("funding_error", None)
        return _totals(result)
    except (ValueError, TypeError, KeyError, OverflowError, InvalidOperation) as exc:
        failed = copy.deepcopy(trade)
        failed.update(funding_complete=False, funding_error=str(exc))
        return failed


def summary(trades: list[dict], arm: str) -> dict:
    selected = [t for t in trades if t["arm"] == arm]
    closed = [t for t in selected if t["status"] == "CLOSED"]
    valid = [t for t in closed if _finite(t.get("net_pnl")) and not t.get("data_error")]
    wins = sum(t["net_pnl"] > 1e-9 for t in valid)
    losses = sum(t["net_pnl"] < -1e-9 for t in valid)
    return {
        "total": len(selected),
        "open": sum(t["status"] == "OPEN" for t in selected),
        "pending": sum(t["status"] == "PENDING" for t in selected),
        "closed": len(closed),
        "expired": sum(t["status"] == "EXPIRED" for t in selected),
        "wins": wins,
        "losses": losses,
        "breakeven": len(valid) - wins - losses,
        "wr": 100 * wins / len(valid) if valid else None,
        "net_pnl": sum(t["net_pnl"] for t in valid),
        "invalid_outcomes": len(closed) - len(valid),
        "funding_pending": sum(not t.get("funding_complete") for t in closed),
        "management_incomplete": sum(
            not t.get("management_complete") for t in selected
        ),
        "data_gaps": sum(
            bool(t.get("data_gap") or t.get("data_error")) for t in selected
        ),
        "cohort": (
            "all baseline candidates"
            if arm == "baseline_shadow"
            else (
                "AI-selected candidates"
                if arm == "ai_shadow"
                else "arm-specific simulated candidates"
            )
        ),
        "comparison_note": "Observational selection; differing eligibility, sizing or management confound arm outcomes",
    }


def comparison(trades: list[dict]) -> dict:
    """Match original opportunities, preserving actual decision/entry latency."""
    baseline = {t["opportunity_id"]: t for t in trades if t["arm"] == "baseline_shadow"}
    selected = {t["opportunity_id"]: t for t in trades if t["arm"] == "ai_shadow"}
    matches = []
    for key in sorted(baseline.keys() & selected.keys()):
        base, ai = baseline[key], selected[key]
        same_window = (
            _finite(base.get("entry_eligible_at"))
            and _finite(ai.get("entry_eligible_at"))
            and base["entry_eligible_at"] == ai["entry_eligible_at"]
        )
        same_sizing = all(
            base.get(k) == ai.get(k)
            for k in ("qty", "limit", "original_stop", "target1", "target2")
        )
        confounds = []
        if not same_window:
            confounds.append("entry_latency")
        if not same_sizing:
            confounds.append("portfolio_sizing_or_levels")
        if any(base.get(k) != ai.get(k) for k in ("cost_model", "management_mode")):
            confounds.append("simulation_assumptions")
        closed = all(
            t["status"] == "CLOSED" and _finite(t.get("net_pnl")) for t in (base, ai)
        )
        matches.append(
            {
                "counterfactual_id": key,
                "baseline_id": base["id"],
                "ai_id": ai["id"],
                "decision_latency_seconds": ai["created_at"] - base["created_at"],
                "same_entry_window": same_window,
                "confounds": confounds,
                "both_closed": closed,
                "net_delta": ai["net_pnl"] - base["net_pnl"] if closed else None,
                "funding_complete": base.get("funding_complete", False)
                and ai.get("funding_complete", False),
            }
        )
    paired = [m for m in matches if m["both_closed"]]
    return {
        "baseline_all": summary(trades, "baseline_shadow"),
        "ai_selected": summary(trades, "ai_shadow"),
        "matches": matches,
        "matched_count": len(matches),
        "paired_closed_count": len(paired),
        "baseline_unselected_count": len(baseline.keys() - selected.keys()),
        "ai_without_baseline_count": len(selected.keys() - baseline.keys()),
        "matched_net_delta": sum(m["net_delta"] for m in paired),
        "pure_filter_effect": False,
        "comparison_basis": (
            "selection_plus_latency_or_sizing"
            if any(m["confounds"] for m in matches)
            else "selection_on_observed_candidates"
        ),
        "note": "No causal AI uplift claim; retain earlier baseline and actual AI decisions, never backdate fills",
    }
