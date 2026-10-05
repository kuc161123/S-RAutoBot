"""Fail-closed deterministic sizing. Percent arguments use percentage points.

Funding alone is a fractional, already normalized 8h rate. This function is a
pure assessment, not an account reservation; the executor must reserve atomically.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import math
import time

from .engine import DATA_GRACE, DAY, H4, PLAN_LIFETIME, VERSION, stop_buffer_pct
from .models import Instrument, Opportunity

MONITORED_QUOTE_MAX_AGE = 60

HARD_CAPS = {
    "heat_pct": 4.0,
    "bucket_pct": 2.0,
    "max_positions": 6,
    "max_per_bucket": 2,
    "single_notional_pct": 25.0,
    "gross_notional_pct": 100.0,
}
PROFILES = {
    name: {"risk_pct": pct, **HARD_CAPS}
    for name, pct in (
        ("ultra_cautious", 0.10),
        ("cautious", 0.25),
        ("balanced", 0.50),
        ("aggressive", 0.75),
        ("extreme", 1.0),
    )
}
MAX_SPREAD_PCT = 0.20
BASE_COST_RATE = Decimal(
    "0.0015"
)  # Roundtrip fee/slippage allowance, plus spread/funding.


def _finite(value: object) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _positive(value: object) -> bool:
    return _finite(value) and value > 0


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _round(value: Decimal, step: Decimal, up: bool) -> Decimal:
    return (value / step).to_integral_value(
        rounding=ROUND_CEILING if up else ROUND_FLOOR
    ) * step


def _diagnostic_float(value: Decimal) -> float | None:
    """Keep optional diagnostics finite and safe for JSON/UI consumers."""
    if not value.is_finite():
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def assess(
    op: Opportunity,
    instrument: Instrument,
    equity: float,
    profile: str = "cautious",
    risk_pct: float | None = None,
    exposures: list[dict] | None = None,
    daily_loss_pct: float = 0,
    weekly_loss_pct: float = 0,
    drawdown_pct: float = 0,
    funding_rate_8h: float | None = None,
    spread_pct: float | None = None,
    now: float | None = None,
    regime_multiplier: float = 1.0,
    *,
    allow_resting: bool = False,
    allow_monitored: bool = False,
) -> dict:
    """Validate evidence, gates, conservative rounding, costs and account caps.

    ``equity`` is the caller's authorized sizing capital, not auto-discovered
    balance. All owned/external positions and outstanding reservations must be
    supplied. None means unknown account state; [] means a reconciled flat book.
    Rejected assessments always carry zero executable quantity, risk and notional.

    ``allow_resting=True`` is an explicit offline/shadow research gate for
    separately identified Tier-1 midpoint zone orders. It does not synthesize
    a confirmation trigger. Stops, targets, costs, sizing and all R:R/portfolio
    limits still apply. ExecutionManager separately rejects this entry style;
    the production runtime never enables this flag.

    ``allow_monitored=True`` separately permits offline Tier-1 zone-arrival
    observations. They require a fresh observation and assessment, the original
    plan deadline, and no confirmation evidence. The quote is an adverse cap
    within 5bp of the observation; its executable tick must respect that cap.
    Both observation and rounded entry must be inside the exact frozen zone.
    All normal costs, R:R and account limits still apply. Neither research flag
    authorizes exchange submission or is enabled by the production runtime.

    R:R diagnostics are finite floats or None when unavailable: rr_target1,
    rr_blended (nominal 50/50), rr_actual_split (lot-rounded sizing hurdle),
    rr_required_target1 and rr_required_blended. A single lot retains the nominal
    50/50 hurdle; zero lots leave rr_actual_split unknown. Thresholds are known
    for eligible tiers. Price diagnostics require the existing evidence, stop,
    zone and trigger gates to pass; split diagnostics use the proposed quantity
    even when another gate rejects it.

    rr_entry_bound is a hypothetical tick-rounded entry satisfying only the T1
    and nominal 50/50 hurdles with the same rounded stop, targets and cost rate.
    rr_entry_relation is at_or_below for longs, at_or_above for shorts, or None
    without a valid bound. rr_zone_compatible is whether that half-line overlaps
    the allowed entry zone (1% directional tolerance for confirmed entries only) at
    a tick with valid stop/target ordering, or None without a valid bound.
    These fields never change entry, sizing or permission. The bound does not
    guarantee the odd-lot hurdle, sizing, a new valid trigger, funding or account
    caps; all checks must pass on a fresh assessment before any trade.
    """
    result = dict(
        allowed=False,
        reasons=[],
        qty=0.0,
        qty_step=0.0,
        risk_cash=0.0,
        risk_pct=0.0,
        notional=0.0,
        entry=0.0,
        stop=0.0,
        target1=0.0,
        target2=0.0,
        rr_target1=None,
        rr_blended=None,
        rr_actual_split=None,
        rr_required_target1=None,
        rr_required_blended=None,
        rr_entry_bound=None,
        rr_entry_relation=None,
        rr_zone_compatible=None,
    )
    reasons = result["reasons"]
    if not isinstance(op, Opportunity) or not isinstance(instrument, Instrument):
        reasons.append("INVALID_CONTRACT")
        return result
    resting = (
        isinstance(op.evidence, dict)
        and op.evidence.get("entry_style") == "resting_limit"
    )
    monitored = isinstance(op.evidence, dict) and (
        op.evidence.get("entry_style") == "monitored_zone"
        or op.evidence.get("trigger_kind") in ("ZONE_ARRIVAL", "ZONE_WATCH")
    )
    if resting and allow_resting is not True:
        reasons.append("RESTING_RESEARCH_ONLY")
    if resting and op.tier != 1:
        reasons.append("RESTING_TIER1_ONLY")
    if monitored and allow_monitored is not True:
        reasons.append("MONITORED_RESEARCH_ONLY")
    if monitored and (op.tier != 1 or isinstance(op.tier, bool)):
        reasons.append("MONITORED_TIER1_ONLY")
    if now is None:
        now = time.time()
    if not _finite(now) or now < 0:
        reasons.append("INVALID_TIME")
    if not _positive(equity):
        reasons.append("INVALID_EQUITY")
    if not _finite(regime_multiplier) or not 0 <= regime_multiplier <= 1:
        reasons.append("INVALID_REGIME_MULTIPLIER")
    elif regime_multiplier == 0:
        reasons.append("REGIME_BLOCKED")
    if not isinstance(profile, str) or profile not in PROFILES:
        reasons.append("UNKNOWN_PROFILE")
    else:
        requested = PROFILES[profile]["risk_pct"] if risk_pct is None else risk_pct
        if not _finite(requested) or not 0.05 <= requested <= 1:
            reasons.append("INVALID_RISK_OVERRIDE")
        else:
            result["risk_pct"] = requested
    if op.side not in ("Buy", "Sell"):
        reasons.append("INVALID_SIDE")
    if (
        not isinstance(op.symbol, str)
        or not op.symbol.strip()
        or not isinstance(instrument.symbol, str)
        or op.symbol.upper() != instrument.symbol.upper()
    ):
        reasons.append("SYMBOL_MISMATCH")
    if not isinstance(op.bucket, str) or not op.bucket.strip():
        reasons.append("UNKNOWN_BUCKET")
    if op.tier not in (1, 2) or isinstance(op.tier, bool):
        reasons.append("TIER_NOT_ELIGIBLE")
    else:
        rr1, blended_min = (
            (Decimal("1.5"), Decimal("2.5"))
            if op.tier == 1
            else (Decimal("1.8"), Decimal("3"))
        )
        result.update(
            rr_required_target1=float(rr1), rr_required_blended=float(blended_min)
        )
    if not isinstance(op.setup, str) or op.setup not in {
        "1",
        "1S",
        "2",
        "2S",
        "2X",
        "2XS",
        "5L",
        "5S",
    }:
        reasons.append("UNSUPPORTED_SETUP")
    elif (op.setup in {"1", "2", "2X", "5L"}) != (op.side == "Buy"):
        reasons.append("SETUP_SIDE_MISMATCH")
    if op.setup in ("2X", "2XS") and op.tier != 1:
        reasons.append("TIER_NOT_ELIGIBLE")
    if op.state != "READY":
        reasons.append("OPPORTUNITY_NOT_READY")
    prices = (
        op.entry,
        op.stop,
        op.target1,
        op.target2,
        op.invalidation,
    )
    if not (resting or monitored):
        prices += (op.confirmation,)
    if not all(_positive(x) for x in prices):
        reasons.append("INVALID_PRICE")
    else:
        result.update(
            entry=op.entry, stop=op.stop, target1=op.target1, target2=op.target2
        )
    fields = (
        instrument.qty_step,
        instrument.min_qty,
        instrument.max_qty,
        instrument.tick_size,
        instrument.min_notional,
        instrument.max_leverage,
        instrument.funding_interval_minutes,
    )
    if not all(_positive(v) for v in fields) or instrument.min_qty > instrument.max_qty:
        reasons.append("INVALID_INSTRUMENT")
    else:
        result["qty_step"] = instrument.qty_step
    losses = (daily_loss_pct, weekly_loss_pct, drawdown_pct)
    if not all(_finite(x) and x >= 0 for x in losses):
        reasons.append("INVALID_LOSS_DATA")
    else:
        if daily_loss_pct >= 2:
            reasons.append("DAILY_LOSS_HALT")
        if weekly_loss_pct >= 4:
            reasons.append("WEEKLY_LOSS_HALT")
        if drawdown_pct >= 12:
            reasons.append("DRAWDOWN_HALT")
    if not _finite(funding_rate_8h):
        reasons.append("FUNDING_UNKNOWN")
    elif (
        op.side == "Buy"
        and funding_rate_8h > 0.0003
        or op.side == "Sell"
        and funding_rate_8h < -0.0003
    ):
        reasons.append("ADVERSE_FUNDING")
    if not _finite(spread_pct) or spread_pct < 0:
        reasons.append("SPREAD_UNKNOWN")
    elif spread_pct > MAX_SPREAD_PCT:
        reasons.append("SPREAD_TOO_WIDE")
    valid_exposures = []
    if not isinstance(exposures, list):
        reasons.append("EXPOSURES_UNKNOWN")
    else:
        for item in exposures:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("symbol"), str)
                or not item["symbol"].strip()
                or not isinstance(item.get("bucket"), str)
                or not item["bucket"].strip()
                or not _finite(item.get("risk_cash"))
                or item["risk_cash"] < 0
                or not _positive(item.get("notional"))
            ):
                reasons.append("EXPOSURE_RISK_OR_NOTIONAL_UNKNOWN")
                continue
            valid_exposures.append(item)
            if (
                isinstance(op.symbol, str)
                and item["symbol"].strip().upper() == op.symbol.strip().upper()
            ):
                reasons.append("SYMBOL_ALREADY_EXPOSED")
    ev = op.evidence
    if not isinstance(ev, dict):
        reasons.append("MISSING_EVIDENCE")
        ev = {}
    if (
        ev.get("engine_version") != VERSION
        or ev.get("structural_valid") is not True
        or ev.get("data_valid") is not True
        or not ev.get("evidence_id")
        or not ev.get(
            "plan_evidence_id" if resting or monitored else "trigger_evidence_id"
        )
        or ev.get("terminal_status") is not None
    ):
        reasons.append("UNVERIFIED_EVIDENCE")
    if ev.get("evidence_id") != op.id:
        reasons.append("EVIDENCE_ID_MISMATCH")
    if monitored:
        if (
            ev.get("entry_style") != "monitored_zone"
            or ev.get("trigger_kind") != "ZONE_ARRIVAL"
            or any(
                ev.get(k) is not None
                for k in ("trigger_closed_at", "trigger_price", "trigger_evidence_id")
            )
            or not isinstance(ev.get("plan_evidence_id"), str)
            or not ev["plan_evidence_id"].strip()
            or not _positive(ev.get("observation_price"))
            or not _positive(ev.get("quote_limit"))
            or op.entry != ev.get("quote_limit")
            or ev.get("entry_zone_low") != ev.get("zone_low")
            or ev.get("entry_zone_high") != ev.get("zone_high")
        ):
            reasons.append("INVALID_MONITORED_EVIDENCE")
    elif resting:
        if ev.get("trigger_kind") != "ZONE_LIMIT" or any(
            ev.get(k) is not None
            for k in ("trigger_closed_at", "trigger_price", "trigger_evidence_id")
        ):
            reasons.append("INVALID_RESTING_EVIDENCE")
    elif ev.get("trigger_kind") not in ("SWING_BREAK", "BREAKOUT_RETEST"):
        reasons.append("MISSING_TRIGGER_KIND")
    start_key = (
        "monitored_started_at"
        if monitored
        else "order_armed_at" if resting else "trigger_closed_at"
    )
    timestamps = [
        op.created_at,
        op.expires_at,
        ev.get(start_key),
        ev.get("daily_closed_at"),
        ev.get("execution_closed_at"),
        ev.get("as_of"),
    ]
    if monitored:
        timestamps.append(ev.get("observation_at"))
    if not all(_finite(x) and x >= 0 for x in timestamps) or not _finite(now):
        reasons.append("MISSING_EVIDENCE_TIME")
    else:
        created, expires, trigger, daily, execution, as_of = timestamps[:6]
        if monitored:
            observation = timestamps[6]
            timing_valid = (
                created == trigger <= observation <= as_of <= now
                and created <= execution <= observation
                and daily <= observation
                and created < expires <= created + PLAN_LIFETIME
            )
            if now - observation > MONITORED_QUOTE_MAX_AGE:
                reasons.append("STALE_OBSERVATION")
        elif resting:
            timing_valid = (
                created == trigger <= execution <= as_of <= now
                and trigger < expires <= created + 2 * DAY
            )
        else:
            timing_valid = (
                created < trigger <= execution <= as_of <= now
                and trigger < expires <= trigger + 2 * H4
            )
        if not timing_valid or daily > as_of:
            reasons.append("INVALID_EVIDENCE_TIME")
        if now >= expires:
            reasons.append("EXPIRED")
        if now - daily > DAY + DATA_GRACE or now - execution > H4 + DATA_GRACE:
            reasons.append("STALE_DATA")
        # Current assessment must not revive an old READY snapshot after price
        # could have moved on an intervening closed execution bar.
        if now - as_of > DATA_GRACE:
            reasons.append("STALE_ASSESSMENT")
    zone_values = [
        ev.get("zone_low"),
        ev.get("zone_high"),
        ev.get("entry_zone_low"),
        ev.get("entry_zone_high"),
    ]
    if not (resting or monitored):
        zone_values.append(ev.get("trigger_price"))
    if not all(_positive(x) for x in zone_values):
        reasons.append("MISSING_ENTRY_EVIDENCE")
    elif zone_values[0] > zone_values[1] or zone_values[2] > zone_values[3]:
        reasons.append("INVALID_ZONE")
    multiplier = ev.get("risk_multiplier", 1.0)
    if not _positive(multiplier) or multiplier > 1:
        reasons.append("INVALID_RISK_MULTIPLIER")
    if reasons:
        result["reasons"] = list(dict.fromkeys(reasons))
        return result

    with localcontext() as context:
        context.prec = 50
        d = Decimal(1 if op.side == "Buy" else -1)
        tick, step = _decimal(instrument.tick_size), _decimal(instrument.qty_step)
        # A monitored quote is a maximum adverse limit: round into that cap,
        # never increase the permitted slippage to reach an executable tick.
        entry = _round(_decimal(op.entry), tick, d < 0 if monitored else d > 0)
        stop = _round(_decimal(op.stop), tick, d < 0)
        t1 = _round(_decimal(op.target1), tick, d < 0)
        t2 = _round(_decimal(op.target2), tick, d < 0)
        invalidation = _decimal(op.invalidation)
        result.update(
            entry=float(entry), stop=float(stop), target1=float(t1), target2=float(t2)
        )
        if min(entry, stop, t1, t2) <= 0 or not (
            d * (entry - stop) > 0
            and d * (t1 - entry) > 0
            and d * (t2 - t1) > 0
            and d * (entry - invalidation) > 0
        ):
            reasons.append("INVALID_ROUNDED_PRICE_ORDER")
        required_stop = invalidation * (
            1 - d * _decimal(stop_buffer_pct(op.symbol, op.bucket))
        )
        if d * (stop - required_stop) > 0:
            reasons.append("STOP_BUFFER_TOO_SMALL")
        low, high = _decimal(ev["entry_zone_low"]), _decimal(ev["entry_zone_high"])
        if not (resting or monitored) and d > 0:
            high *= Decimal("1.01")
        elif not (resting or monitored):
            low *= Decimal("0.99")
        if not low <= entry <= high:
            reasons.append("ENTRY_OUTSIDE_ZONE")
        if monitored:
            observation_price = _decimal(ev["observation_price"])
            quote_limit = _decimal(ev["quote_limit"])
            max_adverse = observation_price * Decimal("0.0005")
            # Accept the analyzer's floating-point 5bp formula as well as the
            # exact decimal boundary, without a general price/zone epsilon.
            float_cap = ev["observation_price"] * (1 + float(d) * 0.0005)
            if _finite(float_cap):
                max_adverse = max(
                    max_adverse, d * (_decimal(float_cap) - observation_price)
                )
            if not 0 <= d * (quote_limit - observation_price) <= max_adverse:
                reasons.append("INVALID_MONITORED_EVIDENCE")
            if not low <= observation_price <= high:
                reasons.append("OBSERVATION_OUTSIDE_ZONE")
        elif not resting:
            trigger_price = _decimal(ev["trigger_price"])
            if d * (trigger_price - _decimal(op.confirmation)) <= 0:
                reasons.append("TRIGGER_NOT_BEYOND_CONFIRMATION")
            if d * (entry - trigger_price) > trigger_price * Decimal("0.005"):
                reasons.append("CHASING")
        if reasons:
            return result
        cost_rate = (
            BASE_COST_RATE
            + _decimal(spread_pct) / 100
            + max(d * _decimal(funding_rate_8h), Decimal(0))
        )
        cost = entry * cost_rate
        unit_risk = abs(entry - stop) + cost
        reward1, reward2 = d * (t1 - entry) - cost, d * (t2 - entry) - cost
        result.update(
            rr_target1=_diagnostic_float(reward1 / unit_risk),
            rr_blended=_diagnostic_float((reward1 + reward2) / (2 * unit_risk)),
        )
        # Diagnostic only: solve both nominal inequalities with cost = entry*k.
        # Round toward a better entry so the displayed bound remains conservative.
        cost_factor = 1 + d * cost_rate
        if cost_factor.is_finite() and cost_factor > 0:
            bounds = [
                (target + required * stop) / ((1 + required) * cost_factor)
                for target, required in ((t1, rr1), ((t1 + t2) / 2, blended_min))
            ]
            bound = _round(min(bounds) if d > 0 else max(bounds), tick, d < 0)
            bound_float = _diagnostic_float(bound)
            if (
                bound_float is not None
                and bound_float > 0
                and d * (bound - stop) > 0
                and d * (t1 - bound) > 0
            ):
                zone_low = max(_round(low, tick, True), tick)
                zone_high = _round(high, tick, False)
                if d > 0:
                    zone_low = max(zone_low, stop + tick)
                    zone_high = min(zone_high, bound, t1 - tick)
                else:
                    zone_low = max(zone_low, bound, t1 + tick)
                    zone_high = min(zone_high, stop - tick)
                result.update(
                    rr_entry_bound=bound_float,
                    rr_entry_relation="at_or_below" if d > 0 else "at_or_above",
                    rr_zone_compatible=zone_low <= zone_high,
                )
        if reward1 / unit_risk < rr1:
            reasons.append("RR_TARGET1")
        if (reward1 + reward2) / (2 * unit_risk) < blended_min:
            reasons.append("RR_BLENDED")
        effective_pct = _decimal(result["risk_pct"])
        effective_pct *= _decimal(regime_multiplier)
        effective_pct *= Decimal("0.5") if op.tier == 2 else 1
        effective_pct *= min(
            _decimal(multiplier), Decimal("0.5") if op.setup in {"2X", "2XS"} else 1
        )
        if drawdown_pct >= 8:
            effective_pct *= Decimal("0.5")
        result["risk_pct"] = float(effective_pct)
        capital = _decimal(equity)
        budget = capital * effective_pct / 100
        # Profile caps can only tighten absolute caps, never weaken them.
        caps = {
            key: _decimal(min(value, PROFILES[profile].get(key, value)))
            for key, value in HARD_CAPS.items()
        }
        bucket = [
            p
            for p in valid_exposures
            if p["bucket"].strip().lower() == op.bucket.strip().lower()
        ]
        heat = sum((_decimal(p["risk_cash"]) for p in valid_exposures), Decimal(0))
        bucket_heat = sum((_decimal(p["risk_cash"]) for p in bucket), Decimal(0))
        gross = sum((_decimal(p["notional"]) for p in valid_exposures), Decimal(0))
        # Each reservation/position consumes a slot; do not net hedges or debts.
        if len(valid_exposures) >= caps["max_positions"]:
            reasons.append("POSITION_CAP")
        if len(bucket) >= caps["max_per_bucket"]:
            reasons.append("BUCKET_POSITION_CAP")
        if heat + budget > capital * caps["heat_pct"] / 100:
            reasons.append("HEAT_CAP")
        if bucket_heat + budget > capital * caps["bucket_pct"] / 100:
            reasons.append("BUCKET_HEAT_CAP")
        qty = _round(budget / unit_risk, step, False)
        actual_risk, notional = qty * unit_risk, qty * entry
        if qty <= 0 or qty < _decimal(instrument.min_qty):
            reasons.append("QUANTITY_BELOW_MINIMUM")
        if qty > _decimal(instrument.max_qty):
            reasons.append("QUANTITY_ABOVE_MAXIMUM")
        if notional < _decimal(instrument.min_notional):
            reasons.append("NOTIONAL_BELOW_MINIMUM")
        if notional > capital * caps["single_notional_pct"] / 100:
            reasons.append("SINGLE_NOTIONAL_CAP")
        if gross + notional > capital * caps["gross_notional_pct"] / 100:
            reasons.append("GROSS_NOTIONAL_CAP")
        if qty > 0:
            units = (qty / step).to_integral_value()
            # Odd lot TP1 rounds up. One lot has no partial exit, retaining the
            # nominal 50/50 hurdle rather than gaining an easier all-T2 hurdle.
            weight1 = (
                (units / 2).to_integral_value(rounding=ROUND_CEILING) / units
                if units > 1
                else Decimal("0.5")
            )
            actual_rr = (weight1 * reward1 + (1 - weight1) * reward2) / unit_risk
            result["rr_actual_split"] = _diagnostic_float(actual_rr)
            if actual_rr < blended_min:
                reasons.append("RR_ROUNDED_SPLIT")
        if (
            actual_risk > budget
            or heat + actual_risk > capital * caps["heat_pct"] / 100
        ):
            reasons.append("ROUNDED_RISK_CAP")
        if bucket_heat + actual_risk > capital * caps["bucket_pct"] / 100:
            reasons.append("ROUNDED_BUCKET_HEAT_CAP")
        if reasons:
            result["reasons"] = list(dict.fromkeys(reasons))
            return result
        result.update(
            allowed=True,
            qty=float(qty),
            risk_cash=float(actual_risk),
            notional=float(notional),
        )
    return result
