"""Causal, numeric Apex candidates. See docs/apex/ENGINE_RULES.txt for scope.

No exchange calls or mutable global state. Replaying the same history origin is
intentional: a truncated warmup is a different evidence set, not a recount.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math

from .models import Candle, Opportunity

DAY = 86_400
H4 = 14_400
VERSION = "apex-numeric-1.2"
ATR_PERIOD = 14
PLAN_LIFETIME = 30 * DAY
DATA_GRACE = 300


def _finite(value: object) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _id(value: object) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()[:24]


def stop_buffer_pct(symbol: str, bucket: str) -> float:
    """Fractional hard-stop buffer from charter 7.3 (not percent points)."""
    if symbol.upper() in {"BTCUSDT", "ETHUSDT", "BNBUSDT", "BTC", "ETH", "BNB"}:
        return 0.01
    if bucket.lower() in {"b4", "memes", "meme", "base", "base_coins"}:
        return 0.025
    return 0.015


def _closed(candles: list[Candle], seconds: int, now: float) -> list[Candle]:
    """Ignore future/in-progress OHLC, reject malformed closed history and gaps."""
    result = []
    for candle in candles:
        if (
            not isinstance(candle, Candle)
            or not isinstance(candle.open_time, int)
            or isinstance(candle.open_time, bool)
        ):
            raise ValueError("invalid candle timestamp")
        if candle.open_time / 1000 + seconds > now:
            continue
        if candle.open_time < 0 or candle.open_time % (seconds * 1000):
            raise ValueError("unaligned candle")
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if (
            not all(_finite(x) for x in values)
            or min(values[:4]) <= 0
            or candle.volume < 0
        ):
            raise ValueError("invalid OHLCV")
        if (
            not candle.low
            <= min(candle.open, candle.close)
            <= max(candle.open, candle.close)
            <= candle.high
        ):
            raise ValueError("inconsistent OHLC")
        if result and candle.open_time - result[-1].open_time != seconds * 1000:
            raise ValueError("unordered, duplicate or missing candle")
        result.append(candle)
    return result


def atr(candles: list[Candle], period: int = ATR_PERIOD) -> list[float | None]:
    """Wilder ATR, first value seeded with the first period true ranges."""
    if not isinstance(period, int) or period < 1:
        raise ValueError("positive ATR period required")
    result: list[float | None] = []
    seed = 0.0
    previous = None
    for i, bar in enumerate(candles):
        tr = bar.high - bar.low
        if i:
            tr = max(
                tr,
                abs(bar.high - candles[i - 1].close),
                abs(bar.low - candles[i - 1].close),
            )
        if not math.isfinite(tr):
            raise ValueError("nonfinite ATR")
        seed += tr if i < period else 0
        if i == period - 1:
            previous = seed / period
        elif i >= period:
            previous = previous + (tr - previous) / period
        result.append(previous)
    return result


@dataclass(frozen=True)
class Pivot:
    id: str
    kind: str
    price: float
    open_time: int
    available_at: float
    index: int
    atr: float


def confirmed_zigzag(
    candles: list[Candle],
    timeframe_seconds: int = DAY,
    now: float | None = None,
    atr_period: int = ATR_PERIOD,
    atr_multiple: float = 2.0,
) -> list[Pivot]:
    """Confirmed extrema only, using prior-bar ATR and a later closing reversal.

    An extremum made on this bar cannot be confirmed by this same bar. Equal
    extrema retain the earliest occurrence. Once appended, pivots never move.
    """
    if (
        timeframe_seconds not in (DAY, H4)
        or not _finite(atr_multiple)
        or atr_multiple <= 0
    ):
        raise ValueError("invalid zigzag configuration")
    if now is None:
        now = candles[-1].open_time / 1000 + timeframe_seconds if candles else 0
    if not _finite(now) or now < 0:
        raise ValueError("invalid analysis time")
    bars = _closed(candles, timeframe_seconds, now)
    values = atr(bars, atr_period)
    pivots: list[Pivot] = []
    direction = 0
    high = low = None
    hi = lo = 0
    for i in range(atr_period - 1, len(bars)):
        bar = bars[i]
        if high is None:
            high, low, hi, lo = bar.high, bar.low, i, i
            continue
        if bar.high > high:
            high, hi = bar.high, i
        if bar.low < low:
            low, lo = bar.low, i
        previous_atr = values[i - 1]
        if previous_atr is None or previous_atr <= 0:
            continue
        threshold = atr_multiple * previous_atr
        low_confirmed = lo < i and bar.close - low >= threshold
        high_confirmed = hi < i and high - bar.close >= threshold
        kind = None
        if direction == 0:
            # An outside range with both tests true does not resolve its order.
            if low_confirmed != high_confirmed:
                kind = "low" if low_confirmed else "high"
        elif direction > 0 and high_confirmed:
            kind = "high"
        elif direction < 0 and low_confirmed:
            kind = "low"
        if kind:
            price, index = (low, lo) if kind == "low" else (high, hi)
            available = bar.open_time / 1000 + timeframe_seconds
            identity = [
                VERSION,
                timeframe_seconds,
                atr_period,
                atr_multiple,
                kind,
                bars[index].open_time,
                price,
                available,
                previous_atr,
            ]
            pivots.append(
                Pivot(
                    _id(identity),
                    kind,
                    price,
                    bars[index].open_time,
                    available,
                    index,
                    previous_atr,
                )
            )
            direction = 1 if kind == "low" else -1
            # Retain the opposite extreme already observed during confirmation
            # lag. Exclude the pivot's own bar: its intrabar order is unknown.
            if kind == "high":
                lo = min(range(index + 1, i + 1), key=lambda j: bars[j].low)
                low = bars[lo].low
                high, hi = bar.high, i
            else:
                hi = max(range(index + 1, i + 1), key=lambda j: bars[j].high)
                high = bars[hi].high
                low, lo = bar.low, i
    return pivots


def validate_impulse(pivots: list[Pivot], complete: bool = True) -> list[str]:
    """Reasons a standard 0..5 impulse (or 0..2/3/4 prefix) is invalid."""
    if (complete and len(pivots) != 6) or (
        not complete and len(pivots) not in (3, 4, 5, 6)
    ):
        return ["PIVOT_COUNT"]
    if any(not _finite(p.price) or p.price <= 0 for p in pivots):
        return ["INVALID_PRICE"]
    if pivots[0].kind not in ("low", "high"):
        return ["PIVOT_KIND"]
    direction = 1 if pivots[0].kind == "low" else -1
    x = [direction * p.price for p in pivots]
    reasons = []
    for i, p in enumerate(pivots):
        expected = "low" if (i % 2 == 0) == (direction > 0) else "high"
        if p.kind != expected:
            reasons.append("NOT_ALTERNATING")
        if (
            not _finite(p.open_time)
            or p.open_time < 0
            or not _finite(p.available_at)
            or p.available_at <= p.open_time / 1000
        ):
            reasons.append("UNCONFIRMED_PIVOT")
        if i and (
            p.open_time <= pivots[i - 1].open_time
            or p.available_at <= pivots[i - 1].available_at
        ):
            reasons.append("PIVOT_ORDER")
        if i and (x[i] - x[i - 1]) * (1 if i % 2 else -1) <= 0:
            reasons.append("NONPOSITIVE_WAVE")
    if x[2] <= x[0]:
        reasons.append("WAVE2_RETRACE_ORIGIN")
    if len(x) >= 4 and x[3] <= x[1]:
        reasons.append("WAVE3_NO_NEW_EXTREME")
    if len(x) >= 5 and x[4] <= x[1]:
        reasons.append("WAVE4_OVERLAP")
    if len(x) == 6:
        if x[5] <= x[3]:
            reasons.append("TRUNCATED_WAVE5_UNSUPPORTED")
        if x[3] - x[2] < min(x[1] - x[0], x[5] - x[4]):
            reasons.append("WAVE3_SHORTEST")
    return list(dict.fromkeys(reasons))


@dataclass(frozen=True)
class _Plan:
    anchors: tuple[Pivot, ...]
    setup: str
    direction: int
    created: float
    zone_low: float
    zone_high: float
    invalidation: float
    target1: float
    target2: float
    confirmation: float
    history_start: int
    risk_multiplier: float = 1.0


def _plans(bars: list[Candle], pivots: list[Pivot]) -> list[_Plan]:
    plans = []

    def add(
        anchors,
        setup,
        d,
        created,
        lo,
        hi,
        invalid,
        t1,
        t2,
        confirm=0,
        start=None,
        multiplier=1,
    ):
        prices = [d * lo, d * hi]
        zone_low, zone_high = min(prices), max(prices)
        values = [zone_low, zone_high, d * invalid, d * t1, d * t2]
        if lo > hi or not all(_finite(x) and x > 0 for x in values):
            return
        plans.append(
            _Plan(
                tuple(anchors),
                setup,
                d,
                created,
                zone_low,
                zone_high,
                d * invalid,
                d * t1,
                d * t2,
                d * confirm,
                anchors[-1].index + 1 if start is None else start,
                multiplier,
            )
        )

    for end in range(3, len(pivots) + 1):
        p = pivots[end - 3 : end]
        d = 1 if p[0].kind == "low" else -1
        x = [d * v.price for v in p]
        if not validate_impulse(p, complete=False):
            # Freeze "move off W2" at the first closed daily breakout known
            # after W2 confirms. No future running maximum enters this plan.
            for bar in bars[p[-1].index + 1 :]:
                closed = bar.open_time / 1000 + DAY
                if closed < p[-1].available_at:
                    continue
                if d * bar.close <= x[2]:
                    break
                if d * bar.close > x[1]:
                    extreme = bar.high if d > 0 else -bar.low
                    move, w1 = extreme - x[2], x[1] - x[0]
                    add(
                        p,
                        "2" if d > 0 else "2S",
                        d,
                        closed,
                        extreme - 0.618 * move,
                        extreme - 0.5 * move,
                        x[2],
                        x[2] + w1,
                        x[2] + 1.618 * w1,
                        x[1],
                    )
                    break
        if end >= 4:
            p = pivots[end - 4 : end]
            d = 1 if p[0].kind == "low" else -1
            x = [d * v.price for v in p]
            w1, move = x[1] - x[0], x[3] - x[2]
            if not validate_impulse(p, complete=False) and move >= 1.618 * w1:
                cutoff = p[-1].available_at
                reached = max(
                    (
                        d * (b.high if d > 0 else b.low)
                        for b in bars[p[2].index :]
                        if b.open_time / 1000 + DAY <= cutoff
                    ),
                    default=x[3],
                )
                targets = [
                    x[2] + ratio * w1
                    for ratio in (1.618, 2.618, 4.236)
                    if x[2] + ratio * w1 > reached
                ]
                if len(targets) >= 2:
                    add(
                        p,
                        "2X" if d > 0 else "2XS",
                        d,
                        cutoff,
                        max(x[3] - 0.5 * move, x[1] + 0.01 * abs(x[1])),
                        x[3] - 0.382 * move,
                        x[1],
                        *targets[:2],
                        multiplier=0.5,
                    )
        if end >= 6:
            p = pivots[end - 6 : end]
            if not validate_impulse(p):
                d = 1 if p[0].kind == "low" else -1
                x = [d * v.price for v in p]
                length = x[5] - x[0]
                lo, hi = x[5] - 0.5 * length, x[5] - 0.382 * length
                if lo <= x[4] <= hi:
                    lo = x[4]  # Use internal W4 as the deep edge, never widen.
                add(
                    p,
                    "1" if d > 0 else "1S",
                    d,
                    p[-1].available_at,
                    lo,
                    hi,
                    min(x[5] - 0.618 * length, x[4]),
                    x[5],
                    (lo + hi) / 2 + length,
                )
        if end >= 8:
            p = pivots[end - 8 : end]
            if validate_impulse(p[:6]):
                continue
            original = 1 if p[0].kind == "low" else -1
            x = [original * v.price for v in p]
            a = x[5] - x[6]
            if a <= 0 or not x[6] < x[7] < x[5]:
                continue
            lo, hi = x[6] + 0.5 * a, x[6] + 0.618 * a
            broken = any(
                original * b.close < x[4] for b in bars[p[5].index + 1 : p[6].index + 1]
            )
            if broken and lo <= x[7] <= hi:
                # Reverse orientation: B is a confirmed frozen endpoint.
                add(
                    p,
                    "5S" if original > 0 else "5L",
                    -original,
                    p[-1].available_at,
                    -hi,
                    -lo,
                    -x[5],
                    -x[7] + a,
                    -x[7] + 1.618 * a,
                )
    return plans


def _in_zone(price: float, low: float, high: float, direction: int) -> bool:
    return low <= price <= high * 1.01 if direction > 0 else low * 0.99 <= price <= high


def _correction_start(plan: _Plan) -> int:
    # Bound the TRIGGER swing to the current approach, not its predecessor.
    # A Type2 add-zone approach begins at the frozen breakout leg, not at an
    # unrelated old W2 correction. Type5 approaches begin at A before B exists.
    if plan.setup in {"2", "2S"}:
        return int((plan.created - DAY) * 1000)
    index = -2 if plan.setup in {"5S", "5L"} else -1
    return plan.anchors[index].open_time


def _evaluate(
    symbol: str,
    plan: _Plan,
    daily: list[Candle],
    execution: list[Candle],
    swings: list[Pivot],
    now: float,
    tier: int,
    bucket: str,
) -> Opportunity:
    d = plan.direction
    buffer = stop_buffer_pct(symbol, bucket)
    stop = plan.invalidation * (1 - d * buffer)
    frozen = {
        "engine_version": VERSION,
        "anchors": [asdict(p) for p in plan.anchors],
        "setup": plan.setup,
        "side": "Buy" if d > 0 else "Sell",
        "zone_low": plan.zone_low,
        "zone_high": plan.zone_high,
        "invalidation": plan.invalidation,
        "target1": plan.target1,
        "target2": plan.target2,
        "created_at": plan.created,
        "risk_multiplier": plan.risk_multiplier,
        "daily_evidence_id": _id(
            [asdict(b) for b in daily if b.open_time / 1000 + DAY <= plan.created]
        ),
    }
    evidence_id = _id([symbol, tier, bucket, frozen])
    evidence = dict(
        frozen,
        evidence_id=evidence_id,
        structural_valid=True,
        data_valid=True,
        daily_closed_at=daily[-1].open_time / 1000 + DAY,
        execution_closed_at=execution[-1].open_time / 1000 + H4 if execution else None,
        as_of=now,
        stop_buffer_pct=buffer,
        trigger_closed_at=None,
        trigger_price=None,
        trigger_evidence_id=None,
        entry_zone_low=plan.zone_low,
        entry_zone_high=plan.zone_high,
        terminal_status=None,
    )
    entry = (plan.zone_low + plan.zone_high) / 2
    confirmation = plan.confirmation
    expires = plan.created + PLAN_LIFETIME
    state, reason = "WAIT", "AWAIT_CLOSED_4H_TRIGGER"
    touched = False
    breakout = None
    retest_done = False
    correction_start = _correction_start(plan)
    evidence["correction_started_at"] = correction_start / 1000
    # This bar is comparison evidence, not a trigger. It was already closed
    # when the plan became available. Dropping it would suppress a valid cross
    # on the first eligible later bar (and its subsequent Type 2 retest).
    previous = next(
        (
            bar
            for bar in reversed(execution)
            if bar.open_time / 1000 + H4 <= plan.created
        ),
        None,
    )
    # Chronological merge prevents a later failure from rewriting the first
    # terminal transition. Daily events win ties (conservative intrabar order).
    events = [(b.open_time / 1000 + DAY, 0, b) for b in daily[plan.history_start :]]
    events += [
        (b.open_time / 1000 + H4, 1, b)
        for b in execution
        if b.open_time / 1000 >= plan.created
    ]
    events.sort(key=lambda e: (e[0], e[1]))

    def invalidate(code: str, at: float):
        nonlocal state, reason
        state, reason = "INVALID", code
        evidence["terminal_status"] = code
        evidence["terminal_at"] = max(at, plan.created)

    for closed, kind, bar in events:
        if closed >= expires:
            invalidate("EXPIRED", expires)
            break
        if (d > 0 and bar.low <= stop) or (d < 0 and bar.high >= stop):
            invalidate("HARD_STOP_BREACHED", closed)
            break
        if kind == 0 and d * (bar.close - plan.invalidation) <= 0:
            invalidate("DAILY_INVALIDATION", closed)
            break
        if plan.setup in {"2X", "2XS", "5S", "5L"} and (
            (d > 0 and bar.low <= plan.invalidation)
            or (d < 0 and bar.high >= plan.invalidation)
        ):
            invalidate("STRUCTURE_INVALIDATED", closed)
            break
        if (d > 0 and bar.high >= plan.target1) or (d < 0 and bar.low <= plan.target1):
            invalidate("TARGET_ALREADY_PASSED", closed)
            break
        if kind == 0 or closed <= plan.created:
            continue
        if state == "READY":
            if not _in_zone(
                bar.close, evidence["entry_zone_low"], evidence["entry_zone_high"], d
            ):
                invalidate("ENTRY_OUTSIDE_ZONE", closed)
                break
            if d * (bar.close - entry) > entry * 0.005:
                invalidate("CHASING", closed)
                break
            continue
        touched = touched or bar.low <= plan.zone_high and bar.high >= plan.zone_low
        known = [
            p
            for p in swings
            if p.available_at <= bar.open_time / 1000
            and p.kind == ("high" if d > 0 else "low")
        ]
        trigger = None
        entry_zone_low, entry_zone_high = plan.zone_low, plan.zone_high
        if (
            len(known) >= 2
            and known[-1].open_time >= correction_start
            and d * (known[-1].price - known[-2].price) < 0
        ):
            confirmation = known[-1].price
            if (
                touched
                and previous is not None
                and d * (previous.close - confirmation) <= 0
                and d * (bar.close - confirmation) > 0
            ):
                trigger = "SWING_BREAK"
        # Type 2 alternative: a *new*, known 4H breakout followed by its first
        # retest, on one of the next three bars. No retroactive daily trigger.
        if plan.setup in {"2", "2S"} and not retest_done:
            level = plan.confirmation
            if breakout is not None:
                distance = (closed - breakout) / H4
                if distance > 3:
                    retest_done = True
                    evidence["retest_status"] = "RETEST_EXPIRED"
                    evidence["retest_ended_at"] = breakout + 3 * H4
                elif (
                    distance >= 1
                    and bar.low <= level * 1.001
                    and bar.high >= level * 0.999
                ):
                    retest_done = True
                    evidence["retest_ended_at"] = closed
                    if d * (bar.close - level) > 0:
                        evidence["retest_status"] = "CONFIRMED"
                        if trigger is None:
                            trigger, confirmation = "BREAKOUT_RETEST", level
                            entry_zone_low = level * 0.999 if d > 0 else level
                            entry_zone_high = level if d > 0 else level * 1.001
                    else:
                        evidence["retest_status"] = "FAILED_FIRST_RETEST"
            if (
                breakout is None
                and previous is not None
                and d * (previous.close - level) <= 0
                and d * (bar.close - level) > 0
            ):
                breakout = closed
                evidence["retest_breakout_at"] = closed
        if trigger:
            if not _in_zone(bar.close, entry_zone_low, entry_zone_high, d):
                # "No ticket" for this attempt is not structural invalidation.
                # A later fresh closed-bar cross may qualify inside the zone.
                evidence["last_entry_attempt"] = {
                    "reason": "TRIGGER_OUTSIDE_ZONE",
                    "at": closed,
                    "kind": trigger,
                    "price": bar.close,
                }
                previous = bar
                continue
            entry, state, reason = bar.close, "READY", trigger
            expires = min(expires, closed + 2 * H4)
            evidence.update(
                trigger_closed_at=closed,
                trigger_price=entry,
                trigger_kind=trigger,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                trigger_evidence_id=_id(
                    [
                        evidence_id,
                        trigger,
                        confirmation,
                        asdict(bar),
                        [p.id for p in known],
                    ]
                ),
            )
        previous = bar
    if state != "INVALID" and now >= expires:
        invalidate("EXPIRED", expires)
    if (
        state == "WAIT"
        and not retest_done
        and breakout is not None
        and now >= breakout + 3 * H4
    ):
        evidence["retest_status"] = "RETEST_EXPIRED"
        evidence["retest_ended_at"] = breakout + 3 * H4
    fresh = (
        now - evidence["daily_closed_at"] <= DAY + DATA_GRACE
        and evidence["execution_closed_at"] is not None
        and now - evidence["execution_closed_at"] <= H4 + DATA_GRACE
    )
    if state != "INVALID" and not fresh:
        state, reason = "WAIT", "DATA_STALE_OR_MISSING"
        evidence["data_valid"] = False
    if state != "INVALID" and (
        tier not in (1, 2) or plan.setup in {"2X", "2XS"} and tier != 1
    ):
        state, reason = "WAIT", "TIER_NOT_ELIGIBLE"
    return Opportunity(
        evidence_id,
        symbol,
        frozen["side"],
        plan.setup,
        state,
        entry,
        stop,
        plan.target1,
        plan.target2,
        plan.invalidation,
        confirmation,
        plan.created,
        expires,
        reason,
        evidence,
        tier,
        bucket,
    )


def analyze(
    symbol: str,
    daily: list[Candle],
    execution: list[Candle],
    now: float,
    tier: int = 1,
    bucket: str = "crypto",
) -> list[Opportunity]:
    """Replay causal daily plans and closed 4H triggers as known at ``now``.

    Invalid input returns no opportunities. A valid plan with absent/stale 4H
    evidence remains WAIT. INVALID tombstones remain in complete-history replay.
    """
    if (
        not isinstance(symbol, str)
        or not symbol.strip()
        or not isinstance(bucket, str)
        or not bucket.strip()
        or not _finite(now)
        or now < 0
        or not isinstance(tier, int)
        or isinstance(tier, bool)
    ):
        return []
    symbol = symbol.strip().upper()
    try:
        d = _closed(daily, DAY, now)
        e = _closed(execution, H4, now)
        if len(d) < ATR_PERIOD + 2:
            return []
        pivots = confirmed_zigzag(d, DAY, now)
        swings = confirmed_zigzag(e, H4, now, atr_multiple=1)
        result = [
            _evaluate(symbol, p, d, e, swings, now, tier, bucket)
            for p in _plans(d, pivots)
        ]
    except (ValueError, TypeError, OverflowError):
        return []
    sides = {op.side for op in result if op.state == "READY"}
    if len(sides) > 1:
        result = [
            (
                replace(op, state="WAIT", reason="CONFLICTING_DIRECTIONS")
                if op.state == "READY"
                else op
            )
            for op in result
        ]
    return sorted(result, key=lambda op: (op.created_at, op.id))
