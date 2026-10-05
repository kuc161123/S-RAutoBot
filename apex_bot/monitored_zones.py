"""Bounded offline wait-then-enter experiment over frozen daily plans.

The pre-registered comparison is a 30-day primary watch and a 48-hour control,
both measured from plan creation. Lifetime is not an optimization grid. No
runtime, exchange, fill, risk sizing, or synthetic 4H confirmation lives here.
Replay complete daily/4H history from the same origin on every analysis call.
Consumers separately track consumed IDs and persist INVALID tombstones.
"""

from __future__ import annotations

from dataclasses import replace

from . import engine
from .models import Candle, Opportunity
from .zone_orders import _evaluate as _evaluate_zone

ENTRY_STYLE = "monitored_zone"
PRIMARY_LIFETIME = engine.PLAN_LIFETIME
CONTROL_LIFETIME = 48 * 60 * 60
QUOTE_MAX_AGE = 60
ADVERSE_CAP_RATE = 0.0005
_QUOTE_WAIT_REASONS = {
    "AWAIT_ZONE",
    "QUOTE_STALE_OR_MISSING",
    "QUOTE_OUTSIDE_ZONE",
    "QUOTE_LIMIT_OUTSIDE_ZONE",
}
_STRUCTURE_SETUPS = {"2X", "2XS", "5S", "5L"}


def _evaluate(
    symbol: str,
    plan: engine._Plan,
    daily: list[Candle],
    execution: list[Candle],
    now: float,
    tier: int,
    bucket: str,
    *,
    lifetime_seconds: float = engine.PLAN_LIFETIME,
) -> Opportunity:
    op = _evaluate_zone(
        symbol,
        plan,
        daily,
        execution,
        now,
        tier,
        bucket,
        lifetime=lifetime_seconds,
        entry_style=ENTRY_STYLE,
    )
    evidence = dict(
        op.evidence,
        monitored_started_at=plan.created,
        lifetime_seconds=float(lifetime_seconds),
        observation_at=None,
        observation_price=None,
        quote_limit=None,
        trigger_kind="ZONE_WATCH",
    )
    # The shared identity includes style and the creation-anchored expiry.
    # Its entry field remains the frozen midpoint, never the observed quote.
    return replace(
        op,
        state="WAIT" if op.state == "READY" else op.state,
        reason="AWAIT_ZONE" if op.state == "READY" else op.reason,
        evidence=evidence,
    )


def analyze_monitored_zones(
    symbol: str,
    daily: list[Candle],
    execution: list[Candle],
    now: float,
    tier: int = 1,
    bucket: str = "crypto",
    *,
    lifetime_seconds: float = engine.PLAN_LIFETIME,
) -> list[Opportunity]:
    """Return Tier-1 WAIT/AWAIT_ZONE watches or persistent invalid tombstones.

    Lifetime must be finite and in [1, engine.PLAN_LIFETIME] seconds. The
    registered durations are PRIMARY_LIFETIME (30d) and CONTROL_LIFETIME (48h).
    Original engine plan identity, zone, stop, targets and invalidation stay
    frozen; the separate watch ID includes entry style and lifetime via expiry.
    Historical touches do not produce arrivals or rearm the deadline. Opposing
    eligible watches WAIT/CONFLICTING_DIRECTIONS and cannot accept quotes.

    ``engine`` and ``_evaluate`` are intentional globals for the parent's plan
    cache. The evaluator receives ``lifetime_seconds`` as a keyword argument.
    Invalid input returns []; missing/stale bars leave nonterminal watches WAIT.
    Times are UNIX seconds, except Candle.open_time (milliseconds).
    """
    if (
        not isinstance(symbol, str)
        or not symbol.strip()
        or not isinstance(bucket, str)
        or not bucket.strip()
        or not engine._finite(now)
        or now < 0
        or not isinstance(tier, int)
        or isinstance(tier, bool)
        or not engine._finite(lifetime_seconds)
        or not 1 <= lifetime_seconds <= engine.PLAN_LIFETIME
    ):
        return []
    symbol = symbol.strip().upper()
    # Equivalent numeric durations have the same identity (48h == 172800.0).
    lifetime_seconds = float(lifetime_seconds)
    try:
        d = engine._closed(daily, engine.DAY, now)
        e = engine._closed(execution, engine.H4, now)
        if len(d) < engine.ATR_PERIOD + 2:
            return []
        pivots = engine.confirmed_zigzag(d, engine.DAY, now)
        result = [
            _evaluate(
                symbol,
                plan,
                d,
                e,
                now,
                tier,
                bucket,
                lifetime_seconds=lifetime_seconds,
            )
            for plan in engine._plans(d, pivots)
            if plan.created <= now
        ]
    except (ValueError, TypeError, OverflowError):
        return []
    sides = {op.side for op in result if op.reason == "AWAIT_ZONE"}
    if len(sides) > 1:
        result = [
            (
                replace(op, reason="CONFLICTING_DIRECTIONS")
                if op.reason == "AWAIT_ZONE"
                else op
            )
            for op in result
        ]
    return sorted(result, key=lambda op: (op.created_at, op.id))


def _without_quote(watch: Opportunity, now: float) -> Opportunity:
    """Reset a transient quote attempt without touching the watch's identity."""
    evidence = dict(
        watch.evidence,
        observation_at=None,
        observation_price=None,
        quote_limit=None,
        trigger_kind="ZONE_WATCH",
        trigger_closed_at=None,
        trigger_price=None,
        trigger_evidence_id=None,
    )
    if engine._finite(now) and now >= 0:
        evidence["as_of"] = now
    return replace(
        watch,
        entry=evidence.get("entry", watch.entry),
        state="WAIT",
        reason=(
            "AWAIT_ZONE"
            if watch.reason in _QUOTE_WAIT_REASONS
            or (watch.state, watch.reason) == ("READY", "ZONE_ARRIVAL")
            else watch.reason
        ),
        evidence=evidence,
    )


def _terminal(watch: Opportunity, code: str, at: float, now: float) -> Opportunity:
    clean = _without_quote(watch, now)
    return replace(
        clean,
        state="INVALID",
        reason=code,
        evidence=dict(
            clean.evidence,
            terminal_status=code,
            terminal_at=at,
            structural_valid=code
            not in {
                "HARD_STOP_BREACHED",
                "DAILY_INVALIDATION",
                "STRUCTURE_INVALIDATED",
            },
        ),
    )


def observe_quote(
    watch: Opportunity,
    price: float | None,
    observed_at: float | None,
    now: float,
) -> Opportunity:
    """Offer a fresh in-zone quote with an adverse 5bp cap, never a fill.

    A quote must be positive, finite, no older than QUOTE_MAX_AGE (60s), not
    future-dated or older than the supporting bars/plan, and use an eligible
    watch with fresh daily/4H data. Both price and cap must satisfy the exact
    inclusive zone bounds: no engine directional 1% tolerance or price clamp.
    READY/ZONE_ARRIVAL sets entry to price * (1 +/- .0005); evidence.entry stays
    the frozen midpoint used by the watch ID. Observation evidence is present
    only for READY. Trigger time/price/ID always remain None.

    Rejected quotes leave WAIT and may be retried without changing ID, levels
    or expiry. R:R is deliberately left to the parent's risk assessment; a
    rejected risk assessment does not invalidate this watch. A new analyzer
    snapshot is required to clear data, tier or directional-conflict gates.
    Apply/persist chronological bar cancellations before calling this helper.
    """
    if watch.state == "INVALID":
        return watch
    clean = _without_quote(watch, now)
    if not engine._finite(now) or now < 0:
        return replace(clean, reason="INVALID_TIME")
    if now >= watch.expires_at:
        return _terminal(watch, "EXPIRED", watch.expires_at, now)
    e = watch.evidence
    if (
        e.get("entry_style") != ENTRY_STYLE
        or e.get("structural_valid") is not True
        or e.get("terminal_status") is not None
        or watch.side not in {"Buy", "Sell"}
        or not engine._finite(e.get("as_of"))
        or not watch.created_at <= e["as_of"] <= now
        or clean.reason != "AWAIT_ZONE"
        or watch.state not in {"WAIT", "READY"}
    ):
        return replace(
            clean,
            reason=(
                clean.reason if clean.reason != "AWAIT_ZONE" else "WATCH_NOT_ELIGIBLE"
            ),
        )
    if watch.tier != 1 or isinstance(watch.tier, bool):
        return replace(clean, reason="TIER_NOT_ELIGIBLE")
    daily_at, execution_at = e.get("daily_closed_at"), e.get("execution_closed_at")
    if not (
        e.get("data_valid") is True
        and engine._finite(daily_at)
        and engine._finite(execution_at)
        and 0 <= now - daily_at <= engine.DAY + engine.DATA_GRACE
        and watch.created_at <= execution_at <= now
        and now - execution_at <= engine.H4 + engine.DATA_GRACE
    ):
        return replace(
            clean,
            reason="DATA_STALE_OR_MISSING",
            evidence=dict(clean.evidence, data_valid=False),
        )
    low, high = e.get("zone_low"), e.get("zone_high")
    if not (
        engine._finite(low)
        and engine._finite(high)
        and 0 < low <= high
        and e.get("entry_zone_low") == low
        and e.get("entry_zone_high") == high
    ):
        return replace(clean, reason="WATCH_NOT_ELIGIBLE")
    if not (
        engine._finite(price)
        and price > 0
        and engine._finite(observed_at)
        and max(watch.created_at, daily_at, execution_at) <= observed_at <= now
        and now - observed_at <= QUOTE_MAX_AGE
    ):
        return replace(clean, reason="QUOTE_STALE_OR_MISSING")
    if not low <= price <= high:
        return replace(clean, reason="QUOTE_OUTSIDE_ZONE")
    direction = 1 if watch.side == "Buy" else -1
    cap = price * (1 + direction * ADVERSE_CAP_RATE)
    if not engine._finite(cap) or not 0 < low <= cap <= high:
        return replace(clean, reason="QUOTE_LIMIT_OUTSIDE_ZONE")
    return replace(
        clean,
        state="READY",
        reason="ZONE_ARRIVAL",
        entry=cap,
        evidence=dict(
            clean.evidence,
            trigger_kind="ZONE_ARRIVAL",
            observation_at=observed_at,
            observation_price=price,
            quote_limit=cap,
        ),
    )


def observe_bar(
    watch: Opportunity,
    bar: Candle,
    now: float,
    interval_seconds: int = 3600,
) -> Opportunity:
    """Cancel on a closed hourly bar; never turn a bar into quote/4H evidence.

    Process bars in chronological order, before quote assessment. Only closes
    at or after creation and at or before now count. Stop, special-setup wick
    invalidation, then T1 passage win within ambiguous OHLC; expiry wins at its
    exact boundary. Ordinary daily-close invalidation stays with the analyzer.
    INVALID is absorbing, including when supplied by the parent's tombstones.
    Future/pre-creation bars are ignored; malformed closed bars block quoting.

    This helper neither advances daily/4H freshness nor claims hourly coverage.
    The parent must persist first terminal results across analyzer cache calls,
    keep frozen levels and IDs, and never offer consumed watches again. A quote
    may separately use hourly close +/- half-spread, observed_at=bar close time
    and now=current tick; its age must still pass observe_quote's 60s limit.
    """
    if watch.state == "INVALID":
        return watch
    if not engine._finite(now) or now < 0:
        return replace(_without_quote(watch, now), reason="INVALID_TIME")
    if (
        not isinstance(interval_seconds, int)
        or isinstance(interval_seconds, bool)
        or interval_seconds < 1
    ):
        return replace(_without_quote(watch, now), reason="INVALID_BAR")
    try:
        closed_bars = engine._closed([bar], interval_seconds, now)
    except (ValueError, TypeError, OverflowError):
        if now >= watch.expires_at:
            return _terminal(watch, "EXPIRED", watch.expires_at, now)
        return replace(_without_quote(watch, now), reason="INVALID_BAR")
    if closed_bars:
        closed = bar.open_time / 1000 + interval_seconds
        if watch.created_at <= closed < watch.expires_at:
            d = 1 if watch.side == "Buy" else -1
            code = None
            if (d > 0 and bar.low <= watch.stop) or (d < 0 and bar.high >= watch.stop):
                code = "HARD_STOP_BREACHED"
            elif watch.setup in _STRUCTURE_SETUPS and (
                (d > 0 and bar.low <= watch.invalidation)
                or (d < 0 and bar.high >= watch.invalidation)
            ):
                code = "STRUCTURE_INVALIDATED"
            elif (d > 0 and bar.high >= watch.target1) or (
                d < 0 and bar.low <= watch.target1
            ):
                code = "TARGET_ALREADY_PASSED"
            if code is not None:
                return _terminal(watch, code, closed, now)
            # A closed bar can revoke an old quote offer, never renew one.
            watch = _without_quote(watch, now)
    if now >= watch.expires_at:
        return _terminal(watch, "EXPIRED", watch.expires_at, now)
    return watch
