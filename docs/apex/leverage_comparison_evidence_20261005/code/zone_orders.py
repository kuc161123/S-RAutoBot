"""Offline/SHADOW resting-limit proposals for the engine's frozen daily plans.

READY means a pending research proposal, never a fill or a confirmed trigger.
This module has no runtime integration, exchange calls, or order state. Replay
the same daily/execution history origin on every call; consumers must track
fills and consumed IDs separately and must not backfill historical entries.

Section 7.2 resting entries are Tier 1 only. One frozen midpoint is an explicit
research simplification of its 30/30/40 ladder; the fixed 48-hour deadline is
our research policy, not a source rule. The midpoint is never optimized to
make R:R pass. Risk assessment remains a separate, explicit research opt-in.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from . import engine
from .models import Candle, Opportunity

ENTRY_STYLE = "resting_limit"
ORDER_LIFETIME = 48 * 60 * 60


def _evaluate(
    symbol: str,
    plan: engine._Plan,
    daily: list[Candle],
    execution: list[Candle],
    now: float,
    tier: int,
    bucket: str,
    *,
    lifetime: float = ORDER_LIFETIME,
    entry_style: str = ENTRY_STYLE,
) -> Opportunity:
    """Evaluate frozen levels and cancellations; defaults preserve zone orders.

    The monitored experiment supplies its own style and lifetime, then replaces
    the proposal state. Neither option changes the original daily plan ID.
    """
    d = plan.direction
    buffer = engine.stop_buffer_pct(symbol, bucket)
    stop = plan.invalidation * (1 - d * buffer)
    entry = (plan.zone_low + plan.zone_high) / 2
    expires = min(plan.created + lifetime, plan.created + engine.PLAN_LIFETIME)
    history_start = daily[plan.history_start].open_time / 1000

    # Match the confirmed engine's frozen plan identity for paired comparison.
    # Do not call its evaluator: its trigger lifecycle is a different experiment.
    frozen = {
        "engine_version": engine.VERSION,
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
        "daily_evidence_id": engine._id(
            [
                asdict(b)
                for b in daily
                if b.open_time / 1000 + engine.DAY <= plan.created
            ]
        ),
    }
    plan_id = engine._id([symbol, tier, bucket, frozen])
    order = {
        "plan_evidence_id": plan_id,
        "entry_style": entry_style,
        "entry": entry,
        "stop": stop,
        "confirmation": plan.confirmation,
        "stop_buffer_pct": buffer,
        "order_armed_at": plan.created,
        "order_expires_at": expires,
        "plan_history_start": history_start,
    }
    evidence_id = engine._id(order)
    evidence = dict(
        frozen,
        **order,
        parent_plan_id=plan_id,
        evidence_id=evidence_id,
        structural_valid=True,
        data_valid=True,
        daily_closed_at=daily[-1].open_time / 1000 + engine.DAY,
        execution_closed_at=(
            execution[-1].open_time / 1000 + engine.H4 if execution else None
        ),
        as_of=now,
        trigger_kind="ZONE_LIMIT",
        trigger_closed_at=None,
        trigger_price=None,
        trigger_evidence_id=None,
        entry_zone_low=plan.zone_low,
        entry_zone_high=plan.zone_high,
        terminal_status=None,
    )

    # Both timeframes can cancel a plan during its confirmation lag. The
    # history boundary is the engine's, not the historical zone geometry or
    # the time the caller first discovers a plan. A touch never implies a fill.
    events = [
        (b.open_time / 1000 + engine.DAY, 0, b) for b in daily[plan.history_start :]
    ]
    events += [
        (b.open_time / 1000 + engine.H4, 1, b)
        for b in execution
        if b.open_time / 1000 >= history_start
    ]
    events.sort(key=lambda event: (event[0], event[1]))
    state, reason = "READY", "ZONE_LIMIT"

    def invalidate(code: str, at: float) -> None:
        nonlocal state, reason
        state, reason = "INVALID", code
        evidence["terminal_status"] = code
        evidence["terminal_at"] = max(at, plan.created)
        evidence["structural_valid"] = code not in {
            "HARD_STOP_BREACHED",
            "DAILY_INVALIDATION",
            "STRUCTURE_INVALIDATED",
        }

    # Evaluate the first terminal event before offering any pending order.
    # Expiry wins at its boundary; daily events win tied closes, and stop /
    # structure precede target passage within an ambiguous OHLC bar.
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
    if state != "INVALID" and now >= expires:
        invalidate("EXPIRED", expires)

    fresh = (
        now - evidence["daily_closed_at"] <= engine.DAY + engine.DATA_GRACE
        and evidence["execution_closed_at"] is not None
        and plan.created <= evidence["execution_closed_at"]
        and now - evidence["execution_closed_at"] <= engine.H4 + engine.DATA_GRACE
    )
    evidence["data_valid"] = fresh
    if state != "INVALID" and not fresh:
        state, reason = "WAIT", "DATA_STALE_OR_MISSING"
    if state != "INVALID" and tier != 1:
        state, reason = "WAIT", "TIER_NOT_ELIGIBLE"
    return Opportunity(
        id=evidence_id,
        symbol=symbol,
        side=frozen["side"],
        setup=plan.setup,
        state=state,
        entry=entry,
        stop=stop,
        target1=plan.target1,
        target2=plan.target2,
        invalidation=plan.invalidation,
        confirmation=plan.confirmation,
        created_at=plan.created,
        expires_at=expires,
        reason=reason,
        evidence=evidence,
        tier=tier,
        bucket=bucket,
    )


def analyze_zones(
    symbol: str,
    daily: list[Candle],
    execution: list[Candle],
    now: float,
    tier: int = 1,
    bucket: str = "crypto",
) -> list[Opportunity]:
    """Return independent pending proposals from complete bars known at ``now``.

    Each available engine plan gets one frozen midpoint and one ID, with a
    deadline of min(created + 48h, created + PLAN_LIFETIME). No touch, retest,
    later discovery, or return to the zone changes the price or rearms expiry.
    Original stops, structure, daily closes and T1 passage cancel permanently.
    Targets remain the engine's two Elliott levels for downstream partial TPs.

    Invalid input returns []; absent/stale data, execution history ending before
    creation, or a tier other than 1 leaves nonterminal plans WAIT. INVALID
    tombstones survive full-history replay.
    Simultaneously READY opposing sides become WAIT/CONFLICTING_DIRECTIONS,
    preserving their frozen order IDs. These proposals are not executable tickets.
    ``plan_evidence_id`` / ``parent_plan_id`` identify the confirmed engine's
    frozen plan; ``id`` / ``evidence_id`` hash the separate resting order.
    All times are UNIX seconds except Candle.open_time (milliseconds).
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
    ):
        return []
    symbol = symbol.strip().upper()
    try:
        d = engine._closed(daily, engine.DAY, now)
        e = engine._closed(execution, engine.H4, now)
        if len(d) < engine.ATR_PERIOD + 2:
            return []
        pivots = engine.confirmed_zigzag(d, engine.DAY, now)
        result = [
            _evaluate(symbol, plan, d, e, now, tier, bucket)
            for plan in engine._plans(d, pivots)
            if plan.created <= now
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
