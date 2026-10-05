"""Offline chronological acceptance diagnostics from hourly Parquet.

No network, credentials, orders, parameter search or source-data writes.
These are historical engineering observations, never live opportunities or
profitability/release evidence. Existing report files are never overwritten.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import FunctionType, SimpleNamespace

from .engine import ATR_PERIOD, DAY, H4, VERSION, analyze
from .models import Candle, Instrument
from .risk import assess
from .replay_management import _ManagementCache
from .simulation import advance, create_trade, summary

HOUR = 3600
ARM = "acceptance_replay"
LIMITATIONS = [
    "Historical engineering diagnostics only; no current opportunities or release authorization",
    "Hourly fills/management instead of production 3m; conservative OHLC ordering and next-hour entry eligibility",
    "Default scans hourly instead of runtime default 60s; finer scans cannot reconstruct intrahour fills or exposure changes",
    "Synthetic quantity/tick limits, assumed 0.01% spread, zero funding for sizing; realized funding unknown",
    "No AI, macro gate or live account; independent symbol books starting flat at 10000",
    "Risk circuit historical path not replayed; closed PnL before funding used for sizing; no portfolio-return inference",
    "Fixed full available contiguous history origin; production persisted history may differ",
    "Only the contiguous segment containing the requested end is replayed; no filling or bridging source gaps",
]


def _iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


class _AnalysisCache:
    """Per-replay cache with owned immutable histories, never cached risk.

    With unchanged prefixes, engine price/plan/retest transitions occur on
    closed 4H/daily boundaries; expiry can also change a snapshot. Between
    those boundaries only evidence.as_of changes. Never cross an expected 4H
    close, expiry, stale history or backwards time. No global monkeypatching.
    ``enabled=False`` is the exact differential-check path.
    """

    def __init__(
        self,
        symbol,
        daily,
        execution,
        enabled=True,
        analyzer=None,
        analyzer_kwargs=None,
    ):
        self.symbol = symbol
        self.analyzer = analyze if analyzer is None else analyzer
        self.analyzer_kwargs = dict(analyzer_kwargs or {})
        self.daily, self.execution = tuple(daily), tuple(execution)
        self.d_times = [b.open_time / 1000 + DAY for b in daily]
        self.e_times = [b.open_time / 1000 + H4 for b in execution]
        self.enabled = enabled
        self.key = None
        self.ops = []
        self.at = self.until = 0
        self.calls = self.hits = 0
        self.terminal_hits = self.plan_hits = 0
        # A private function namespace avoids changing the engine module seen
        # by another replay/runtime. Only its pure call sites are memoized.
        # Custom analyzers remain on their own path.
        if enabled and getattr(self.analyzer, "__name__", None) in {
            "analyze",
            "analyze_zones",
            "analyze_monitored_zones",
        }:
            namespace = self.analyzer.__globals__
            if "_evaluate" in namespace and (
                "_plans" in namespace or "engine" in namespace
            ):
                self.analyzer = self._memoized_engine(self.analyzer)

    def _memoized_engine(self, analyzer):
        original = analyzer.__globals__
        zone = analyzer.__name__ in {"analyze_zones", "analyze_monitored_zones"}
        engine_namespace = vars(original["engine"]) if zone else original
        evaluate, make_plans = original["_evaluate"], engine_namespace["_plans"]
        terminal = {}
        last_plan_key, last_plans = None, None
        closed_cache, swing_cache = {}, {}

        def closed(candles, seconds, now):
            key = (tuple(candles), now // seconds)
            prior = closed_cache.get(seconds)
            if prior is None or prior[0] != key:
                prior = (key, engine_namespace["_closed"](candles, seconds, now))
                closed_cache[seconds] = prior
            return list(prior[1])

        # Both analyze and confirmed_zigzag validate the same immutable
        # prefix. Memoize exact inputs; do not replace the validation rules.
        swing_function = engine_namespace["confirmed_zigzag"]
        swing_namespace = dict(swing_function.__globals__, _closed=closed)
        raw_swings = FunctionType(
            swing_function.__code__,
            swing_namespace,
            swing_function.__name__,
            swing_function.__defaults__,
            swing_function.__closure__,
        )

        def swings(
            candles,
            timeframe_seconds=DAY,
            now=None,
            atr_period=ATR_PERIOD,
            atr_multiple=2.0,
        ):
            if now is None:
                now = candles[-1].open_time / 1000 + timeframe_seconds if candles else 0
            key = (tuple(candles), now // timeframe_seconds, atr_period, atr_multiple)
            prior = swing_cache.get(timeframe_seconds)
            if prior is None or prior[0] != key:
                prior = (
                    key,
                    raw_swings(
                        candles, timeframe_seconds, now, atr_period, atr_multiple
                    ),
                )
                swing_cache[timeframe_seconds] = prior
            return list(prior[1])

        def plans(bars, pivots):
            nonlocal last_plan_key, last_plans
            key = (tuple(bars), tuple(pivots))
            if key == last_plan_key:
                self.plan_hits += 1
            else:
                last_plan_key, last_plans = key, make_plans(bars, pivots)
            return list(last_plans)

        def evaluate_plan(symbol, plan, daily, execution, *tail, **kwargs):
            now, tier, bucket = tail[-3:]
            key = (symbol, plan, tier, bucket, tuple(sorted(kwargs.items())))
            prior = terminal.get(key)
            if prior is not None and now >= prior.evidence["as_of"]:
                # Owned histories are immutable and prefixes only append. A
                # terminal transition cannot be rewritten by later events.
                # Refresh only the three unfrozen snapshot timestamps.
                self.terminal_hits += 1
                evidence = dict(
                    deepcopy(prior.evidence),
                    as_of=now,
                    daily_closed_at=daily[-1].open_time / 1000 + DAY,
                    execution_closed_at=(
                        execution[-1].open_time / 1000 + H4 if execution else None
                    ),
                )
                if zone:
                    # Unlike confirmed snapshots, zone proposals report
                    # freshness even on terminal tombstones.
                    evidence["data_valid"] = bool(
                        now - evidence["daily_closed_at"]
                        <= DAY + engine_namespace["DATA_GRACE"]
                        and evidence["execution_closed_at"] is not None
                        and now - evidence["execution_closed_at"]
                        <= H4 + engine_namespace["DATA_GRACE"]
                    )
                return replace(prior, evidence=evidence)
            op = evaluate(symbol, plan, daily, execution, *tail, **kwargs)
            if (
                op.state == "INVALID"
                and op.evidence.get("terminal_at", float("inf")) <= now
            ):
                terminal[key] = op
            return op

        namespace = dict(
            original,
            _evaluate=evaluate_plan,
            _plans=plans,
            _closed=closed,
            confirmed_zigzag=swings,
        )
        if zone:
            namespace["engine"] = SimpleNamespace(
                **dict(
                    engine_namespace,
                    _plans=plans,
                    _closed=closed,
                    confirmed_zigzag=swings,
                )
            )
        optimized = FunctionType(
            analyzer.__code__,
            namespace,
            analyzer.__name__,
            analyzer.__defaults__,
            analyzer.__closure__,
        )
        optimized.__kwdefaults__ = deepcopy(analyzer.__kwdefaults__)
        return optimized

    def histories(self, now):
        return (
            list(self.daily[: bisect_right(self.d_times, now)]),
            list(self.execution[: bisect_right(self.e_times, now)]),
        )

    def get(self, now):
        key = (bisect_right(self.d_times, now), bisect_right(self.e_times, now))
        if self.enabled and key == self.key and self.at <= now < self.until:
            self.hits += 1
        else:
            daily, execution = self.histories(now)
            self.ops = self.analyzer(
                self.symbol,
                daily,
                execution,
                now,
                bucket="majors" if self.symbol in {"BTCUSDT", "ETHUSDT"} else "alts",
                **self.analyzer_kwargs,
            )
            self.calls += 1
            self.key, self.at = key, now
            # Grace deadlines can elapse between boundaries with missing bars.
            # Cache only while both prefixes are strictly fresh without grace.
            fresh = (
                daily
                and execution
                and now < self.d_times[key[0] - 1] + DAY
                and now < self.e_times[key[1] - 1] + H4
            )
            self.until = (
                min(
                    [(now // H4 + 1) * H4]
                    + [
                        op.expires_at
                        for op in self.ops
                        if op.state != "INVALID" and op.expires_at > now
                    ]
                )
                if fresh
                else now
            )
        # Frozen Opportunity still contains mutable evidence. Return owned
        # dictionaries, preserving all identity/entry/trigger fields exactly.
        return [
            replace(op, evidence=dict(deepcopy(op.evidence), as_of=now))
            for op in self.ops
        ]


def _source(path, days, end):
    import pandas as pd

    if end is not None and not isinstance(end, (str, datetime)):
        raise ValueError("end must be a datetime or UTC date/time string")
    columns = ["open", "high", "low", "close", "volume"]
    frame = pd.read_parquet(path, columns=["start", *columns])
    if frame.empty:
        raise ValueError("Empty hourly source")
    if pd.api.types.is_numeric_dtype(frame["start"]):
        raise ValueError("Source start must be datetime, not ambiguous numeric epochs")
    frame["start"] = pd.to_datetime(frame["start"], utc=True, errors="raise")
    frame = frame.set_index("start").sort_index()
    if frame.index.hasnans or not frame.index.is_unique:
        raise ValueError("Missing or duplicate source timestamps")
    if not (frame.index == frame.index.floor("h")).all():
        raise ValueError("Source timestamps must align to UTC hours")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid source OHLCV") from exc
    finite = frame.notna() & frame.ne(float("inf")) & frame.ne(-float("inf"))
    valid = (
        finite.all(axis=1)
        & frame[columns[:4]].gt(0).all(axis=1)
        & frame.volume.ge(0)
        & frame.low.le(frame[["open", "close"]].min(axis=1))
        & frame.high.ge(frame[["open", "close"]].max(axis=1))
    )
    if not valid.all():
        raise ValueError(f"Invalid source OHLCV at {frame.index[~valid][0]}")
    source_end = frame.index[-1] + pd.Timedelta(hours=1)
    try:
        requested_end = source_end if end is None else pd.Timestamp(end)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid replay end") from exc
    if pd.isna(requested_end):
        raise ValueError("Missing replay end")
    requested_end = (
        requested_end.tz_localize("UTC")
        if requested_end.tzinfo is None
        else requested_end.tz_convert("UTC")
    )
    if requested_end != requested_end.floor("h"):
        raise ValueError("Replay end must align to a UTC hour")
    if not frame.index[0] < requested_end <= source_end:
        raise ValueError("Replay end is outside source coverage")
    gaps = []
    for previous, current in zip(frame.index, frame.index[1:]):
        if current - previous > pd.Timedelta(hours=1):
            gaps.append(
                dict(
                    start=(previous + pd.Timedelta(hours=1)).isoformat(),
                    end=current.isoformat(),
                    missing_hours=int((current - previous) / pd.Timedelta(hours=1)) - 1,
                )
            )
    prior = frame.loc[frame.index < requested_end]
    if prior.index[-1] + pd.Timedelta(hours=1) != requested_end:
        raise ValueError("Replay end falls inside a source gap")
    starts = prior.index[prior.index.to_series().diff() > pd.Timedelta(hours=1)]
    segment_start = starts[-1] if len(starts) else prior.index[0]
    segment = prior.loc[segment_start:]
    requested_start = requested_end - pd.Timedelta(days=days)
    start = max(requested_start, segment_start)

    def candles(result):
        return [
            Candle(int(t.timestamp() * 1000), *row)
            for t, row in zip(
                result.index, result[columns].itertuples(index=False, name=None)
            )
        ]

    def aggregate(freq, count):
        grouped = segment.resample(freq)
        result = grouped.agg(
            dict(open="first", high="max", low="min", close="last", volume="sum")
        )
        return candles(result[grouped["open"].count() == count])

    return (
        aggregate("1D", 24),
        aggregate("4h", 4),
        # Include the preceding completed hourly bar solely as a quote at
        # window start. Decisions/fills still begin at start, never earlier.
        candles(segment.loc[start - pd.Timedelta(hours=1) :]),
        start.timestamp(),
        requested_end.timestamp(),
        dict(
            source_start=frame.index[0].isoformat(),
            source_end=source_end.isoformat(),
            requested_start=requested_start.isoformat(),
            requested_end=requested_end.isoformat(),
            segment_start=segment_start.isoformat(),
            source_gaps=gaps,
            excluded_before_segment_hours=int(
                (segment_start - frame.index[0]) / pd.Timedelta(hours=1)
            ),
            window_truncated=start != requested_start,
            excluded_requested_hours=int(
                (start - requested_start) / pd.Timedelta(hours=1)
            ),
            history_policy="All complete bars from a fixed contiguous segment origin; no rolling 500-bar truncation",
        ),
    )


def replay(
    path,
    days=90,
    *,
    end=None,
    runoff_days=0,
    scan_seconds=HOUR,
    max_scans=None,
    max_seconds=None,
    use_cache=True,
    entry_style="confirmed",
    monitor_days=30,
):
    """Retry unconsumed READY IDs each scan until invalidation/expiry.

    The requested source window is days + runoff_days ending at end (or the
    source end). Decisions in [start, end - runoff_days * DAY) follow management
    of bars closing at that time. Truncated source coverage never moves this
    cutoff. With runoff, pending entries expire at the cutoff; only fills whose
    modeled timestamp precedes it are allowed. OPEN trades are managed through
    end, never forced closed. Zero runoff preserves the prior end behavior.
    A new decision can fill only in a subsequent fully evidenced hourly bar.
    Explicit ``end`` selects an earlier source segment. Budgets stop a labeled
    incomplete chronological prefix, never silently count unfinished as flat.
    """
    if type(days) is not int or not 1 <= days <= 1460:
        raise ValueError("days must be 1..1460")
    if type(runoff_days) is not int or not 0 <= runoff_days <= 1460:
        raise ValueError("runoff_days must be 0..1460")
    if type(use_cache) is not bool:
        raise ValueError("use_cache must be bool")
    if entry_style not in {"confirmed", "resting_limit", "monitored_zone"}:
        raise ValueError(
            "entry_style must be confirmed, resting_limit or monitored_zone"
        )
    if type(monitor_days) is not int or monitor_days not in {2, 30}:
        raise ValueError("monitor_days must be the predefined 2 or 30 day arm")
    analyzer, arm = analyze, ARM
    analyzer_kwargs = {}
    if entry_style == "resting_limit":
        from .zone_orders import analyze_zones

        analyzer, arm = analyze_zones, "resting_shadow"
    elif entry_style == "monitored_zone":
        from .monitored_zones import analyze_monitored_zones, observe_bar, observe_quote

        analyzer, arm = analyze_monitored_zones, "monitored_shadow"
        analyzer_kwargs = {"lifetime_seconds": monitor_days * DAY}
        if scan_seconds != HOUR:
            raise ValueError("monitored hourly quote experiment requires hourly scans")
    if (
        type(scan_seconds) is not int
        or not 60 <= scan_seconds <= H4
        or not (
            HOUR % scan_seconds == 0
            or scan_seconds % HOUR == 0
            and H4 % scan_seconds == 0
        )
    ):
        raise ValueError("scan_seconds must divide one hour or be 1, 2 or 4 hours")
    if max_scans is not None and (type(max_scans) is not int or max_scans < 1):
        raise ValueError("max_scans must be positive")
    if max_seconds is not None and (
        type(max_seconds) not in (int, float) or not 0 < max_seconds < float("inf")
    ):
        raise ValueError("max_seconds must be finite and positive")
    started = perf_counter()
    path = Path(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    daily, execution, hourly, start, end, data = _source(path, days + runoff_days, end)
    entry_cutoff = end - runoff_days * DAY
    symbol = path.stem
    cache = _AnalysisCache(
        symbol, daily, execution, use_cache, analyzer, analyzer_kwargs
    )
    management = _ManagementCache(daily, execution) if use_cache else None
    advance_trade = management.advance if management else advance
    instrument = Instrument(symbol, 0.0001, 0.0001, 1000000, 0.0001, 5)
    trades, plans = {}, {}
    states, reasons, rejection = Counter(), Counter(), Counter()
    state_ids, reason_ids, rejection_ids = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    events, consumed = [], set()
    watch_terminal = {}
    scans = empty_scans = warmup_scans = attempted = rejected = 0
    last_time, stop_reason = start, None
    bars = {b.open_time // 1000 + HOUR: b for b in hourly}
    initial_d, initial_e = cache.histories(start)

    def event(kind, now, record, **extra):
        events.append(
            dict(
                kind=kind,
                time=now,
                time_utc=_iso(now),
                id=record["id"],
                symbol=symbol,
                setup=record["setup"],
                side=record["side"],
                direction="LONG" if record["side"] == "Buy" else "SHORT",
                **extra,
            )
        )

    for now in range(int(start), int(end) + 1, min(HOUR, scan_seconds)):
        if max_seconds is not None and perf_counter() - started >= max_seconds:
            stop_reason = "MAX_SECONDS"
            break
        if max_scans is not None and scans >= max_scans and now < entry_cutoff:
            stop_reason = "MAX_SCANS"
            break
        last_time = now
        # Management first: fills/exits change eligibility at this scan. Never
        # give the just-closed candle to a trade created by the following scan.
        if now in bars:
            d, e = cache.histories(now)
            for key, old in list(trades.items()):
                if old["status"] not in {"PENDING", "OPEN"}:
                    continue
                # Confirmed/resting fills are timestamped at the hourly CLOSE;
                # monitored IOC fills occur at its OPEN. Resolve the last
                # entry-period IOC before expiring any still-pending order.
                cutoff_pending = (
                    runoff_days > 0 and now >= entry_cutoff
                    and old["status"] == "PENDING"
                )
                prior_ioc = (
                    old.get("entry_style") == "monitored_zone"
                    and bars[now].open_time / 1000 < entry_cutoff
                )
                if (
                    cutoff_pending and not prior_ioc
                    and old["expires_at"] >= entry_cutoff
                ):
                    trade = deepcopy(old)
                else:
                    trade = advance_trade(
                        old, [bars[now]], now, interval_ms=HOUR * 1000, daily=d, execution=e
                    )
                if cutoff_pending and trade["status"] == "PENDING":
                    trade.update(
                        status="EXPIRED", closed_at=entry_cutoff,
                        exit_reason="ENTRY_CUTOFF",
                    )
                trades[key] = trade
                record = plans[trade["opportunity_id"]]
                if trade.get("opened_at") is not None and old.get("opened_at") is None:
                    event("FILLED", now, record, trade_id=key)
                if trade["status"] != old["status"]:
                    event(
                        "TRADE_STATE",
                        now,
                        record,
                        trade_id=key,
                        state=trade["status"],
                        reason=trade.get("exit_reason"),
                    )
                if trade.get("data_gap") or trade.get("data_error"):
                    event(
                        "DATA_ISSUE",
                        now,
                        record,
                        trade_id=key,
                        reason=trade.get("data_gap") or trade.get("data_error"),
                    )
        if now >= entry_cutoff or now % scan_seconds:
            continue
        scans += 1
        ops = cache.get(now)
        if entry_style == "monitored_zone":
            observed = []
            for watch in ops:
                if watch.id in watch_terminal:
                    watch = replace(
                        watch_terminal[watch.id],
                        evidence=dict(watch_terminal[watch.id].evidence, as_of=now),
                    )
                elif now in bars:
                    watch = observe_bar(watch, bars[now], now, interval_seconds=HOUR)
                    if watch.state == "INVALID":
                        watch_terminal[watch.id] = watch
                observed.append(watch)
            # A hard stop may cancel a watch between 4H closes. Its cached
            # opposite-direction conflict must stop blocking the surviving
            # watch immediately; apply cancellations to the whole set first.
            if any(w.reason == "CONFLICTING_DIRECTIONS" for w in observed):
                eligible_sides = {
                    w.side
                    for w in observed
                    if w.state != "INVALID"
                    and w.reason in {"AWAIT_ZONE", "CONFLICTING_DIRECTIONS"}
                }
                if len(eligible_sides) <= 1:
                    observed = [
                        (
                            replace(w, reason="AWAIT_ZONE")
                            if w.state != "INVALID"
                            and w.reason == "CONFLICTING_DIRECTIONS"
                            else w
                        )
                        for w in observed
                    ]
            ops = [
                (
                    observe_quote(
                        w,
                        bars[now].close
                        * (1 + (1 if w.side == "Buy" else -1) * 0.00005),
                        now,
                        now,
                    )
                    if w.state != "INVALID" and now in bars
                    else w
                )
                for w in observed
            ]
        empty_scans += not ops
        warmup_scans += bisect_right(cache.d_times, now) < ATR_PERIOD + 2
        for op in ops:
            states[op.state] += 1
            reasons[op.reason] += 1
            state_ids[op.state].add(op.id)
            reason_ids[op.reason].add(op.id)
            record = plans.setdefault(
                op.id,
                dict(
                    id=op.id,
                    setup=op.setup,
                    side=op.side,
                    created_at=op.created_at,
                    parent_plan_id=op.evidence.get("parent_plan_id", op.id),
                    entry_style=entry_style,
                    first_observed_at=now,
                    first_ready_at=None,
                    attempts=0,
                    accepted_at=None,
                    rejection_reasons={},
                    states=[],
                ),
            )
            if (record.get("last_state"), record.get("last_reason")) != (
                op.state,
                op.reason,
            ):
                event(
                    "PLAN_STATE",
                    now,
                    record,
                    state=op.state,
                    reason=op.reason,
                    trigger_at=op.evidence.get("trigger_closed_at"),
                    terminal_at=op.evidence.get("terminal_at"),
                )
            record.update(
                last_state=op.state,
                last_reason=op.reason,
                last_observed_at=now,
                expires_at=op.expires_at,
                entry=op.entry,
                stop=op.stop,
                target1=op.target1,
                target2=op.target2,
                invalidation=op.invalidation,
                trigger_at=op.evidence.get("trigger_closed_at"),
                order_armed_at=op.evidence.get("order_armed_at"),
                observation_at=op.evidence.get("observation_at"),
                observation_price=op.evidence.get("observation_price"),
            )
            if op.state not in record["states"]:
                record["states"].append(op.state)
            if op.state != "READY":
                continue
            if record["first_ready_at"] is None:
                record["first_ready_at"] = now
                record["first_ready_evidence_at"] = (
                    op.evidence.get("observation_at")
                    if entry_style == "monitored_zone"
                    else (
                        op.evidence.get("order_armed_at")
                        if entry_style == "resting_limit"
                        else op.evidence.get("trigger_closed_at")
                    )
                )
                event(
                    "FIRST_READY",
                    now,
                    record,
                    trigger_at=op.evidence.get("trigger_closed_at"),
                )
            if op.id in consumed or now >= op.expires_at:
                continue
            record["attempts"] += 1
            attempted += 1
            exposures = [
                t for t in trades.values() if t["status"] in {"PENDING", "OPEN"}
            ]
            equity = 10000 + sum(
                t.get("net_pnl_before_funding", 0)
                for t in trades.values()
                if t["status"] == "CLOSED"
            )
            # Only the explicit offline resting comparison enables this gate.
            # The default confirmed path keeps the ordinary assess contract.
            policy = (
                {"allow_resting": True}
                if entry_style == "resting_limit"
                else (
                    {"allow_monitored": True} if entry_style == "monitored_zone" else {}
                )
            )
            sizing = assess(
                op,
                instrument,
                equity,
                exposures=exposures,
                funding_rate_8h=0,
                spread_pct=0.01,
                now=now,
                **policy,
            )
            record["last_risk_assessment"] = deepcopy(sizing)
            if sizing["allowed"]:
                trade = create_trade(
                    op, dict(sizing, qty_step=instrument.qty_step), arm, now
                )
                # create_trade starts with its default 3m interval. Correct
                # metadata even if a diagnostic stops before its first advance.
                trade["interval_ms"] = HOUR * 1000
                trade["entry_eligible_at"] = ((now + HOUR - 1) // HOUR) * HOUR
                trades[trade["id"]] = trade
                consumed.add(op.id)
                record["accepted_at"] = now
                record["accepted_opportunity"] = op.to_dict()
                record["accepted_risk_assessment"] = deepcopy(sizing)
                event(
                    "ACCEPTED",
                    now,
                    record,
                    attempt=record["attempts"],
                    trade_id=trade["id"],
                )
            else:
                rejected += 1
                codes = list(dict.fromkeys(sizing["reasons"]))
                rejection.update(codes)
                for reason in codes:
                    rejection_ids[reason].add(op.id)
                    record["rejection_reasons"][reason] = (
                        record["rejection_reasons"].get(reason, 0) + 1
                    )
                event(
                    "RISK_REJECTED",
                    now,
                    record,
                    attempt=record["attempts"],
                    reasons=codes,
                )
    closed = [t for t in trades.values() if t["status"] == "CLOSED"]
    ready = [p for p in plans.values() if p["first_ready_at"] is not None]
    filled = [t for t in trades.values() if t.get("opened_at") is not None]
    data.update(
        daily_origin=_iso(daily[0].open_time / 1000) if daily else None,
        execution_origin=_iso(execution[0].open_time / 1000) if execution else None,
        initial_daily_bars=len(initial_d),
        initial_execution_bars=len(initial_e),
        daily_minimum_bars=ATR_PERIOD + 2,
        warmup_scans=warmup_scans,
        empty_scans=empty_scans,
    )

    def side_count(items):
        items = list(items)
        return {side: sum(p["side"] == side for p in items) for side in ("Buy", "Sell")}

    return {
        "symbol": symbol,
        "engine_version": VERSION,
        "source_sha256": source_hash,
        "entry_style": entry_style,
        "arm": arm,
        "start": _iso(start),
        "end": _iso(end),
        "entry_end": _iso(entry_cutoff),
        "entry_days": days,
        "runoff_days": runoff_days,
        "entry_cutoff": entry_cutoff,
        "window_days": days + runoff_days,
        "replayed_through": _iso(last_time),
        "complete": stop_reason is None,
        "stop_reason": stop_reason,
        "scan_seconds": scan_seconds,
        "scans": scans,
        "data": data,
        "limitations": LIMITATIONS
        + (
            [
                "Monitored quote is last hourly close plus/minus half an assumed 0.01% spread; no live bid/ask evidence",
                "Monitored IOC resolves at next hourly OPEN with 3bp adverse slippage and strict 5bp quote cap; no intrahour touch fills",
                "One accepted attempt per plan including an unfilled IOC; no liquidity/partial fill model; not a continuous tick replay",
            ]
            if entry_style == "monitored_zone"
            else []
        ),
        "monitor_days": monitor_days if entry_style == "monitored_zone" else None,
        "ready_candidates": len(ready),
        "ready_by_setup_side": dict(
            Counter(p["setup"] + ":" + p["side"] for p in ready)
        ),
        "states_observed": dict(states),
        "reasons_observed": dict(reasons),
        "unique_plans_by_state": {k: len(v) for k, v in state_ids.items()},
        "unique_plans_by_reason": {k: len(v) for k, v in reason_ids.items()},
        "last_plan_states": dict(Counter(p["last_state"] for p in plans.values())),
        "risk_rejections": dict(rejection),
        "unique_plans_by_risk_rejection": {k: len(v) for k, v in rejection_ids.items()},
        "funnel": dict(
            unique_plans=len(plans),
            unique_ready=len(ready),
            risk_attempts=attempted,
            unique_plans_created_in_window=sum(
                start <= p["created_at"] < end for p in plans.values()
            ),
            invalid_when_first_observed=sum(
                p["states"][0] == "INVALID" for p in plans.values()
            ),
            ready_evidence_before_window=sum(
                p.get("first_ready_evidence_at") is not None
                and p["first_ready_evidence_at"] < start
                for p in ready
            ),
            rejected_attempts=rejected,
            unique_attempted=sum(p["attempts"] > 0 for p in plans.values()),
            unique_accepted=len(consumed),
            unique_filled=len(filled),
            accepted_after_retry=sum(
                p["accepted_at"] is not None and p["attempts"] > 1
                for p in plans.values()
            ),
            plans_by_side=side_count(plans.values()),
            ready_by_side=side_count(ready),
            accepted_by_side=side_count(
                [p for p in plans.values() if p["accepted_at"] is not None]
            ),
            filled_by_side=side_count(filled),
        ),
        "counting": "States/reasons count scan observations; unique counts count IDs ever observed and can overlap; event observation times are separate from trigger/terminal times",
        "plans": list(plans.values()),
        "events": events,
        "summary": summary(list(trades.values()), arm),
        "open_at_end": (
            sum(t["status"] == "OPEN" for t in trades.values())
            if stop_reason is None else None
        ),
        "pending_at_end": (
            sum(t["status"] == "PENDING" for t in trades.values())
            if stop_reason is None else None
        ),
        "closed_net_before_funding": sum(t["net_pnl_before_funding"] for t in closed),
        "data_issues": sum(
            bool(t.get("data_gap") or t.get("data_error")) for t in trades.values()
        ),
        "closed_trades": [
            {
                k: t.get(k)
                for k in (
                    "opportunity_id",
                    "symbol",
                    "setup",
                    "side",
                    "opened_at",
                    "closed_at",
                    "exit_reason",
                    "net_pnl_before_funding",
                )
            }
            for t in closed
        ],
        "trades": list(trades.values()),
        "performance": dict(
            engine_calls=cache.calls,
            cache_hits=cache.hits,
            terminal_plan_cache_hits=cache.terminal_hits,
            daily_plan_cache_hits=cache.plan_hits,
            management_validation_hits=management.validation_hits if management else 0,
            management_pivot_hits=management.pivot_hits if management else 0,
            management_pivot_builds=management.pivot_builds if management else 0,
            elapsed_seconds=perf_counter() - started,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--days", type=int, default=90, help="Entry window: 1..1460 days")
    parser.add_argument(
        "--runoff-days", type=int, default=0,
        help="Additional management days, 0..1460; pending entries expire at cutoff",
    )
    parser.add_argument("--end", help="Exclusive UTC hour, e.g. 2026-05-25T01:00:00Z")
    parser.add_argument("--scan-seconds", type=int, default=HOUR)
    parser.add_argument(
        "--max-scans", type=int, help="Per-symbol diagnostic scan budget"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Per-symbol soft time budget, checked between steps",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Differential verification of the same replay",
    )
    parser.add_argument(
        "--entry-style",
        choices=("confirmed", "resting_limit", "monitored_zone"),
        default="confirmed",
        help="Explicit offline comparison; resting_limit uses its own resting_shadow book",
    )
    parser.add_argument("--monitor-days", type=int, choices=(2, 30), default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {
        p.resolve() for p in args.files
    }:
        parser.error("output must be a new file; existing research is preserved")
    try:
        result = dict(
            purpose="Engineering acceptance only; not evidence of profitability",
            limitations=LIMITATIONS,
            results=[
                replay(
                    path,
                    args.days,
                    end=args.end,
                    runoff_days=args.runoff_days,
                    scan_seconds=args.scan_seconds,
                    max_scans=args.max_scans,
                    max_seconds=args.max_seconds,
                    use_cache=not args.no_cache,
                    entry_style=args.entry_style,
                    monitor_days=args.monitor_days,
                )
                for path in args.files
            ],
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
