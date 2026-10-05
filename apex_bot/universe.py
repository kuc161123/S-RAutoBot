"""Pure, persisted liquidity selection. Times are UNIX seconds except launchTime.

The caller supplies a complete instruments/tickers fetch and books whose depths
are already summed within 25 bps of mid. No exchange, position, or file I/O lives
here. Removing a member only revokes entry permission; position draining belongs
to the caller. Persist the entire returned state between refreshes/restarts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re


DAY = 86400
SAMPLE_INTERVAL = 6 * 3600
HISTORY_SECONDS = 7 * DAY
MAX_SAMPLES = 28
STABLECOIN_BASES = frozenset(
    {
        "USDT",
        "USDC",
        "USDE",
        "DAI",
        "TUSD",
        "FDUSD",
        "BUSD",
        "USDD",
        "USDP",
        "PYUSD",
        "FRAX",
        "LUSD",
        "GUSD",
        "SUSD",
        "USDJ",
        "USDS",
        "DOLA",
        "MIM",
        "USD1",
        "USD0",
        "RLUSD",
        "UST",
        "USTC",
        "EURC",
        "EURT",
        "EURS",
    }
)
_SYMBOL = re.compile(r"[A-Z0-9]+USDT", re.ASCII)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _symbol(value):
    return isinstance(value, str) and _SYMBOL.fullmatch(value) is not None


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    # Avoid overflowing the sum of two finite turnover values.
    return (
        values[middle]
        if len(values) % 2
        else values[middle - 1] / 2 + values[middle] / 2
    )


def _history(previous, at):
    snapshots = previous.get("snapshots", {})
    if not isinstance(snapshots, dict):
        raise ValueError("Invalid persisted snapshots")
    result = {}
    for symbol, rows in snapshots.items():
        if not _symbol(symbol) or not isinstance(rows, list):
            raise ValueError("Invalid persisted snapshots")
        clean = {}
        for row in rows:
            timestamp = _number(row.get("as_of")) if isinstance(row, dict) else None
            turnover = (
                _number(row.get("turnover24h")) if isinstance(row, dict) else None
            )
            if timestamp is None or timestamp < 0 or turnover is None or turnover <= 0:
                raise ValueError("Invalid persisted turnover sample")
            if at - HISTORY_SECONDS < timestamp <= at:
                clean.setdefault(timestamp, turnover)
        spaced = []
        for timestamp, turnover in sorted(clean.items()):
            if not spaced or timestamp - spaced[-1]["as_of"] >= SAMPLE_INTERVAL:
                spaced.append({"as_of": timestamp, "turnover24h": turnover})
        if spaced:
            result[symbol] = spaced[-MAX_SAMPLES:]
    return result


def _eligibility(
    symbol,
    instrument,
    ticker,
    book,
    now,
    *,
    min_listing_days,
    min_turnover_usdt,
    max_spread_bps,
    min_depth_usdt,
    max_book_age,
):
    reasons = []
    if not _symbol(symbol):
        reasons.append("invalid_symbol")
    if instrument is None:
        return {"status": "ineligible", "reasons": ["not_listed"]}
    if instrument.get("symbol", symbol) != symbol:
        reasons.append("metadata_symbol_mismatch")
    if instrument.get("status") != "Trading":
        reasons.append("not_trading")
    if (
        instrument.get("contractType") != "LinearPerpetual"
        or instrument.get("quoteCoin") != "USDT"
        or instrument.get("settleCoin") != "USDT"
    ):
        reasons.append("not_usdt_perpetual")
    if str(instrument.get("symbolType", "")).lower() == "xstocks":
        reasons.append("non_crypto_instrument")
    base = instrument.get("baseCoin")
    if not isinstance(base, str) or symbol != base + "USDT":
        reasons.append("invalid_base_coin")
    if isinstance(base, str) and base in STABLECOIN_BASES:
        reasons.append("stablecoin_base")
    launched = _number(instrument.get("launchTime"))
    if (
        launched is None
        or launched <= 0
        or now - launched / 1000 < min_listing_days * DAY
    ):
        reasons.append("listing_too_young_or_unknown")
    if instrument.get("isPreListing") is not False:
        reasons.append("prelisting_or_unknown")
    ticker = ticker or {}
    turnover = _number(ticker.get("turnover24h"))
    if turnover is None or turnover <= 0 or turnover < min_turnover_usdt:
        reasons.append("low_or_invalid_turnover")
    bid, ask = _number(ticker.get("bid1Price")), _number(ticker.get("ask1Price"))
    spread = None
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        reasons.append("invalid_bid_ask")
    else:
        spread = (ask - bid) / (bid / 2 + ask / 2) * 10000
        if spread > max_spread_bps:
            reasons.append("wide_spread")
    pending = []
    depths = [None, None]
    if not isinstance(book, dict):
        pending.append("book_missing")
    else:
        timestamp = _number(book.get("as_of"))
        depths = [
            _number(book.get(key)) for key in ("bid_depth_usdt", "ask_depth_usdt")
        ]
        if (
            timestamp is None
            or timestamp < 0
            or not 0 <= now - timestamp <= max_book_age
        ):
            pending.append("book_stale_or_invalid_time")
        else:
            if "band_bps" in book and _number(book["band_bps"]) != 25:
                pending.append("book_invalid_band")
            if any(depth is None or depth < 0 for depth in depths):
                pending.append("book_invalid_depth")
            elif any(depth < min_depth_usdt for depth in depths):
                reasons.append("insufficient_depth")
            if "spread_bps" in book:
                book_spread = _number(book["spread_bps"])
                if book_spread is None or book_spread < 0:
                    pending.append("book_invalid_spread")
                elif book_spread > max_spread_bps:
                    reasons.append("wide_book_spread")
    return {
        "status": (
            "ineligible"
            if reasons
            else "blocked_pending_fresh" if pending else "eligible"
        ),
        "reasons": reasons + pending,
        "turnover24h": turnover,
        "spread_bps": spread,
        "bid_depth_usdt": depths[0] if not reasons and not pending else None,
        "ask_depth_usdt": depths[1] if not reasons and not pending else None,
    }


def refresh_universe(
    previous: dict,
    metadata: dict[str, dict],
    tickers: list[dict],
    books: dict[str, dict],
    now: float,
    *,
    target=50,
    snapshot_as_of=None,
    min_listing_days=30,
    min_turnover_usdt=20_000_000,
    max_spread_bps=10,
    min_depth_usdt=25_000,
    max_book_age=120,
    max_snapshot_age=120,
    replacement_ratio=1.5,
    confirmation_observations=2,
    max_daily_replacements=5,
) -> dict:
    """Select entries without mutating inputs; return strictly JSON-safe state.

    Raw instruments use Bybit fields, including millisecond ``launchTime``.
    ``snapshot_as_of`` and optional ticker ``as_of`` use seconds. Untimestamped
    raw tickers are attested by the caller to have been fetched at ``now``.
    Empty, incomplete, stale, future, or out-of-order market snapshots raise
    ValueError; the caller must retain previous state on that failure.

    Turnovers for ALL observed ASCII USDT symbols survive for seven days (at
    most 28 samples each). Samples and confirmation observations are >=6h apart,
    anchored to actual observations rather than bucket boundaries. Partial
    history is labelled available_24h_median with its exact sample_count.

    Missing/stale/malformed books reserve incumbent slots but block entries.
    Eligible details include validated bid/ask depths; blocked depths are null.
    Fresh shallow books and other hard failures remove members immediately;
    qualified hole fills do not consume the daily healthy replacement budget.
    Healthy replacements pair strongest challengers with weakest incumbents,
    require consecutive observations against the SAME incumbent, and run only
    on a new six-hour observation. ``last_rotation_day`` is the UTC budget day.
    """
    now = _number(now)
    numeric = tuple(
        _number(x)
        for x in (
            min_listing_days,
            min_turnover_usdt,
            max_spread_bps,
            min_depth_usdt,
            max_book_age,
            max_snapshot_age,
            replacement_ratio,
        )
    )
    if (
        now is None
        or now < 0
        or any(x is None or x <= 0 for x in numeric)
        or any(
            not isinstance(x, int) or isinstance(x, bool)
            for x in (target, confirmation_observations, max_daily_replacements)
        )
        or target < 1
        or confirmation_observations < 1
        or max_daily_replacements < 0
        or numeric[-1] <= 1
    ):
        raise ValueError("Invalid universe policy or time")
    (
        min_listing_days,
        min_turnover_usdt,
        max_spread_bps,
        min_depth_usdt,
        max_book_age,
        max_snapshot_age,
        replacement_ratio,
    ) = numeric
    if (
        not isinstance(previous, dict)
        or not isinstance(metadata, dict)
        or not metadata
        or not isinstance(tickers, list)
        or not tickers
        or not isinstance(books, dict)
        or any(
            not isinstance(s, str)
            or not isinstance(row, dict)
            or not {"status", "contractType", "quoteCoin", "settleCoin"} <= row.keys()
            for s, row in metadata.items()
        )
    ):
        raise ValueError("Empty or invalid universe snapshot")
    try:
        day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        raise ValueError("Invalid universe time") from None
    at = now if snapshot_as_of is None else _number(snapshot_as_of)
    if at is None or at < 0 or not 0 <= now - at <= max_snapshot_age:
        raise ValueError("Stale or future universe snapshot")
    by_symbol = {}
    timestamps = [at]
    for ticker in tickers:
        symbol = ticker.get("symbol") if isinstance(ticker, dict) else None
        if not isinstance(symbol, str) or symbol not in metadata or symbol in by_symbol:
            raise ValueError("Incomplete or duplicate ticker/metadata snapshot")
        if not {"turnover24h", "bid1Price", "ask1Price"} <= ticker.keys():
            raise ValueError("Incomplete ticker fields")
        if "as_of" in ticker:
            timestamp = _number(ticker["as_of"])
            if (
                timestamp is None
                or timestamp < 0
                or not 0 <= now - timestamp <= max_snapshot_age
            ):
                raise ValueError("Stale or future ticker snapshot")
            timestamps.append(timestamp)
        by_symbol[symbol] = ticker
    at = min(timestamps)
    required = {
        s
        for s, row in metadata.items()
        if row.get("status") == "Trading"
        and row.get("contractType") == "LinearPerpetual"
        and row.get("quoteCoin") == row.get("settleCoin") == "USDT"
    }
    if required - by_symbol.keys():
        raise ValueError("Incomplete ticker snapshot")
    old_active = previous.get("active_symbols", [])
    old_at = _number(previous.get("as_of", 0))
    if (
        not isinstance(old_active, list)
        or any(not _symbol(s) for s in old_active)
        or len(set(old_active)) != len(old_active)
        or old_at is None
        or not 0 <= old_at <= at
    ):
        raise ValueError("Invalid previous state or out-of-order snapshot")
    snapshots = _history(previous, at)
    last_observation = _number(previous.get("last_observation_at"))
    if previous.get("last_observation_at") is not None and (
        last_observation is None or not 0 <= last_observation <= at
    ):
        raise ValueError("Invalid previous observation time")
    if last_observation is None:
        last_observation = max(
            (row["as_of"] for rows in snapshots.values() for row in rows), default=None
        )
    observation = last_observation is None or at - last_observation >= SAMPLE_INTERVAL
    for symbol, ticker in sorted(by_symbol.items()):
        turnover = _number(ticker.get("turnover24h"))
        if _symbol(symbol) and turnover is not None and turnover > 0:
            rows = snapshots.setdefault(symbol, [])
            if not rows or (observation and at - rows[-1]["as_of"] >= SAMPLE_INTERVAL):
                rows.append({"as_of": at, "turnover24h": turnover})
            snapshots[symbol] = rows[-MAX_SAMPLES:]

    eligibility = {}
    for symbol in sorted(metadata.keys() | set(old_active)):
        details = _eligibility(
            symbol,
            metadata.get(symbol),
            by_symbol.get(symbol),
            books.get(symbol),
            now,
            min_listing_days=min_listing_days,
            min_turnover_usdt=min_turnover_usdt,
            max_spread_bps=max_spread_bps,
            min_depth_usdt=min_depth_usdt,
            max_book_age=max_book_age,
        )
        samples = snapshots.get(symbol, [])
        details.update(
            score=_median([r["turnover24h"] for r in samples]) if samples else None,
            sample_count=len(samples),
            rank=None,
            score_basis=(
                "7d_median" if len(samples) == MAX_SAMPLES else "available_24h_median"
            ),
        )
        eligibility[symbol] = details
    ranked = sorted(
        (s for s, d in eligibility.items() if d["score"] is not None),
        key=lambda s: (-eligibility[s]["score"], s),
    )
    # Ranks compare qualified candidates and retained book-pending incumbents.
    ranked = [
        s
        for s in ranked
        if eligibility[s]["status"] == "eligible"
        or (s in old_active and eligibility[s]["status"] == "blocked_pending_fresh")
    ]
    for rank, symbol in enumerate(ranked, 1):
        eligibility[symbol]["rank"] = rank
    order = lambda s: (-eligibility[s]["score"], s)
    active = {s for s in old_active if eligibility[s]["status"] != "ineligible"}
    if len(active) > target:
        raise ValueError("Target reduction requires an explicit membership migration")
    candidates = [s for s in ranked if eligibility[s]["status"] == "eligible"]
    for symbol in candidates:
        if len(active) >= target:
            break
        active.add(symbol)

    daily_count = previous.get("day_replacements", 0)
    if (
        not isinstance(daily_count, int)
        or isinstance(daily_count, bool)
        or daily_count < 0
    ):
        raise ValueError("Invalid previous replacement count")
    if previous.get("last_rotation_day") != day:
        daily_count = 0
    previous_streaks = previous.get("challenger_streaks", {})
    if not isinstance(previous_streaks, dict):
        raise ValueError("Invalid previous challenger streaks")
    streaks = {}
    weak = sorted(
        (
            s
            for s in active
            if s in old_active and eligibility[s]["status"] == "eligible"
        ),
        key=order,
        reverse=True,
    )
    for challenger in [s for s in candidates if s not in active]:
        if not weak:
            break
        incumbent = weak.pop(0)
        if (
            eligibility[challenger]["score"] / eligibility[incumbent]["score"]
            < replacement_ratio
        ):
            break
        prior = previous_streaks.get(challenger, {})
        if not isinstance(prior, dict):
            raise ValueError("Invalid previous challenger streak")
        count = prior.get("count", 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Invalid previous challenger count")
        same_pair = (
            prior.get("incumbent") == incumbent
            and prior.get("as_of") == last_observation
        )
        count = count if same_pair else 0
        if observation:
            count = min(count + 1, confirmation_observations)
        if (
            observation
            and count >= confirmation_observations
            and daily_count < max_daily_replacements
        ):
            active.remove(incumbent)
            active.add(challenger)
            daily_count += 1
        elif count:
            streaks[challenger] = {
                "incumbent": incumbent,
                "count": count,
                "as_of": at if observation else prior["as_of"],
            }

    active_symbols = sorted(active, key=order)
    blocked = {
        s: d["reasons"] for s, d in eligibility.items() if d["status"] != "eligible"
    }
    watchlist = []
    for symbol in candidates:
        if symbol in active:
            continue
        count = streaks.get(symbol, {}).get("count", 0)
        reason = (
            "daily_replacement_limit"
            if count >= confirmation_observations
            else "awaiting_confirmation" if count else "insufficient_replacement_margin"
        )
        watchlist.append(
            {
                "symbol": symbol,
                "score": eligibility[symbol]["score"],
                "rank": eligibility[symbol]["rank"],
                "reason": reason,
                "confirmation_count": count,
            }
        )
    pending = any(s in blocked for s in active)
    return {
        "as_of": at,
        "status": (
            "degraded"
            if pending
            else "ready" if len(active) == target else "underfilled"
        ),
        "target": target,
        "active_symbols": active_symbols,
        "members": {s: dict(eligibility[s]) for s in active_symbols},
        "eligibility": eligibility,
        "watchlist": watchlist,
        "blocked": blocked,
        "blocked_symbols": sorted(blocked),
        "block_reasons": blocked,
        "policy": {
            "target": target,
            "min_listing_days": min_listing_days,
            "min_turnover_usdt": min_turnover_usdt,
            "max_spread_bps": max_spread_bps,
            "min_depth_usdt": min_depth_usdt,
            "depth_band_bps": 25,
            "max_book_age": max_book_age,
            "max_snapshot_age": max_snapshot_age,
            "replacement_ratio": replacement_ratio,
            "confirmation_observations": confirmation_observations,
            "max_daily_replacements": max_daily_replacements,
        },
        "basis": {
            "method": "median_turnover24h",
            "history_days": 7,
            "sample_interval_seconds": SAMPLE_INTERVAL,
            "max_samples": MAX_SAMPLES,
            "cold_start": not active
            or any(eligibility[s]["sample_count"] < MAX_SAMPLES for s in active),
        },
        "snapshots": snapshots,
        "last_observation_at": at if observation else last_observation,
        "challenger_streaks": streaks,
        "last_rotation_day": day,
        "day_replacements": daily_count,
        "added": [s for s in active_symbols if s not in old_active],
        "removed": sorted(set(old_active) - active),
    }


def entry_symbols(
    state: dict, now: float, max_age=DAY, *, required_policy=None
) -> list[str]:
    """Return entry-authorized members, or [] for missing/stale/invalid state.

    Entry permission expires at age >= max_age. Book freshness is checked at
    refresh time; the runtime must still perform
    its usual live quote/entry checks. This helper never grants execution or
    capital authority and never treats a removed symbol as an exit instruction.
    """
    now, max_age = _number(now), _number(max_age)
    if (
        not isinstance(state, dict)
        or now is None
        or now < 0
        or max_age is None
        or max_age < 0
    ):
        return []
    if required_policy is not None:
        stored_policy = state.get("policy", {})
        if (
            not isinstance(required_policy, dict)
            or not isinstance(stored_policy, dict)
            or any(
                stored_policy.get(k) != value for k, value in required_policy.items()
            )
        ):
            return []
    at = _number(state.get("as_of"))
    if (
        at is None
        or at < 0
        or not 0 <= now - at < max_age
        or state.get("status") not in ("ready", "underfilled", "degraded")
    ):
        return []
    active, members = state.get("active_symbols"), state.get("members")
    if not isinstance(active, list) or not isinstance(members, dict):
        return []
    return list(
        dict.fromkeys(
            s
            for s in active
            if _symbol(s)
            and isinstance(members.get(s), dict)
            and members[s].get("status") == "eligible"
        )
    )
