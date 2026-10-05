"""Strategy risk accounting shared by runtime and Telegram controls.

Daily/weekly loss gates conservatively include the entire current unrealized
loss, even for positions opened in an earlier period. They are risk gates,
not reports of calendar-period investment returns. Deposits never become P&L.
"""

import math
from datetime import datetime, timezone


def finite(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def loss_metrics(state, equity, arm, now, mode="live", *, include_state=False):
    if not finite(equity) or equity <= 0 or not finite(now):
        raise ValueError("Risk accounting capital or time unavailable")
    records = list(state["trades" if arm else "orders"].values())
    records = [r for r in records if not arm or r.get("arm") == arm]
    closed = []
    for r in records:
        if r.get("status") != "CLOSED":
            continue
        pnl = r.get("net_pnl", r.get("net_pnl_before_funding"))
        stamp = r.get("closed_at")
        if not finite(pnl) or not finite(stamp) or not 0 < stamp <= now:
            raise ValueError("Closed trade accounting unavailable")
        closed.append((stamp, pnl))
    mtm = 0.0
    if arm:
        for r in records:
            if r.get("status") != "OPEN":
                continue
            mark, mark_at = r.get("mark_price"), r.get("mark_at")
            if (
                not finite(mark)
                or mark <= 0
                or not finite(mark_at)
                or not 0 <= now - mark_at <= 600
            ):
                raise ValueError("Open simulation valuation stale")
            if (
                r.get("side") not in {"Buy", "Sell"}
                or not finite(r.get("entry"))
                or r["entry"] <= 0
                or not finite(r.get("remaining"))
                or r["remaining"] <= 0
                or not finite(r.get("net_pnl"))
            ):
                raise ValueError("Invalid open simulated position or cashflow")
            sign = 1 if r["side"] == "Buy" else -1
            mtm += r.get("net_pnl", 0) + sign * (mark - r["entry"]) * r["remaining"]
    else:
        for p in state.get("account", {}).get("positions", []):
            try:
                size = float(p["size"])
                value = float(p.get("unrealisedPnl")) if size else 0.0
            except (ValueError, TypeError, KeyError):
                raise ValueError("Exchange unrealized P&L unavailable") from None
            if not math.isfinite(size) or size < 0 or not math.isfinite(value):
                raise ValueError("Invalid exchange valuation")
            mtm += value
        # Realized partial exits and fees belong to the strategy while open.
        for r in records:
            if r.get("status") == "OPEN":
                value = r.get("net_pnl_before_funding")
                if value is None:
                    value = r.get("gross_pnl", 0) - r.get("fees", 0)
                if not finite(value) or not finite(r.get("funding", 0)):
                    raise ValueError("Partial exit accounting unavailable")
                mtm += value + r.get("funding", 0)
    if not finite(mtm):
        raise ValueError("Strategy valuation is not finite")
    stamp = datetime.fromtimestamp(now, timezone.utc)
    today = stamp.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    week = today - stamp.weekday() * 86400
    key = arm or mode
    base = state.get("risk_reference_equities", {}).get(
        key, state.get("risk_reference_equity", equity)
    )
    if not finite(base) or base <= 0:
        raise ValueError("Risk reference capital unavailable")
    running = peak = base
    for _, pnl in sorted(closed):
        running += pnl
        peak = max(peak, running)
    if not finite(running) or not finite(peak):
        raise ValueError("Strategy capital is not finite")
    previous = state.get("risk_circuits", {}).get(key, {})
    prior_peak = previous.get("strategy_peak", base)
    if not finite(prior_peak) or prior_peak <= 0:
        raise ValueError("Persisted strategy peak is invalid")
    peak = max(peak, prior_peak)
    running += mtm
    if not finite(running):
        raise ValueError("Strategy valuation overflow")
    peak = max(peak, running)
    conservative_open_loss = min(0, mtm)
    result = {
        "daily_loss_pct": max(
            0,
            -(sum(p for t, p in closed if t >= today) + conservative_open_loss)
            / equity
            * 100,
        ),
        "weekly_loss_pct": max(
            0,
            -(sum(p for t, p in closed if t >= week) + conservative_open_loss)
            / equity
            * 100,
        ),
        "drawdown_pct": max(0, (peak - running) / peak * 100),
    }
    if not all(finite(value) for value in result.values()):
        raise ValueError("Risk metric overflow")
    if include_state:
        result.update(strategy_peak=peak, strategy_value=running)
    return result
