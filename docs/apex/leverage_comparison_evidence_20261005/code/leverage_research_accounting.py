"""Offline USDT perpetual margin comparison; never an exchange liquidation model.

Leverage changes collateral requirements, not quantities, P&L or funding. The
caller owns the shared portfolio, entry guard, historical funding, and mark-price
stress checks. Last-price candles cannot establish actual exchange liquidations.
No production globals, strategy files or earlier research results are modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math

from . import risk, simulation
from .replay_management import _ManagementCache, _clone

__all__ = ["LeverageModel", "account_snapshot", "guard", "pending_requirement"]

DEFAULT_FEE_RATE = 0.00055
LIQUIDATION_RESERVE_RATE = 0.002


def _number(value, name, *, minimum=None, positive=False):
    if not simulation._finite(value):
        raise ValueError(f"{name} must be a finite number")
    if (positive and value <= 0) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}")
    return value


def _leverage(value):
    return _number(value, "leverage", minimum=1)


def _fee(value):
    _number(value, "fee rate", minimum=0)
    if value >= 1:
        raise ValueError("fee rate must be below 1")
    return value


def _record(trade, *, allow_unresolved=False):
    if not isinstance(trade, Mapping):
        raise ValueError("Trade must be a mapping")
    if not allow_unresolved:
        for flag in ("data_error", "data_gap", "funding_error"):
            if trade.get(flag):
                raise ValueError(f"Trade has an unresolved {flag}")
    for key in ("id", "symbol"):
        if not isinstance(trade.get(key), str) or not trade[key]:
            raise ValueError(f"Missing trade {key}")
    if trade.get("side") not in ("Buy", "Sell"):
        raise ValueError("Invalid perpetual side")
    if trade.get("research_market", "perpetual") != "perpetual":
        raise ValueError("Trade market must be perpetual")
    status = trade.get("status")
    if status not in ("PENDING", "OPEN", "CLOSED", "EXPIRED", "CANCELLED"):
        raise ValueError("Invalid trade status")
    qty = _number(trade.get("qty"), "qty", positive=True)
    remaining = _number(trade.get("remaining"), "remaining", minimum=0)
    if remaining > qty:
        raise ValueError("Remaining quantity exceeds initial quantity")
    _number(trade.get("limit"), "limit", positive=True)
    net = _number(trade.get("net_pnl"), "net_pnl")
    for key in ("gross_pnl", "net_pnl_before_funding", "funding", "fees"):
        if key in trade:
            _number(trade[key], key, minimum=0 if key == "fees" else None)
    if all(k in trade for k in ("gross_pnl", "fees", "funding")):
        if not math.isclose(
            net,
            trade["gross_pnl"] - trade["fees"] + trade["funding"],
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ValueError("Inconsistent trade cashflows")
    if "research_entry_fee_rate" in trade:
        _fee(trade["research_entry_fee_rate"])
    opened = trade.get("opened_at") is not None
    if not opened:
        if (
            status not in ("PENDING", "EXPIRED", "CANCELLED")
            or trade.get("entry") is not None
        ):
            raise ValueError("Unfilled trade has an entry or filled status")
        if any(
            trade.get(k, 0) != 0
            for k in (
                "net_pnl",
                "net_pnl_before_funding",
                "gross_pnl",
                "funding",
                "fees",
            )
        ):
            raise ValueError("Unfilled trade has cashflows")
        if status == "PENDING" and remaining != qty:
            raise ValueError("Pending quantity must be entirely unfilled")
    else:
        _number(trade["opened_at"], "opened_at", minimum=0)
        _number(trade.get("entry"), "entry", positive=True)
        if status not in ("OPEN", "CLOSED"):
            raise ValueError("Filled trade must be OPEN or CLOSED")
        if (status == "OPEN" and remaining == 0) or (
            status == "CLOSED" and remaining != 0
        ):
            raise ValueError("Remaining quantity does not match trade status")
    if status in ("PENDING", "OPEN"):
        _fee(trade.get("research_entry_fee_rate"))
    if trade.get("closed_at") is not None:
        _number(trade["closed_at"], "closed_at", minimum=0)
        if opened and trade["closed_at"] < trade["opened_at"]:
            raise ValueError("Trade closed before entry")


def pending_requirement(
    qty,
    entry,
    leverage,
    fee_rate=DEFAULT_FEE_RATE,
    *,
    upper_fill_price=None,
):
    """Reserve initial margin, entry fee and conservative estimated close fee.

    The close-fee reserve uses entry*(1+1/leverage) for BOTH directions (the
    higher unadjusted short bankruptcy price). It is collateral held aside, not
    an expense charged to equity. This is a research convention, not exact UTA
    order-margin replication. No unrealized profit funds new isolated positions.
    Monitored short IOC orders must supply their upper entry-zone boundary as
    upper_fill_price: a favorable higher fill can require MORE collateral.
    """
    _number(qty, "qty", positive=True)
    _number(entry, "entry", positive=True)
    _leverage(leverage)
    _fee(fee_rate)
    if upper_fill_price is not None:
        _number(upper_fill_price, "upper fill price", positive=True)
        if upper_fill_price < entry:
            raise ValueError("Upper fill price is below entry")
    price = entry if upper_fill_price is None else upper_fill_price
    result = qty * price * (1 / leverage + fee_rate * (2 + 1 / leverage))
    return _number(result, "pending requirement", minimum=0)


def guard(entry, stop, side, leverage, maintenance_rate):
    """Mirror runtime's isolated stop-buffer rule using an ASSUMED MMR.

    buffer = entry * (1/leverage - maintenance_rate - .002).
    Accept only when buffer >= 2*abs(entry-stop). The estimated boundary uses
    this deliberately conservative buffer, omits maintenance deductions and is
    NOT an exact liquidation price. Funding erosion, mark/last divergence,
    historical tier changes and intrabar sequence require separate stress tests.
    Unknown/invalid inputs reject; the caller must not silently assume an MMR.
    """
    result = dict(
        allowed=False,
        reasons=[],
        buffer_price=None,
        stop_distance=None,
        estimated_liquidation_price=None,
        maintenance_rate=maintenance_rate,
        reserve_rate=LIQUIDATION_RESERVE_RATE,
        minimum_buffer_multiple=2,
        liquidation_verified=False,
    )
    try:
        _number(entry, "entry", positive=True)
        _number(stop, "stop", positive=True)
        _leverage(leverage)
        _number(maintenance_rate, "maintenance rate", minimum=0)
        if maintenance_rate >= 1:
            raise ValueError("maintenance rate must be below 1")
        if side not in ("Buy", "Sell"):
            raise ValueError("Invalid perpetual side")
        sign = 1 if side == "Buy" else -1
        if sign * (entry - stop) <= 0:
            raise ValueError("Stop must be on the loss side of entry")
        buffer = entry * (1 / leverage - maintenance_rate - LIQUIDATION_RESERVE_RATE)
        distance = abs(entry - stop)
        for value in (buffer, distance, entry - sign * buffer):
            _number(value, "guard calculation")
        result.update(
            buffer_price=buffer,
            stop_distance=distance,
            estimated_liquidation_price=entry - sign * buffer,
        )
        if buffer < 2 * distance:
            result["reasons"].append("ESTIMATED_LIQUIDATION_BUFFER_INSUFFICIENT")
        else:
            result["allowed"] = True
    except (ValueError, TypeError, OverflowError) as exc:
        result["reasons"].append("INVALID_LIQUIDATION_GUARD_INPUT: " + str(exc))
    return result


class LeverageModel:
    """Frozen long/short fills and risk sizing with isolated research fee globals.

    Leverage is intentionally absent: it belongs to account collateral and entry
    guards, never quantity multiplication. ``advance`` does not simulate forced
    liquidation. Callers must label mark-price stress coverage separately.
    """

    def __init__(self, daily, execution, fee_rate=DEFAULT_FEE_RATE):
        self._fee_rate = _fee(fee_rate)
        self.assess = _clone(
            risk.assess,
            BASE_COST_RATE=Decimal(str(fee_rate)) * 2 + Decimal("0.0006"),
        )
        self.assess.__annotations__ = dict(risk.assess.__annotations__)
        self._management = _ManagementCache(daily, execution)
        exit_trade = _clone(simulation._exit, FEE_RATE=fee_rate)
        manage = _clone(simulation._manage, _exit=exit_trade)
        self._advance = _clone(
            self._management.advance,
            FEE_RATE=fee_rate,
            _exit=exit_trade,
            _manage=manage,
        )

    @property
    def market(self):
        return "perpetual"

    @property
    def fee_rate(self):
        return self._fee_rate

    @property
    def entry_fee_rate(self):
        return self._fee_rate

    def advance(self, trade, bars, now, **kwargs):
        if not isinstance(trade, Mapping):
            raise ValueError("Trade must be a mapping")
        if trade.get("research_entry_fee_rate", self.fee_rate) != self.fee_rate:
            raise ValueError("Trade fee rate does not match research model")
        if trade.get("opened_at") is not None and (
            "research_entry_fee_rate" not in trade or "research_market" not in trade
        ):
            raise ValueError("Opened trade must originate from a research model")
        prepared = dict(trade, research_entry_fee_rate=self.fee_rate)
        _record(prepared, allow_unresolved=True)
        result = self._advance(prepared, bars, now, **kwargs)
        result["research_market"] = "perpetual"
        result["cost_model"] = (
            f"offline-research/perpetual/fee={self.fee_rate:g}"
            "/frozen-fill-and-slippage-rules/no-liquidation-engine"
        )
        result["research_assumptions"] = (
            "Long/short research only; explicit isolated collateral accounting; "
            "leverage never multiplies quantity; funding separately supplied; "
            "no actual mark-price liquidation simulation."
        )
        for fill in result.get("fills", []):
            fill["fee_asset"] = "quote"
        return result


def account_snapshot(trades, quotes, leverage, initial=10000):
    """Shared USDT book with isolated initial-margin reservations, no borrowing.

    wallet_balance = initial + all realized net cashflows (including fees,
    partial exits and supplied funding); equity adds SIGNED unrealized P&L.
    open_margin = remaining*entry/leverage. cash is wallet minus open_margin;
    available_cash further subtracts pending margin/fees and open close-fee
    reserves. Unrealized profits never become spendable cash. Negative totals
    are returned, never clamped; affordability/solvency belongs to the runner.

    Margin releases proportionally at partial exits. Open collateral remains at
    entry value, not current mark. Funding is charged to wallet; no automatic
    isolated margin replenishment or funding-deduction fallback is simulated.
    The caller must flag insufficient wallet/funding and missing mark coverage.
    """
    _leverage(leverage)
    _number(initial, "initial", minimum=0)
    if not isinstance(quotes, Mapping):
        raise ValueError("quotes must map symbols to current closed prices")
    for symbol, quote in quotes.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Invalid quote symbol")
        _number(quote, f"quote for {symbol}", positive=True)
    realized_net = unrealized = open_margin = gross_notional = 0.0
    reserved_pending = reserved_close_fees = 0.0
    open_count = pending_count = 0
    identities = set()
    for trade in trades:
        _record(trade)
        if trade["id"] in identities:
            raise ValueError(f"Duplicate trade id: {trade['id']}")
        identities.add(trade["id"])
        if "research_leverage" in trade:
            if _leverage(trade["research_leverage"]) != leverage:
                raise ValueError("Trade leverage does not match account")
        if trade.get("opened_at") is not None:
            realized_net += trade["net_pnl"]
        if trade["status"] == "OPEN":
            symbol = trade["symbol"]
            if symbol not in quotes:
                raise ValueError(f"Missing current closed quote for {symbol}")
            remaining, entry = trade["remaining"], trade["entry"]
            sign = 1 if trade["side"] == "Buy" else -1
            open_margin += remaining * entry / leverage
            gross_notional += remaining * quotes[symbol]
            unrealized += sign * remaining * (quotes[symbol] - entry)
            reserved_close_fees += (
                remaining
                * entry
                * (1 + 1 / leverage)
                * trade["research_entry_fee_rate"]
            )
            open_count += 1
        elif trade["status"] == "PENDING":
            upper_fill_price = None
            if trade["side"] == "Sell" and trade.get("entry_style") == "monitored_zone":
                upper_fill_price = _number(
                    trade.get("zone_high"),
                    "short upper fill price",
                    positive=True,
                )
            reserved_pending += pending_requirement(
                trade["qty"],
                trade["limit"],
                leverage,
                trade["research_entry_fee_rate"],
                upper_fill_price=upper_fill_price,
            )
            pending_count += 1
    wallet = initial + realized_net
    cash = wallet - open_margin
    result = dict(
        equity=wallet + unrealized,
        wallet_balance=wallet,
        cash=cash,
        available_cash=cash - reserved_pending - reserved_close_fees,
        open_margin=open_margin,
        initial_margin=open_margin,
        reserved_pending=reserved_pending,
        reserved_close_fees=reserved_close_fees,
        realized_net=realized_net,
        unrealized=unrealized,
        gross_notional=gross_notional,
        open_count=open_count,
        pending_count=pending_count,
    )
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite account total")
    return result
