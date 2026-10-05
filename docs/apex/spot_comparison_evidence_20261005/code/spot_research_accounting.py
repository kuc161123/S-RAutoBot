"""OFFLINE long-only spot versus fully collateralized 1x perpetual accounting.

Spot entry fees are a fractional base-fee equivalent: simulator quantities are
NET owned base, with gross purchases of net / (1 - fee). Frozen synthetic lot
rules apply to net quantities; gross purchases/fees need not meet venue rounding.
Frozen fills, slippage and management remain unchanged. Research only, not live;
no liquidity, real spot filters, borrowing, leverage or automatic funding model.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math

from . import risk, simulation
from .replay_management import _ManagementCache, _clone

__all__ = ["ResearchModel", "account_snapshot"]


def _market(market):
    if market not in ("spot", "perpetual"):
        raise ValueError("market must be 'spot' or 'perpetual'")


def _number(value, name, *, minimum=None, positive=False):
    if not simulation._finite(value):
        raise ValueError(f"{name} must be a finite number")
    if (positive and value <= 0) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}")
    return value


def _record(trade, market, *, allow_unresolved=False):
    """Validate financial inputs without inferring missing cashflows or prices."""
    if not isinstance(trade, Mapping):
        raise ValueError("Trade must be a mapping")
    if not allow_unresolved:
        for flag in ("data_error", "data_gap"):
            if trade.get(flag):
                raise ValueError(f"Trade has a simulation {flag}")
    for key in ("id", "symbol"):
        if not isinstance(trade.get(key), str) or not trade[key]:
            raise ValueError(f"Missing trade {key}")
    if trade.get("side") != "Buy":
        raise ValueError("Research accounting is long-only (Buy)")
    if trade.get("research_market", market) != market:
        raise ValueError("Trade market does not match account market")
    status = trade.get("status")
    if status not in ("PENDING", "OPEN", "CLOSED", "EXPIRED", "CANCELLED"):
        raise ValueError("Invalid trade status")
    qty = _number(trade.get("qty"), "qty", positive=True)
    remaining = _number(trade.get("remaining"), "remaining", minimum=0)
    _number(trade.get("limit"), "limit", positive=True)
    if remaining > qty:
        raise ValueError("Remaining quantity exceeds owned base")
    _number(trade.get("net_pnl"), "net_pnl")
    for key in ("gross_pnl", "net_pnl_before_funding", "funding", "fees"):
        if key in trade:
            _number(trade[key], key, minimum=0 if key == "fees" else None)
    if "research_entry_fee_rate" in trade:
        rate = _number(trade["research_entry_fee_rate"], "entry fee rate", minimum=0)
        if market == "perpetual" and rate >= 1:
            raise ValueError("Perpetual entry fee rate must be below 1")
    opened = trade.get("opened_at") is not None
    if not opened:
        if status not in ("PENDING", "EXPIRED", "CANCELLED") or trade.get("entry") is not None:
            raise ValueError("Unfilled trade has an entry or filled status")
        if any(trade.get(key, 0) != 0 for key in (
            "net_pnl", "net_pnl_before_funding", "gross_pnl", "funding", "fees",
        )):
            raise ValueError("Unfilled trades must have zero cashflows and fees")
        if status == "PENDING":
            if remaining != qty:
                raise ValueError("Pending quantity must be entirely unfilled")
            _number(trade.get("research_entry_fee_rate"), "entry fee rate", minimum=0)
    else:
        _number(trade["opened_at"], "opened_at", minimum=0)
        _number(trade.get("entry"), "entry", positive=True)
        if status not in ("OPEN", "CLOSED"):
            raise ValueError("Filled trade must be OPEN or CLOSED")
        if (status == "OPEN" and remaining == 0) or (status == "CLOSED" and remaining != 0):
            raise ValueError("Remaining quantity does not match trade status")
    if trade.get("closed_at") is not None:
        _number(trade["closed_at"], "closed_at", minimum=0)
        if opened and trade["closed_at"] < trade["opened_at"]:
            raise ValueError("Trade closed before entry")


class ResearchModel:
    """Own isolated fee-adjusted functions and a frozen-history management cache.

    ``assess`` has the exact ``risk.assess`` interface and unchanged gates, with
    the fee/slippage allowance replaced. The caller supplies ``spread_pct=.01``
    (risk adds this separately) and selects long opportunities. ``create_trade``
    remains ``simulation.create_trade``. For a newly created pending trade, the
    runner stores ``research_entry_fee_rate=model.entry_fee_rate`` before taking
    a snapshot; ``advance`` also attaches it to its returned copy.
    """

    def __init__(self, market, daily, execution, fee_rate=None):
        _market(market)
        fee = (.001 if market == "spot" else .00055) if fee_rate is None else fee_rate
        _number(fee, "fee_rate", minimum=0)
        if fee >= 1:
            raise ValueError("fee_rate must be below 1")
        self._market = market
        self._fee_rate = fee
        self._entry_fee_rate = fee / (1 - fee) if market == "spot" else fee
        self.assess = _clone(
            risk.assess,
            BASE_COST_RATE=(
                Decimal(str(self.entry_fee_rate)) + Decimal(str(fee)) + Decimal("0.0006")
            ),
        )
        self.assess.__annotations__ = dict(risk.assess.__annotations__)
        self._management = _ManagementCache(daily, execution)
        exit_trade = _clone(simulation._exit, FEE_RATE=fee)
        manage = _clone(simulation._manage, _exit=exit_trade)
        self._advance = _clone(
            self._management.advance,
            FEE_RATE=self.entry_fee_rate,
            _exit=exit_trade,
            _manage=manage,
        )

    @property
    def market(self):
        return self._market

    @property
    def fee_rate(self):
        return self._fee_rate

    @property
    def entry_fee_rate(self):
        return self._entry_fee_rate

    def advance(self, trade, bars, now, **kwargs):
        """Run frozen replay with isolated fees; add metadata, never resize fills.

        All keywords are forwarded to simulation.advance, including interval_ms,
        daily and execution. Owned histories enable caching, not implicit future
        management inputs. Invalid candles retain simulation's data_error result.
        """
        if not isinstance(trade, Mapping):
            raise ValueError("Trade must be a mapping")
        rate = trade.get("research_entry_fee_rate", self.entry_fee_rate)
        if rate != self.entry_fee_rate:
            raise ValueError("Trade entry fee rate does not match research model")
        if trade.get("opened_at") is not None and (
            "research_entry_fee_rate" not in trade or "research_market" not in trade
        ):
            raise ValueError("Opened trade must originate from this research model")
        prepared = dict(trade, research_entry_fee_rate=self.entry_fee_rate)
        # Replay can repair flagged history; account valuation must reject it.
        _record(prepared, self.market, allow_unresolved=True)
        result = self._advance(prepared, bars, now, **kwargs)
        result["research_market"] = self.market
        result["cost_model"] = (
            f"offline-research/{self.market}/entry-fee={self.entry_fee_rate:g}"
            f"/exit-fee={self.fee_rate:g}/frozen-fill-and-slippage-rules"
        )
        result["research_assumptions"] = (
            "Research only, not live; net synthetic lot quantities; full entry "
            "principal committed; funding only if separately supplied. "
            + ("Fractional withheld base-fee equivalent; gross purchase rounding omitted."
               if self.market == "spot" else "1x collateral; fees charged in quote.")
        )
        for fill in result.get("fills", []):
            if fill["reason"] != "ENTRY":
                continue
            gross = fill["qty"] / (1 - self.fee_rate) if self.market == "spot" else fill["qty"]
            fill["gross_purchase_qty"] = gross
            fill["fee_asset"] = "base" if self.market == "spot" else "quote"
            if self.market == "spot":
                fill["fee_base_qty"] = gross - fill["qty"]
        return result


def account_snapshot(trades, quotes, initial=10000, market="spot"):
    """Value an iterable of trades using explicit symbol -> current closed price.

    Returns equity, cash, reserved_pending, available_cash, unrealized,
    realized_net, invested_value, open_count and pending_count. Realized net
    includes fees and partial exits of EVERY opened trade, including time-zero
    entries. invested_value is remaining base marked at the supplied quote;
    cash subtracts remaining ENTRY principal for both spot and 1x perpetuals.
    No hypothetical liquidation fee is deducted from unrealized P&L.

    Pending reservations require the stored research_entry_fee_rate, including
    custom fees. Terminal unfilled trades cost zero and reserve nothing. Missing
    or invalid inputs and unresolved data flags raise ValueError; inputs are
    never mutated. Negative cash
    is reported, not funded or clamped: the runner must enforce affordability.
    """
    _market(market)
    _number(initial, "initial", minimum=0)
    if not isinstance(quotes, Mapping):
        raise ValueError("quotes must map symbols to current closed prices")
    for symbol, quote in quotes.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Invalid quote symbol")
        _number(quote, f"quote for {symbol}", positive=True)
    realized_net = unrealized = principal = invested_value = reserved = 0.0
    open_count = pending_count = 0
    identities = set()
    for trade in trades:
        _record(trade, market)
        if trade["id"] in identities:
            raise ValueError(f"Duplicate trade id: {trade['id']}")
        identities.add(trade["id"])
        if trade.get("opened_at") is not None:
            realized_net += trade["net_pnl"]
        if trade["status"] == "OPEN":
            symbol = trade["symbol"]
            if symbol not in quotes:
                raise ValueError(f"Missing current closed quote for {symbol}")
            remaining = trade["remaining"]
            principal += remaining * trade["entry"]
            invested_value += remaining * quotes[symbol]
            unrealized += remaining * (quotes[symbol] - trade["entry"])
            open_count += 1
        elif trade["status"] == "PENDING":
            reserved += trade["qty"] * trade["limit"] * (1 + trade["research_entry_fee_rate"])
            pending_count += 1
    cash = initial + realized_net - principal
    result = dict(
        equity=initial + realized_net + unrealized,
        cash=cash,
        reserved_pending=reserved,
        available_cash=cash - reserved,
        unrealized=unrealized,
        realized_net=realized_net,
        invested_value=invested_value,
        open_count=open_count,
        pending_count=pending_count,
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite account total")
    return result
