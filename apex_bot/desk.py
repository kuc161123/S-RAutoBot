"""Live Hyperliquid trading desk that mirrors the Apex shadow book.

Every new PENDING trade in the followed shadow book becomes a live order. Signal
levels (limit entry, stop, target 1, target 2) come unchanged from the Apex engine.
The desk owns only sizing, leverage and exchange mechanics:

* size = risk dollars / (stop distance + estimated round-trip fees), rounded down;
* leverage = min(user cap or per-symbol setting, exchange max, liquidation-safe cap),
  where the liquidation-safe cap keeps the liquidation price well beyond the stop;
* the stop-loss always lives on the exchange as a reduce-only stop-market trigger,
  sized to the full position, and is re-placed if it ever goes missing;
* target 1 closes half (one lot stays intact, odd lots round up, as in the shadow
  book) and moves the stop to the entry price; target 2 closes the rest;
* when the shadow book ratchets its stop, the live stop follows (tighten only);
  when the shadow book exits for a structural reason, the live position is closed.

State lives in the shared Apex state document under "desk" and is written right
after every exchange action, so a restart never forgets an order it placed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone

from .hl_client import (
    MARKET_SLIPPAGE,
    MIN_NOTIONAL_USD,
    STOP_SLIPPAGE,
    HLError,
    cloid_for,
    map_coin,
    round_price,
    round_size,
    size_step,
)

LOG = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "risk_mode": "usd",  # usd | pct
    "risk_usd": 50.0,
    "risk_pct": 1.0,
    "max_leverage": 5,
    "symbol_leverage": {},  # coin -> leverage
    "max_positions": 6,
    "daily_loss_usd": 200.0,  # 0 disables
    "direction": "both",  # both | long | short
    "blocked": [],  # coins
    "book": "shadow_book",
    "layer": "book",  # book | all (Shadow Book layers: BOOK only, or BOOK + SIGNAL)
    "paused": False,
    "watchlist_alerts": "auto",  # auto | on | off (auto = on only while not live)
    "shadow_alerts": "auto",  # auto | on | off
    "fee_rate": 0.00045,
}

LIMITS = {
    "risk_usd": (1.0, 100000.0),
    "risk_pct": (0.05, 10.0),
    "max_leverage": (1, 50),
    "max_positions": (1, 30),
    "daily_loss_usd": (0.0, 1000000.0),
}

BOOKS = {
    "shadow_book": "Apex Kapital Shadow Book",
    "baseline_shadow": "Railway engine · rules book",
    "ai_shadow": "Railway engine · AI book",
}
ACTIVE = ("PLACING", "ORDERED", "OPEN")
# Shadow exits the exchange orders already cover; anything else is structural.
PRICE_EXITS = {"STOP", "TP1", "TP2", "UNFILLED", "TARGET_BEFORE_ENTRY"}
LIQ_BUFFER = 1.5  # liquidation distance must be >= 1.5x the stop distance
PRICE_SANITY = 0.05  # skip if Hyperliquid mid differs >5% from the signal entry
TICK_SECONDS = 4


class Skip(Exception):
    def __init__(self, reason: str, *, retry: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.retry = retry


def desk_state(state: dict) -> dict:
    desk = state.setdefault("desk", {})
    desk.setdefault("settings", {})
    desk.setdefault("trades", {})
    desk.setdefault("since", time.time())
    return desk


def normalize_settings(raw) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    settings["symbol_leverage"] = {}
    settings["blocked"] = []
    if isinstance(raw, dict):
        for key, default in DEFAULT_SETTINGS.items():
            value = raw.get(key, default)
            if key in LIMITS:
                low, high = LIMITS[key]
                try:
                    value = type(default)(value)
                except (TypeError, ValueError):
                    value = default
                if not (isinstance(value, (int, float)) and math.isfinite(value)):
                    value = default
                value = min(max(value, low), high)
            elif key == "symbol_leverage":
                value = {
                    str(k): int(v)
                    for k, v in (value or {}).items()
                    if isinstance(v, (int, float)) and 1 <= int(v) <= 50
                }
            elif key == "blocked":
                value = sorted({str(x) for x in (value or []) if str(x)})
            elif key == "risk_mode" and value not in ("usd", "pct"):
                value = default
            elif key == "direction" and value not in ("both", "long", "short"):
                value = default
            elif key == "book" and value not in BOOKS:
                value = default
            elif key == "layer" and value not in ("book", "all"):
                value = default
            elif key == "paused":
                value = bool(value)
            elif key in ("watchlist_alerts", "shadow_alerts"):
                if value is True:
                    value = "on"
                elif value is False:
                    value = "off"
                if value not in ("auto", "on", "off"):
                    value = default
            elif key == "fee_rate":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = default
                if not 0 <= value <= 0.005:
                    value = default
            settings[key] = value
    return settings


def risk_dollars(settings: dict, equity: float) -> float:
    if settings["risk_mode"] == "pct":
        return max(0.0, equity) * settings["risk_pct"] / 100.0
    return settings["risk_usd"]


def liquidation_safe_leverage(stop_fraction: float, exchange_max: int) -> int:
    """Highest isolated leverage whose liquidation sits >= LIQ_BUFFER x the stop.

    Isolated liquidation distance ~= 1/L - maintenance, where Hyperliquid's
    maintenance margin is half the initial margin at the asset's max leverage.
    """
    maintenance = 1.0 / (2.0 * max(1, exchange_max))
    bound = 1.0 / (LIQ_BUFFER * stop_fraction + maintenance)
    return max(1, int(math.floor(bound)))


def plan_entry(signal: dict, settings: dict, asset: dict, coin: str, equity: float, available: float, mid: float | None) -> dict:
    """Pure sizing and leverage plan. Raises Skip with a readable reason."""
    side = signal.get("side")
    if side not in ("Buy", "Sell"):
        raise Skip("Unknown direction")
    long = side == "Buy"
    entry = float(signal["limit"])
    stop = float(signal["stop"])
    tp1 = float(signal["target1"])
    tp2 = float(signal["target2"])
    sign = 1 if long else -1
    if not (sign * (entry - stop) > 0 and sign * (tp1 - entry) > 0 and sign * (tp2 - tp1) >= 0):
        raise Skip("Signal levels are not in a valid order")
    if mid is None or not math.isfinite(mid) or mid <= 0:
        raise Skip("No Hyperliquid price for this coin", retry=True)
    if abs(mid / entry - 1) > PRICE_SANITY:
        raise Skip(f"Hyperliquid price {mid:g} is more than {PRICE_SANITY:.0%} from the signal entry")
    if sign * (mid - stop) <= 0:
        raise Skip("Price is already through the stop")
    if sign * (mid - tp1) >= 0:
        raise Skip("Price already reached target 1")
    risk = risk_dollars(settings, equity)
    if risk <= 0:
        raise Skip("Risk amount is zero (check equity or risk setting)", retry=True)
    distance = abs(entry - stop)
    fees = settings["fee_rate"] * (entry + stop)
    szd = asset["sz_decimals"]
    size = round_size(risk / (distance + fees), szd)
    if size <= 0:
        raise Skip("Position size rounds to zero at this risk")
    notional = size * entry
    if size * min(entry, stop) < MIN_NOTIONAL_USD * 1.05:
        raise Skip(
            f"Order value ${notional:.2f} is too small for Hyperliquid's ${MIN_NOTIONAL_USD:.0f} minimum "
            "(raise the risk per trade)"
        )
    stop_fraction = distance / entry
    exchange_max = max(1, int(asset["max_leverage"]))
    wanted = int(settings["symbol_leverage"].get(coin, settings["max_leverage"]))
    safe = liquidation_safe_leverage(stop_fraction, exchange_max)
    leverage = max(1, min(wanted, exchange_max, safe))
    margin = notional / leverage
    if margin > available * 0.95:
        raise Skip(
            f"Not enough free margin: needs ${margin:,.2f} at {leverage}x, ${available:,.2f} available",
            retry=True,
        )
    lots = int(round(size / size_step(szd)))
    share = float(signal.get("tp1_share") or 50.0)
    tp1_lots = min(lots - 1, max(1, math.ceil(lots * share / 100.0))) if lots > 1 else 0
    tp1_size = round_size(tp1_lots * size_step(szd), szd)
    if tp1_size * tp1 < MIN_NOTIONAL_USD * 1.05 or (size - tp1_size) * tp2 < MIN_NOTIONAL_USD * 1.05:
        tp1_size = 0.0  # too small to split; target 2 takes it all, stop still moves at target 1
    return {
        "coin": coin,
        "side": side,
        "is_buy": long,
        "entry": round_price(entry, szd),
        "stop": round_price(stop, szd),
        "target1": round_price(tp1, szd),
        "target2": round_price(tp2, szd),
        "size": size,
        "tp1_size": tp1_size,
        "sz_decimals": szd,
        "notional": notional,
        "risk_usd": size * (distance + fees),
        "leverage": leverage,
        "leverage_wanted": wanted,
        "leverage_capped_by": (
            "liquidation safety" if leverage == safe and safe < min(wanted, exchange_max)
            else "exchange maximum" if leverage == exchange_max and exchange_max < wanted
            else None
        ),
        "margin": margin,
    }


def _stop_order(trade: dict, size: float, price: float, version: int) -> dict:
    long = trade["is_buy"]
    szd = trade["sz_decimals"]
    limit = price * (1 - STOP_SLIPPAGE) if long else price * (1 + STOP_SLIPPAGE)
    return {
        "coin": trade["coin"],
        "is_buy": not long,
        "size": round_size(size, szd),
        "limit_px": round_price(limit, szd),
        "trigger_px": round_price(price, szd),
        "kind": "sl",
        "reduce_only": True,
        "cloid": cloid_for(trade["id"], f"sl{version}"),
    }


def _target_order(trade: dict, size: float, price: float, role: str) -> dict:
    long = trade["is_buy"]
    szd = trade["sz_decimals"]
    limit = price * (1 - STOP_SLIPPAGE) if long else price * (1 + STOP_SLIPPAGE)
    return {
        "coin": trade["coin"],
        "is_buy": not long,
        "size": round_size(size, szd),
        "limit_px": round_price(limit, szd),
        "trigger_px": round_price(price, szd),
        "kind": "tp",
        "reduce_only": True,
        "cloid": cloid_for(trade["id"], role),
    }


def short_id(trade_id: str) -> str:
    import hashlib

    return hashlib.sha256(trade_id.encode()).hexdigest()[:6].upper()


def _utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


class Desk:
    """Owns the live Hyperliquid book. One instance per leader process."""

    def __init__(self, store, client, *, network: str = "mainnet", clock=time.time, source=None):
        self.store = store
        self.client = client
        self.source = source
        self.network = network
        self.clock = clock
        self.lock = asyncio.Lock()
        self.assets: dict = {}
        self.assets_at = 0.0
        self.snapshot: dict = {"status": "starting" if client else "no_keys"}
        self.announced = False
        self._sl_failures: dict = {}
        self._deferred_notified: set = set()

    @property
    def configured(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ helpers
    async def settings(self) -> dict:
        state = await self.store.read()
        return normalize_settings(state.get("desk", {}).get("settings"))

    async def _save(self, trade_id: str, changes: dict | None = None, event: tuple | None = None, new: dict | None = None):
        def write(tx):
            desk = desk_state(tx.state)
            if new is not None:
                desk["trades"][trade_id] = new
            elif changes:
                desk["trades"].setdefault(trade_id, {}).update(changes)
            if event:
                key, payload, text = event
                tx.event(key, "desk", payload, text)

        await self.store.update(write)

    async def _notify(self, key: str, text: str, payload: dict | None = None):
        def write(tx):
            tx.event(key, "desk", payload or {}, text)

        await self.store.update(write)

    async def _refresh_assets(self, force=False):
        if force or not self.assets or self.clock() - self.assets_at > 3600:
            self.assets = await self.client.assets()
            self.assets_at = self.clock()

    # --------------------------------------------------------------------- tick
    async def tick(self):
        if not self.client:
            self.snapshot = {"status": "no_keys", "at": self.clock()}
            return
        await self.store.assert_leader()
        async with self.lock:
            await self._refresh_assets()
            account = await self.client.account()
            orders = await self.client.open_orders()
            mids = await self.client.mids()
            state = await self.store.read()
            desk = desk_state(state)
            settings = normalize_settings(desk.get("settings"))
            if self.source is not None and settings["book"] == "shadow_book":
                await self.source.refresh()
            self.snapshot = {
                "status": "live",
                "at": self.clock(),
                "equity": account["equity"],
                "available": account["available"],
                "margin_used": account["margin_used"],
                "positions": account["positions"],
                "orders": orders,
                "mids": mids,
            }
            if not self.announced:
                self.announced = True
                await self._notify(
                    f"desk:online:{int(self.clock() // 60)}",
                    "🏦 Apex Desk online · Hyperliquid LIVE\n"
                    f"Account {self.client.address[:6]}…{self.client.address[-4:]} · equity ${account['equity']:,.2f}\n"
                    f"Signals: {BOOKS[settings['book']]} · {self.source_status(settings)}\n"
                    f"Risk {self._risk_label(settings)} · max leverage {settings['max_leverage']}x · "
                    f"{'⏸ paused' if settings['paused'] else '▶️ taking signals'}\n"
                    + (
                        f"Markets: crypto + {', '.join(d for d in getattr(self.client, 'dexs', ()) if d)} "
                        f"(shared collateral {getattr(self.client, 'dex_abstraction', None) or 'n/a'})\n"
                        if len(getattr(self.client, "dexs", ())) > 1
                        else ""
                    )
                    + "Open /dashboard for controls.",
                )
            for trade in list(desk["trades"].values()):
                if trade.get("status") in ACTIVE:
                    try:
                        await self._manage(trade, state, account, orders, mids)
                    except Exception as exc:  # noqa: BLE001 - one trade must not block others
                        LOG.warning("manage %s failed: %s", trade.get("coin"), type(exc).__name__)
                        await self._notify(
                            f"desk:{trade['id']}:fault:{int(self.clock() // 1800)}",
                            f"⚠️ {trade.get('coin')} management retrying ({exc if isinstance(exc, HLError) else type(exc).__name__}). "
                            "The exchange stop-loss stays in place.",
                        )
            if not settings["paused"]:
                await self._intake(state, settings, account, orders, mids)
            await self._cleanup_orphans(state, orders)
            if len(desk["trades"]) > 1200:
                await self._prune()

    async def _prune(self, keep: int = 1000):
        def write(tx):
            trades = desk_state(tx.state)["trades"]
            done = sorted(
                (t for t in trades.values() if t.get("status") not in ACTIVE),
                key=lambda t: t.get("closed_at") or t.get("created_at") or 0,
            )
            for t in done[: max(0, len(trades) - keep)]:
                trades.pop(t["id"], None)

        await self.store.update(write)

    def source_status(self, settings=None) -> str:
        if settings and settings.get("book") != "shadow_book":
            return "engine connected"
        if self.source is None or not self.source.configured:
            return "Shadow Book NOT connected (add APEX_SHADOW_BOOK_CSV_URL or a sheet ID + Google key)"
        if self.source.ok:
            return f"Shadow Book read OK ({len(self.source.signals)} rows)"
        return f"Shadow Book unavailable: {self.source.error or 'waiting for first read'}"

    def _shadow(self, state, shadow_id) -> dict:
        if str(shadow_id).startswith("shadowbook:"):
            if self.source is None or not self.source.ok:
                return {}
            return self.source.signals.get(shadow_id, {})
        return state.get("trades", {}).get(shadow_id, {})

    def _risk_label(self, settings):
        if settings["risk_mode"] == "pct":
            return f"{settings['risk_pct']:g}% of equity"
        return f"${settings['risk_usd']:,.2f} per trade"

    # ------------------------------------------------------------------- intake
    def _candidates(self, state: dict, settings: dict) -> list[dict]:
        now = self.clock()
        desk = desk_state(state)
        out = []
        if settings["book"] == "shadow_book":
            if self.source is None or not self.source.ok:
                return []
            for signal in self.source.signals.values():
                if (
                    signal["status"] == "PENDING"
                    and signal.get("entry_type", "LIMIT") == "LIMIT"
                    and (settings["layer"] == "all" or signal.get("layer") == "BOOK")
                    and signal["id"] not in desk["trades"]
                    and signal["expires_at"] > now + 60
                    and signal.get("created_at", 0) > now - 72 * 3600
                ):
                    out.append(signal)
            out.sort(key=lambda t: t.get("created_at", 0))
            return out
        for trade in state.get("trades", {}).values():
            if (
                trade.get("arm") == settings["book"]
                and trade.get("status") == "PENDING"
                and trade.get("entry_style", "confirmed") == "confirmed"
                and trade.get("id") not in desk["trades"]
                and float(trade.get("expires_at", 0)) > now + 60
            ):
                out.append(trade)
        out.sort(key=lambda t: t.get("created_at", 0))
        return out

    async def _intake(self, state, settings, account, orders, mids):
        desk = desk_state(state)
        active = [t for t in desk["trades"].values() if t.get("status") in ACTIVE]
        busy = {t["coin"] for t in active}
        today = _utc_day(self.clock())
        realized_today = sum(
            t.get("pnl", 0.0)
            for t in desk["trades"].values()
            if t.get("status") == "CLOSED" and _utc_day(t.get("closed_at", 0)) == today
        )
        for signal in self._candidates(state, settings):
            trade_id = signal["id"]
            direct = signal.get("coin")
            if direct and direct in self.assets and not self.assets[direct].get("delisted"):
                coin = direct
            else:
                coin = map_coin(signal.get("symbol", ""), self.assets)
            try:
                if coin is None:
                    raise Skip(f"{signal.get('symbol')} is not listed on Hyperliquid")
                if coin in settings["blocked"]:
                    raise Skip(f"{coin} is on your blocklist")
                long = signal.get("side") == "Buy"
                if settings["direction"] == "long" and not long:
                    raise Skip("Shorts are switched off")
                if settings["direction"] == "short" and long:
                    raise Skip("Longs are switched off")
                if coin in busy or coin in account["positions"]:
                    raise Skip(f"Already have a position or order on {coin}")
                if len(active) >= settings["max_positions"]:
                    raise Skip(f"Max open positions reached ({settings['max_positions']})", retry=True)
                if settings["daily_loss_usd"] > 0 and realized_today <= -settings["daily_loss_usd"]:
                    raise Skip(
                        f"Daily loss stop hit (${-realized_today:,.2f} lost today, limit ${settings['daily_loss_usd']:,.0f})",
                        retry=True,
                    )
                asset = self.assets[coin]
                plan = plan_entry(signal, settings, asset, coin, account["equity"], account["available"], mids.get(coin))
            except Skip as skip:
                await self._skip(signal, coin, skip)
                continue
            trade = await self._open(signal, plan, settings)
            if trade and trade.get("status") in ACTIVE:
                active.append(trade)
                busy.add(coin)
                account["available"] -= plan["margin"]

    async def _skip(self, signal, coin, skip: Skip):
        from .desk_panel import skip_text

        trade_id = signal["id"]
        text = skip_text(signal, coin, skip.reason)
        if skip.retry:
            marker = (trade_id, skip.reason.split("(")[0])
            if marker in self._deferred_notified:
                return
            self._deferred_notified.add(marker)
            await self._notify(f"desk:{trade_id}:defer:{short_id(marker[1])}", text)
            return
        record = {
            "id": trade_id,
            "shadow_id": trade_id,
            "symbol": signal.get("symbol"),
            "coin": coin,
            "side": signal.get("side"),
            "status": "SKIPPED",
            "reason": skip.reason,
            "created_at": self.clock(),
            "closed_at": self.clock(),
        }
        await self._save(trade_id, new=record, event=(f"desk:{trade_id}:skip", {"id": trade_id, "reason": skip.reason}, text))

    async def _open(self, signal, plan, settings):
        from .desk_panel import signal_card

        trade_id = signal["id"]
        trade = {
            "id": trade_id,
            "shadow_id": trade_id,
            "symbol": signal.get("symbol"),
            "setup": signal.get("setup"),
            "setup_text": signal.get("setup_text"),
            "source_id": signal.get("source_id"),
            "layer": signal.get("layer"),
            "coin": plan["coin"],
            "side": plan["side"],
            "is_buy": plan["is_buy"],
            "status": "PLACING",
            "created_at": self.clock(),
            "expires_at": float(signal["expires_at"]),
            "entry": plan["entry"],
            "stop": plan["stop"],
            "original_stop": plan["stop"],
            "target1": plan["target1"],
            "target2": plan["target2"],
            "size": plan["size"],
            "tp1_size": plan["tp1_size"],
            "sz_decimals": plan["sz_decimals"],
            "risk_usd": plan["risk_usd"],
            "notional": plan["notional"],
            "leverage": plan["leverage"],
            "leverage_note": plan["leverage_capped_by"],
            "margin": plan["margin"],
            "entry_cloid": cloid_for(trade_id, "entry"),
            "entry_oid": None,
            "sl_oid": None,
            "sl_version": 0,
            "tp1_oid": None,
            "tp2_oid": None,
            "tp1_done": False,
            "filled": 0.0,
            "avg_entry": None,
            "close_reason": None,
            "notes": [],
        }
        await self._save(trade_id, new=trade)
        try:
            await self.client.set_leverage(plan["coin"], plan["leverage"], isolated=True)
            statuses = await self.client.place(
                [
                    {
                        "coin": plan["coin"],
                        "is_buy": plan["is_buy"],
                        "size": plan["size"],
                        "limit_px": plan["entry"],
                        "kind": "limit",
                        "tif": "Gtc",
                        "reduce_only": False,
                        "cloid": trade["entry_cloid"],
                    }
                ]
            )
        except HLError as exc:
            await self._save(
                trade_id,
                {"status": "ERROR", "reason": str(exc), "closed_at": self.clock()},
                (f"desk:{trade_id}:error", {"id": trade_id}, f"⚠️ {plan['coin']} order not placed: {exc}"),
            )
            return None
        status = statuses[0] if statuses else {"ok": False, "error": "No response"}
        if not status["ok"]:
            await self._save(
                trade_id,
                {"status": "ERROR", "reason": status["error"], "closed_at": self.clock()},
                (f"desk:{trade_id}:error", {"id": trade_id}, f"⚠️ {plan['coin']} order rejected: {status['error']}"),
            )
            return None
        changes = {"status": "ORDERED", "entry_oid": status["oid"], "placed_at": self.clock()}
        trade.update(changes)
        await self._save(trade_id, changes, (f"desk:{trade_id}:placed", {"id": trade_id, "plan": plan}, signal_card(trade, signal)))
        return trade

    # ------------------------------------------------------------------- manage
    async def _manage(self, trade, state, account, orders, mids):
        coin = trade["coin"]
        pos = account["positions"].get(coin)
        sign = 1 if trade["is_buy"] else -1
        held = abs(pos["size"]) if pos and pos["size"] * sign > 0 else 0.0
        open_oids = {o["oid"] for o in orders if o.get("coin") == coin}
        shadow = self._shadow(state, trade["shadow_id"])
        if trade["status"] == "PLACING":
            await self._resolve_placing(trade, held)
            return
        if trade["status"] == "ORDERED":
            await self._manage_ordered(trade, held, pos, open_oids, orders, shadow, mids.get(coin))
            return
        await self._manage_open(trade, held, pos, open_oids, orders, shadow, mids.get(coin))

    async def _resolve_placing(self, trade, held):
        # Crashed between saving and the exchange answer: ask by client order ID.
        try:
            result = await self.client.order_status(trade["entry_cloid"])
        except HLError:
            return
        order = result.get("order", {}).get("order") if isinstance(result, dict) else None
        if order and order.get("oid"):
            await self._save(trade["id"], {"status": "ORDERED", "entry_oid": order["oid"]})
        elif held > 0:
            await self._save(trade["id"], {"status": "ORDERED"})
        else:
            await self._save(
                trade["id"],
                {"status": "CANCELLED", "reason": "Order was never accepted", "closed_at": self.clock()},
            )

    async def _manage_ordered(self, trade, held, pos, open_oids, orders, shadow, mid):
        from .desk_panel import fill_text, cancel_text

        tid = trade["id"]
        if held > 0:
            await self._ensure_stop(trade, held, orders, mid)
        entry_open = trade.get("entry_oid") in open_oids
        sign = 1 if trade["is_buy"] else -1
        reason = None
        if self.clock() >= trade["expires_at"]:
            reason = "Entry window expired"
        elif shadow.get("status") in ("EXPIRED", "CLOSED"):
            reason = "The engine closed this signal before the entry filled"
        elif mid is not None and sign * (mid - trade["stop"]) <= 0:
            reason = "Price went through the stop before the entry filled"
        elif mid is not None and held == 0 and sign * (mid - trade["target1"]) >= 0:
            reason = "Price reached target 1 before the entry filled"
        if entry_open and reason:
            await self.client.cancel(trade["coin"], trade["entry_oid"])
            entry_open = False
        if entry_open:
            return
        if held > 0:
            changes = {
                "status": "OPEN",
                "filled": held,
                "avg_entry": pos.get("entry_px") or trade["entry"],
                "opened_at": self.clock(),
            }
            trade.update(changes)
            await self._save(tid, changes)
            await self._place_targets(trade, held, open_oids)
            await self._notify(f"desk:{tid}:filled", fill_text(trade, held, pos), {"id": tid})
        else:
            await self._save(
                tid,
                {"status": "CANCELLED", "reason": reason or "Entry order no longer on the book", "closed_at": self.clock()},
                (f"desk:{tid}:cancelled", {"id": tid}, cancel_text(trade, reason or "Entry order no longer on the book")),
            )

    async def _place_targets(self, trade, held, open_oids=frozenset()):
        roles, orders = [], []
        tp1 = trade["tp1_size"] if trade["tp1_size"] and trade["tp1_size"] < held else 0.0
        if tp1 > 0 and not trade.get("tp1_done") and trade.get("tp1_oid") not in open_oids:
            roles.append("tp1_oid")
            orders.append(_target_order(trade, tp1, trade["target1"], "tp1"))
        if trade.get("tp2_oid") not in open_oids:
            roles.append("tp2_oid")
            orders.append(_target_order(trade, held, trade["target2"], f"tp2v{trade.get('tp2_version', 0) + 1}"))
        if not orders:
            return
        try:
            statuses = await self.client.place(orders)
        except HLError as exc:
            statuses = [{"ok": False, "error": str(exc), "oid": None}] * len(orders)
        changes = {"tp2_version": trade.get("tp2_version", 0) + (1 if "tp2_oid" in roles else 0)}
        errors = []
        for role, status in zip(roles, statuses):
            if status["ok"]:
                changes[role] = status["oid"]
            else:
                errors.append(status.get("error") or "rejected")
        if tp1 <= 0 and not trade.get("tp1_done"):
            changes["tp1_size"] = 0.0
        trade.update(changes)
        await self._save(trade["id"], changes)
        if errors:
            await self._notify(
                f"desk:{trade['id']}:tp-error:{int(self.clock() // 900)}",
                f"⚠️ {trade['coin']} target order not placed ({errors[0]}). The stop-loss stays active; retrying.",
            )

    async def _ensure_stop(self, trade, held, orders, mid):
        """Invariant: a reduce-only stop covering the whole position is on the book."""
        current = next((o for o in orders if o.get("oid") == trade.get("sl_oid")), None)
        if current and current["size"] + 1e-12 >= held:
            self._sl_failures.pop(trade["id"], None)
            return True
        sign = 1 if trade["is_buy"] else -1
        if mid is not None and sign * (mid - trade["stop"]) <= 0:
            await self._close(trade, held, "Price is through the stop and no stop order was active")
            return False
        version = trade.get("sl_version", 0) + 1
        try:
            statuses = await self.client.place([_stop_order(trade, held, trade["stop"], version)])
            ok = statuses and statuses[0]["ok"]
        except HLError:
            ok = False
            statuses = []
        if not ok:
            failures = self._sl_failures.get(trade["id"], 0) + 1
            self._sl_failures[trade["id"]] = failures
            if failures >= 2:
                await self._close(trade, held, "Stop-loss could not be placed twice; closed for safety")
            return False
        self._sl_failures.pop(trade["id"], None)
        old = trade.get("sl_oid")
        changes = {"sl_oid": statuses[0]["oid"], "sl_version": version}
        trade.update(changes)
        await self._save(trade["id"], changes)
        if old and current:
            await self.client.cancel(trade["coin"], old)
        return True

    async def _move_stop(self, trade, held, new_stop, mid, why):
        from .desk_panel import stop_text

        sign = 1 if trade["is_buy"] else -1
        new_stop = round_price(new_stop, trade["sz_decimals"])
        if sign * (new_stop - trade["stop"]) <= 0:
            return  # never loosen
        if mid is not None and sign * (mid - new_stop) <= 0:
            await self._close(trade, held, f"{why}: price already beyond the new stop")
            return
        version = trade.get("sl_version", 0) + 1
        statuses = await self.client.place([_stop_order(trade, held, new_stop, version)])
        if not statuses or not statuses[0]["ok"]:
            return
        old = trade.get("sl_oid")
        changes = {"sl_oid": statuses[0]["oid"], "sl_version": version, "stop": new_stop}
        previous = trade["stop"]
        trade.update(changes)
        await self._save(
            trade["id"],
            changes,
            (f"desk:{trade['id']}:stop:{new_stop}", {"id": trade["id"], "stop": new_stop}, stop_text(trade, previous, new_stop, why)),
        )
        if old:
            await self.client.cancel(trade["coin"], old)

    async def _manage_open(self, trade, held, pos, open_oids, orders, shadow, mid):
        tid = trade["id"]
        if held <= 0:
            await self._finalize(trade, open_oids, orders)
            return
        if not await self._ensure_stop(trade, held, orders, mid):
            return
        if trade.get("tp2_oid") not in open_oids:
            await self._place_targets(trade, held, open_oids)
        tp1_filled = (
            not trade.get("tp1_done")
            and (
                (trade.get("tp1_oid") and trade["tp1_oid"] not in open_oids and held < trade["filled"] - 1e-12)
                or shadow.get("tp1_done")
            )
        )
        if tp1_filled:
            trade["tp1_done"] = True
            trade["filled_after_tp1"] = held
            await self._save(tid, {"tp1_done": True, "filled_after_tp1": held})
            await self._move_stop(trade, held, trade.get("avg_entry") or trade["entry"], mid, "Target 1 hit — stop to entry")
        shadow_stop = shadow.get("stop")
        if isinstance(shadow_stop, (int, float)) and trade.get("tp1_done"):
            sign = 1 if trade["is_buy"] else -1
            if sign * (shadow_stop - trade["stop"]) > 0:
                await self._move_stop(trade, held, shadow_stop, mid, "Trailing stop moved by the engine")
        price_exit = shadow.get("price_exit", shadow.get("exit_reason") in PRICE_EXITS)
        if shadow.get("status") == "CLOSED" and shadow.get("exit_reason") and not price_exit:
            from .notifications import describe_reason

            reason = shadow["exit_reason"]
            label = "Shadow Desk exit" if shadow.get("source") == "shadow_book" else "Engine exit"
            await self._close(trade, held, f"{label}: {describe_reason(reason)[:160]}")

    async def _close(self, trade, held, why):
        """Reduce-only aggressive limit (Hyperliquid's market order)."""
        mid = (self.snapshot.get("mids") or {}).get(trade["coin"])
        if mid is None:
            mids = await self.client.mids()
            mid = mids.get(trade["coin"])
        if mid is None:
            return False
        long = trade["is_buy"]
        px = round_price(mid * (1 - MARKET_SLIPPAGE) if long else mid * (1 + MARKET_SLIPPAGE), trade["sz_decimals"])
        attempt = trade.get("close_attempts", 0) + 1
        await self._save(trade["id"], {"close_reason": why, "close_attempts": attempt})
        trade["close_reason"] = why
        statuses = await self.client.place(
            [
                {
                    "coin": trade["coin"],
                    "is_buy": not long,
                    "size": round_size(held, trade["sz_decimals"]),
                    "limit_px": px,
                    "kind": "limit",
                    "tif": "Ioc",
                    "reduce_only": True,
                    "cloid": cloid_for(trade["id"], f"close{attempt}"),
                }
            ]
        )
        return bool(statuses and statuses[0]["ok"])

    async def _finalize(self, trade, open_oids, orders):
        from .desk_panel import close_text

        tid = trade["id"]
        for o in orders:
            if o.get("coin") == trade["coin"] and o.get("oid") in {trade.get("sl_oid"), trade.get("tp1_oid"), trade.get("tp2_oid"), trade.get("entry_oid")}:
                try:
                    await self.client.cancel(trade["coin"], o["oid"])
                except HLError:
                    pass
        start = int(float(trade.get("placed_at") or trade["created_at"]) * 1000)
        pnl, fees, last_oid = 0.0, 0.0, None
        try:
            fills = await self.client.fills_since(start - 60000)
        except HLError:
            fills = None
        if fills is not None:
            for f in fills:
                if f["coin"] != trade["coin"]:
                    continue
                pnl += f["closed_pnl"]
                fees += f["fee"]
                if f["closed_pnl"] != 0 or (f.get("dir") or "").startswith("Close"):
                    last_oid = f["oid"]
        net = pnl - fees
        if trade.get("close_reason"):
            reason = trade["close_reason"]
        elif last_oid and last_oid == trade.get("tp2_oid"):
            reason = "Target 2 hit"
        elif last_oid and last_oid == trade.get("sl_oid"):
            moved = trade["stop"] != trade.get("original_stop")
            reason = ("Stop hit at entry (break-even)" if trade.get("tp1_done") else "Trailing stop hit") if moved else "Stop-loss hit"
        else:
            reason = "Position closed on the exchange"
        risk = trade.get("risk_usd") or 0
        changes = {
            "status": "CLOSED",
            "closed_at": self.clock(),
            "pnl": round(net, 2),
            "fees": round(fees, 4),
            "r": round(net / risk, 2) if risk else None,
            "exit": reason,
            "pnl_complete": fills is not None,
        }
        trade.update(changes)
        await self._save(tid, changes, (f"desk:{tid}:closed", {"id": tid, "pnl": net}, close_text(trade)))

    async def _cleanup_orphans(self, state, orders):
        """Cancel our own leftover orders whose trade is no longer active."""
        desk = desk_state(state)
        known = {}
        for t in desk["trades"].values():
            for key in ("entry_oid", "sl_oid", "tp1_oid", "tp2_oid"):
                if t.get(key):
                    known[t[key]] = t
        for o in orders:
            trade = known.get(o.get("oid"))
            if trade and trade.get("status") not in ACTIVE:
                try:
                    await self.client.cancel(o["coin"], o["oid"])
                except HLError:
                    pass

    # ----------------------------------------------------------- user actions
    async def close_trade(self, trade_id: str, why: str = "Closed from Telegram") -> str:
        if not self.client:
            return "Hyperliquid is not connected."
        async with self.lock:
            state = await self.store.read()
            trade = desk_state(state)["trades"].get(trade_id)
            if not trade or trade.get("status") not in ACTIVE:
                return "That trade is no longer active."
            account = await self.client.account()
            pos = account["positions"].get(trade["coin"])
            sign = 1 if trade["is_buy"] else -1
            held = abs(pos["size"]) if pos and pos["size"] * sign > 0 else 0.0
            if trade.get("entry_oid") and trade["status"] in ("ORDERED", "PLACING"):
                try:
                    await self.client.cancel(trade["coin"], trade["entry_oid"])
                except HLError:
                    pass
            if held > 0:
                if trade["status"] != "OPEN":
                    await self._save(trade_id, {"status": "OPEN", "filled": held, "avg_entry": pos.get("entry_px")})
                ok = await self._close(trade, held, why)
                return f"Closing {trade['coin']} ({held:g}) at market{'…' if ok else ' failed — check /positions'}."
            await self._save(
                trade_id,
                {"status": "CANCELLED", "reason": why, "closed_at": self.clock()},
            )
            return f"Cancelled the pending {trade['coin']} entry."

    async def close_all(self) -> str:
        state = await self.store.read()
        ids = [t["id"] for t in desk_state(state)["trades"].values() if t.get("status") in ACTIVE]
        if not ids:
            return "No open bot positions or pending entries."
        results = [await self.close_trade(i, "Close-all from Telegram") for i in ids]
        return "\n".join(results)
