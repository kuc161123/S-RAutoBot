"""Hyperliquid perpetuals boundary: precision, client order IDs and an async SDK wrapper.

Importing this module performs no I/O. The official SDK is synchronous, so every
network call runs in a worker thread. Secrets never appear in exceptions, logs or
return values: callers only see sanitized HLError messages.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal

MAINNET_URL = "https://api.hyperliquid.xyz"
MIN_NOTIONAL_USD = 10.0
# Market-style orders on Hyperliquid are aggressive limits; this bounds slippage.
MARKET_SLIPPAGE = 0.05
STOP_SLIPPAGE = 0.10

_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}\Z")
_KEY = re.compile(r"(0x)?[0-9a-fA-F]{64}\Z")


READ_RETRIES = 2
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _describe(exc) -> str:
    """Exception type plus HTTP status: enough to diagnose, never the payload."""
    code = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} {code}" if type(code) is int else type(exc).__name__


def _transient(exc) -> bool:
    code = getattr(exc, "status_code", None)
    if type(code) is int:
        return code in TRANSIENT_STATUS
    try:
        import requests
    except ImportError:  # pragma: no cover - the SDK depends on requests
        return False
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def _backoff(exc, attempt: int) -> float:
    base = 2.0 if getattr(exc, "status_code", None) == 429 else 0.5
    return base * (attempt + 1)


class HLError(Exception):
    """Sanitized exchange failure. Never carries credentials or request payloads."""


def valid_address(value) -> bool:
    return isinstance(value, str) and bool(_ADDRESS.fullmatch(value))


def valid_private_key(value) -> bool:
    return isinstance(value, str) and bool(_KEY.fullmatch(value))


def round_size(size: float, sz_decimals: int) -> float:
    """Round DOWN to the asset's size precision: never exceed the planned risk."""
    if not math.isfinite(size) or size <= 0:
        return 0.0
    step = Decimal(1).scaleb(-int(sz_decimals))
    return float(Decimal(str(size)).quantize(step, rounding=ROUND_DOWN))


def size_step(sz_decimals: int) -> float:
    return float(Decimal(1).scaleb(-int(sz_decimals)))


def round_price(price: float, sz_decimals: int, direction: str = "nearest") -> float:
    """Valid perp price: at most 5 significant figures and 6 - szDecimals decimals.

    Integer prices are always valid regardless of significant figures.
    direction: nearest | down | up.
    """
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Price must be positive and finite")
    value = Decimal(repr(float(price)))
    max_decimals = max(0, 6 - int(sz_decimals))
    sig_decimals = max(0, 4 - value.adjusted())
    decimals = min(max_decimals, sig_decimals)
    rounding = {"nearest": ROUND_HALF_UP, "down": ROUND_DOWN, "up": ROUND_UP}[
        direction
    ]
    quantum = Decimal(1).scaleb(-decimals)
    result = value.quantize(quantum, rounding=rounding)
    if result <= 0:
        result = quantum
    return float(result)


def cloid_for(trade_id: str, role: str) -> str:
    """Stable 16-byte client order ID: the same trade/role always maps to one ID."""
    digest = hashlib.sha256(f"apex|{trade_id}|{role}".encode()).hexdigest()
    return "0x" + digest[:32]


def coin_candidates(symbol: str) -> list[str]:
    """Bybit USDT-perp symbol -> Hyperliquid coin names to try, most likely first."""
    symbol = str(symbol).upper()
    if not symbol.endswith("USDT"):
        return []
    base = symbol[:-4]
    if not base:
        return []
    if base.startswith("1000000") or base.startswith("10000"):
        # Different contract multipliers; prices would not match. Never guess.
        return []
    if base.startswith("1000"):
        return ["k" + base[4:]]
    if base.endswith("1000"):
        return ["k" + base[:-4]]
    return [base]


def map_coin(symbol: str, assets: dict) -> str | None:
    for coin in coin_candidates(symbol):
        asset = assets.get(coin)
        if asset and not asset.get("delisted"):
            return coin
    return None


def _num(value, default=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_statuses(response) -> list[dict]:
    """Normalize an /exchange order response into one dict per order.

    Each item: {"ok": bool, "oid": int|None, "filled": float, "avg_px": float|None,
    "error": str|None, "state": "resting"|"filled"|"waiting"|"error"}.
    """
    if not isinstance(response, dict) or response.get("status") != "ok":
        detail = response.get("response") if isinstance(response, dict) else None
        raise HLError(f"Exchange rejected the request: {str(detail)[:160]}")
    data = response.get("response", {}).get("data", {})
    out = []
    for status in data.get("statuses", []):
        if isinstance(status, dict) and "resting" in status:
            out.append(
                {
                    "ok": True,
                    "state": "resting",
                    "oid": status["resting"].get("oid"),
                    "filled": 0.0,
                    "avg_px": None,
                    "error": None,
                }
            )
        elif isinstance(status, dict) and "filled" in status:
            fill = status["filled"]
            out.append(
                {
                    "ok": True,
                    "state": "filled",
                    "oid": fill.get("oid"),
                    "filled": _num(fill.get("totalSz")),
                    "avg_px": _num(fill.get("avgPx"), None),
                    "error": None,
                }
            )
        elif isinstance(status, dict) and "error" in status:
            out.append(
                {
                    "ok": False,
                    "state": "error",
                    "oid": None,
                    "filled": 0.0,
                    "avg_px": None,
                    "error": str(status["error"])[:200],
                }
            )
        else:
            out.append(
                {
                    "ok": True,
                    "state": "waiting",
                    "oid": None,
                    "filled": 0.0,
                    "avg_px": None,
                    "error": None,
                }
            )
    return out


class HyperliquidClient:
    """Async facade over hyperliquid-python-sdk for one account and API wallet.

    The API wallet signs; orders are placed for account_address. An API wallet can
    trade but cannot withdraw funds.
    """

    def __init__(self, account_address: str, private_key: str, base_url: str = MAINNET_URL, timeout: float = 10.0, perp_dexs=("xyz",)):
        if not valid_address(account_address):
            raise ValueError("HL_ACCOUNT_ADDRESS must be a 0x-prefixed 40-hex address")
        if not valid_private_key(private_key):
            raise ValueError("HL_API_PRIVATE_KEY must be a 64-hex private key")
        self.address = account_address
        self._key = private_key if private_key.startswith("0x") else "0x" + private_key
        self.base_url = base_url
        self.timeout = timeout
        self._info = None
        self._exchange = None
        self.signer = None
        self.wanted_dexs = tuple(d for d in perp_dexs if d)
        self.dexs = ("",)
        self.dex_abstraction = None  # None = not attempted; "on" or an error note
        self._abstraction_mode = "unknown"
        self._abstraction_at = 0.0

    def _build(self):
        try:
            import eth_account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info

            wallet = eth_account.Account.from_key(self._key)
            dexs = [""]
            if self.wanted_dexs:
                probe = Info(self.base_url, skip_ws=True, perp_dexs=[""], timeout=self.timeout)
                listed = {d.get("name") for d in (probe.perp_dexs() or [])[1:] if isinstance(d, dict)}
                dexs += [d for d in self.wanted_dexs if d in listed]
            info = Info(self.base_url, skip_ws=True, perp_dexs=dexs, timeout=self.timeout)
            exchange = Exchange(
                wallet,
                self.base_url,
                account_address=self.address,
                perp_dexs=dexs,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - sanitize SDK/network errors
            raise HLError(f"Hyperliquid connection failed ({type(exc).__name__})") from None
        self._info, self._exchange, self.signer = info, exchange, wallet.address
        self.dexs = tuple(dexs)
        if len(dexs) > 1 and self.dex_abstraction is None:
            # Lets builder-deployed markets (e.g. xyz: stocks/commodities) use the
            # main perp USDC balance as collateral instead of a separate transfer.
            try:
                result = exchange.agent_enable_dex_abstraction()
                ok = isinstance(result, dict) and result.get("status") == "ok"
                self.dex_abstraction = "on" if ok else f"not enabled ({str(result)[:80]})"
            except Exception as exc:  # noqa: BLE001
                self.dex_abstraction = f"not enabled ({type(exc).__name__})"

    async def connect(self, force: bool = False):
        if force or self._exchange is None:
            await asyncio.to_thread(self._build)

    async def _call(self, fn, *args, retries: int = 0, **kwargs):
        """Run one SDK call off the event loop; errors become sanitized HLErrors.

        retries > 0 is only for read requests: a rate limit (429), a 5xx or a
        network blip is retried after a short pause. Exchange actions are never
        retried here — the desk reconciles them by client order ID instead.
        """
        if self._exchange is None:
            await self.connect()
        for attempt in range(retries + 1):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except HLError:
                raise
            except Exception as exc:  # noqa: BLE001 - never leak payloads/URLs
                if attempt < retries and _transient(exc):
                    await asyncio.sleep(_backoff(exc, attempt))
                    continue
                raise HLError(f"Hyperliquid request failed ({_describe(exc)})") from None

    async def _read(self, fn):
        return await self._call(fn, retries=READ_RETRIES)

    async def assets(self) -> dict:
        """coin -> {sz_decimals, max_leverage, only_isolated, delisted}."""
        await self.connect(force=True)  # refresh name->asset map for new listings
        out = {}
        for dex in self.dexs:
            meta = await self._read(lambda d=dex: self._info.meta(dex=d))
            for item in meta.get("universe", []):
                name = item.get("name")
                if not isinstance(name, str):
                    continue
                out[name] = {
                    "sz_decimals": int(item.get("szDecimals", 0)),
                    "max_leverage": int(item.get("maxLeverage", 1) or 1),
                    "only_isolated": bool(item.get("onlyIsolated", False)),
                    "delisted": bool(item.get("isDelisted", False)),
                    "dex": dex,
                }
        return out

    async def account(self) -> dict:
        total = {"equity": 0.0, "margin_used": 0.0, "available": 0.0, "positions": {}, "mode": "standard"}
        for dex in self.dexs:
            part = await self._account(dex)
            total["equity"] += part["equity"]
            total["margin_used"] += part["margin_used"]
            total["available"] += part["available"]
            total["positions"].update(part["positions"])
        mode = await self._abstraction()
        if mode in ("unifiedAccount", "portfolioMargin"):
            # Unified/portfolio accounts keep collateral in the spot USDC balance;
            # the perp clearinghouse's "withdrawable" understates free margin there.
            spot = await self._read(lambda: self._info.spot_user_state(self.address))
            usdc = next(
                (b for b in (spot or {}).get("balances", []) if b.get("coin") == "USDC"),
                {},
            )
            usdc_total, hold = _num(usdc.get("total")), _num(usdc.get("hold"))
            upnl = sum(p["upnl"] for p in total["positions"].values())
            total["equity"] = usdc_total + upnl
            total["available"] = max(total["available"], usdc_total - hold)
            total["mode"] = mode
        return total

    async def _abstraction(self) -> str:
        now = time.monotonic()
        if self._abstraction_at and now - self._abstraction_at < 600:
            return self._abstraction_mode
        try:
            mode = await self._read(lambda: self._info.query_user_abstraction_state(self.address))
        except HLError:
            mode = self._abstraction_mode
        self._abstraction_mode = mode if isinstance(mode, str) else "unknown"
        self._abstraction_at = now
        return self._abstraction_mode

    async def _account(self, dex: str) -> dict:
        state = await self._read(lambda: self._info.user_state(self.address, dex))
        summary = state.get("marginSummary", {}) if isinstance(state, dict) else {}
        positions = {}
        for row in state.get("assetPositions", []) if isinstance(state, dict) else []:
            p = row.get("position", {}) if isinstance(row, dict) else {}
            size = _num(p.get("szi"))
            if not p.get("coin") or size == 0:
                continue
            lev = p.get("leverage") or {}
            positions[p["coin"]] = {
                "size": size,
                "entry_px": _num(p.get("entryPx"), None),
                "liq_px": _num(p.get("liquidationPx"), None),
                "upnl": _num(p.get("unrealizedPnl")),
                "margin": _num(p.get("marginUsed")),
                "value": _num(p.get("positionValue")),
                "leverage": int(_num(lev.get("value"), 0)),
                "lev_type": lev.get("type"),
            }
        return {
            "equity": _num(summary.get("accountValue")),
            "margin_used": _num(summary.get("totalMarginUsed")),
            "available": _num(state.get("withdrawable")) if isinstance(state, dict) else 0.0,
            "positions": positions,
        }

    async def open_orders(self) -> list[dict]:
        rows = []
        for dex in self.dexs:
            part = await self._read(lambda d=dex: self._info.frontend_open_orders(self.address, d))
            rows += part if isinstance(part, list) else []
        out = []
        for o in rows if isinstance(rows, list) else []:
            out.append(
                {
                    "coin": o.get("coin"),
                    "oid": o.get("oid"),
                    "cloid": o.get("cloid"),
                    "side": o.get("side"),
                    "size": _num(o.get("sz")),
                    "orig_size": _num(o.get("origSz")),
                    "limit_px": _num(o.get("limitPx"), None),
                    "trigger_px": _num(o.get("triggerPx"), None),
                    "is_trigger": bool(o.get("isTrigger")),
                    "reduce_only": bool(o.get("reduceOnly")),
                    "order_type": o.get("orderType"),
                }
            )
        return out

    async def mids(self) -> dict:
        out = {}
        for dex in self.dexs:
            mids = await self._read(lambda d=dex: self._info.all_mids(d))
            out.update({k: _num(v, None) for k, v in (mids or {}).items()})
        return out

    async def fills_since(self, start_ms: int) -> list[dict]:
        rows = await self._read(
            lambda: self._info.user_fills_by_time(self.address, int(start_ms))
        )
        out = []
        for f in rows if isinstance(rows, list) else []:
            out.append(
                {
                    "coin": f.get("coin"),
                    "oid": f.get("oid"),
                    "px": _num(f.get("px")),
                    "size": _num(f.get("sz")),
                    "side": f.get("side"),
                    "time": int(_num(f.get("time"))),
                    "closed_pnl": _num(f.get("closedPnl")),
                    "fee": _num(f.get("fee")),
                    "dir": f.get("dir"),
                }
            )
        return out

    async def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[tuple]:
        """[(t_ms, high, low), …] oldest first (Hyperliquid returns up to 5000 bars)."""
        rows = await self._read(
            lambda: self._info.candles_snapshot(coin, interval, int(start_ms), int(end_ms))
        )
        out = []
        for c in rows if isinstance(rows, list) else []:
            high, low = _num(c.get("h"), None), _num(c.get("l"), None)
            if high is not None and low is not None:
                out.append((int(_num(c.get("t"))), high, low))
        out.sort()
        return out

    async def set_leverage(self, coin: str, leverage: int, isolated: bool = True):
        response = await self._call(
            lambda: self._exchange.update_leverage(int(leverage), coin, is_cross=not isolated)
        )
        if not isinstance(response, dict) or response.get("status") != "ok":
            raise HLError("Leverage update rejected")

    def _wire(self, order: dict) -> dict:
        from hyperliquid.utils.types import Cloid

        kind = order["kind"]
        if kind == "limit":
            order_type = {"limit": {"tif": order.get("tif", "Gtc")}}
        elif kind in ("sl", "tp"):
            order_type = {
                "trigger": {
                    "triggerPx": order["trigger_px"],
                    "isMarket": True,
                    "tpsl": kind,
                }
            }
        else:
            raise ValueError("Unknown order kind")
        wire = {
            "coin": order["coin"],
            "is_buy": bool(order["is_buy"]),
            "sz": order["size"],
            "limit_px": order["limit_px"],
            "order_type": order_type,
            "reduce_only": bool(order.get("reduce_only", False)),
        }
        if order.get("cloid"):
            wire["cloid"] = Cloid.from_str(order["cloid"])
        return wire

    async def place(self, orders: list[dict]) -> list[dict]:
        wires = [self._wire(o) for o in orders]
        response = await self._call(lambda: self._exchange.bulk_orders(wires))
        return parse_statuses(response)

    async def cancel(self, coin: str, oid: int) -> bool:
        response = await self._call(lambda: self._exchange.cancel(coin, int(oid)))
        if not isinstance(response, dict) or response.get("status") != "ok":
            return False
        statuses = response.get("response", {}).get("data", {}).get("statuses", [])
        return all(s == "success" for s in statuses)

    async def order_status(self, cloid: str) -> dict:
        from hyperliquid.utils.types import Cloid

        result = await self._read(
            lambda: self._info.query_order_by_cloid(self.address, Cloid.from_str(cloid))
        )
        return result if isinstance(result, dict) else {}
