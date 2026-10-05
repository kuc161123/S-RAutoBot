"""Async Bybit V5 USDT-linear adapter; the caller owns the aiohttp session.

Public reads need no credentials. No account configuration is changed: the runtime
must verify isolated margin, one-way positions and leverage <= 5 before trading.
Writes return exchange acknowledgements, not proof of a fill/cancel/protection.
After AmbiguousOrderError, reconcile by order-link ID; never assume an absent
history record proves a timed-out submission failed (history is eventually consistent).

API references: https://bybit-exchange.github.io/docs/v5/guide and /v5/order/create-order.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import random
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import aiohttp
from yarl import URL

from .models import Candle, Instrument


class BybitError(RuntimeError):
    """Base for sanitized adapter errors (never contains response text or headers)."""


class DataUnavailable(BybitError):
    """A read failed, or its data cannot safely be used."""


class CredentialsError(BybitError):
    """A private operation requires both explicitly supplied credentials."""


class BybitAPIError(BybitError):
    """Confirmed HTTP/API rejection. ret_code and http_status are safe to inspect."""

    def __init__(self, ret_code: int | None = None, http_status: int | None = None):
        self.ret_code = ret_code
        self.http_status = http_status
        super().__init__(
            f"Bybit rejected request (code={ret_code}, HTTP={http_status})"
        )


class OrderRejectedError(BybitAPIError):
    """A mutation was explicitly rejected; the adapter did not replay it."""


class AmbiguousOrderError(BybitError):
    """A mutation may have reached Bybit; reconcile before any further action."""

    def __init__(self, order_link_id: str | None = None):
        self.order_link_id = order_link_id
        super().__init__(
            "Bybit mutation outcome is unknown; reconciliation is required"
        )


_DAY_MS = 86_400_000
_MINUTE_INTERVALS = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720"}
_TRANSIENT_CODES = {429, 10000, 10006, 10016}
_AMBIGUOUS_CODES = {
    10000,
    10016,
    10014,
    110072,
}  # Timeout, server error, duplicate IDs.
_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


def _decimal(value: Any, name: str, *, zero: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{name} must be a finite decimal") from None
    if not result.is_finite() or (result < 0 if zero else result <= 0):
        raise ValueError(
            f"{name} must be {'nonnegative' if zero else 'positive'} and finite"
        )
    return result


def _number(value: Any, name: str, *, zero: bool = False) -> float:
    try:
        result = float(_decimal(value, name, zero=zero))
        if not math.isfinite(result) or (not zero and result == 0):
            raise ValueError
        return result
    except (ValueError, OverflowError):
        raise DataUnavailable(f"Invalid {name}") from None


def _integer(value: Any, name: str) -> int:
    # Do not silently truncate floating-point timestamps or accept bools.
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise DataUnavailable(f"Invalid {name}")
    try:
        return int(value)
    except (ValueError, OverflowError):
        raise DataUnavailable(f"Invalid {name}") from None


def _symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]+USDT", symbol):
        raise ValueError("symbol must be an uppercase USDT symbol")
    return symbol


def _link_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,36}", value):
        raise ValueError(
            "order_link_id must contain 1-36 ASCII letters, digits, '-' or '_'"
        )
    return value


def _interval(interval: str) -> str:
    if interval not in _MINUTE_INTERVALS | {"D", "W", "M"}:
        raise ValueError("Unsupported Bybit candle interval")
    return interval


def _floor_time(timestamp: int, interval: str) -> int:
    _interval(interval)
    if interval in _MINUTE_INTERVALS or interval == "D":
        width = int(interval) * 60_000 if interval != "D" else _DAY_MS
        return timestamp // width * width
    try:
        date = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date = (
            date - timedelta(days=date.weekday())
            if interval == "W"
            else date.replace(day=1)
        )
        return int(date.timestamp() * 1000)
    except (ValueError, OverflowError, OSError):
        raise DataUnavailable("Invalid candle timestamp") from None


def _next_time(timestamp: int, interval: str) -> int:
    if interval == "M":
        try:
            date = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
            return int(
                date.replace(
                    year=date.year + (date.month == 12),
                    month=date.month % 12 + 1,
                    day=1,
                ).timestamp()
                * 1000
            )
        except (ValueError, OverflowError, OSError):
            raise DataUnavailable("Invalid candle timestamp") from None
    return timestamp + (
        {"D": _DAY_MS, "W": 7 * _DAY_MS}.get(interval) or int(interval) * 60_000
    )


def latest_closed_open_time(interval: str, now_ms: int) -> int:
    """UTC start of the most recently closed bar (Monday weeks/calendar months)."""
    return _floor_time(_floor_time(_integer(now_ms, "now_ms"), interval) - 1, interval)


def validate_candles(
    candles: list[Candle], interval: str = "D", now_ms: int | None = None
) -> list[Candle]:
    """Sort/deduplicate, discard the current bar, and require fresh contiguous OHLCV.

    An explicitly supplied now_ms should use the exchange clock. The client does
    this automatically; standalone callers may supply their deterministic clock.
    Equal duplicates are harmless; conflicting duplicates and future bars fail.
    Short histories are allowed (e.g. newly listed contracts); gaps are not.
    """
    now = _integer(now_ms if now_ms is not None else int(time.time() * 1000), "now_ms")
    current = _floor_time(now, interval)
    unique: dict[int, Candle] = {}
    for candle in candles:
        if not isinstance(candle, Candle) or type(candle.open_time) is not int:
            raise DataUnavailable("Malformed candle")
        stamp = _integer(candle.open_time, "candle timestamp")
        if stamp != _floor_time(stamp, interval) or stamp > current:
            raise DataUnavailable("Misaligned or future candle")
        o, h, low, c = [
            _number(value, "candle price")
            for value in (candle.open, candle.high, candle.low, candle.close)
        ]
        volume = _number(candle.volume, "candle volume", zero=True)
        if low > min(o, c) or h < max(o, c) or low > h:
            raise DataUnavailable("Inconsistent candle OHLC")
        if stamp == current:
            continue
        candle = Candle(stamp, o, h, low, c, volume)
        if stamp in unique and unique[stamp] != candle:
            raise DataUnavailable("Conflicting duplicate candle")
        unique[stamp] = candle
    result = sorted(unique.values(), key=lambda candle: candle.open_time)
    if not result or result[-1].open_time != latest_closed_open_time(interval, now):
        raise DataUnavailable("Candle history is empty or stale")
    if any(
        _next_time(a.open_time, interval) != b.open_time
        for a, b in zip(result, result[1:])
    ):
        raise DataUnavailable("Gap in candle history")
    return result


class BybitClient:
    """Use an existing aiohttp.ClientSession; construction performs no I/O.

    GETs: at most four attempts with bounded exponential backoff and jitter.
    POSTs: exactly one attempt, including throttling and timestamp failures.
    """

    RECV_WINDOW_MS = 5_000
    REQUEST_TIMEOUT_SECONDS = 15
    GET_ATTEMPTS = 4
    MAX_BACKOFF_SECONDS = 8.0
    CLOCK_REFRESH_SECONDS = 60
    MAX_PAGES = 1_000
    MAX_CANDLE_RANGE = 50_000
    EXECUTION_LOOKBACK_MS = 7 * _DAY_MS
    ACCOUNTING_LOOKBACK_MS = 730 * _DAY_MS

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str = "https://api.bybit.com",
        api_key: str = "",
        api_secret: str = "",
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("base_url must be an HTTPS origin without credentials")
        self.session = session
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_secret = api_secret
        # Python 3.9 binds Lock to a loop on construction; stay inert until first I/O.
        self._clock_lock: asyncio.Lock | None = None
        self._clock_anchor: float | None = None
        self._server_ms = 0.0
        self.instrument_metadata = {}

    def _require_credentials(self) -> None:
        if not (
            isinstance(self._api_key, str)
            and self._api_key.strip()
            and isinstance(self._api_secret, str)
            and self._api_secret.strip()
        ):
            raise CredentialsError(
                "Private Bybit methods require api_key and api_secret"
            )

    def _now_ms(self) -> int:
        if self._clock_anchor is None:
            raise DataUnavailable("Exchange clock has not been synchronized")
        # Monotonic elapsed time prevents wall-clock jumps from breaking signatures.
        return int(self._server_ms + (time.monotonic() - self._clock_anchor) * 1000)

    async def _ensure_clock(self) -> None:
        if self._clock_lock is None:
            self._clock_lock = asyncio.Lock()
        async with self._clock_lock:
            if (
                self._clock_anchor is not None
                and time.monotonic() - self._clock_anchor < self.CLOCK_REFRESH_SECONDS
            ):
                return
            # Each successful attempt supplies its own timing, excluding retry sleeps.
            result = await self._request("GET", "/v5/market/time", clock_sample=True)
            try:
                if "timeNano" in result:
                    server_ms = _integer(result["timeNano"], "server time") / 1_000_000
                else:
                    server_ms = _integer(result.get("timeSecond"), "server time") * 1000
                valid = math.isfinite(server_ms) and server_ms > 0
            except OverflowError:
                valid = False
            if not valid:
                raise DataUnavailable("Invalid server time")
            self._server_ms = server_ms
            self._clock_anchor = self._sample_midpoint

    def _headers(self, payload: str) -> dict[str, str]:
        timestamp = str(self._now_ms())
        window = str(self.RECV_WINDOW_MS)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            (timestamp + self._api_key + window + payload).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": window,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
        }

    async def _backoff(self, attempt: int, headers: dict[str, str]) -> None:
        delay = 0.25 * 2**attempt
        try:
            retry_after = float(headers.get("Retry-After", "0"))
            if math.isfinite(retry_after):
                delay = max(delay, retry_after)
        except (TypeError, ValueError):
            pass
        try:
            reset = float(headers.get("X-Bapi-Limit-Reset-Timestamp", "0"))
            now = (
                self._now_ms() if self._clock_anchor is not None else time.time() * 1000
            )
            if math.isfinite(reset):
                delay = max(delay, (reset - now) / 1000)
        except (TypeError, ValueError):
            pass
        await asyncio.sleep(
            min(self.MAX_BACKOFF_SECONDS, delay + random.uniform(0, 0.25))
        )

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        private: bool = False,
        clock_sample: bool = False,
    ) -> dict:
        params = params or {}
        write = method == "POST"
        link = params.get("orderLinkId")
        if private:
            self._require_credentials()
        payload = (
            json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if write
            else urlencode(sorted(params.items()), quote_via=quote, safe="")
        )
        # yarl must not re-encode the already-signed query (notably cursor '+'/'=').
        url = URL(
            self.base_url + path + ("?" + payload if payload and not write else ""),
            encoded=True,
        )
        attempts = 1 if write else self.GET_ATTEMPTS
        for attempt in range(attempts):
            if private:
                await self._ensure_clock()
            headers = self._headers(payload) if private else {}
            if write:
                headers["Content-Type"] = "application/json"
            retry_headers: dict[str, str] = {}
            started = time.monotonic()
            try:
                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    data=payload.encode("utf-8") if write else None,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT_SECONDS),
                    allow_redirects=False,
                ) as response:
                    status = response.status
                    retry_headers = {
                        key: response.headers.get(key, "")
                        for key in ("Retry-After", "X-Bapi-Limit-Reset-Timestamp")
                    }
                    try:
                        body = await response.json(content_type=None)
                    except (ValueError, UnicodeError, aiohttp.ContentTypeError):
                        body = None
                finished = time.monotonic()
            except asyncio.CancelledError:
                # Cancellation during a mutation also cannot establish non-submission.
                if write:
                    raise AmbiguousOrderError(link) from None
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                if write:
                    raise AmbiguousOrderError(link) from None
                transient = isinstance(
                    exc,
                    (
                        aiohttp.ClientConnectionError,
                        aiohttp.ClientPayloadError,
                        asyncio.TimeoutError,
                    ),
                )
                if isinstance(exc, (aiohttp.ClientSSLError, aiohttp.InvalidURL)):
                    transient = False
                if not transient or attempt == attempts - 1:
                    raise DataUnavailable("Bybit read transport failed") from None
                await self._backoff(attempt, retry_headers)
                continue

            if status in _TRANSIENT_HTTP:
                if write:
                    if status == 429:
                        raise OrderRejectedError(http_status=status)
                    raise AmbiguousOrderError(link)
                if attempt == attempts - 1:
                    raise DataUnavailable(
                        f"Bybit read retries exhausted (HTTP={status})"
                    )
                await self._backoff(attempt, retry_headers)
                continue
            if not 200 <= status < 300:
                if write and not 400 <= status < 500:
                    raise AmbiguousOrderError(link)
                error = OrderRejectedError if write else BybitAPIError
                raise error(http_status=status)
            if not isinstance(body, dict) or type(body.get("retCode")) is not int:
                if write:
                    raise AmbiguousOrderError(link)
                raise DataUnavailable("Malformed Bybit response envelope")
            code = body["retCode"]
            if code:
                if code in {-1, 10002} and private:
                    self._clock_anchor = None
                if write:
                    if code in _AMBIGUOUS_CODES:
                        raise AmbiguousOrderError(link)
                    raise OrderRejectedError(code, status)
                if code in _TRANSIENT_CODES or (private and code in {-1, 10002}):
                    if attempt == attempts - 1:
                        raise DataUnavailable(
                            f"Bybit read retries exhausted (code={code})"
                        )
                    await self._backoff(attempt, retry_headers)
                    continue
                raise BybitAPIError(code, status)
            if not isinstance(body.get("result"), dict):
                if write:
                    raise AmbiguousOrderError(link)
                raise DataUnavailable("Malformed Bybit result")
            if clock_sample:
                if finished - started >= 2:
                    # An asymmetric long round trip could put a midpoint estimate
                    # >=1s ahead of the server, outside Bybit's accepted time window.
                    raise DataUnavailable("Exchange clock sample is too slow")
                self._sample_midpoint = (started + finished) / 2
            return body["result"]
        raise DataUnavailable("Bybit read retries exhausted")

    @staticmethod
    def _rows(result: dict) -> list[dict]:
        rows = result.get("list")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise DataUnavailable("Malformed Bybit result list")
        return rows

    @staticmethod
    def _category(result: dict) -> None:
        if result.get("category", "linear") != "linear":
            raise DataUnavailable("Unexpected market category")

    async def _pages(
        self, path: str, params: dict, *, private: bool = False
    ) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()
        query = dict(params)
        for _ in range(self.MAX_PAGES):
            result = await self._request("GET", path, query, private=private)
            self._category(result)
            rows.extend(self._rows(result))
            cursor = result.get("nextPageCursor", "")
            if cursor == "":
                return rows
            if not isinstance(cursor, str) or cursor in seen:
                raise DataUnavailable("Invalid or repeated pagination cursor")
            seen.add(cursor)
            query["cursor"] = cursor
        raise DataUnavailable("Bybit pagination limit exceeded")

    async def candles(
        self,
        symbol: str,
        interval: str = "D",
        limit: int = 500,
        now_ms: int | None = None,
    ) -> list[Candle]:
        _symbol(symbol)
        _interval(interval)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if now_ms is None:
            await self._ensure_clock()
            now_ms = self._now_ms()
        now_ms = _integer(now_ms, "now_ms")
        result = await self._request(
            "GET",
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "end": _floor_time(now_ms, interval) - 1,
            },
        )
        return validate_candles(self._candle_rows(result, symbol), interval, now_ms)[
            -limit:
        ]

    @classmethod
    def _candle_rows(cls, result: dict, symbol: str) -> list[Candle]:
        cls._category(result)
        if result.get("symbol") != symbol or not isinstance(result.get("list"), list):
            raise DataUnavailable("Malformed candle response")
        candles = []
        for row in result["list"]:
            if not isinstance(row, list) or len(row) != 7:
                raise DataUnavailable("Malformed candle row")
            stamp = _integer(row[0], "candle timestamp")
            prices = [_number(value, "candle price") for value in row[1:5]]
            volume = _number(row[5], "candle volume", zero=True)
            _number(row[6], "candle turnover", zero=True)
            candles.append(Candle(stamp, *prices, volume))
        return candles

    async def candle_range(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Candle]:
        """Every complete candle with open >= start_ms and close <= end_ms.

        Partial boundary candles are excluded (start rounds up, end down). End
        may not exceed the synchronized server clock. Empty ranges and requests
        exceeding 50,000 bars fail before querying klines. Missing listings/data
        fail rather than silently shortening restart recovery history.
        """
        _symbol(symbol)
        _interval(interval)
        start, end = _integer(start_ms, "start_ms"), _integer(end_ms, "end_ms")
        if start >= end:
            raise ValueError("Candle range must have start_ms < end_ms")
        first = _floor_time(start, interval)
        if first < start:
            first = _next_time(first, interval)
        last = latest_closed_open_time(interval, end)
        if first > last:
            raise ValueError("Candle range contains no complete interval")
        expected = 0
        stamp = first
        while stamp <= last:
            expected += 1
            if expected > self.MAX_CANDLE_RANGE:
                raise ValueError("Candle range exceeds 50000 bars; split the request")
            stamp = _next_time(stamp, interval)
        await self._ensure_clock()
        if end > self._now_ms():
            raise ValueError("Candle range end_ms is in the future")
        unique: dict[int, Candle] = {}
        cursor = last
        for _ in range(self.MAX_PAGES):
            result = await self._request(
                "GET",
                "/v5/market/kline",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": first,
                    "end": cursor,
                    "limit": 1000,
                },
            )
            rows = self._candle_rows(result, symbol)
            if not rows:
                raise DataUnavailable("Candle range is missing required history")
            for candle in rows:
                stamp = candle.open_time
                if not first <= stamp <= last:
                    raise DataUnavailable("Out-of-range or unclosed recovery candle")
                if stamp in unique and unique[stamp] != candle:
                    raise DataUnavailable("Conflicting duplicate recovery candle")
                unique[stamp] = candle
            oldest = min(candle.open_time for candle in rows)
            if oldest > cursor:
                raise DataUnavailable("Candle pagination made no progress")
            if oldest == first:
                break
            cursor = oldest - 1
        else:
            raise DataUnavailable("Candle pagination limit exceeded")
        candles = validate_candles(list(unique.values()), interval, end)
        if len(candles) != expected or candles[0].open_time != first:
            raise DataUnavailable("Candle range is missing required history")
        return candles

    async def instruments(self) -> dict[str, Instrument]:
        """Trading USDT linear perpetuals. max_qty is the exchange's LIMIT maximum.

        max_leverage is exchange metadata, not the runtime's <=5 risk cap.
        No default tick/lot/notional/funding values are synthesized.
        """
        rows = await self._pages(
            "/v5/market/instruments-info",
            {
                "category": "linear",
                "status": "Trading",
                "limit": 1000,
            },
        )
        result = {}
        metadata = {}
        for row in rows:
            if not (
                row.get("status") == "Trading"
                and row.get("quoteCoin") == "USDT"
                and row.get("settleCoin") == "USDT"
                and row.get("contractType") == "LinearPerpetual"
            ):
                continue
            try:
                symbol = _symbol(row["symbol"])
                lot, price, leverage = (
                    row["lotSizeFilter"],
                    row["priceFilter"],
                    row["leverageFilter"],
                )
                instrument = Instrument(
                    symbol=symbol,
                    qty_step=_number(lot["qtyStep"], "quantity step"),
                    min_qty=_number(lot["minOrderQty"], "minimum quantity"),
                    max_qty=_number(lot["maxOrderQty"], "maximum limit quantity"),
                    tick_size=_number(price["tickSize"], "tick size"),
                    min_notional=_number(
                        lot["minNotionalValue"], "minimum notional", zero=True
                    ),
                    funding_interval_minutes=_integer(
                        row["fundingInterval"], "funding interval"
                    ),
                    max_leverage=_number(leverage["maxLeverage"], "maximum leverage"),
                )
            except (KeyError, TypeError, ValueError):
                raise DataUnavailable("Malformed instrument metadata") from None
            if (
                instrument.min_qty > instrument.max_qty
                or instrument.qty_step > instrument.max_qty
                or instrument.funding_interval_minutes <= 0
            ):
                raise DataUnavailable("Inconsistent instrument metadata")
            if symbol in result and result[symbol] != instrument:
                raise DataUnavailable("Conflicting instrument metadata")
            result[symbol] = instrument
            metadata[symbol] = {
                k: row.get(k)
                for k in (
                    "symbol",
                    "status",
                    "quoteCoin",
                    "settleCoin",
                    "contractType",
                    "baseCoin",
                    "launchTime",
                    "isPreListing",
                    "symbolType",
                )
            }
        if not result:
            raise DataUnavailable("No trading USDT linear perpetual instruments")
        self.instrument_metadata = metadata
        return result

    async def tickers(self) -> dict:
        """Unsigned complete linear ticker snapshot for liquidity selection."""
        await self._ensure_clock()
        result = await self._request(
            "GET", "/v5/market/tickers", {"category": "linear"}
        )
        self._category(result)
        rows = self._rows(result)
        if not rows:
            raise DataUnavailable("Empty market ticker snapshot")
        by_symbol = {}
        for row in rows:
            symbol = row.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                raise DataUnavailable("Missing ticker symbol")
            if symbol in by_symbol:
                raise DataUnavailable("Duplicate market ticker")
            by_symbol[symbol] = row
        # This endpoint has no per-snapshot server timestamp. Use receipt time
        # in the runtime's clock domain, not the signing clock's offset.
        return {"as_of": time.time(), "list": list(by_symbol.values())}

    async def liquidity_book(self, symbol: str, band_bps: float = 25) -> dict:
        """Executable displayed depth within a narrow band, not total book size."""
        if not math.isfinite(band_bps) or not 0 < band_bps <= 100:
            raise ValueError("Invalid depth band")
        await self._ensure_clock()
        result = await self._request(
            "GET",
            "/v5/market/orderbook",
            {
                "category": "linear",
                "symbol": _symbol(symbol),
                "limit": 200,
            },
        )
        if result.get("s") != symbol:
            raise DataUnavailable("Mismatched order book")
        stamp = _integer(result.get("ts"), "book timestamp")
        age_ms = self._now_ms() - stamp
        if not 0 <= age_ms <= 120000:
            raise DataUnavailable("Stale or future order book")
        observed_at = time.time() - age_ms / 1000
        sides = []
        for key, descending in (("b", True), ("a", False)):
            rows = result.get(key)
            if not isinstance(rows, list) or not rows:
                raise DataUnavailable("Missing order book side")
            levels = []
            for row in rows:
                if not isinstance(row, list) or len(row) != 2:
                    raise DataUnavailable("Malformed order book level")
                levels.append(
                    (_number(row[0], "book price"), _number(row[1], "book size"))
                )
            prices = [p for p, _ in levels]
            if len(set(prices)) != len(prices) or prices != sorted(
                prices, reverse=descending
            ):
                raise DataUnavailable("Unordered order book")
            sides.append(levels)
        bids, asks = sides
        if bids[0][0] >= asks[0][0]:
            raise DataUnavailable("Crossed order book")
        mid = bids[0][0] / 2 + asks[0][0] / 2
        fraction = band_bps / 10000
        book = {
            "as_of": observed_at,
            "exchange_as_of": stamp / 1000,
            "band_bps": band_bps,
            "bid_depth_usdt": sum(p * q for p, q in bids if p >= mid * (1 - fraction)),
            "ask_depth_usdt": sum(p * q for p, q in asks if p <= mid * (1 + fraction)),
            "bid": bids[0][0],
            "ask": asks[0][0],
            "spread_bps": (asks[0][0] - bids[0][0]) / mid * 10000,
        }
        if not all(
            math.isfinite(book[k])
            for k in ("bid_depth_usdt", "ask_depth_usdt", "spread_bps")
        ):
            raise DataUnavailable("Nonfinite order book depth")
        return book

    async def ticker(self, symbol: str) -> dict:
        result = await self._request(
            "GET",
            "/v5/market/tickers",
            {
                "category": "linear",
                "symbol": _symbol(symbol),
            },
        )
        self._category(result)
        rows = self._rows(result)
        if len(rows) != 1 or rows[0].get("symbol") != symbol:
            raise DataUnavailable("Missing or mismatched ticker")
        return rows[0]

    async def positions(self) -> list[dict]:
        """All returned rows, including zeros; settleCoin queries normally omit flats."""
        return await self._pages(
            "/v5/position/list",
            {
                "category": "linear",
                "settleCoin": "USDT",
                "limit": 200,
            },
            private=True,
        )

    async def position_info(self, symbol: str) -> list[dict]:
        """Symbol-specific settings, including zero-size rows, leverage and positionIdx."""
        rows = await self._pages(
            "/v5/position/list",
            {
                "category": "linear",
                "symbol": _symbol(symbol),
                "limit": 200,
            },
            private=True,
        )
        if not rows or any(row.get("symbol") != symbol for row in rows):
            raise DataUnavailable("Missing or mismatched position settings")
        return rows

    async def risk_limits(self, symbol: str) -> list[dict]:
        """Raw exchange risk tiers; no inferred tier, leverage or margin deduction."""
        rows = await self._pages(
            "/v5/market/risk-limit",
            {
                "category": "linear",
                "symbol": _symbol(symbol),
            },
        )
        if not rows or any(row.get("symbol") != symbol for row in rows):
            raise DataUnavailable("Missing or mismatched risk tiers")
        return rows

    async def open_orders(self) -> list[dict]:
        """All active USDT orders, including conditional orders; preserve raw fields."""
        return await self._pages(
            "/v5/order/realtime",
            {
                "category": "linear",
                "settleCoin": "USDT",
                "openOnly": 0,
                "limit": 50,
            },
            private=True,
        )

    async def account_info(self) -> dict:
        """Raw UTA account fields, including the global marginMode."""
        return await self._request("GET", "/v5/account/info", private=True)

    async def equity(self) -> float:
        """Actual USDT coin equity for the supported USDT-settled perpetual account.

        Require reported coin equity == walletBalance + unrealisedPnl using exact
        decimals. Borrowing or unexplained option/other adjustments are unsupported.
        totalEquity/usdValue are USD marks, not USDT units, and are never substituted.
        This is capital including deposits, not strategy PnL; attribution belongs
        to the ledger's fills and cashflows, never a difference of equity snapshots.
        """
        result = await self._request(
            "GET",
            "/v5/account/wallet-balance",
            {
                "accountType": "UNIFIED",
                "coin": "USDT",
            },
            private=True,
        )
        rows = self._rows(result)
        if len(rows) != 1 or rows[0].get("accountType") != "UNIFIED":
            raise DataUnavailable("Missing UNIFIED wallet")
        coins = rows[0].get("coin")
        if (
            not isinstance(coins, list)
            or len(coins) != 1
            or not isinstance(coins[0], dict)
            or coins[0].get("coin") != "USDT"
        ):
            raise DataUnavailable(
                "Expected exactly one USDT coin wallet; other settlement currencies are unsupported"
            )
        coin = coins[0]
        try:
            wallet = _decimal(
                coin.get("walletBalance"), "USDT wallet balance", zero=True
            )
            reported = _decimal(coin.get("equity"), "USDT coin equity")
            pnl = Decimal(str(coin.get("unrealisedPnl")))
            if not pnl.is_finite():
                raise ValueError
            for field in ("spotBorrow", "borrowAmount", "accruedInterest"):
                if (
                    field in coin
                    and _decimal(coin[field], "USDT borrowing", zero=True) != 0
                ):
                    raise DataUnavailable(
                        "Borrowed USDT is unsupported for perpetual sizing equity"
                    )
        except (ValueError, InvalidOperation):
            raise DataUnavailable(
                "Missing or invalid USDT coin equity components"
            ) from None
        if wallet + pnl != reported:
            raise DataUnavailable(
                "USDT coin equity does not equal wallet balance plus unrealised PnL"
            )
        return _number(reported, "USDT coin equity")

    async def executions(self, symbol: str, start_ms: int | None = None) -> list[dict]:
        """Raw fills, sorted by time/id, covering at most the most recent seven days.

        An older start is rejected, never silently truncated. All cursor pages
        share the same explicit time bounds, so retries do not move the window.
        REST does not promise execPnl: use closed_pnl and transaction_log alongside
        execFee/execType/closedSize. Any additional exchange fields are preserved.
        """
        self._require_credentials()
        _symbol(symbol)
        await self._ensure_clock()
        end = self._now_ms()
        start = (
            end - self.EXECUTION_LOOKBACK_MS
            if start_ms is None
            else _integer(start_ms, "start_ms")
        )
        if not end - self.EXECUTION_LOOKBACK_MS <= start <= end:
            raise ValueError("start_ms must be within the most recent seven days")
        rows = await self._pages(
            "/v5/execution/list",
            {
                "category": "linear",
                "symbol": symbol,
                "startTime": start,
                "endTime": end,
                "limit": 100,
            },
            private=True,
        )
        unique = {}
        for row in rows:
            stamp = _integer(row.get("execTime"), "execution time")
            key = row.get("execId")
            if (
                row.get("symbol") != symbol
                or not isinstance(key, str)
                or not key
                or not start <= stamp <= end
            ):
                raise DataUnavailable("Malformed or out-of-window execution")
            if key in unique and unique[key] != row:
                raise DataUnavailable("Conflicting execution records")
            unique[key] = row
        return sorted(
            unique.values(), key=lambda row: (int(row["execTime"]), row["execId"])
        )

    async def _accounting_pages(
        self, path: str, params: dict, start: int, end: int
    ) -> list[dict]:
        if not 0 <= start <= end or end - start > self.ACCOUNTING_LOOKBACK_MS:
            raise ValueError(
                "Accounting range must be ordered and no longer than 730 days"
            )
        rows = []
        # Both endpoints impose a <=7 day request window. Use disjoint inclusive
        # millisecond windows and reset the cursor for every window.
        while start <= end:
            window_end = min(end, start + 7 * _DAY_MS)
            rows.extend(
                await self._pages(
                    path,
                    {
                        **params,
                        "startTime": start,
                        "endTime": window_end,
                    },
                    private=True,
                )
            )
            start = window_end + 1
        return rows

    @staticmethod
    def _unique_records(rows: list[dict], key_name: str, time_name: str) -> list[dict]:
        unique = {}
        for row in rows:
            key = row.get(key_name)
            _integer(row.get(time_name), "accounting timestamp")
            if not isinstance(key, str) or not key:
                raise DataUnavailable("Missing accounting record identity")
            if key in unique and unique[key] != row:
                raise DataUnavailable("Conflicting accounting records")
            unique[key] = row
        return sorted(
            unique.values(), key=lambda row: (int(row[time_name]), row[key_name])
        )

    async def closed_pnl(self, symbol: str, start_ms: int | None = None) -> list[dict]:
        """Exchange closed-PnL records, including closedPnl/openFee/closeFee.

        Default: last seven days; explicit starts support up to 730 days via
        seven-day windows. Preserve raw accounting semantics: do not add fees
        again to closedPnl or treat these cumulative order records as fills.
        """
        self._require_credentials()
        _symbol(symbol)
        await self._ensure_clock()
        end = self._now_ms()
        start = (
            end - 7 * _DAY_MS if start_ms is None else _integer(start_ms, "start_ms")
        )
        rows = await self._accounting_pages(
            "/v5/position/closed-pnl",
            {
                "category": "linear",
                "symbol": symbol,
                "limit": 100,
            },
            start,
            end,
        )
        if any(row.get("symbol") != symbol for row in rows):
            raise DataUnavailable("Mismatched closed-PnL symbol")
        return self._unique_records(rows, "orderId", "createdTime")

    async def transaction_log(
        self, start_ms: int, end_ms: int | None = None
    ) -> list[dict]:
        """Raw UNIFIED linear USDT cashflows, including unattributed settlements.

        Positive funding means received; positive fee means paid; Bybit's change
        is cashFlow + funding - fee. Funding may have no orderLinkId, so the ledger
        must attribute by symbol/side/time and owned exposure, not invent a link.
        Explicit ranges up to 730 days are paginated in <=7-day request windows.
        """
        self._require_credentials()
        start = _integer(start_ms, "start_ms")
        if end_ms is None:
            await self._ensure_clock()
            end_ms = self._now_ms()
        end = _integer(end_ms, "end_ms")
        rows = await self._accounting_pages(
            "/v5/account/transaction-log",
            {
                "accountType": "UNIFIED",
                "category": "linear",
                "currency": "USDT",
                "limit": 50,
            },
            start,
            end,
        )
        if any(
            row.get("currency") != "USDT"
            or row.get("category") != "linear"
            or not start
            <= _integer(row.get("transactionTime"), "transaction time")
            <= end
            for row in rows
        ):
            raise DataUnavailable("Mismatched or out-of-window transaction")
        return self._unique_records(rows, "id", "transactionTime")

    async def order(self, symbol: str, order_link_id: str) -> dict | None:
        """Look up realtime first, then history (Bybit's default seven-day window).

        None means not currently visible in these endpoints, not safe to resubmit.
        Cancelled/rejected retention can be shorter than the history window.
        """
        return await self._order_lookup(symbol, "orderLinkId", _link_id(order_link_id))

    async def order_by_id(self, symbol: str, order_id: str) -> dict | None:
        """Raw exchange-ID receipt, checking realtime then order history.

        Supports generated TP/SL orders with empty orderLinkId. Preserve actual
        parentOrderLinkId/createType/stopOrderType/reduceOnly fields; neither fill
        classification nor ownership is inferred here. The executor must verify
        the receipt against its owned entry and fills. Missing/delayed history
        returns None, never proof of non-execution or permission to replay.

        Uses Bybit's default seven-day history window. The custom order-link ID
        length limit does not apply to exchange order IDs.
        """
        if (
            not isinstance(order_id, str)
            or not order_id
            or any(
                char.isspace() or ord(char) < 32 or ord(char) == 127
                for char in order_id
            )
        ):
            raise ValueError(
                "order_id must be a nonempty exchange identifier without whitespace"
            )
        return await self._order_lookup(symbol, "orderId", order_id)

    async def _order_lookup(
        self, symbol: str, identity_field: str, identity: str
    ) -> dict | None:
        params = {
            "category": "linear",
            "symbol": _symbol(symbol),
            identity_field: identity,
            "limit": 50,
        }
        for path in ("/v5/order/realtime", "/v5/order/history"):
            rows = await self._pages(path, params, private=True)
            if any(
                row.get("symbol") != symbol or row.get(identity_field) != identity
                for row in rows
            ):
                raise DataUnavailable("Mismatched order lookup")
            if rows:
                if any(row != rows[0] for row in rows[1:]):
                    raise DataUnavailable("Conflicting order lookup")
                return rows[0]
        return None

    @staticmethod
    def _order_params(symbol: str, side: str, qty: Any, order_link_id: str) -> dict:
        if side not in ("Buy", "Sell"):
            raise ValueError("side must be Buy or Sell")
        return {
            "category": "linear",
            "symbol": _symbol(symbol),
            "side": side,
            "qty": format(_decimal(qty, "qty"), "f"),
            "positionIdx": 0,
            "orderLinkId": _link_id(order_link_id),
        }

    @staticmethod
    def _protection(stop: Any, target: Any) -> dict:
        return {
            "stopLoss": format(_decimal(stop, "stop"), "f"),
            "takeProfit": format(_decimal(target, "target"), "f"),
            "tpslMode": "Full",
            "slOrderType": "Market",
            "tpOrderType": "Market",
            "slTriggerBy": "LastPrice",
            "tpTriggerBy": "LastPrice",
        }

    async def _order_write(self, path: str, params: dict) -> dict:
        result = await self._request("POST", path, params, private=True)
        if (
            not isinstance(result.get("orderId"), str)
            or not result["orderId"]
            or result.get("orderLinkId") != params["orderLinkId"]
        ):
            raise AmbiguousOrderError(params["orderLinkId"])
        return result

    async def submit_limit(
        self,
        symbol: str,
        side: str,
        qty: Any,
        price: Any,
        stop: Any,
        target: Any,
        order_link_id: str,
    ) -> dict:
        """Submit one GTC limit with initial Full market TP/SL in one-way mode.

        Callers size/quantize against fresh instruments; decimals are preserved,
        never silently rounded or expressed in scientific notation.
        """
        params = self._order_params(symbol, side, qty, order_link_id)
        entry = _decimal(price, "price")
        protection = self._protection(stop, target)
        sl, tp = Decimal(protection["stopLoss"]), Decimal(protection["takeProfit"])
        if not (sl < entry < tp if side == "Buy" else tp < entry < sl):
            raise ValueError(
                "stop and target must bracket entry for the requested side"
            )
        params.update(
            orderType="Limit",
            timeInForce="GTC",
            price=format(entry, "f"),
            reduceOnly=False,
            **protection,
        )
        return await self._order_write("/v5/order/create", params)

    async def reduce_market(
        self, symbol: str, side: str, qty: Any, order_link_id: str
    ) -> dict:
        """side is the closing order side; quantity must be explicitly positive."""
        params = self._order_params(symbol, side, qty, order_link_id)
        params.update(orderType="Market", timeInForce="IOC", reduceOnly=True)
        return await self._order_write("/v5/order/create", params)

    async def cancel(self, symbol: str, order_link_id: str) -> dict:
        return await self._order_write(
            "/v5/order/cancel",
            {
                "category": "linear",
                "symbol": _symbol(symbol),
                "orderLinkId": _link_id(order_link_id),
            },
        )

    async def protect(self, symbol: str, stop: Any, target: Any) -> dict:
        return await self._request(
            "POST",
            "/v5/position/trading-stop",
            {
                "category": "linear",
                "symbol": _symbol(symbol),
                "positionIdx": 0,
                **self._protection(stop, target),
            },
            private=True,
        )

    async def funding_history(
        self, symbol: str, start_ms: int | None = None, end_ms: int | None = None
    ) -> list[dict]:
        """All actual settlements in [start_ms, end_ms], ascending, default seven days.

        Pagination uses returned settlement timestamps, so changing instrument
        funding intervals do not skip records or fabricate intervening rates.
        """
        _symbol(symbol)
        if end_ms is None:
            await self._ensure_clock()
            end_ms = self._now_ms()
        end = _integer(end_ms, "end_ms")
        start = (
            max(0, end - 7 * _DAY_MS)
            if start_ms is None
            else _integer(start_ms, "start_ms")
        )
        if start > end:
            raise ValueError("start_ms must not exceed end_ms")
        records: dict[int, dict] = {}
        for _ in range(self.MAX_PAGES):
            params = {
                "category": "linear",
                "symbol": symbol,
                "startTime": start,
                "endTime": end,
                "limit": 200,
            }
            result = await self._request("GET", "/v5/market/funding/history", params)
            self._category(result)
            rows = self._rows(result)
            if not rows:
                break
            stamps = []
            for row in rows:
                stamp = _integer(row.get("fundingRateTimestamp"), "funding timestamp")
                try:
                    rate = Decimal(str(row["fundingRate"]))
                    if not rate.is_finite():
                        raise ValueError
                except (KeyError, ValueError, InvalidOperation):
                    raise DataUnavailable("Invalid funding rate") from None
                if row.get("symbol") != symbol or not start <= stamp <= end:
                    raise DataUnavailable("Mismatched or out-of-window funding record")
                if stamp in records and records[stamp] != row:
                    raise DataUnavailable("Conflicting funding records")
                records[stamp] = row
                stamps.append(stamp)
            end = min(stamps) - 1
            if len(rows) < 200 or end < start:
                break
        else:
            raise DataUnavailable("Funding pagination limit exceeded")
        return [records[stamp] for stamp in sorted(records)]
