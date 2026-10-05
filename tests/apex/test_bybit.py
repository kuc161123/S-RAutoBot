"""Offline adapter contract tests. No credentials, sockets, storage or exchange I/O."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import hmac
import importlib
import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import aiohttp
import pytest
from yarl import URL

from apex_bot import bybit
from apex_bot.models import Candle, Instrument


DAY = 86_400_000
NOW = 1_735_689_600_000  # 2025-01-01 UTC


def ms(iso):
    return int(
        datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000
    )


def ok(result):
    return {"retCode": 0, "retMsg": "OK", "result": result}


def page(rows, cursor="", **extra):
    return ok({"list": rows, "nextPageCursor": cursor, **extra})


def bar(timestamp, **changes):
    values = dict(
        stamp=str(timestamp),
        open="10",
        high="12",
        low="9",
        close="11",
        volume="2",
        turnover="22",
    )
    values.update(changes)
    return list(values.values())


def instrument(symbol="BTCUSDT", **changes):
    row = {
        "symbol": symbol,
        "status": "Trading",
        "quoteCoin": "USDT",
        "settleCoin": "USDT",
        "contractType": "LinearPerpetual",
        "fundingInterval": 60,
        "priceFilter": {"tickSize": "0.00000001"},
        "lotSizeFilter": {
            "qtyStep": "0.001",
            "minOrderQty": "0.003",
            "maxOrderQty": "876.5",
            "maxMktOrderQty": "95.5",
            "minNotionalValue": "7.25",
        },
        "leverageFilter": {"maxLeverage": "37.5"},
    }
    row.update(changes)
    return row


class Reply:
    def __init__(self, body, status=200, headers=None, on_read=None):
        self.body, self.status = body, status
        self.headers = headers or {}
        self.on_read = on_read

    async def json(self, **kwargs):
        assert kwargs == {"content_type": None}
        if self.on_read:
            self.on_read()
        if isinstance(self.body, BaseException):
            raise self.body
        return deepcopy(self.body)


class ExchangeContext:
    def __init__(self, event):
        self.event = event

    async def __aenter__(self):
        if isinstance(self.event, BaseException):
            raise self.event
        return self.event if isinstance(self.event, Reply) else Reply(self.event)

    async def __aexit__(self, *args):
        return False


class Session:
    """Scripted aiohttp context-manager protocol; unexpected requests fail immediately."""

    def __init__(self, events=(), clock_events=()):
        self.events = list(events)
        self.clock_events = list(clock_events)
        self.calls = []

    def request(self, method, url, **kwargs):
        assert isinstance(url, URL)
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"].total == 15
        call = {
            "method": method,
            "url": url,
            "path": url.path,
            "query": {k: v[0] for k, v in parse_qs(url.raw_query_string).items()},
            **kwargs,
        }
        self.calls.append(call)
        if url.path == "/v5/market/time":
            event = (
                self.clock_events.pop(0)
                if self.clock_events
                else ok({"timeNano": str(NOW * 1_000_000)})
            )
        else:
            assert self.events, f"Unscripted request: {method} {url.path}"
            event = self.events.pop(0)
        return ExchangeContext(event)

    @property
    def business_calls(self):
        return [call for call in self.calls if call["path"] != "/v5/market/time"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    async def forbid_network(*args, **kwargs):
        raise AssertionError("These tests must never make a network request")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbid_network)
    clock = SimpleNamespace(mono=100.0, wall=NOW / 1000 + 9999, sleeps=[])
    monkeypatch.setattr(
        bybit,
        "time",
        SimpleNamespace(monotonic=lambda: clock.mono, time=lambda: clock.wall),
    )

    async def sleep(delay):
        clock.sleeps.append(delay)

    monkeypatch.setattr(bybit.asyncio, "sleep", sleep)
    monkeypatch.setattr(bybit.random, "uniform", lambda lo, hi: 0.125)
    return clock


def client(events=(), *, private=False, clock_events=()):
    session = Session(events, clock_events)
    adapter = bybit.BybitClient(
        session,
        api_key="fixture-key" if private else "",
        api_secret="fixture-secret" if private else "",
    )
    return adapter, session


def test_public_ticker_is_raw_unsigned_and_constructor_is_inert():
    ticker = {
        "symbol": "BTCUSDT",
        "lastPrice": "12.000",
        "fundingRate": "-0.00003",
        "newField": "kept",
    }
    api, session = client([page([ticker], category="linear")])
    assert session.calls == []
    assert asyncio.run(api.ticker("BTCUSDT")) == ticker
    assert session.calls[0]["headers"] == {}
    assert session.calls[0]["query"] == {"category": "linear", "symbol": "BTCUSDT"}


def test_candles_sort_deduplicate_and_exclude_incomplete():
    rows = [
        bar(NOW),
        bar(NOW - DAY),
        bar(NOW - 3 * DAY),
        bar(NOW - 2 * DAY),
        bar(NOW - DAY),
    ]
    api, session = client([page(rows, category="linear", symbol="BTCUSDT")])
    result = asyncio.run(api.candles("BTCUSDT", now_ms=NOW + 123, limit=3))
    assert result == [Candle(NOW - n * DAY, 10, 12, 9, 11, 2) for n in (3, 2, 1)]
    assert session.calls[0]["query"]["end"] == str(NOW - 1)
    assert session.calls[0]["query"]["limit"] == "3"


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [bar(NOW)],
        [bar(NOW - 2 * DAY)],  # empty, open only, stale
        [bar(NOW - 3 * DAY), bar(NOW - DAY)],  # missing closed bar
        [bar(NOW - DAY), bar(NOW - DAY, close="10.5")],
        [bar(NOW - DAY + 1)],
        [bar(NOW + DAY)],
        [bar(NOW - DAY, high="9")],
        [bar(NOW - DAY, low="11")],
        [bar(NOW - DAY, volume="-1")],
        [bar(NOW - DAY, turnover="NaN")],
        [bar(NOW - DAY, close="Infinity")],
        [bar(NOW - DAY, open="0")],
        [bar(NOW - DAY, close="1e9999")],
        [bar(NOW - DAY)[:-1]],
        [bar(NOW - DAY, stamp="1735603200000.5")],
    ],
)
def test_candles_fail_closed_on_bad_history(rows):
    api, session = client([page(rows, category="linear", symbol="BTCUSDT")])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.candles("BTCUSDT", now_ms=NOW))
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "interval,now,expected",
    [
        ("D", "2024-03-01T00:00:00", "2024-02-29T00:00:00"),
        ("240", "2024-03-01T05:00:00", "2024-03-01T00:00:00"),
        ("60", "2024-03-01T00:59:59", "2024-02-29T23:00:00"),
        ("W", "2024-03-04T00:00:00", "2024-02-26T00:00:00"),
        ("W", "2024-03-03T23:59:59", "2024-02-19T00:00:00"),
        ("M", "2024-03-01T00:00:00", "2024-02-01T00:00:00"),
        ("M", "2025-01-01T00:00:00", "2024-12-01T00:00:00"),
    ],
)
def test_utc_boundaries(interval, now, expected):
    assert bybit.latest_closed_open_time(interval, ms(now)) == ms(expected)


def test_calendar_month_contiguity_and_standalone_freshness():
    candles = [
        Candle(ms(date), 10, 12, 9, 11)
        for date in ("2024-01-01", "2024-02-01", "2024-03-01")
    ]
    assert bybit.validate_candles(candles, "M", ms("2024-04-01")) == candles
    with pytest.raises(bybit.DataUnavailable, match="Gap"):
        bybit.validate_candles(candles[::2], "M", ms("2024-04-01"))


def test_candles_use_server_clock_without_credentials():
    api, session = client([page([bar(NOW - DAY)], category="linear", symbol="BTCUSDT")])
    assert len(asyncio.run(api.candles("BTCUSDT"))) == 1
    assert [call["path"] for call in session.calls] == [
        "/v5/market/time",
        "/v5/market/kline",
    ]
    assert all(not call["headers"] for call in session.calls)


def test_candle_range_recovers_2000_closed_bars_backwards_without_gaps():
    width = 180_000
    stamps = [NOW - n * width for n in range(2000, 0, -1)]
    api, session = client(
        [
            page([bar(t) for t in reversed(stamps[1000:])], symbol="BTCUSDT"),
            page([bar(t) for t in reversed(stamps[:1000])], symbol="BTCUSDT"),
        ]
    )
    result = asyncio.run(api.candle_range("BTCUSDT", "3", stamps[0], NOW))
    assert [c.open_time for c in result] == stamps
    calls = session.business_calls
    assert len(calls) == 2
    assert calls[1]["query"]["end"] == str(stamps[1000] - 1)
    assert all(call["query"]["start"] == str(stamps[0]) for call in calls)
    assert all(call["headers"] == {} for call in session.calls)


def test_candle_range_rejects_gap_between_2000_bar_pages():
    width = 180_000
    stamps = [NOW - n * width for n in range(2000, 0, -1)]
    api, _ = client(
        [
            page([bar(t) for t in reversed(stamps[1000:])], symbol="BTCUSDT"),
            page([bar(t) for t in reversed(stamps[:999])], symbol="BTCUSDT"),
        ]
    )
    with pytest.raises(bybit.DataUnavailable, match="Gap"):
        asyncio.run(api.candle_range("BTCUSDT", "3", stamps[0], NOW))


@pytest.mark.parametrize(
    "start,end",
    [(NOW, NOW), (NOW, NOW - 1), (NOW - 1, NOW), (NOW - 50_001 * 60_000, NOW)],
)
def test_candle_range_bounds_fail_before_io(start, end):
    api, session = client()
    with pytest.raises(ValueError):
        asyncio.run(api.candle_range("BTCUSDT", "1", start, end))
    assert session.calls == []


def test_candle_range_rejects_future_end_before_kline_read():
    api, session = client()
    with pytest.raises(ValueError, match="future"):
        asyncio.run(api.candle_range("BTCUSDT", "D", NOW - DAY, NOW + DAY))
    assert session.business_calls == []


def test_candle_range_accepts_the_exact_50000_bar_budget():
    api, session = client([page([], symbol="BTCUSDT")])
    with pytest.raises(bybit.DataUnavailable, match="missing required history"):
        asyncio.run(api.candle_range("BTCUSDT", "1", NOW - 50_000 * 60_000, NOW))
    assert len(session.business_calls) == 1


@pytest.mark.parametrize(
    "interval,start,end,stamps",
    [
        ("3", NOW - 3 * 180_000 + 1, NOW - 1, [NOW - 2 * 180_000]),
        ("W", ms("2024-12-09") + 1, ms("2024-12-30") - 1, [ms("2024-12-16")]),
        ("M", ms("2024-01-15"), ms("2024-04-15"), [ms("2024-02-01"), ms("2024-03-01")]),
    ],
)
def test_candle_range_excludes_partial_boundaries_using_utc_anchors(
    interval, start, end, stamps
):
    api, session = client([page([bar(t) for t in reversed(stamps)], symbol="BTCUSDT")])
    candles = asyncio.run(api.candle_range("BTCUSDT", interval, start, end))
    assert [c.open_time for c in candles] == stamps
    assert int(session.business_calls[0]["query"]["start"]) == stamps[0]


@pytest.mark.parametrize("conflicting", [False, True])
def test_candle_range_duplicate_overlap_must_agree(conflicting):
    duplicate = bar(NOW - DAY, close="10.5") if conflicting else bar(NOW - DAY)
    api, _ = client(
        [
            page([bar(NOW - DAY)], symbol="BTCUSDT"),
            page([duplicate, bar(NOW - 2 * DAY)], symbol="BTCUSDT"),
        ]
    )
    if conflicting:
        with pytest.raises(bybit.DataUnavailable, match="Conflicting"):
            asyncio.run(api.candle_range("BTCUSDT", "D", NOW - 2 * DAY, NOW))
    else:
        assert (
            len(asyncio.run(api.candle_range("BTCUSDT", "D", NOW - 2 * DAY, NOW))) == 2
        )


@pytest.mark.parametrize("rows", [[bar(NOW)], [bar(NOW + DAY)], []])
def test_candle_range_never_returns_partial_or_unclosed_history(rows):
    api, _ = client([page(rows, symbol="BTCUSDT")])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.candle_range("BTCUSDT", "D", NOW - DAY, NOW))


def test_instruments_page_filter_and_use_exchange_metadata():
    ignored = [
        instrument(status="PreLaunch"),
        instrument(contractType="LinearFutures"),
        instrument(quoteCoin="USDC"),
        instrument(settleCoin="USDC"),
    ]
    api, session = client(
        [
            page([instrument(), *ignored], "next +/%="),
            page([instrument("ETHUSDT", fundingInterval=240)]),
        ]
    )
    result = asyncio.run(api.instruments())
    assert result["BTCUSDT"] == Instrument(
        "BTCUSDT", 0.001, 0.003, 876.5, 1e-8, 7.25, 60, 37.5
    )
    assert result["ETHUSDT"].funding_interval_minutes == 240
    assert session.calls[1]["query"]["cursor"] == "next +/%="
    assert len(result) == 2


@pytest.mark.parametrize(
    "field", ["fundingInterval", "priceFilter", "lotSizeFilter", "leverageFilter"]
)
def test_instruments_never_invent_missing_filters(field):
    row = instrument()
    del row[field]
    api, _ = client([page([row])])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.instruments())


@pytest.mark.parametrize(
    "event",
    [
        asyncio.TimeoutError(),
        aiohttp.ServerDisconnectedError("secret URL"),
        aiohttp.ClientPayloadError("truncated"),
        Reply(None, 429),
        Reply(None, 503),
        {"retCode": 10006},
        {"retCode": 10000},
        {"retCode": 10016},
        {"retCode": 429},
    ],
)
def test_get_retries_transient_failures_only(event, offline):
    api, session = client([event, page([{"symbol": "BTCUSDT"}])])
    assert asyncio.run(api.ticker("BTCUSDT")) == {"symbol": "BTCUSDT"}
    assert len(session.business_calls) == 2
    assert offline.sleeps == [0.375]


@pytest.mark.parametrize(
    "event,error",
    [
        (Reply(None, 401), bybit.BybitAPIError),
        (Reply(None, 403), bybit.BybitAPIError),
        ({"retCode": 10004}, bybit.BybitAPIError),
        ({"retCode": 110007}, bybit.BybitAPIError),
        (Reply(ValueError("bad json")), bybit.DataUnavailable),
        ({}, bybit.DataUnavailable),
        ({"retCode": False, "result": {}}, bybit.DataUnavailable),
        (ok(None), bybit.DataUnavailable),
    ],
)
def test_get_does_not_retry_confirmed_errors_or_malformed_data(event, error, offline):
    api, session = client([event])
    with pytest.raises(error):
        asyncio.run(api.ticker("BTCUSDT"))
    assert len(session.calls) == 1
    assert offline.sleeps == []


def test_get_retry_budget_backoff_jitter_and_rate_headers(offline):
    events = [
        Reply(None, 429, {"Retry-After": "2"}),
        Reply(None, 429, {"Retry-After": "999"}),
        Reply(None, 503),
        Reply(None, 503),
    ]
    api, session = client(events)
    with pytest.raises(bybit.DataUnavailable, match="exhausted"):
        asyncio.run(api.ticker("BTCUSDT"))
    assert len(session.calls) == 4
    assert offline.sleeps == [2.125, 8, 1.125]


def test_signed_get_uses_exact_wire_query_even_for_opaque_cursor():
    cursor = "opaque%3A+ /?=&雪"
    api, session = client([page([], cursor), page([])], private=True)
    assert asyncio.run(api.positions()) == []
    call = session.business_calls[1]
    query = call["url"].raw_query_string
    assert "cursor=opaque%253A%2B%20%2F%3F%3D%26%E9%9B%AA" in query
    headers = call["headers"]
    expected = hmac.new(
        b"fixture-secret",
        (str(NOW) + "fixture-key5000" + query).encode(),
        hashlib.sha256,
    ).hexdigest()
    assert headers["X-BAPI-SIGN"] == expected
    assert headers["X-BAPI-RECV-WINDOW"] == "5000"
    assert "fixture-key" not in str(call["url"])
    assert call["data"] is None

    # Construct aiohttp's real request object without a connector or any I/O.
    async def wire_request():
        request = aiohttp.ClientRequest("GET", call["url"], headers=headers)
        assert request.url.raw_query_string == query

    asyncio.run(wire_request())


def test_submit_limit_exact_signed_bytes_decimal_format_and_initial_protection():
    api, session = client(
        [ok({"orderId": "exchange-id", "orderLinkId": "stable_1"})], private=True
    )
    result = asyncio.run(
        api.submit_limit(
            "BTCUSDT", "Buy", Decimal("1E-8"), "1E+2", "9E+1", "1.2E+2", "stable_1"
        )
    )
    assert result["orderId"] == "exchange-id"
    call = session.business_calls[0]
    body = json.loads(call["data"])
    assert body == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "qty": "0.00000001",
        "price": "100",
        "stopLoss": "90",
        "takeProfit": "120",
        "orderLinkId": "stable_1",
        "positionIdx": 0,
        "orderType": "Limit",
        "timeInForce": "GTC",
        "reduceOnly": False,
        "tpslMode": "Full",
        "tpOrderType": "Market",
        "slOrderType": "Market",
        "tpTriggerBy": "LastPrice",
        "slTriggerBy": "LastPrice",
    }
    expected = hmac.new(
        b"fixture-secret",
        str(NOW).encode() + b"fixture-key5000" + call["data"],
        hashlib.sha256,
    ).hexdigest()
    assert call["headers"]["X-BAPI-SIGN"] == expected
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["method"] == "POST"


@pytest.mark.parametrize(
    "event",
    [
        asyncio.TimeoutError("secret"),
        aiohttp.ServerDisconnectedError("secret"),
        aiohttp.ClientPayloadError("partial response"),
        asyncio.CancelledError(),
        Reply(None, 408),
        Reply(None, 500),
        Reply(None, 502),
        Reply(None, 504),
        Reply(None, 307),
        {"retCode": 10000},
        {"retCode": 10016},
        {"retCode": 10014},
        {"retCode": 110072},
        Reply(ValueError("bad json")),
        {},
        ok(None),
        ok({}),
        ok({"orderId": "exchange-id", "orderLinkId": "some-other-order"}),
    ],
)
def test_ambiguous_post_is_never_replayed(event, offline):
    api, session = client([event], private=True)
    with pytest.raises(bybit.AmbiguousOrderError) as error:
        asyncio.run(api.reduce_market("BTCUSDT", "Sell", ".2", "stable-id"))
    assert error.value.order_link_id == "stable-id"
    assert len(session.business_calls) == 1
    assert session.business_calls[0]["method"] == "POST"
    assert offline.sleeps == []


@pytest.mark.parametrize(
    "event,code",
    [
        ({"retCode": 110007, "retMsg": "fixture-secret"}, 110007),
        ({"retCode": 10004}, 10004),
        ({"retCode": 10002}, 10002),
        ({"retCode": 10006}, 10006),
        (Reply(None, 429), None),
        (Reply(None, 403), None),
    ],
)
def test_confirmed_post_rejections_are_distinct_and_not_retried(event, code, offline):
    api, session = client([event], private=True)
    with pytest.raises(bybit.OrderRejectedError) as error:
        asyncio.run(api.cancel("BTCUSDT", "stable-id"))
    assert error.value.ret_code == code
    assert isinstance(error.value, bybit.BybitAPIError)
    assert len(session.business_calls) == 1
    assert offline.sleeps == []
    assert "fixture-secret" not in str(error.value)


def test_reduce_cancel_protect_payloads_and_return_raw_acknowledgements():
    api, session = client(
        [
            ok({"orderId": "1", "orderLinkId": "exit-1"}),
            ok({"orderId": "2", "orderLinkId": "entry-1"}),
            ok({}),
        ],
        private=True,
    )

    async def scenario():
        await api.reduce_market("BTCUSDT", "Sell", ".25", "exit-1")
        await api.cancel("BTCUSDT", "entry-1")
        assert await api.protect("BTCUSDT", "100", "120") == {}

    asyncio.run(scenario())
    calls = session.business_calls
    reduce = json.loads(calls[0]["data"])
    assert reduce["reduceOnly"] is True and reduce["positionIdx"] == 0
    assert reduce["orderType"] == "Market" and reduce["timeInForce"] == "IOC"
    assert "takeProfit" not in reduce and "stopLoss" not in reduce
    assert [call["path"] for call in calls] == [
        "/v5/order/create",
        "/v5/order/cancel",
        "/v5/position/trading-stop",
    ]
    protect = json.loads(calls[2]["data"])
    assert protect["tpslMode"] == "Full" and protect["slTriggerBy"] == "LastPrice"
    assert protect["tpOrderType"] == protect["slOrderType"] == "Market"


@pytest.mark.parametrize("operation", ["cancel", "protect"])
def test_cancel_and_protection_timeouts_are_ambiguous(operation):
    api, session = client([asyncio.TimeoutError()], private=True)
    coro = (
        api.cancel("BTCUSDT", "id")
        if operation == "cancel"
        else api.protect("BTCUSDT", 9, 12)
    )
    with pytest.raises(bybit.AmbiguousOrderError):
        asyncio.run(coro)
    assert len(session.business_calls) == 1


@pytest.mark.parametrize(
    "method,args",
    [
        ("positions", ()),
        ("position_info", ("BTCUSDT",)),
        ("open_orders", ()),
        ("account_info", ()),
        ("equity", ()),
        ("executions", ("BTCUSDT",)),
        ("closed_pnl", ("BTCUSDT",)),
        ("transaction_log", (NOW - DAY, NOW)),
        ("order", ("BTCUSDT", "id")),
        ("submit_limit", ("BTCUSDT", "Buy", 1, 10, 9, 12, "id")),
        ("reduce_market", ("BTCUSDT", "Sell", 1, "id")),
        ("cancel", ("BTCUSDT", "id")),
        ("protect", ("BTCUSDT", 9, 12)),
    ],
)
def test_private_methods_fail_before_any_io_without_credentials(method, args):
    api, session = client()
    with pytest.raises(bybit.CredentialsError):
        asyncio.run(getattr(api, method)(*args))
    assert session.calls == []


@pytest.mark.parametrize("link", ["", "x" * 37, "with space", "has/slash", "雪", None])
def test_invalid_order_link_ids_fail_before_io(link):
    api, session = client(private=True)
    with pytest.raises(ValueError):
        asyncio.run(api.reduce_market("BTCUSDT", "Sell", 1, link))
    assert session.calls == []


@pytest.mark.parametrize("qty", [0, -1, float("nan"), float("inf"), True, "garbage"])
def test_invalid_quantities_fail_before_io(qty):
    api, session = client(private=True)
    with pytest.raises(ValueError):
        asyncio.run(api.reduce_market("BTCUSDT", "Sell", qty, "id"))
    assert session.calls == []


def test_server_clock_cache_midpoint_monotonic_and_refresh(offline):
    clock_sample = Reply(
        ok({"timeNano": str(NOW * 1_000_000)}),
        on_read=lambda: setattr(offline, "mono", 100.2),
    )
    api, session = client(
        [ok({}), ok({}), ok({})],
        private=True,
        clock_events=[clock_sample, ok({"timeNano": str((NOW + 70_000) * 1_000_000)})],
    )

    async def scenario():
        await api.account_info()
        offline.wall += 999_999  # Wall-clock adjustment must not change signatures.
        offline.mono = 101.2
        await api.account_info()
        offline.mono = 170.0
        await api.account_info()

    asyncio.run(scenario())
    stamps = [
        int(call["headers"]["X-BAPI-TIMESTAMP"]) for call in session.business_calls
    ]
    assert abs(stamps[0] - (NOW + 100)) <= 1
    assert stamps[1] - stamps[0] == 1000
    assert stamps[2] == NOW + 70_000
    assert sum(call["path"] == "/v5/market/time" for call in session.calls) == 2


def test_timestamp_get_rejection_refreshes_clock_and_resigns(offline):
    api, session = client(
        [{"retCode": 10002}, ok({"marginMode": "ISOLATED_MARGIN"})],
        private=True,
        clock_events=[
            ok({"timeSecond": str(NOW // 1000)}),
            ok({"timeNano": str((NOW + 1000) * 1_000_000)}),
        ],
    )
    assert asyncio.run(api.account_info())["marginMode"] == "ISOLATED_MARGIN"
    stamps = [call["headers"]["X-BAPI-TIMESTAMP"] for call in session.business_calls]
    assert stamps == [str(NOW), str(NOW + 1000)]
    assert len(offline.sleeps) == 1


def test_slow_or_missing_clock_prevents_submission(offline):
    slow = Reply(
        ok({"timeNano": str(NOW * 1_000_000)}),
        on_read=lambda: setattr(offline, "mono", 103),
    )
    api, session = client(private=True, clock_events=[slow])
    with pytest.raises(bybit.DataUnavailable, match="clock sample"):
        asyncio.run(api.reduce_market("BTCUSDT", "Sell", 1, "id"))
    assert session.business_calls == []


def test_read_cancellation_propagates_without_retry():
    api, session = client([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(api.ticker("BTCUSDT"))
    assert len(session.calls) == 1


@pytest.mark.parametrize("method", ["positions", "open_orders"])
def test_private_list_pagination_retains_raw_and_zero_rows(method):
    zero = {"symbol": "BTCUSDT", "size": "0", "leverage": "3", "positionIdx": 0}
    other = {"symbol": "ETHUSDT", "size": "1", "unfamiliar": "preserved"}
    api, session = client([page([zero], "next"), page([other])], private=True)
    assert asyncio.run(getattr(api, method)()) == [zero, other]
    assert session.business_calls[1]["query"]["cursor"] == "next"
    assert session.business_calls[0]["query"]["settleCoin"] == "USDT"
    if method == "open_orders":
        assert session.business_calls[0]["query"]["openOnly"] == "0"
        assert "orderFilter" not in session.business_calls[0]["query"]


def test_position_info_keeps_flat_and_hedge_settings_for_runtime_to_verify():
    rows = [
        {"symbol": "BTCUSDT", "size": "0", "positionIdx": n, "leverage": "5"}
        for n in (1, 2)
    ]
    api, session = client([page(rows)], private=True)
    assert asyncio.run(api.position_info("BTCUSDT")) == rows
    assert session.business_calls[0]["query"]["symbol"] == "BTCUSDT"
    assert "settleCoin" not in session.business_calls[0]["query"]


def test_risk_limits_public_paginated_preserves_tiers_and_deductions():
    rows = [
        {
            "symbol": "BTCUSDT",
            "id": n,
            "riskLimitValue": str(n * 100000),
            "maintenanceMargin": 0.005 * n,
            "initialMargin": 0.01 * n,
            "maxLeverage": "50",
            "mmDeduction": "500",
        }
        for n in (1, 2)
    ]
    api, session = client([page(rows[:1], "next"), page(rows[1:])])
    assert asyncio.run(api.risk_limits("BTCUSDT")) == rows
    assert all(not call["headers"] for call in session.calls)


@pytest.mark.parametrize(
    "events", [[page([], "same"), page([], "same")], [page([], None)], [ok({})]]
)
def test_invalid_or_repeating_pagination_never_returns_partial_success(events):
    api, _ = client(events, private=True)
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.positions())


def test_pagination_limit_fails_instead_of_returning_truncated_results():
    api, session = client([page([], "next")], private=True)
    api.MAX_PAGES = 1
    with pytest.raises(bybit.DataUnavailable, match="pagination limit"):
        asyncio.run(api.positions())
    assert len(session.business_calls) == 1


def test_order_lookup_falls_back_to_history_and_preserves_status():
    order = {
        "symbol": "BTCUSDT",
        "orderLinkId": "id",
        "orderId": "1",
        "orderStatus": "Filled",
        "cumExecQty": ".1",
    }
    api, session = client([page([]), page([], "next"), page([order])], private=True)
    assert asyncio.run(api.order("BTCUSDT", "id")) == order
    assert [call["path"] for call in session.business_calls] == [
        "/v5/order/realtime",
        "/v5/order/history",
        "/v5/order/history",
    ]


def test_order_lookup_returns_none_only_after_both_empty_endpoints():
    api, session = client([page([]), page([])], private=True)
    assert asyncio.run(api.order("BTCUSDT", "id")) is None
    assert len(session.business_calls) == 2


def test_order_lookup_short_circuits_realtime_and_propagates_errors():
    order = {"symbol": "BTCUSDT", "orderLinkId": "id", "orderStatus": "New"}
    api, session = client([page([order])], private=True)
    assert asyncio.run(api.order("BTCUSDT", "id")) == order
    assert len(session.business_calls) == 1
    api, session = client([{"retCode": 10004}], private=True)
    with pytest.raises(bybit.BybitAPIError):
        asyncio.run(api.order("BTCUSDT", "id"))
    assert len(session.business_calls) == 1


def protective_receipt(**changes):
    return {
        "symbol": "BTCUSDT",
        "orderId": "exchange-generated-stop",
        "orderLinkId": "",
        "parentOrderLinkId": "apx-owned-entry",
        "side": "Sell",
        "positionIdx": 0,
        "orderStatus": "Filled",
        "orderType": "Market",
        "createType": "CreateByStopLoss",
        "stopOrderType": "StopLoss",
        "reduceOnly": True,
        "closeOnTrigger": True,
        "qty": "2",
        "cumExecQty": "2",
        "leavesQty": "0",
        **changes,
    }


def test_order_by_id_recovers_blank_link_protection_from_paginated_history():
    receipt = protective_receipt()
    api, session = client(
        [page([]), page([], "receipt-cursor"), page([receipt])], private=True
    )
    assert asyncio.run(api.order_by_id("BTCUSDT", receipt["orderId"])) == receipt
    calls = session.business_calls
    assert [c["path"] for c in calls] == [
        "/v5/order/realtime",
        "/v5/order/history",
        "/v5/order/history",
    ]
    assert calls[-1]["query"]["cursor"] == "receipt-cursor"
    for call in calls:
        assert (
            call["method"] == "GET" and call["query"]["orderId"] == receipt["orderId"]
        )
        assert "orderLinkId" not in call["query"] and "openOnly" not in call["query"]
        query = call["url"].raw_query_string
        expected = hmac.new(
            b"fixture-secret",
            (str(NOW) + "fixture-key5000" + query).encode(),
            hashlib.sha256,
        ).hexdigest()
        assert call["headers"]["X-BAPI-SIGN"] == expected


def test_order_by_id_short_circuits_realtime_with_exact_raw_receipt():
    receipt = protective_receipt(
        orderStatus="Untriggered", cumExecQty="0", leavesQty="2"
    )
    api, session = client([page([receipt])], private=True)
    assert asyncio.run(api.order_by_id("BTCUSDT", receipt["orderId"])) == receipt
    assert len(session.business_calls) == 1


def test_order_by_id_does_not_fabricate_parent_link_or_protective_ownership():
    receipt = {
        "symbol": "BTCUSDT",
        "orderId": "manual-exit",
        "orderLinkId": "",
        "createType": "CreateByUser",
        "stopOrderType": "",
        "reduceOnly": True,
    }
    api, _ = client([page([]), page([receipt])], private=True)
    result = asyncio.run(api.order_by_id("BTCUSDT", "manual-exit"))
    assert result == receipt and "parentOrderLinkId" not in result


def test_order_by_id_returns_none_only_after_both_empty_reads():
    api, session = client([page([]), page([])], private=True)
    assert asyncio.run(api.order_by_id("BTCUSDT", "missing-exchange-id")) is None
    assert len(session.business_calls) == 2


@pytest.mark.parametrize(
    "changes,category",
    [
        ({"orderId": "foreign-id"}, "linear"),
        ({"symbol": "ETHUSDT"}, "linear"),
        ({}, "spot"),
    ],
)
def test_order_by_id_rejects_mismatched_receipts(changes, category):
    api, _ = client(
        [page([]), page([protective_receipt(**changes)], category=category)],
        private=True,
    )
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.order_by_id("BTCUSDT", "exchange-generated-stop"))


def test_order_by_id_conflicting_receipts_are_not_accepted():
    api, _ = client(
        [
            page([protective_receipt()], "next"),
            page([protective_receipt(parentOrderLinkId="foreign-entry")]),
        ],
        private=True,
    )
    with pytest.raises(bybit.DataUnavailable, match="Conflicting"):
        asyncio.run(api.order_by_id("BTCUSDT", "exchange-generated-stop"))


def test_order_by_id_retries_transient_reads_without_any_mutation():
    receipt = protective_receipt()
    api, session = client(
        [
            Reply(None, 429),
            page([]),
            aiohttp.ServerDisconnectedError(),
            page([receipt]),
        ],
        private=True,
    )
    assert asyncio.run(api.order_by_id("BTCUSDT", receipt["orderId"])) == receipt
    assert len(session.business_calls) == 4
    assert all(c["method"] == "GET" for c in session.calls)


@pytest.mark.parametrize(
    "event,error",
    [({"retCode": 10004}, bybit.BybitAPIError), (ok({}), bybit.DataUnavailable)],
)
def test_order_by_id_errors_never_become_absent_order(event, error):
    api, session = client([event], private=True)
    with pytest.raises(error):
        asyncio.run(api.order_by_id("BTCUSDT", "exchange-generated-stop"))
    assert len(session.business_calls) == 1


@pytest.mark.parametrize("order_id", [None, "", " ", " id", "id\n", "id\x00", 12, True])
def test_order_by_id_invalid_identifier_fails_before_io(order_id):
    api, session = client(private=True)
    with pytest.raises(ValueError):
        asyncio.run(api.order_by_id("BTCUSDT", order_id))
    assert session.calls == []


def test_order_by_id_requires_credentials_and_does_not_use_custom_link_length_limit():
    api, session = client()
    with pytest.raises(bybit.CredentialsError):
        asyncio.run(api.order_by_id("BTCUSDT", "exchange-generated-stop"))
    assert session.calls == []
    receipt = protective_receipt(orderId="opaque-exchange-identifier-" + "1" * 36)
    api, _ = client([page([receipt])], private=True)
    assert asyncio.run(api.order_by_id("BTCUSDT", receipt["orderId"])) == receipt


def wallet(coin_row=None, **account_fields):
    return page(
        [
            {
                "accountType": "UNIFIED",
                "totalEquity": "",
                "totalWalletBalance": "",
                "coin": [
                    (
                        coin_row
                        if coin_row is not None
                        else {
                            "coin": "USDT",
                            "equity": "123.45",
                            "walletBalance": "150",
                            "unrealisedPnl": "-26.55",
                            "spotBorrow": "0",
                            "borrowAmount": "0",
                            "accruedInterest": "0",
                            "usdValue": "123.44",
                        }
                    )
                ],
                **account_fields,
            }
        ]
    )


@pytest.mark.parametrize("total", ["", "999999"])
def test_isolated_equity_uses_actual_usdt_coin_units_not_usd_mark(total):
    api, session = client([wallet(totalEquity=total)], private=True)
    assert asyncio.run(api.equity()) == 123.45
    assert session.business_calls[0]["query"] == {
        "accountType": "UNIFIED",
        "coin": "USDT",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"equity": ""},
        {"equity": None},
        {"equity": "NaN"},
        {"equity": "Infinity"},
        {"equity": "-1"},
        {"equity": "0"},
        {"equity": "999"},
        {"walletBalance": ""},
        {"walletBalance": "NaN"},
        {"unrealisedPnl": None},
        {"unrealisedPnl": "Infinity"},
        {"coin": "USDC"},
        {"coin": "USD"},
        {"spotBorrow": "1"},
        {"borrowAmount": "2"},
        {"accruedInterest": ".1"},
    ],
)
def test_equity_rejects_missing_inconsistent_nonpositive_or_wrong_currency(changes):
    coin = {
        "coin": "USDT",
        "equity": "123.45",
        "walletBalance": "150",
        "unrealisedPnl": "-26.55",
        **changes,
    }
    api, _ = client([wallet(coin, totalEquity="123.45")], private=True)
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.equity())


@pytest.mark.parametrize(
    "coins",
    [
        [],
        None,
        [{"coin": "USDT"}, {"coin": "USDT"}],
        [{"coin": "USDT"}, {"coin": "BTC"}],
    ],
)
def test_equity_never_guesses_missing_or_ambiguous_coin_wallet(coins):
    api, _ = client([wallet(coin=coins)], private=True)
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.equity())


def test_manual_deposit_changes_capital_without_manufacturing_trading_pnl():
    before = {
        "coin": "USDT",
        "walletBalance": "1000.1",
        "unrealisedPnl": "0.2",
        "equity": "1000.3",
    }
    after = {**before, "walletBalance": "1500.1", "equity": "1500.3"}
    api, session = client([wallet(before), wallet(after)], private=True)

    async def scenario():
        assert await api.equity() == 1000.3
        assert await api.equity() == 1500.3

    asyncio.run(scenario())
    assert all(
        call["path"] == "/v5/account/wallet-balance" for call in session.business_calls
    )


def test_executions_page_deduplicate_and_preserve_fee_fields():
    first = {
        "symbol": "BTCUSDT",
        "execId": "a",
        "execTime": str(NOW - 20),
        "execFee": "0.01",
        "execType": "Trade",
        "closedSize": "1",
    }
    second = {
        **first,
        "execId": "b",
        "execTime": str(NOW - 10),
        "execType": "Funding",
        "execFee": "-0.02",
    }
    api, session = client([page([second], "next"), page([first, second])], private=True)
    assert asyncio.run(api.executions("BTCUSDT", NOW - 100)) == [first, second]
    assert all(
        call["query"]["startTime"] == str(NOW - 100)
        and call["query"]["endTime"] == str(NOW)
        for call in session.business_calls
    )


def test_executions_reject_old_lookback_instead_of_silently_dropping_fills():
    api, session = client(private=True)
    with pytest.raises(ValueError, match="seven days"):
        asyncio.run(api.executions("BTCUSDT", NOW - 8 * DAY))
    assert session.business_calls == []


def funding(stamp, rate="-0.00012"):
    return {
        "symbol": "BTCUSDT",
        "fundingRate": rate,
        "fundingRateTimestamp": str(stamp),
    }


def test_required_funding_history_pages_using_actual_timestamps_not_fixed_intervals():
    # More than one page, with a change from hourly to four-hourly settlements.
    stamps = [NOW - n * 3_600_000 for n in range(200)]
    old = [stamps[-1] - 4 * 3_600_000, stamps[-1] - 8 * 3_600_000]
    api, session = client(
        [page([funding(t) for t in stamps]), page([funding(t, "0.0001") for t in old])]
    )
    result = asyncio.run(api.funding_history("BTCUSDT", old[-1], NOW))
    assert len(result) == 202
    assert [int(row["fundingRateTimestamp"]) for row in result] == sorted(stamps + old)
    assert session.calls[1]["query"]["endTime"] == str(stamps[-1] - 1)
    assert all(call["query"]["startTime"] == str(old[-1]) for call in session.calls)
    assert all(call["headers"] == {} for call in session.calls)
    assert result[0]["fundingRate"] == "0.0001"


@pytest.mark.parametrize(
    "rows",
    [[funding(NOW + 1)], [funding(NOW, "NaN")], [funding(NOW), funding(NOW, "0.1")]],
)
def test_funding_history_rejects_bad_or_conflicting_settlements(rows):
    api, _ = client([page(rows)])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.funding_history("BTCUSDT", NOW - DAY, NOW))


def test_closed_pnl_raw_paginated_and_no_double_fee_calculation():
    row = {
        "symbol": "BTCUSDT",
        "orderId": "closed-1",
        "createdTime": str(NOW - 100),
        "updatedTime": str(NOW - 50),
        "closedPnl": "-1.25",
        "openFee": ".03",
        "closeFee": ".02",
    }
    api, session = client([page([], "next"), page([row])], private=True)
    assert asyncio.run(api.closed_pnl("BTCUSDT", NOW - DAY)) == [row]
    assert session.business_calls[0]["path"] == "/v5/position/closed-pnl"
    assert session.business_calls[1]["query"]["cursor"] == "next"


def transaction(identifier, stamp, **extra):
    return {
        "id": identifier,
        "transactionTime": str(stamp),
        "category": "linear",
        "currency": "USDT",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "type": "SETTLEMENT",
        "orderLinkId": "",
        "funding": "-0.5",
        "fee": "0",
        "cashFlow": "0",
        "change": "-0.5",
        **extra,
    }


def test_transaction_log_pages_windows_preserves_funding_sign_and_unowned_rows():
    start, boundary, end = NOW - 8 * DAY, NOW - DAY, NOW
    first = transaction("1", start)
    second = transaction("2", boundary, funding="0.2", change="0.2", side="Sell")
    third = transaction(
        "3",
        end,
        symbol="ETHUSDT",
        orderLinkId="external-order",
        type="TRADE",
        funding="",
        fee="0.03",
        cashFlow="5",
        change="4.97",
    )
    api, session = client(
        [page([first], "next"), page([second]), page([third])], private=True
    )
    assert asyncio.run(api.transaction_log(start, end)) == [first, second, third]
    calls = session.business_calls
    assert calls[0]["query"]["endTime"] == str(boundary)
    assert calls[1]["query"]["cursor"] == "next"
    assert calls[2]["query"]["startTime"] == str(boundary + 1)
    assert "cursor" not in calls[2]["query"]
    assert all(
        call["query"]["currency"] == "USDT" and call["query"]["category"] == "linear"
        for call in calls
    )
    assert all("type" not in call["query"] for call in calls)


def test_accounting_duplicate_records_not_double_counted_and_conflicts_fail():
    row = transaction("same", NOW)
    api, _ = client([page([row], "next"), page([row])], private=True)
    assert asyncio.run(api.transaction_log(NOW - DAY, NOW)) == [row]
    api, _ = client([page([row, {**row, "funding": "-1"}])], private=True)
    with pytest.raises(bybit.DataUnavailable, match="Conflicting"):
        asyncio.run(api.transaction_log(NOW - DAY, NOW))


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "USDC"},
        {"category": "spot"},
        {"transactionTime": str(NOW + 1)},
        {"id": ""},
    ],
)
def test_transaction_log_rejects_mixed_or_unidentifiable_accounting(changes):
    api, _ = client([page([transaction("1", NOW, **changes)])], private=True)
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.transaction_log(NOW - DAY, NOW))


def test_accounting_range_budget_is_explicit():
    api, session = client(private=True)
    with pytest.raises(ValueError, match="730 days"):
        asyncio.run(api.transaction_log(NOW - 731 * DAY, NOW))
    assert session.calls == []


def test_rate_reset_header_works_without_retry_after(offline):
    api, session = client(
        [
            Reply(None, 429, {"X-Bapi-Limit-Reset-Timestamp": str(NOW + 2_000)}),
            ok({"marginMode": "ISOLATED_MARGIN"}),
        ],
        private=True,
    )
    asyncio.run(api.account_info())
    assert offline.sleeps == [2.125]
    assert len(session.business_calls) == 2


def test_clock_retries_do_not_include_backoff_in_midpoint(offline, monkeypatch):
    async def sleep(delay):
        offline.mono += delay

    monkeypatch.setattr(bybit.asyncio, "sleep", sleep)
    api, session = client([ok({})], private=True, clock_events=[Reply(None, 503)])
    asyncio.run(api.account_info())
    assert session.business_calls[0]["headers"]["X-BAPI-TIMESTAMP"] == str(NOW)


def test_concurrent_private_reads_share_one_clock_sample():
    api, session = client([ok({}), ok({}), ok({})], private=True)

    async def scenario():
        await asyncio.gather(*(api.account_info() for _ in range(3)))

    asyncio.run(scenario())
    assert sum(call["path"] == "/v5/market/time" for call in session.calls) == 1


@pytest.mark.parametrize(
    "clock_result",
    [
        {},
        {"timeNano": "NaN"},
        {"timeNano": "0"},
        {"timeNano": "9" * 1000},
        {"timeSecond": "-1"},
    ],
)
def test_malformed_clock_prevents_any_private_submission(clock_result):
    api, session = client(private=True, clock_events=[ok(clock_result)])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.reduce_market("BTCUSDT", "Sell", 1, "id"))
    assert session.business_calls == []


def test_mixed_spot_or_wrong_symbol_responses_are_never_used():
    api, _ = client([page([bar(NOW - DAY)], category="spot", symbol="BTCUSDT")])
    with pytest.raises(bybit.DataUnavailable, match="category"):
        asyncio.run(api.candles("BTCUSDT", now_ms=NOW))
    api, _ = client([page([{"symbol": "ETHUSDT"}], category="linear")])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.ticker("BTCUSDT"))


def test_closed_pnl_default_seven_days_is_one_window():
    api, session = client([page([])], private=True)
    assert asyncio.run(api.closed_pnl("BTCUSDT")) == []
    assert len(session.business_calls) == 1
    assert (
        int(session.business_calls[0]["query"]["endTime"])
        - int(session.business_calls[0]["query"]["startTime"])
        == 7 * DAY
    )


def test_execution_default_bound_and_partial_credentials():
    api, session = client([page([])], private=True)
    assert asyncio.run(api.executions("BTCUSDT")) == []
    assert session.business_calls[0]["query"]["startTime"] == str(NOW - 7 * DAY)
    for key, secret in (("key", ""), ("", "secret"), ("  ", "secret")):
        session = Session()
        api = bybit.BybitClient(session, api_key=key, api_secret=secret)
        with pytest.raises(bybit.CredentialsError):
            asyncio.run(api.account_info())
        assert session.calls == []


def test_protection_is_mandatory_and_brackets_entry():
    api, session = client(private=True)
    for side, stop, target in (
        ("Buy", None, 12),
        ("Buy", 0, 12),
        ("Buy", 11, 12),
        ("Sell", 9, 12),
        ("Sell", 11, 10),
    ):
        with pytest.raises(ValueError):
            asyncio.run(api.submit_limit("BTCUSDT", side, 1, 10, stop, target, "id"))
    assert session.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://api.bybit.com",
        "https://user:secret@api.bybit.com",
        "https://api.bybit.com?secret=x",
        "https://api.bybit.com/v5",
    ],
)
def test_base_url_cannot_embed_credentials_or_redirect_payload(url):
    with pytest.raises(ValueError):
        bybit.BybitClient(Session(), base_url=url)


def test_errors_and_logs_do_not_expose_raw_response_urls_or_headers(caplog):
    api, _ = client(
        [aiohttp.ServerDisconnectedError("https://secret.example?api_key=fixture-key")]
        * 4
    )
    with pytest.raises(bybit.DataUnavailable) as error:
        asyncio.run(api.ticker("BTCUSDT"))
    assert "secret.example" not in str(error.value)
    assert "fixture-key" not in str(error.value)
    assert error.value.__suppress_context__ is True
    assert caplog.records == []


def test_module_import_does_not_construct_session_or_read_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import must be inert")

    # Use a separate namespace to avoid replacing the classes used by other tests.
    source = importlib.util.find_spec("apex_bot.bybit").loader.get_source(
        "apex_bot.bybit"
    )
    monkeypatch.setattr(aiohttp, "ClientSession", forbidden)
    namespace = {"__name__": "apex_bot._bybit_import_test", "__package__": "apex_bot"}
    exec(compile(source, "bybit.py", "exec"), namespace)
    assert "BybitClient" in namespace


def test_universe_metadata_and_all_tickers_are_unsigned_and_complete():
    meta = instrument(
        baseCoin="BTC", launchTime="1234567890000", isPreListing=False, symbolType=""
    )
    api, session = client(
        [
            page([meta]),
            page(
                [
                    {"symbol": "BTCUSDT", "turnover24h": "50000000"},
                    {"symbol": "ETHUSDT"},
                ],
                category="linear",
            ),
        ]
    )

    async def run():
        await api.instruments()
        result = await api.tickers()
        assert api.instrument_metadata["BTCUSDT"]["launchTime"] == "1234567890000"
        assert api.instrument_metadata["BTCUSDT"]["isPreListing"] is False
        assert result["as_of"] == bybit.time.time()
        assert len(result["list"]) == 2

    asyncio.run(run())
    assert all("X-BAPI-API-KEY" not in c["headers"] for c in session.business_calls)


@pytest.mark.parametrize(
    "rows", [[], [{"symbol": "BTCUSDT"}, {"symbol": "BTCUSDT"}], [{}]]
)
def test_invalid_universe_tickers_are_not_a_successful_empty_snapshot(rows):
    api, _ = client([page(rows, category="linear")])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.tickers())


def book_payload(**changes):
    result = {
        "s": "BTCUSDT",
        "ts": NOW,
        "b": [["99.99", "1000"], ["99.9", "500"], ["98", "999999"]],
        "a": [["100.01", "1000"], ["100.1", "500"], ["102", "999999"]],
    }
    result.update(changes)
    return ok(result)


def test_liquidity_depth_counts_only_narrow_band_both_sides():
    api, session = client([book_payload()])
    result = asyncio.run(api.liquidity_book("BTCUSDT"))
    assert result["bid_depth_usdt"] == pytest.approx(149940)
    assert result["ask_depth_usdt"] == pytest.approx(150060)
    assert result["spread_bps"] == pytest.approx(2)
    assert result["band_bps"] == 25 and result["as_of"] == bybit.time.time()
    assert result["exchange_as_of"] == NOW / 1000
    assert session.business_calls[0]["query"] == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "limit": "200",
    }
    assert "X-BAPI-API-KEY" not in session.business_calls[0]["headers"]


@pytest.mark.parametrize(
    "changes",
    [
        {"s": "ETHUSDT"},
        {"ts": NOW - 120001},
        {"ts": NOW + 1},
        {"b": []},
        {"a": [["99", "2"]]},
        {"b": [["99", "2"], ["99.9", "3"]]},
        {"b": [["99", "2"], ["99", "3"]]},
        {"b": [["99", "0"]]},
        {"a": [["100.01", "nan"]]},
        {"b": [["99.99", "1e308"]]},
    ],
)
def test_invalid_liquidity_book_fails_closed(changes):
    api, _ = client([book_payload(**changes)])
    with pytest.raises(bybit.DataUnavailable):
        asyncio.run(api.liquidity_book("BTCUSDT"))
