"""Offline official-calendar fixtures and closed-market-data context contracts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from apex_bot import context as ctx
from apex_bot.models import Candle


def stamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


NOW = stamp("2026-10-05T12:00:00Z")
MEETINGS = [
    ("January", "27-28"),
    ("March", "17-18*"),
    ("April", "28-29"),
    ("June", "16-17*"),
    ("July", "28-29"),
    ("September", "15-16*"),
    ("October", "27-28"),
    ("December", "8-9*"),
]


def fomc(year=2026, meetings=None):
    entries = MEETINGS if meetings is None else meetings
    return (
        f"<html><h4>{year} FOMC Meetings</h4>"
        + "".join(
            f'<div class="fomc-meeting"><div><strong>{month}</strong></div><div>{days}</div></div>'
            for month, days in entries
        )
        + "</html>"
    ).encode()


def bls(cpi="20261014T083000", jobs="20261106T083000", all_day=False):
    property_name = "DTSTART;VALUE=DATE" if all_day else "DTSTART;TZID=America/New_York"
    return (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
        + "".join(
            f"BEGIN:VEVENT\r\nUID:{kind}\r\n{property_name}:{day}\r\nSUMMARY:{kind}\r\nEND:VEVENT\r\n"
            for kind, day in [
                ("Consumer Price Index", cpi),
                ("Employment Situation", jobs),
            ]
        )
        + "END:VCALENDAR\r\n"
    ).encode()


def bars(now=NOW, slope=1):
    day = int(now // ctx.DAY) * ctx.DAY
    return [
        Candle(
            (day - (240 - i) * ctx.DAY) * 1000,
            1000 + slope * i,
            1001 + slope * i,
            999 + slope * i,
            1000 + slope * i,
        )
        for i in range(240)
    ]


@pytest.fixture
def fetch(monkeypatch):
    payloads = {ctx.BLS_URL: bls(), ctx.FOMC_URL: fomc()}
    calls = []

    async def get(session, url, **kwargs):
        calls.append((url, kwargs))
        value = payloads[url]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(ctx, "_get_bytes", get)
    return payloads, calls


def test_context_api_import_and_default_build(fetch):
    service = ctx.ContextService(object())
    result = asyncio.run(service.build(bars(), NOW))
    assert result["data_complete"] and result["fresh"]
    assert result["risk_state"] == "risk_on"
    assert result["long_multiplier"] == 1 and result["short_multiplier"] == 0.5
    assert result["as_of"] == NOW and result["expires_at"] == NOW + 3600
    assert not result["event_blackout"] and not result["live_blocked"]
    assert {source["url"] for source in result["sources"]} == {
        ctx.BLS_URL,
        ctx.FOMC_URL,
    }
    assert all(
        len(source["sha256"]) == 64 and source["verified_at"] == NOW
        for source in result["sources"]
    )
    assert result["event_coverage"]["scope"] == "US_MACRO"
    assert result["event_coverage"]["crypto_events_verified"] is False
    assert all(call[1]["headers"]["Cache-Control"] == "no-cache" for call in fetch[1])


@pytest.mark.parametrize(
    "slope,state", [(1, "risk_on"), (-1, "risk_off"), (0, "neutral")]
)
def test_transparent_btc_policy(slope, state):
    result = ctx.btc_regime(bars(slope=slope), NOW)
    assert result["risk_state"] == state
    assert result["btc_evidence"]["bars"] == 240
    assert result["btc_evidence"]["daily_closed_at"] == int(NOW // ctx.DAY) * ctx.DAY


@pytest.mark.parametrize(
    "change", ["short", "gap", "duplicate", "unclosed", "stale", "nan"]
)
def test_btc_invalid_or_insufficient_data_blocks(change, fetch):
    data = bars()
    if change == "short":
        data = data[:219]
    elif change == "gap":
        data.pop(30)
    elif change == "duplicate":
        data[31] = data[30]
    elif change == "unclosed":
        last = data[-1]
        data.append(Candle(last.open_time + ctx.DAY * 1000, 1000, 1001, 999, 1000))
    elif change == "stale":
        data = data[:-1]
    else:
        last = data[-1]
        data[-1] = Candle(last.open_time, 1000, 1001, 999, float("nan"))
    result = asyncio.run(ctx.ContextService(object()).build(data, NOW))
    assert not result["data_complete"] and result["live_blocked"]
    assert result["expires_at"] == NOW


@pytest.mark.parametrize("url", [ctx.BLS_URL, ctx.FOMC_URL])
def test_missing_calendar_blocks_without_reusing_prior_success(url, fetch):
    service = ctx.ContextService(object())
    assert asyncio.run(service.build(bars(), NOW))["data_complete"]
    fetch[0][url] = RuntimeError("Authorization=secret")
    failed = asyncio.run(service.build(bars(), NOW + 1))
    assert (
        not failed["data_complete"]
        and failed["live_blocked"]
        and failed["event_blackout"]
    )
    assert failed["expires_at"] == NOW + 1
    assert "secret" not in str(failed)
    assert any(
        s["status"] == "unavailable" and "verified_at" not in s
        for s in failed["sources"]
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"<html>Access denied</html>",
        b"BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR",
        bls().replace(b"DTSTART", b"MISSING"),
        bls().replace(b"20261014", b"20261340"),
        bls().replace(b"END:VCALENDAR", b""),
        bls().replace(b"Employment Situation", b"Other Release"),
        bls("20250114T083000", "20250206T083000"),
    ],
)
def test_bls_empty_malformed_missing_or_stale_dates_fail_closed(raw):
    with pytest.raises(ctx._Failure):
        ctx.parse_bls(raw, NOW)


@pytest.mark.parametrize(
    "date,now,expected",
    [
        ("20260109T083000", "2026-01-01T00:00:00Z", "2026-01-09T13:30:00Z"),
        ("20260313T083000", "2026-03-01T00:00:00Z", "2026-03-13T12:30:00Z"),
        ("20261106T083000", "2026-11-01T00:00:00Z", "2026-11-06T13:30:00Z"),
    ],
)
def test_bls_eastern_release_times_follow_dst(date, now, expected):
    events = ctx.parse_bls(bls(date, date), stamp(now))
    assert events[0]["start"] == stamp(expected) - 3600
    assert events[0]["end"] == stamp(expected) + 3600


@pytest.mark.parametrize("day,hours", [("20260308", 23), ("20261101", 25)])
def test_all_day_ics_events_have_dst_aware_local_midnights(day, hours):
    start, end = ctx._ics_start("DTSTART;VALUE=DATE", day)
    assert end - start == hours * 3600


def test_bls_all_day_and_folded_summary():
    raw = bls("20261014", "20261106", all_day=True).replace(
        b"SUMMARY:Consumer Price Index", b"SUMMARY:Consumer Price\r\n  Index"
    )
    events = ctx.parse_bls(raw, NOW)
    assert events[0]["start"] == stamp("2026-10-14T04:00:00Z")
    assert events[0]["end"] == stamp("2026-10-15T04:00:00Z")


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"<html>no meetings</html>",
        fomc(2025),
        fomc(meetings=MEETINGS[:5]),
        fomc().replace(b"27-28</div>", b"TBD</div>", 1),
        fomc().replace(b"17-18*", b"99-99"),
        fomc(meetings=MEETINGS + [MEETINGS[0]]),
    ],
)
def test_fomc_empty_malformed_missing_year_or_dates_fail_closed(raw):
    with pytest.raises(ctx._Failure):
        ctx.parse_fomc(raw, NOW)


def test_fomc_decision_day_and_cross_month_range():
    meetings = [*MEETINGS]
    meetings[2] = ("Apr/May", "30-1")
    events = ctx.parse_fomc(fomc(meetings=meetings), NOW)
    assert len(events) == 8
    assert events[2]["start"] == stamp("2026-05-01T04:00:00Z")
    assert events[6]["start"] == stamp("2026-10-28T04:00:00Z")
    assert events[6]["end"] == stamp("2026-10-29T04:00:00Z")


def test_year_boundary_requires_next_year_schedule():
    with pytest.raises(ctx._Failure, match="fomc_missing_year"):
        ctx.parse_fomc(fomc(), stamp("2027-01-01T04:30:00Z"))


@pytest.mark.parametrize(
    "now,blackout,kind",
    [
        ("2026-10-14T11:29:00Z", False, None),
        ("2026-10-14T11:30:00Z", True, "BLS_CPI"),
        ("2026-10-14T13:29:00Z", True, "BLS_CPI"),
        ("2026-10-28T04:00:00Z", True, "FOMC_DECISIONS"),
        ("2026-10-29T03:59:00Z", True, "FOMC_DECISIONS"),
    ],
)
def test_service_blackout_windows(now, blackout, kind, fetch):
    now = stamp(now)
    # Ensure the current feed contains the following monthly release as well.
    november = bls("20261113T083000", "20261106T083000")
    fetch[0][ctx.BLS_URL] = bls().replace(b"END:VCALENDAR", b"") + november.split(
        b"VERSION:2.0\r\n"
    )[1].replace(b"UID:Employment Situation", b"UID:Employment Situation-next")
    # The second fixture's jobs event duplicates the first: keep just one.
    first = fetch[0][ctx.BLS_URL]
    start = first.index(b"BEGIN:VEVENT\r\nUID:Employment Situation-next")
    end = first.index(b"END:VEVENT", start) + len(b"END:VEVENT\r\n")
    fetch[0][ctx.BLS_URL] = first[:start] + first[end:]
    result = asyncio.run(ctx.ContextService(object()).build(bars(now), now))
    assert result["data_complete"]
    assert result["event_blackout"] is blackout
    if kind:
        assert result["active_events"][0]["kind"] == kind
    else:
        assert result["expires_at"] == stamp("2026-10-14T11:30:00Z")
