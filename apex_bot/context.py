"""Autonomous, deliberately scoped US macro calendar and BTC market context.

This is a NEW deterministic policy, not Apex's DXY/BTC.D regime rule. It covers
BLS CPI/jobs releases and scheduled FOMC decisions only. Crypto-specific events,
unscheduled decisions and other releases are not proven clear by this service.
No manual feed is required. Runtime persists this result, including source hashes.
"""

from __future__ import annotations

import asyncio
import calendar
import math
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

from .models import Candle
from .references import _Failure, _canonical, _digest, _get_bytes, _timestamp


BLS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
# The proposed FOMCmeetingcalendars.htm URL is not the current calendar.
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
DAY = 86400
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
CALENDAR_TTL = 3600
POLICY_VERSION = "btc-ema200-trend20-us-macro-v1"
EVENT_COVERAGE = {
    "scope": "US_MACRO",
    "included": ["BLS_CPI", "BLS_EMPLOYMENT_SITUATION", "FOMC_DECISIONS"],
    "excluded": [
        "crypto_specific_events",
        "unscheduled_policy_actions",
        "other_economic_releases",
    ],
    "crypto_events_verified": False,
}


def _day_window(day):
    start = datetime.combine(day, datetime.min.time(), NY)
    # Construct each local midnight separately: DST days are 23 or 25 hours.
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), NY)
    return start.timestamp(), end.timestamp()


def _ics_start(property_name, value):
    params = {}
    for item in property_name.split(";")[1:]:
        key, sep, val = item.partition("=")
        if not sep or key.upper() in params:
            raise _Failure("bls_invalid_date")
        params[key.upper()] = val.strip('"')
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        return _day_window(datetime.strptime(value, "%Y%m%d").date())
    tzid = params.get("TZID", "America/New_York")
    if tzid not in (
        "America/New_York",
        "US/Eastern",
        "Eastern Standard Time",
        "UTC",
        "Etc/UTC",
    ):
        raise _Failure("bls_unknown_timezone")
    if value.endswith("Z"):
        dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    else:
        zone = UTC if tzid in ("UTC", "Etc/UTC") else NY
        dt = datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=zone)
        # Ambiguous/nonexistent local times cannot safely define a release window.
        if dt.fold == 0 and dt.utcoffset() != dt.replace(fold=1).utcoffset():
            raise _Failure("bls_ambiguous_time")
        if dt.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != dt.replace(
            tzinfo=None
        ):
            raise _Failure("bls_invalid_date")
    return dt.timestamp() - 3600, dt.timestamp() + 3600


def parse_bls(raw, now):
    """Strict finite ICS events; no recurrence expansion or empty-calendar inference."""
    try:
        text = raw.decode("utf-8-sig")
        text = re.sub(r"\r?\n[ \t]", "", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if (
            not lines
            or lines[0] != "BEGIN:VCALENDAR"
            or lines[-1] != "END:VCALENDAR"
            or lines.count("BEGIN:VCALENDAR") != 1
            or "VERSION:2.0" not in lines
        ):
            raise _Failure("bls_invalid_calendar")
        active, events, count = None, [], 0
        for line in lines[1:-1]:
            if line == "BEGIN:VEVENT":
                if active is not None:
                    raise _Failure("bls_invalid_calendar")
                active = {}
            elif line == "END:VEVENT":
                if active is None or "DTSTART" not in active or "SUMMARY" not in active:
                    raise _Failure("bls_missing_event_date")
                count += 1
                start, end = _ics_start(*active["DTSTART"])
                summary = active["SUMMARY"][1].replace("\\n", " ")
                if not summary.strip():
                    raise _Failure("bls_missing_summary")
                kind = (
                    "BLS_CPI"
                    if "consumer price index" in summary.lower()
                    else (
                        "BLS_EMPLOYMENT_SITUATION"
                        if "employment situation" in summary.lower()
                        else None
                    )
                )
                if kind and active.get("STATUS", ("", ""))[1] != "CANCELLED":
                    events.append(
                        {"kind": kind, "start": start, "end": end, "source": BLS_URL}
                    )
                active = None
            elif active is not None:
                name, sep, value = line.partition(":")
                key = name.split(";")[0].upper()
                if (
                    not sep
                    or key in active
                    or key in ("RRULE", "RDATE", "EXDATE", "RECURRENCE-ID")
                ):
                    raise _Failure("bls_unsupported_event")
                active[key] = (name, value)
        if active is not None or count == 0 or count > 5000:
            raise _Failure("bls_incomplete_calendar")
        # Both monthly releases must have an upcoming/current event, reasonably
        # near now. An archived feed or a partial feed cannot imply 'no events'.
        for kind in EVENT_COVERAGE["included"][:2]:
            upcoming = [
                event
                for event in events
                if event["kind"] == kind and event["end"] > now
            ]
            if not upcoming or min(e["start"] for e in upcoming) > now + 45 * DAY:
                raise _Failure("bls_no_upcoming_release")
        unique = {(e["kind"], e["start"], e["end"]) for e in events}
        if len(unique) != len(events):
            raise _Failure("bls_duplicate_release")
        return events
    except _Failure:
        raise
    except (ValueError, UnicodeError, OverflowError):
        raise _Failure("bls_invalid_calendar") from None


class _CalendarText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.ignored = max(0, self.ignored - 1)

    def handle_data(self, data):
        if not self.ignored and data.strip():
            self.parts.append(" ".join(data.split()))


def parse_fomc(raw, now):
    """Parse the official year / month / day-range calendar, not news date links."""
    try:
        parser = _CalendarText()
        parser.feed(raw.decode("utf-8"))
        text = "\n".join(parser.parts)
        headings = list(re.finditer(r"\b(20\d{2})\s+FOMC\s+Meetings\b", text))
        year = datetime.fromtimestamp(now, NY).year
        needed = {year, datetime.fromtimestamp(now + CALENDAR_TTL, NY).year}
        months = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
        months.update(
            {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
        )
        events = []
        for wanted in needed:
            matches = [
                (i, match)
                for i, match in enumerate(headings)
                if int(match[1]) == wanted
            ]
            if len(matches) != 1:
                raise _Failure("fomc_missing_year")
            index, match = matches[0]
            section = text[
                match.end() : (
                    headings[index + 1].start()
                    if index + 1 < len(headings)
                    else len(text)
                )
            ]
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            dates = []
            for i, line in enumerate(lines):
                names = line.lower().split("/")
                if not all(name in months for name in names):
                    continue
                if len(names) > 2 or i + 1 >= len(lines):
                    raise _Failure("fomc_missing_date")
                days = re.fullmatch(
                    r"(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\s*\*?", lines[i + 1]
                )
                if not days:
                    raise _Failure("fomc_missing_date")
                first, last = int(days[1]), int(days[2] or days[1])
                start = datetime(wanted, months[names[0]], first).date()
                end = datetime(wanted, months[names[-1]], last).date()
                if not 0 <= (end - start).days <= 2:
                    raise _Failure("fomc_invalid_date")
                dates.append(end)
            if (
                not 6 <= len(dates) <= 12
                or len(set(dates)) != len(dates)
                or {((d.month - 1) // 3) for d in dates} != {0, 1, 2, 3}
            ):
                raise _Failure("fomc_incomplete_year")
            for day in dates:
                start, end = _day_window(day)
                events.append(
                    {
                        "kind": "FOMC_DECISIONS",
                        "start": start,
                        "end": end,
                        "source": FOMC_URL,
                        "time_precision": "full_New_York_decision_date",
                    }
                )
        return events
    except _Failure:
        raise
    except (ValueError, UnicodeError, OverflowError):
        raise _Failure("fomc_invalid_calendar") from None


def btc_regime(bars, now):
    if not isinstance(bars, list) or len(bars) < 220:
        raise _Failure("btc_history_incomplete")
    previous = None
    for bar in bars:
        if (
            not isinstance(bar, Candle)
            or isinstance(bar.open_time, bool)
            or not isinstance(bar.open_time, int)
            or bar.open_time % (DAY * 1000)
            or bar.open_time / 1000 + DAY > now
            or previous is not None
            and bar.open_time - previous != DAY * 1000
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or x <= 0
                for x in (bar.open, bar.high, bar.low, bar.close)
            )
            or not bar.low
            <= min(bar.open, bar.close)
            <= max(bar.open, bar.close)
            <= bar.high
        ):
            raise _Failure("btc_invalid_closed_bars")
        previous = bar.open_time
    closed_at = bars[-1].open_time / 1000 + DAY
    if closed_at != math.floor(now / DAY) * DAY:
        raise _Failure("btc_daily_stale")
    closes = [float(bar.close) for bar in bars[-500:]]
    ema = sum(closes[:200]) / 200
    for close in closes[200:]:
        ema += (2 / 201) * (close - ema)
    trend = closes[-1] / closes[-21] - 1
    state = (
        "risk_on"
        if closes[-1] > ema and trend > 0
        else ("risk_off" if closes[-1] < ema and trend < 0 else "neutral")
    )
    return {
        "risk_state": state,
        "long_multiplier": 1.0 if state == "risk_on" else 0.5,
        "short_multiplier": 1.0 if state == "risk_off" else 0.5,
        "btc_evidence": {
            "daily_closed_at": closed_at,
            "bars": len(closes),
            "ema200": ema,
            "trend20": trend,
            "last_close": closes[-1],
            "seed": "SMA_first_200_of_latest_up_to_500_closed_daily_bars",
            "sha256": _digest(
                _canonical([[bar.open_time, bar.close] for bar in bars[-500:]])
            ),
        },
    }


class ContextService:
    def __init__(self, session):
        self.session = session

    async def build(self, btc_daily: list[Candle], now) -> dict:
        now = _timestamp(now)
        result = {
            "as_of": now,
            "asof": now,
            "expires_at": now,
            "data_complete": False,
            "fresh": False,
            "live_blocked": True,
            "risk_state": "unknown",
            "long_multiplier": 0.0,
            "short_multiplier": 0.0,
            "event_blackout": True,
            "blackout_scopes": ["ALL"],
            "sources": [],
            "event_coverage": dict(EVENT_COVERAGE),
            "policy_version": POLICY_VERSION,
        }
        failures, events = [], []
        try:
            result.update(btc_regime(btc_daily, now))
        except _Failure as exc:
            failures.append(str(exc))
        for url, parse in ((BLS_URL, parse_bls), (FOMC_URL, parse_fomc)):
            try:
                raw = await _get_bytes(
                    self.session,
                    url,
                    limit=1024 * 1024,
                    headers={
                        "Accept": "text/calendar,text/html",
                        "Cache-Control": "no-cache",
                    },
                )
                parsed = parse(raw, now)
                events.extend(parsed)
                result["sources"].append(
                    {
                        "url": url,
                        "verified_at": now,
                        "sha256": _digest(raw),
                        "parsed_events": len(parsed),
                        "status": "ok",
                    }
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = (
                    str(exc) if isinstance(exc, _Failure) else "calendar_unavailable"
                )
                failures.append(reason)
                result["sources"].append(
                    {
                        "url": url,
                        "attempted_at": now,
                        "status": "unavailable",
                        "reason": reason,
                    }
                )
        if failures:
            result["reason"] = "Live entries blocked: " + ", ".join(failures)
            return result
        active = [event for event in events if event["start"] <= now < event["end"]]
        # Do not let a cached non-blackout context survive an event boundary or
        # the next daily bar. Each build fetches both calendars anew.
        boundaries = [
            edge
            for event in events
            for edge in (event["start"], event["end"])
            if edge > now
        ]
        result.update(
            data_complete=True,
            fresh=True,
            live_blocked=bool(active),
            event_blackout=bool(active),
            blackout_scopes=["ALL"] if active else [],
            expires_at=min(
                now + CALENDAR_TTL, (math.floor(now / DAY) + 1) * DAY, *boundaries
            ),
            active_events=active,
            reason=(
                "US macro blackout: " + ", ".join(event["kind"] for event in active)
                if active
                else "US macro calendars complete; no scheduled CPI/jobs/FOMC window now."
            )
            + " BTC EMA200/trend20 policy: "
            + result["risk_state"]
            + ". Crypto-specific events are not verified by this service.",
        )
        return result
