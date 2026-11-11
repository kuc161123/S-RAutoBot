"""Apex Kapital Shadow Book (Google Sheet) as the desk's signal source.

The Claude Shadow Desk writes every trade to the Sheet's "Trades" tab: a row per
idea with the Hyperliquid market, side, limit entry, stop, TP1, TP2, TP1 share,
"Valid until", status (PENDING → OPEN/TP1 → CLOSED, or EXPIRED) and layer
(BOOK = the capped book whose equity is reported, SIGNAL = the wider sample).

Two read-only ways to reach it, chosen by configuration:
* Sheets API with a Google service account (the Sheet shared with its e-mail
  as Viewer) — APEX_SHADOW_BOOK_SHEET_ID + APEX_GOOGLE_SERVICE_ACCOUNT_JSON;
* a "Publish to the web" CSV link of the Trades tab — APEX_SHADOW_BOOK_CSV_URL.

Rows are parsed by header name, so column order may change; header names may not.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import math
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

LOG = logging.getLogger(__name__)

HEADERS = {
    "id": "ID",
    "opened": "Opened UTC",
    "market": "Market (HL)",
    "side": "Side",
    "setup": "Setup (EW count)",
    "entry_type": "Entry type",
    "entry": "Entry",
    "stop": "Stop",
    "tp1": "TP1",
    "tp2": "TP2",
    "tp1_share": "TP1 share %",
    "valid_until": "Valid until UTC",
    "status": "Status",
    "tp1_hit": "TP1 hit UTC",
    "exit_reason": "Exit reason",
    "layer": "Layer",
}
REQUIRED = ("id", "market", "side", "entry", "stop", "tp1", "tp2", "status")
PREFIX = "shadowbook:"
_SHEET_ID = re.compile(r"[A-Za-z0-9_-]{20,128}\Z")
# Exit reasons the exchange orders already reproduce; anything else is a desk decision.
_PRICE_EXIT = re.compile(r"^\s*(FILL\s+(AND|&)\s+STOP|STOP|TP1|TP2|BE\b|BREAK-?EVEN|EXPIRED|TARGET)", re.I)


class SourceError(Exception):
    pass


def valid_sheet_id(value) -> bool:
    return isinstance(value, str) and bool(_SHEET_ID.fullmatch(value))


def valid_csv_url(value) -> bool:
    try:
        url = urlparse(value)
    except (TypeError, ValueError):
        return False
    query = url.query or ""
    return (
        url.scheme == "https"
        and url.hostname == "docs.google.com"
        and url.path.startswith("/spreadsheets/")
        and ("output=csv" in query or "format=csv" in query)
    )


def _number(value):
    if value is None:
        return None
    text = str(value).replace(",", "").replace("$", "").strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _time(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def setup_title(text: str) -> str:
    """First clause of the desk's setup text, e.g. 'C — sell the retest of a broken floor'."""
    text = " ".join(str(text or "").split())
    if not text:
        return "Elliott setup"
    head = re.split(r"(?<=[a-z0-9)])\.\s", text, maxsplit=1)[0]
    return head[:90] + ("…" if len(head) > 90 else "")


def is_price_exit(reason: str) -> bool:
    return bool(_PRICE_EXIT.match(str(reason or "")))


def parse_rows(rows: list[list[str]]) -> dict:
    """Header row + data rows -> {signal_id: signal dict in the desk's shape}."""
    if not rows:
        return {}
    header = [str(h).strip() for h in rows[0]]
    index = {}
    for key, name in HEADERS.items():
        if name in header:
            index[key] = header.index(name)
    missing = [HEADERS[k] for k in REQUIRED if k not in index]
    if missing:
        raise SourceError("Shadow Book header missing: " + ", ".join(missing))

    def cell(row, key):
        i = index.get(key)
        return row[i] if i is not None and i < len(row) else ""

    out = {}
    for row in rows[1:]:
        trade_id = str(cell(row, "id")).strip()
        market = str(cell(row, "market")).strip()
        side = str(cell(row, "side")).strip().lower()
        if not trade_id or not market or side not in ("long", "short"):
            continue
        entry, stop = _number(cell(row, "entry")), _number(cell(row, "stop"))
        tp1, tp2 = _number(cell(row, "tp1")), _number(cell(row, "tp2"))
        if None in (entry, stop, tp1, tp2):
            continue
        share = _number(cell(row, "tp1_share"))
        status = str(cell(row, "status")).strip().upper()
        reason = str(cell(row, "exit_reason")).strip()
        opened = _time(cell(row, "opened"))
        expires = _time(cell(row, "valid_until"))
        signal_id = PREFIX + trade_id
        out[signal_id] = {
            "id": signal_id,
            "source_id": trade_id,
            "arm": "shadow_book",
            "source": "shadow_book",
            "coin": market,
            "symbol": market,
            "side": "Buy" if side == "long" else "Sell",
            "setup": None,
            "setup_text": setup_title(cell(row, "setup")),
            "entry_type": str(cell(row, "entry_type")).strip().upper() or "LIMIT",
            "entry_style": "confirmed",
            "status": status,
            "limit": entry,
            "stop": stop,
            "target1": tp1,
            "target2": tp2,
            "tp1_share": share if share and 0 < share < 100 else 50.0,
            "created_at": opened or 0.0,
            "expires_at": expires or ((opened or 0.0) + 48 * 3600),
            "tp1_done": status == "TP1" or bool(str(cell(row, "tp1_hit")).strip()),
            "exit_reason": reason,
            "price_exit": is_price_exit(reason),
            "layer": str(cell(row, "layer")).strip().upper() or "BOOK",
        }
    return out


class ShadowBookSource:
    """Polls the Shadow Book. refresh() never raises; check .ok and .error."""

    RANGE = "Trades!A1:AI3000"

    def __init__(self, session, *, sheet_id: str = "", credentials_json: str = "", csv_url: str = "", refresh_seconds: int = 60, clock=time.time):
        if sheet_id and not valid_sheet_id(sheet_id):
            raise ValueError("Invalid Shadow Book sheet ID")
        if csv_url and not valid_csv_url(csv_url):
            raise ValueError("Shadow Book CSV link must be a docs.google.com published CSV link")
        if sheet_id and not credentials_json:
            raise ValueError("Shadow Book sheet ID needs a Google service account")
        self.session = session
        self.sheet_id = sheet_id
        self.csv_url = csv_url
        self.credentials_json = credentials_json
        self.refresh_seconds = refresh_seconds
        self.clock = clock
        self.signals: dict = {}
        self.ok = False
        self.error: str | None = None
        self.fetched_at = 0.0
        self.ok_at = 0.0  # last successful read
        self.newest_at = 0.0  # newest "Opened UTC" in that read
        self._reader = None

    @property
    def configured(self) -> bool:
        return bool(self.sheet_id or self.csv_url)

    @property
    def mode(self) -> str:
        return "sheets_api" if self.sheet_id else "csv" if self.csv_url else "none"

    async def _rows_api(self):
        from .references import ReferenceReader

        if self._reader is None:
            self._reader = ReferenceReader(self.session, None, self.credentials_json, ())
        try:
            token = await asyncio.to_thread(self._reader._token_sync)
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"Google sign-in failed ({getattr(exc, 'args', ['auth'])[0]})") from None
        import aiohttp

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{self.sheet_id}/values/{self.RANGE}"
        async with self.session.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=20, connect=5),
        ) as response:
            if response.status in (401, 403):
                raise SourceError("Google denied access: share the Sheet with the service account e-mail as Viewer")
            if response.status == 404:
                raise SourceError("Shadow Book sheet not found (check APEX_SHADOW_BOOK_SHEET_ID)")
            if response.status != 200:
                raise SourceError(f"Google Sheets error HTTP {response.status}")
            body = await response.json()
        return body.get("values", [])

    async def _rows_csv(self):
        import aiohttp

        # Ask Google for the newest published copy, never a cached one.
        url = self.csv_url + ("&" if "?" in self.csv_url else "?") + f"_ts={int(self.clock())}"
        async with self.session.get(
            url,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            allow_redirects=True,
            max_redirects=3,
            timeout=aiohttp.ClientTimeout(total=20, connect=5),
        ) as response:
            host = response.url.host or ""
            if not (host == "docs.google.com" or host.endswith(".googleusercontent.com")):
                raise SourceError("Shadow Book CSV redirected outside Google")
            if response.status != 200:
                raise SourceError(f"Shadow Book CSV error HTTP {response.status}")
            data = await response.content.read(8_000_000)
        text = data.decode("utf-8-sig", errors="replace")
        if text.lstrip().startswith("<"):
            raise SourceError("Shadow Book link returned a web page, not CSV (publish the Trades tab as CSV)")
        return list(csv.reader(io.StringIO(text)))

    async def refresh(self, force: bool = False) -> bool:
        if not self.configured:
            self.ok, self.error = False, "not configured"
            return False
        if not force and self.clock() - self.fetched_at < self.refresh_seconds:
            return self.ok
        self.fetched_at = self.clock()
        try:
            rows = await (self._rows_api() if self.sheet_id else self._rows_csv())
            self.signals = parse_rows(rows)
            self.ok, self.error = True, None
            self.ok_at = self.clock()
            self.newest_at = max((sig.get("created_at") or 0 for sig in self.signals.values()), default=0)
        except SourceError as exc:
            self.ok, self.error = False, str(exc)
        except Exception as exc:  # noqa: BLE001 - keep trading loop alive
            self.ok, self.error = False, f"Shadow Book read failed ({type(exc).__name__})"
        if not self.ok:
            LOG.warning("Shadow Book unavailable: %s", self.error)
        return self.ok
