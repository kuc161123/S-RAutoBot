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

The same Sheet's "BotRules" tab says which setups the bot may trade: one row per
Side + Setup letter with Active TRUE/FALSE and a "Rules version". It is read on
every poll with Trades; a new Trades row is placed only when an active rule has the
same Side and Setup letter (the first character of "Setup (EW count)").
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
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

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
RULES_TAB = "BotRules"
RULES_GID = "423805298"  # the BotRules tab's id in the Shadow Book (APEX_SHADOW_BOOK_RULES_GID overrides)
RULES_HEADERS = {
    "id": "Rule ID",
    "active": "Active",
    "side": "Side",
    "setup": "Setup",
    "name": "Setup name",
    "version": "Rules version",
}
RULES_REQUIRED = ("active", "side", "setup")
_SHEET_ID = re.compile(r"[A-Za-z0-9_-]{20,128}\Z")
# Exit reasons the exchange orders already reproduce; anything else is a desk decision.
_PRICE_EXIT = re.compile(r"^\s*(FILL\s+(AND|&)\s+STOP|STOP|TP1|TP2|BE\b|BREAK-?EVEN|EXPIRED|TARGET)", re.I)


MAX_CSV_BYTES = 8_000_000


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


def csv_links(value) -> list[str]:
    """APEX_SHADOW_BOOK_CSV_URL may list several links (space or comma separated),
    tried in order: the live sheet export first, the published copy as fallback."""
    return [part for part in re.split(r"[\s,]+", str(value or "").strip()) if part]


def valid_csv_links(value) -> bool:
    links = csv_links(value)
    return bool(links) and len(links) <= 3 and all(valid_csv_url(link) for link in links)


def link_kind(url: str) -> str:
    return "live sheet" if "/export" in url or "/gviz/" in url else "published copy"


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


def setup_letter(text) -> str:
    """BotRules key: the first character of 'Setup (EW count)', e.g. 'B — wave-(iv)…' -> 'B'."""
    text = str(text or "").strip()
    return text[:1].upper()


def _rule_side(text) -> str:
    side = str(text or "").strip().lower()
    return {"long": "Long", "short": "Short"}.get(side, str(text or "").strip())


def parse_rules(rows: list[list[str]]) -> dict:
    """BotRules tab -> {"rules": [...], "version": str}.

    Row 1 holds the headers (found by name); rule rows run from row 2 to the first
    blank row. The notes and change log further down the tab are never read. A row
    whose Active, Side and Setup are all empty also ends the table, because Google's
    by-name CSV drops blank rows.
    """
    if not rows:
        raise SourceError("BotRules tab is empty")
    header = [str(h).strip() for h in rows[0]]
    index = {key: header.index(name) for key, name in RULES_HEADERS.items() if name in header}
    missing = [RULES_HEADERS[k] for k in RULES_REQUIRED if k not in index]
    if missing:
        raise SourceError("BotRules header missing: " + ", ".join(missing))

    def cell(row, key):
        i = index.get(key)
        return str(row[i]).strip() if i is not None and i < len(row) else ""

    rules = []
    for row in rows[1:]:
        if not any(str(c).strip() for c in row):
            break
        if not (cell(row, "active") or cell(row, "side") or cell(row, "setup")):
            break
        rules.append(
            {
                "id": cell(row, "id"),
                "active": cell(row, "active").upper() == "TRUE",
                "side": _rule_side(cell(row, "side")),
                "setup": setup_letter(cell(row, "setup")),
                "name": cell(row, "name"),
                "version": cell(row, "version"),
            }
        )
    versions = sorted({r["version"] for r in rules if r["version"]})
    return {"rules": rules, "version": versions[-1] if versions else ""}


def rule_for(rules, side: str, letter: str):
    """The active rule matching a signal ('Buy'/'Sell' or 'Long'/'Short' + letter), or None."""
    want = {"buy": "Long", "long": "Long", "sell": "Short", "short": "Short"}.get(str(side or "").lower())
    letter = str(letter or "").strip().upper()
    if not want or not letter:
        return None
    for rule in rules or ():
        if rule.get("active") and rule.get("side") == want and rule.get("setup") == letter:
            return rule
    return None


def active_rule_names(rules) -> list[str]:
    return [f"{r['side']} {r['setup']}" for r in rules or () if r.get("active")]


def rules_links(trade_links, gid: str = RULES_GID, tab: str = RULES_TAB) -> list[str]:
    """BotRules CSV links derived from the Trades links: each link with the BotRules
    tab id (exact cell text, blank rows kept), then Google's by-name CSV of the live
    sheet as a fallback in case the tab is ever recreated with a new id."""
    out = []
    for link in trade_links:
        url = urlparse(link)
        if gid:
            query = [(k, v) for k, v in parse_qsl(url.query, keep_blank_values=True) if k != "gid"]
            query.append(("gid", gid))
            out.append(urlunparse(url._replace(query=urlencode(query), fragment="")))
    for link in trade_links:
        url = urlparse(link)
        parts = url.path.split("/")
        # /spreadsheets/d/<id>/... (a "Publish to the web" copy is /d/e/<id> and has no by-name CSV)
        if len(parts) > 3 and parts[1] == "spreadsheets" and parts[2] == "d" and parts[3] != "e":
            query = urlencode({"tqx": "out:csv", "headers": "1", "sheet": tab})
            out.append(urlunparse(url._replace(path=f"/spreadsheets/d/{parts[3]}/gviz/tq", query=query, fragment="")))
    seen, unique = set(), []
    for link in out:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return unique


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
            "setup_letter": setup_letter(cell(row, "setup")),
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

    def __init__(
        self,
        session,
        *,
        sheet_id: str = "",
        credentials_json: str = "",
        csv_url: str = "",
        refresh_seconds: int = 60,
        clock=time.time,
        rules_gid: str = RULES_GID,
        rules_tab: str = RULES_TAB,
    ):
        if sheet_id and not valid_sheet_id(sheet_id):
            raise ValueError("Invalid Shadow Book sheet ID")
        if csv_url and not valid_csv_links(csv_url):
            raise ValueError("Shadow Book CSV link must be a docs.google.com published CSV link")
        if sheet_id and not credentials_json:
            raise ValueError("Shadow Book sheet ID needs a Google service account")
        self.session = session
        self.sheet_id = sheet_id
        self.csv_url = csv_url
        self.csv_urls = csv_links(csv_url)
        self.link_used = ""  # "live sheet" / "published copy" on the last good read
        self.link_errors: dict = {}  # kind -> last error, for links that failed
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
        # BotRules: which Side + Setup letters the bot may trade. rules_ok is False
        # until a read succeeds with at least one rule row; the desk then places nothing new.
        self.rules_tab = rules_tab or RULES_TAB
        self.rules_urls = rules_links(self.csv_urls, rules_gid or "", self.rules_tab)
        self.rules: list = []
        self.rules_version = ""
        self.rules_ok = False
        self.rules_error: str | None = "waiting for first read"
        self.rules_at = 0.0  # last good BotRules read
        self.rules_link_used = ""

    @property
    def configured(self) -> bool:
        return bool(self.sheet_id or self.csv_url)

    @property
    def mode(self) -> str:
        return "sheets_api" if self.sheet_id else "csv" if self.csv_url else "none"

    async def _rows_api(self, cell_range: str | None = None):
        from .references import ReferenceReader

        if self._reader is None:
            self._reader = ReferenceReader(self.session, None, self.credentials_json, ())
        try:
            token = await asyncio.to_thread(self._reader._token_sync)
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"Google sign-in failed ({getattr(exc, 'args', ['auth'])[0]})") from None
        import aiohttp

        url = (
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.sheet_id}/values/"
            + quote(cell_range or self.RANGE, safe="!:")
        )
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
        """First link that answers with CSV wins; the others are fallbacks."""
        first = None
        for link in self.csv_urls:
            try:
                rows = await self._rows_from(link)
            except Exception as exc:  # noqa: BLE001 - try the next link
                error = str(exc) if isinstance(exc, SourceError) else f"read failed ({type(exc).__name__})"
                self.link_errors[link_kind(link)] = error
                first = first or exc
                continue
            self.link_errors.pop(link_kind(link), None)
            self.link_used = link_kind(link)
            return rows
        raise first or SourceError("No Shadow Book link configured")

    async def _rows_from(self, link: str):
        import aiohttp

        # Ask Google for the newest copy, never a cached one.
        url = link + ("&" if "?" in link else "?") + f"_ts={int(self.clock())}"
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
            # Read to the end. content.read(n) returns only what has arrived so far
            # (often the first chunk of a 170 KB sheet), which silently dropped the
            # newest rows.
            chunks, size = [], 0
            async for chunk in response.content.iter_chunked(65536):
                size += len(chunk)
                if size > MAX_CSV_BYTES:
                    raise SourceError("Shadow Book CSV is larger than 8 MB")
                chunks.append(chunk)
            data = b"".join(chunks)
        text = data.decode("utf-8-sig", errors="replace")
        if text.lstrip().startswith("<"):
            raise SourceError("Shadow Book link returned a web page, not CSV (publish the Trades tab as CSV)")
        return list(csv.reader(io.StringIO(text)))

    async def _rules_rows(self):
        if self.sheet_id:
            name = self.rules_tab.replace("'", "''")
            return await self._rows_api(f"'{name}'!A1:Z200")
        first = None
        for link in self.rules_urls:
            try:
                rows = await self._rows_from(link)
                parsed = parse_rules(rows)  # wrong tab or headers -> try the next link
            except Exception as exc:  # noqa: BLE001 - try the next link
                first = first or exc
                continue
            self.rules_link_used = "by tab name" if "/gviz/" in link else link_kind(link)
            return parsed
        if first is not None:
            raise first
        raise SourceError("No BotRules link configured")

    async def _refresh_rules(self):
        try:
            result = await self._rules_rows()
            if self.sheet_id:
                result = parse_rules(result)
            if not result["rules"]:
                raise SourceError("BotRules has no rule rows")
            self.rules, self.rules_version = result["rules"], result["version"]
            self.rules_ok, self.rules_error = True, None
            self.rules_at = self.clock()
        except SourceError as exc:
            self.rules_ok, self.rules_error = False, str(exc).replace("Shadow Book CSV", "BotRules")
        except Exception as exc:  # noqa: BLE001 - keep trading loop alive
            self.rules_ok, self.rules_error = False, f"BotRules read failed ({type(exc).__name__})"
        if not self.rules_ok:
            # Never trade on old rules: a failed read means no new entries this cycle.
            self.rules = []
            LOG.warning("BotRules unavailable: %s", self.rules_error)

    async def refresh(self, force: bool = False) -> bool:
        if not self.configured:
            self.ok, self.error = False, "not configured"
            self.rules_ok, self.rules_error, self.rules = False, "not configured", []
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
        await self._refresh_rules()
        return self.ok
