"""Read-only source ingestion; reference contents are untrusted evidence, never policy.

Only explicitly configured file IDs are read. Folders, shortcuts, links in text,
and instructions in cells are never traversed or executed. Every observed source
version is archived in the NEW bot database, not in the external Apex ledger.

Context is an optional, externally supplied feed in shadow mode. It is not an
autonomous macro-research provider. Consumers must block live entries unless
``fresh`` is true and ``expires_at`` has not passed, then apply the explicit
blackout scopes. A source's verified_at is its publisher's assertion, not an
independent verification of the underlying news by this module.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import ipaddress
import json
import math
import re
import socket
import ssl
import time
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit


READ_SCOPES = (
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
)
SHEET_RANGES = ("Settings!A1:E180", "Counts!A1:K250", "Levels!A1:Q100", "CCR!A1:K50")
_SHAPES = ((180, 5), (250, 11), (100, 17), (50, 11))
_DOC = "application/vnd.google-apps.document"
_SHEET = "application/vnd.google-apps.spreadsheet"
_TOKEN_URI = "https://oauth2.googleapis.com/token"
_METADATA_FIELDS = "id,name,mimeType,version,modifiedTime,trashed"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_CONTEXT_BYTES = 64 * 1024
MAX_FILES = 32
MAX_AGE = 6 * 3600
MAX_EXPIRY = 24 * 3600
_TRANSIENT = {408, 429, 500, 502, 503, 504}


class _Failure(Exception):
    """Only locally defined, non-sensitive reason codes cross the API boundary."""


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _Failure("duplicate_json_key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise _Failure("non_finite_json")

    try:
        return json.loads(
            raw, object_pairs_hook=unique, parse_constant=invalid_constant
        )
    except (ValueError, UnicodeError, RecursionError):
        raise _Failure("invalid_json") from None


def _timestamp(value):
    try:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError()
            value = dt.timestamp()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError()
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError()
        return value
    except (ValueError, OverflowError, TypeError):
        raise _Failure("invalid_timestamp") from None


async def _get_bytes(session, url, *, limit, params=None, headers=None, **kwargs):
    """Bound both streamed/decompressed bytes and elapsed time; never follow redirects."""
    import aiohttp

    for attempt in range(3):
        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=15, connect=5),
                raise_for_status=False,
                **kwargs,
            ) as response:
                if response.status in _TRANSIENT:
                    if attempt == 2:
                        raise _Failure("transient_http_error")
                elif response.status != 200:
                    # Do not expose URLs, headers, tokens, or server error bodies.
                    if 300 <= response.status < 400:
                        raise _Failure("redirect_rejected")
                    raise _Failure(
                        "access_denied"
                        if response.status in (401, 403)
                        else (
                            "source_not_found"
                            if response.status == 404
                            else "http_error"
                        )
                    )
                else:
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) > limit:
                        raise _Failure("response_too_large")
                    data = bytearray()
                    async for chunk in response.content.iter_chunked(16384):
                        data.extend(chunk)
                        if len(data) > limit:
                            raise _Failure("response_too_large")
                    return bytes(data)
        except (
            asyncio.TimeoutError,
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
        ):
            if attempt == 2:
                raise _Failure("network_unavailable") from None
        await asyncio.sleep(0.25 * 2**attempt)
    raise _Failure("network_unavailable")


class ReferenceReader:
    """credentials_json comes from env configuration only; never files or ADC.

    refresh() returns status/connected/files. State is references[file_id] with
    current, checked_at, last_success_at, error and snapshot. Failed reads retain
    the old snapshot/asof but set current=False. Consumers MUST check current;
    checked_at records an attempt, not a successful source observation.
    """

    def __init__(self, session, store, credentials_json="", file_ids=tuple()):
        self.session = session
        self.store = store
        self._credentials_json = credentials_json
        self._credentials = None
        self._lock = None  # Construct on the running loop (Python 3.9 compatibility).
        # A string is not an iterable of file IDs. Reject all malformed config
        # before making even an authentication request.
        if (
            not isinstance(file_ids, (tuple, list))
            or len(file_ids) > MAX_FILES
            or any(
                not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", x)
                for x in file_ids
            )
        ):
            raise ValueError("Invalid reference file ID allowlist")
        self.file_ids = tuple(dict.fromkeys(file_ids))

    def _token_sync(self):
        # Lazy dependency: a bot without Drive configuration can still start.
        try:
            from google.oauth2.service_account import Credentials
            from google.auth.transport.requests import Request
            import requests
        except ImportError:
            raise _Failure("google_auth_unavailable") from None
        try:
            if self._credentials is None:
                if (
                    not isinstance(self._credentials_json, str)
                    or len(self._credentials_json) > 65536
                ):
                    raise _Failure("invalid_credentials")
                info = _json(self._credentials_json)
                if (
                    not isinstance(info, dict)
                    or info.get("type") != "service_account"
                    or info.get("token_uri") != _TOKEN_URI
                    or info.get("universe_domain", "googleapis.com") != "googleapis.com"
                    or info.get("subject")
                ):
                    raise _Failure("invalid_credentials")
                self._credentials = Credentials.from_service_account_info(
                    info, scopes=READ_SCOPES
                )
            if not self._credentials.valid:
                with requests.Session() as auth_session:
                    auth_session.trust_env = False
                    request = Request(session=auth_session)

                    def token_request(url, method="GET", body=None, headers=None, **_):
                        if url != _TOKEN_URI or method != "POST":
                            raise _Failure("invalid_auth_endpoint")
                        return request(
                            url,
                            method=method,
                            body=body,
                            headers=headers,
                            timeout=10,
                            allow_redirects=False,
                        )

                    self._credentials.refresh(token_request)
            token = self._credentials.token
            if not isinstance(token, str) or not token:
                raise _Failure("authentication_failed")
            return token
        except _Failure:
            raise
        except Exception:
            raise _Failure("authentication_failed") from None

    async def _metadata(self, file_id, headers):
        if file_id not in self.file_ids:
            raise _Failure("file_not_allowlisted")
        value = _json(
            await _get_bytes(
                self.session,
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                params={"fields": _METADATA_FIELDS, "supportsAllDrives": "true"},
                headers=headers,
                limit=MAX_METADATA_BYTES,
            )
        )
        if (
            not isinstance(value, dict)
            or value.get("id") != file_id
            or value.get("trashed") is not False
            or value.get("mimeType") not in (_DOC, _SHEET)
            or not re.fullmatch(r"[0-9]+", str(value.get("version", "")))
            or not isinstance(value.get("name"), str)
        ):
            raise _Failure("invalid_source_metadata")
        _timestamp(value.get("modifiedTime"))
        return {key: value[key] for key in _METADATA_FIELDS.split(",")}

    async def _sheet(self, file_id, headers):
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{file_id}"
        # Only the four requested tabs' properties; no gridData or other tabs.
        metadata = _json(
            await _get_bytes(
                self.session,
                base,
                limit=MAX_METADATA_BYTES,
                headers=headers,
                params=[("ranges", r) for r in SHEET_RANGES]
                + [
                    ("includeGridData", "false"),
                    (
                        "fields",
                        "spreadsheetId,sheets(properties(sheetId,title,gridProperties(rowCount,columnCount)))",
                    ),
                ],
            )
        )
        titles = [r.split("!")[0] for r in SHEET_RANGES]
        if not isinstance(metadata, dict) or metadata.get("spreadsheetId") != file_id:
            raise _Failure("invalid_sheet_metadata")
        sheets = metadata.get("sheets")
        if (
            not isinstance(sheets, list)
            or len(sheets) != len(titles)
            or sorted(s.get("properties", {}).get("title", "") for s in sheets)
            != sorted(titles)
        ):
            raise _Failure("invalid_sheet_metadata")
        values = _json(
            await _get_bytes(
                self.session,
                base + "/values:batchGet",
                limit=MAX_SOURCE_BYTES,
                headers=headers,
                params=[("ranges", r) for r in SHEET_RANGES]
                + [
                    ("majorDimension", "ROWS"),
                    ("valueRenderOption", "UNFORMATTED_VALUE"),
                    ("dateTimeRenderOption", "SERIAL_NUMBER"),
                    (
                        "fields",
                        "spreadsheetId,valueRanges(range,majorDimension,values)",
                    ),
                ],
            )
        )
        if not isinstance(values, dict) or values.get("spreadsheetId") != file_id:
            raise _Failure("invalid_sheet_values")
        ranges = values.get("valueRanges")
        if not isinstance(ranges, list) or len(ranges) != len(SHEET_RANGES):
            raise _Failure("invalid_sheet_values")
        normalized = []
        for block, requested, (rows, cols) in zip(ranges, SHEET_RANGES, _SHAPES):
            if (
                not isinstance(block, dict)
                or block.get("range", "").replace("'", "") != requested
                or block.get("majorDimension") != "ROWS"
            ):
                raise _Failure("unexpected_sheet_range")
            data = block.get("values", [])
            if not isinstance(data, list) or len(data) > rows:
                raise _Failure("sheet_range_overflow")
            for row in data:
                if not isinstance(row, list) or len(row) > cols:
                    raise _Failure("sheet_range_overflow")
                if any(
                    not isinstance(x, (str, int, float, bool))
                    or isinstance(x, float)
                    and not math.isfinite(x)
                    for x in row
                ):
                    raise _Failure("invalid_sheet_values")
            normalized.append({"range": requested, "values": data})
        content = {"metadata": metadata, "ranges": normalized}
        if len(_canonical(content)) > MAX_SOURCE_BYTES:
            raise _Failure("response_too_large")
        return content

    async def _snapshot(self, file_id, headers):
        for attempt in range(3):
            before = await self._metadata(file_id, headers)
            if before["mimeType"] == _DOC:
                raw = await _get_bytes(
                    self.session,
                    f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
                    params={"mimeType": "text/plain"},
                    headers=headers,
                    limit=MAX_SOURCE_BYTES,
                )
                content = {"text": raw.decode("utf-8")}
            else:
                content = await self._sheet(file_id, headers)
            after = await self._metadata(file_id, headers)
            if before == after:
                now = time.time()
                if _timestamp(after["modifiedTime"]) > now + 60:
                    raise _Failure("future_source_timestamp")
                content_hash = _digest(_canonical(content))
                provenance = {
                    "file_id": file_id,
                    "revision": str(after["version"]),
                    "source_modified_at": after["modifiedTime"],
                    "mime_type": after["mimeType"],
                    "name": after["name"],
                    "ranges": list(SHEET_RANGES) if after["mimeType"] == _SHEET else [],
                    "sha256": content_hash,
                }
                return {
                    **provenance,
                    "provenance_sha256": _digest(_canonical(provenance)),
                    "asof": now,
                    "content": content,
                    "trust": "untrusted_reference",
                    "policy_authority": False,
                }
            if attempt < 2:
                await asyncio.sleep(0.25 * 2**attempt)
        raise _Failure("source_changed_during_read")

    async def _record(self, file_id, snapshot=None, error=None):
        checked_at = time.time()

        def record(tx):
            failure = error
            references = tx.state.setdefault("references", {})
            prior = references.get(file_id, {})
            old = prior.get("snapshot")
            if (
                old
                and snapshot
                and (
                    int(snapshot["revision"]) < int(old["revision"])
                    or _timestamp(snapshot["source_modified_at"])
                    < _timestamp(old["source_modified_at"])
                    or snapshot["asof"] < old["asof"]
                )
            ):
                failure = "source_revision_regressed"
            if failure:
                references[file_id] = {
                    **prior,
                    "current": False,
                    "status": "unavailable",
                    "checked_at": checked_at,
                    "error": failure,
                }
                if prior.get("current") is not False or prior.get("error") != failure:
                    tx.event(
                        f"reference-error:{file_id}:{checked_at}",
                        "reference_unavailable",
                        {"file_id": file_id, "error": failure, "asof": checked_at},
                        "Apex reference unavailable; cached evidence is not current. Review data access.",
                    )
                return {"status": "unavailable", "current": False, "error": failure}
            changed = (
                old is None
                or old.get("provenance_sha256") != snapshot["provenance_sha256"]
            )
            if changed:
                # Also archive an existing snapshot when upgrading a pre-audit store.
                for version in (old, snapshot):
                    if version:
                        version_key = version.get("provenance_sha256") or _digest(
                            _canonical(version)
                        )
                        tx.event(
                            f"reference-version:{file_id}:{version_key}",
                            "reference_version",
                            copy.deepcopy(version),
                        )
                tx.event(
                    f"reference-change:{file_id}:{snapshot['provenance_sha256']}:{checked_at}",
                    "reference_changed",
                    {
                        "file_id": file_id,
                        "previous_revision": old.get("revision") if old else None,
                        "revision": snapshot["revision"],
                        "sha256": snapshot["sha256"],
                    },
                    "Apex reference source changed. Review the new evidence; live policy was not changed.",
                )
            references[file_id] = {
                "current": True,
                "status": "ok",
                "error": None,
                "checked_at": checked_at,
                "last_success_at": snapshot["asof"],
                "snapshot": copy.deepcopy(snapshot),
            }
            return {
                "status": "updated" if changed else "unchanged",
                "current": True,
                "revision": snapshot["revision"],
                "sha256": snapshot["sha256"],
                "asof": snapshot["asof"],
            }

        return await self.store.update(record)

    async def refresh(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            # Removed IDs remain historical evidence, but cannot remain current.
            def retire(tx):
                for file_id, entry in tx.state.get("references", {}).items():
                    if file_id not in self.file_ids and entry.get("current"):
                        entry.update(
                            current=False,
                            status="not_allowlisted",
                            error="not_allowlisted",
                        )
                        tx.event(
                            f"reference-retired:{file_id}:{time.time()}",
                            "reference_retired",
                            {"file_id": file_id},
                        )

            await self.store.update(retire)
            if not self.file_ids:
                return {"status": "not_configured", "connected": False, "files": {}}
            auth_error = None
            if not self._credentials_json:
                auth_error = "credentials_missing"
            else:
                try:
                    token = await asyncio.to_thread(self._token_sync)
                except _Failure as exc:
                    auth_error = str(exc)
                except Exception:
                    auth_error = "authentication_failed"
            statuses = {}
            for file_id in self.file_ids:
                snapshot, error = None, auth_error
                if not error:
                    try:
                        snapshot = await self._snapshot(
                            file_id, {"Authorization": "Bearer " + token}
                        )
                    except _Failure as exc:
                        error = str(exc)
                    except Exception:
                        error = "source_read_failed"
                # Store failures propagate: no successful status without a durable audit.
                statuses[file_id] = await self._record(file_id, snapshot, error)
            successes = sum(s["current"] for s in statuses.values())
            return {
                "status": (
                    "ok"
                    if successes == len(statuses)
                    else "partial" if successes else "unavailable"
                ),
                "connected": successes > 0,
                "files": statuses,
            }


def _public_url(url):
    if (
        not isinstance(url, str)
        or len(url) > 2048
        or any(ord(c) <= 32 or ord(c) == 127 for c in url)
        or "\\" in url
    ):
        raise _Failure("invalid_public_url")
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().rstrip(".")
        if (
            parts.scheme != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
            or parts.port not in (None, 443)
            or "%" in host
            or host == "localhost"
            or host.endswith((".localhost", ".local", ".internal", ".home.arpa"))
        ):
            raise ValueError()
        host = host.encode("idna").decode("ascii")
        try:
            _public_ip(host)
        except ValueError:
            if "." not in host or not re.fullmatch(r"[a-z0-9.-]+", host):
                raise ValueError()
        return parts, host
    except (ValueError, UnicodeError):
        raise _Failure("invalid_public_url") from None


def _public_ip(address):
    ip = ipaddress.ip_address(address)
    if (
        not ip.is_global
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or getattr(ip, "ipv4_mapped", None)
        or getattr(ip, "sixtofour", None)
        or getattr(ip, "teredo", None)
    ):
        raise _Failure("non_public_address")
    return str(ip)


async def _resolve_public(host):
    try:
        return [_public_ip(host)]
    except ValueError:
        pass
    try:
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM),
            5,
        )
    except (OSError, asyncio.TimeoutError):
        raise _Failure("dns_unavailable") from None
    if not addresses:
        raise _Failure("dns_unavailable")
    # Reject mixed public/private answers, not just the selected first address.
    return list(dict.fromkeys(_public_ip(item[4][0]) for item in addresses))


async def load_context(session, url, now):
    """Validate a public JSON feed; missing/invalid data never defaults to neutral.

    Required JSON fields: asof (or timestamp), expires_at, long_multiplier,
    short_multiplier, risk_state, event_blackout, blackout_scopes and sources.
    Times are UNIX seconds or timezone-aware ISO-8601. blackout_scopes is a list
    of ALL, LONG, SHORT or exact uppercase USDT symbols; it must be nonempty for
    a blackout and empty otherwise. sources is a nonempty list of objects with
    public HTTPS url and verified_at, fresh within six hours. Unknown fields are
    not forwarded. Source links are validated, never crawled.
    """
    if not url:
        return {
            "status": "unavailable",
            "fresh": False,
            "live_blocked": True,
            "reason": "context_not_configured",
        }
    try:
        now = _timestamp(now)
        parts, host = _public_url(url)
        addresses = await _resolve_public(host)
        # Do not use a caller's ambient credentials, cookies or proxy for a
        # public feed. Production should pass its ordinary unauthenticated session.
        if (
            getattr(session, "trust_env", False)
            or getattr(session, "auth", None)
            or getattr(session, "_default_proxy", None)
            or getattr(session, "_default_proxy_auth", None)
            or any(
                str(k).lower() in ("authorization", "cookie", "proxy-authorization")
                for k in getattr(session, "headers", {})
            )
            or len(getattr(session, "cookie_jar", ()))
        ):
            raise _Failure("unsafe_public_session")
        address = addresses[0]
        authority = f"[{address}]" if ":" in address else address
        pinned_url = urlunsplit(
            ("https", authority, parts.path or "/", parts.query, "")
        )
        raw = await _get_bytes(
            session,
            pinned_url,
            limit=MAX_CONTEXT_BYTES,
            headers={
                "Host": f"[{host}]" if ":" in host else host,
                "Accept": "application/json",
            },
            server_hostname=host,
            ssl=ssl.create_default_context(),
            proxy=None,
        )
        value = _json(raw)
        if not isinstance(value, dict):
            raise _Failure("invalid_context")
        asof = _timestamp(value.get("asof", value.get("timestamp")))
        if "timestamp" in value and _timestamp(value["timestamp"]) != asof:
            raise _Failure("conflicting_timestamp")
        expires = _timestamp(value.get("expires_at"))
        if not 0 <= now - asof <= MAX_AGE or not now < expires <= now + MAX_EXPIRY:
            raise _Failure("context_stale_or_future")
        multipliers = {}
        for key in ("long_multiplier", "short_multiplier"):
            item = value.get(key)
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or not 0 <= item <= 1
            ):
                raise _Failure("invalid_multiplier")
            multipliers[key] = float(item)
        risk_state = value.get("risk_state")
        if risk_state not in ("risk_on", "neutral", "risk_off"):
            raise _Failure("invalid_risk_state")
        blackout, scopes = value.get("event_blackout"), value.get("blackout_scopes")
        if (
            type(blackout) is not bool
            or not isinstance(scopes, list)
            or len(scopes) > 80
            or any(
                not isinstance(s, str)
                or not re.fullmatch(r"ALL|LONG|SHORT|[A-Z0-9]{2,30}USDT", s)
                for s in scopes
            )
            or bool(scopes) != blackout
            or len(set(scopes)) != len(scopes)
        ):
            raise _Failure("invalid_blackout")
        sources = value.get("sources")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 12:
            raise _Failure("invalid_sources")
        normalized, resolved = [], {host}
        for source in sources:
            if not isinstance(source, dict):
                raise _Failure("invalid_sources")
            _, source_host = _public_url(source.get("url"))
            verified = _timestamp(source.get("verified_at"))
            if not 0 <= now - verified <= MAX_AGE:
                raise _Failure("source_stale_or_future")
            if source_host not in resolved:
                await _resolve_public(source_host)
                resolved.add(source_host)
            normalized.append({"url": source["url"], "verified_at": verified})
        fresh_until = min(
            expires,
            asof + MAX_AGE,
            *(source["verified_at"] + MAX_AGE for source in normalized),
        )
        return {
            "status": "ok",
            "fresh": True,
            "live_blocked": False,
            "asof": asof,
            "timestamp": asof,
            "expires_at": expires,
            **multipliers,
            "fresh_until": fresh_until,
            "risk_state": risk_state,
            "event_blackout": blackout,
            "blackout_scopes": scopes,
            "sources": normalized,
            "sha256": _digest(raw),
            "provider_url": url,
            "provenance": "external_provider_assertion",
            "policy_authority": False,
        }
    except _Failure as exc:
        reason = str(exc)
    except Exception:
        reason = "context_unavailable"
    return {
        "status": "unavailable",
        "fresh": False,
        "live_blocked": True,
        "reason": reason,
    }
