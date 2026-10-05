"""Standalone, plain-text Telegram boundary; importing this module does no I/O.

The parent owns the ClientSession, polling lease, durable cursor and outbox.
It must also persist actor-bound, expiring, single-use confirmation tokens and
enforce policy atomically in confirm_change. No Telegram path calls
change_setting directly. See docs/apex/TELEGRAM_AUDIT.txt for the service contract.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from decimal import Decimal
from typing import Protocol, TypedDict

import aiohttp


PROFILES = ("ultra_cautious", "cautious", "balanced", "aggressive", "extreme")
VIEWS = (
    "dashboard",
    "status",
    "universe",
    "positions",
    "opportunities",
    "why",
    "waves",
    "risk",
    "profile",
    "performance",
    "research",
    "observer",
    "system",
    "telemetry",
    "help",
)
COMMANDS = {view: f"Show {view}" for view in VIEWS}
COMMANDS.update(start="Open dashboard", pause="Preview pause", resume="Preview resume")
HELP = (
    "🤖 Apex controls\n"
    "/start · /dashboard — dashboard\n"
    "/status · /positions\n"
    "/universe — selected symbols and rotation status (read-only)\n"
    "/opportunities · /why SYMBOL · /waves SYMBOL\n"
    "/risk — risk and all profiles\n"
    "/risk 0.25 — preview 0.25% per trade (0.05–1%)\n"
    "/profile NAME — preview a profile\n"
    "/performance · /research · /system · /telemetry\n"
    "/observer — capacity-rejected shadow outcomes\n"
    "/pause · /resume — preview, then confirm\n"
    "🔎 /start only opens the dashboard.\n"
    "Research shows stored offline analysis.\n"
    "Changes require confirmation; resume remains subject to service safety checks."
)
_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,56}\Z")
_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9._/-]{0,31}\Z")
_PERCENT = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)%?\Z")


class Proposal(TypedDict):
    token: str
    summary: str


class TelegramService(Protocol):
    async def snapshot(self, view: str, symbol: str | None = None) -> str:
        """Committed state, plain text; risk/profile include exact amounts/caps."""
        ...

    async def change_setting(self, key: str, value: object, actor: str) -> str:
        """Parent API only; the Telegram controller never invokes this directly."""
        ...

    async def propose_change(self, key: str, value: object, actor: str) -> Proposal:
        """Persist a preview without applying it; summary includes prospective effect."""
        ...

    async def confirm_change(self, token: str, actor: str) -> str:
        """Atomically validate actor/expiry/policy, consume token, and apply once."""
        ...

    async def cancel_change(self, token: str, actor: str) -> str:
        """Atomically revoke this actor's token without changing settings."""
        ...


class TelegramError(Exception):
    """Sanitized transport error; retry policy belongs to the durable outbox.

    uncertain means Telegram may have accepted the request. Retrying sends can
    produce duplicates. sent_message_ids lists acknowledged chunks, not a
    guarantee that other chunks were not delivered. next_offset is a safe cursor
    for updates completed before a poll failed (the failed update is replayable).
    """

    def __init__(
        self,
        message: str = "Telegram request failed",
        *,
        retry_after: float | None = None,
        status: int | None = None,
        uncertain: bool = False,
        sent_message_ids: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status
        self.uncertain = uncertain
        self.sent_message_ids = sent_message_ids
        self.next_offset: int | None = None


class _CommandError(Exception):
    """Only fixed, safe user-facing messages belong here."""


def _positive_id(value: object) -> bool:
    return type(value) is int and value > 0


def _chunks(text: str) -> list[str]:
    """Lossless chunks bounded to 3900 UTF-16 units, including astral emoji."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A nonempty plain-text message is required")
    chunks = []
    start = units = 0
    for index, char in enumerate(text):
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > 3900:
            chunks.append(text[start:index])
            start, units = index, 0
        units += width
    chunks.append(text[start:])
    return chunks


class TelegramController:
    """One controller per bot token; no constructor polling or command publishing."""

    WRITE_INTERVAL = 1.05
    MAX_MUTATION_AGE = 120
    SERVICE_TIMEOUT = 15

    def __init__(
        self,
        session: aiohttp.ClientSession,
        token: str,
        chat_id: str,
        allowed_user_ids: set[int],
        service: TelegramService,
        *,
        bot_username: str | None = "AutotradingBot222_bot",
    ):
        # Credentials are supplied by the parent, never loaded from local files.
        if not isinstance(token, str) or not re.fullmatch(
            r"[0-9]+:[A-Za-z0-9_-]+", token
        ):
            raise ValueError("Invalid Telegram credential configuration")
        self._session = session
        self._token = token
        self.chat_id = str(chat_id).strip()
        self._users = frozenset(uid for uid in allowed_user_ids if _positive_id(uid))
        self._enabled = bool(
            self._users and re.fullmatch(r"-?[1-9][0-9]*", self.chat_id)
        )
        self._service = service
        expected = (
            bot_username.lstrip("@") if isinstance(bot_username, str) else bot_username
        )
        if expected is not None and not re.fullmatch(r"[A-Za-z0-9_]{5,32}", expected):
            raise ValueError("Invalid expected bot username")
        self._expected_username = expected
        self._bot_username: str | None = None
        self._identity_failed = False
        self._poll_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._next_write = 0.0

    async def initialize(self, *, publish_commands: bool = True) -> None:
        """Verify getMe identity, optionally publish commands; never starts polling.

        This deployment step must precede addressed-command use. A constructor
        username is an expected identity, never self-asserted verification.
        """
        if not self._enabled:
            raise TelegramError("Telegram authorization is not configured")
        self._bot_username = None
        self._identity_failed = True
        identity = await self._request("getMe", {}, get=True)
        username = identity.get("username") if isinstance(identity, dict) else None
        if (
            not isinstance(identity, dict)
            or identity.get("is_bot") is not True
            or not _positive_id(identity.get("id"))
            or str(identity["id"]) != self._token.split(":", 1)[0]
            or not isinstance(username, str)
            or not re.fullmatch(r"[A-Za-z0-9_]{5,32}", username)
            or (
                self._expected_username is not None
                and username != self._expected_username
            )
        ):
            raise TelegramError("Telegram bot identity did not match the expected bot")
        self._bot_username = username
        self._identity_failed = False
        if not publish_commands:
            return
        await self._request(
            "setMyCommands",
            {
                "commands": [
                    {"command": key, "description": value}
                    for key, value in COMMANDS.items()
                ],
                "scope": {"type": "chat", "chat_id": self.chat_id},
            },
        )

    async def poll_once(self, offset: int) -> int:
        """Fetch/process one batch; parent durably checkpoints the returned cursor.

        Only messages and callback queries are requested. No pending updates are
        dropped. Errors preserve the failing update for at-least-once handling.
        The lock only coordinates this instance; deployment needs a single-owner
        lease. Deployment replaces the existing production poller for this token;
        the old poller must stop first. Never run them concurrently.
        """
        if type(offset) is not int or offset < 0:
            raise ValueError("Offset must be a nonnegative integer")
        if not self._enabled or self._identity_failed:
            return offset
        async with self._poll_lock:
            updates = await self._request(
                "getUpdates",
                {
                    "offset": offset,
                    "timeout": 20,
                    "allowed_updates": json.dumps(["message", "callback_query"]),
                },
                get=True,
            )
            if not isinstance(updates, list):
                raise TelegramError("Invalid Telegram updates response")
            valid = [
                u
                for u in updates
                if isinstance(u, dict)
                and type(u.get("update_id")) is int
                and u["update_id"] >= offset
            ]
            for update in sorted(valid, key=lambda u: u["update_id"]):
                leader_check = getattr(self._service, "assert_leader", None)
                if leader_check is not None:
                    await leader_check()
                if update["update_id"] < offset:
                    continue
                try:
                    await self._handle_update(update)
                except TelegramError as error:
                    error.next_offset = offset
                    raise
                offset = update["update_id"] + 1
            return offset

    async def send(self, text: str, keyboard: dict | None = None) -> str:
        """Send plain text, return the last chunk's message ID as a string.

        The caller owns durable delivery, backoff and event/chunk deduplication.
        There is no internal retry: ambiguous failures can otherwise duplicate
        notifications. Only the final chunk carries the supplied keyboard.
        """
        if not self._enabled or self._identity_failed:
            raise TelegramError("Telegram authorization is not configured")
        async with self._send_lock:
            return await self._send_chunks(_chunks(self._redact(text)), keyboard)

    def _redact(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("A plain-text service response is required")
        return text.replace(self._token, "[redacted]")

    async def _send_chunks(self, chunks: list[str], keyboard: dict | None) -> str:
        ids = []
        try:
            for index, chunk in enumerate(chunks):
                payload = self._text_payload(chunk)
                if keyboard is not None and index == len(chunks) - 1:
                    payload["reply_markup"] = keyboard
                result = await self._request("sendMessage", payload)
                if not isinstance(result, dict) or not _positive_id(
                    result.get("message_id")
                ):
                    raise TelegramError(
                        "Invalid Telegram message response", uncertain=True
                    )
                ids.append(str(result["message_id"]))
        except TelegramError as error:
            error.sent_message_ids = tuple(ids)
            raise
        return ids[-1]

    def _text_payload(self, text: str) -> dict:
        return {
            "chat_id": self.chat_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }

    async def _render(
        self, text: str, keyboard: dict | None, message: dict | None
    ) -> None:
        chunks = _chunks(self._redact(text))
        async with self._send_lock:
            if message is not None and _positive_id(message.get("message_id")):
                payload = self._text_payload(chunks[0])
                payload.update(
                    message_id=message["message_id"],
                    reply_markup=(
                        keyboard
                        if len(chunks) == 1 and keyboard is not None
                        else {"inline_keyboard": []}
                    ),
                )
                try:
                    await self._request("editMessageText", payload)
                except TelegramError as error:
                    # An explicit uneditable/deleted message can be replaced.
                    # Never duplicate on timeouts, 429s, or server errors.
                    if error.status not in (400, 403) or error.uncertain:
                        raise
                else:
                    if len(chunks) > 1:
                        await self._send_chunks(chunks[1:], keyboard)
                    return
            await self._send_chunks(chunks, keyboard)

    def _authorized(self, message: object, sender: object) -> bool:
        if (
            not self._enabled
            or self._identity_failed
            or not isinstance(message, dict)
            or not isinstance(sender, dict)
        ):
            return False
        chat = message.get("chat")
        return bool(
            isinstance(chat, dict)
            and type(chat.get("id")) is int
            and str(chat["id"]) == self.chat_id
            and _positive_id(sender.get("id"))
            and sender["id"] in self._users
            and not sender.get("is_bot")
            and not message.get("sender_chat")
        )

    @staticmethod
    def _button(label: str, data: str) -> dict:
        return {"text": label, "callback_data": data}

    def _keyboard(self, view: str, symbol: str | None = None) -> dict:
        button = self._button
        if view in ("risk", "profile"):
            rows = [[button(name, f"p:profile:{name}")] for name in PROFILES]
            rows.append([button("✏️ Custom %", "custom_risk")])
        elif view == "universe":
            rows = []
        else:
            rows = [
                [
                    button("📊 Status", "v:status"),
                    button("📂 Positions", "v:positions"),
                ],
                [
                    button("🔎 Opportunities", "v:opportunities"),
                    button("🌊 Waves", "v:waves"),
                ],
                [button("💡 Why", "v:why"), button("🛡 Risk", "v:risk")],
                [
                    button("🎚 Profile", "v:profile"),
                    button("📈 Performance", "v:performance"),
                ],
                [button("🧪 Research", "v:research"), button("⚙️ System", "v:system")],
                [
                    button("🌐 Universe", "v:universe"),
                    button("🔬 Observer", "v:observer"),
                ],
                [button("⏸ Pause", "p:paused:1"), button("▶️ Resume", "p:paused:0")],
                [button("📡 Telemetry", "v:telemetry"), button("❓ Help", "v:help")],
            ]
        refresh = f"v:{view}" + (f":{symbol}" if symbol else "")
        rows.append(
            [button("🔄 Refresh", refresh), button("🏠 Dashboard", "v:dashboard")]
        )
        return {"inline_keyboard": rows}

    async def _view(self, view: str, symbol: str | None, message: dict | None) -> None:
        if view == "help":
            text = HELP
        else:
            text = await asyncio.wait_for(
                self._service.snapshot(view, symbol),
                timeout=self.SERVICE_TIMEOUT,
            )
        if view in ("risk", "profile"):
            text += (
                "\n\n🎚 Profiles (mild → aggressive):\n"
                + " · ".join(PROFILES)
                + "\n✏️ /risk 0.25 previews 0.25% (allowed: 0.05–1%)."
                + "\nSelect a profile to inspect its prospective effect before confirming."
            )
        await self._render(text, self._keyboard(view, symbol), message)

    def _fresh(self, message: dict, *, callback: bool = False) -> None:
        timestamp = (
            message.get("edit_date", message.get("date"))
            if callback
            else message.get("date")
        )
        if (
            type(timestamp) is not int
            or timestamp <= 0
            or not -5 <= time.time() - timestamp <= self.MAX_MUTATION_AGE
            or message.get("forward_origin")
            or message.get("forward_date")
        ):
            raise _CommandError(
                "⌛ This control is stale. Open /dashboard and request a fresh preview."
            )

    async def _propose(
        self, key: str, value: object, actor: str, message: dict | None
    ) -> None:
        proposal = await asyncio.wait_for(
            self._service.propose_change(key, value, actor),
            timeout=self.SERVICE_TIMEOUT,
        )
        if not isinstance(proposal, dict):
            raise ValueError("Invalid service proposal")
        token, summary = proposal.get("token"), proposal.get("summary")
        if (
            not isinstance(token, str)
            or not _TOKEN.fullmatch(token)
            or not isinstance(summary, str)
            or not summary.strip()
        ):
            raise ValueError("Invalid service proposal")
        # The entire authoritative summary is delivered before the confirm button.
        await self._render(
            "🔎 Proposed change — review before confirming\n\n"
            + summary
            + "\n\nConfirm applies this proposal once, subject to current safety checks.",
            {
                "inline_keyboard": [
                    [
                        self._button("✅ Confirm", f"confirm:{token}"),
                        self._button("✖️ Cancel", f"cancel:{token}"),
                    ]
                ]
            },
            message,
        )

    async def _handle_update(self, update: dict) -> None:
        callback = update.get("callback_query")
        if callback is not None:
            if not isinstance(callback, dict):
                return
            message, sender = callback.get("message"), callback.get("from")
            allowed = self._authorized(message, sender)
            query_id = callback.get("id")
            if not isinstance(query_id, str) or not query_id or len(query_id) > 256:
                return
            # Always dismiss the spinner, including denied/malformed callbacks.
            # A stale query cannot be safely actioned if Telegram rejects the ack.
            try:
                await self._request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query_id,
                        "text": "Opening…" if allowed else "Not authorized",
                        "cache_time": 0,
                    },
                )
            except TelegramError as error:
                if error.status == 400:
                    return
                raise
            if not allowed:
                return
            actor = f"telegram:{self.chat_id}:{sender['id']}"
            target = message
        else:
            message = update.get("message")
            sender = message.get("from") if isinstance(message, dict) else None
            if not self._authorized(message, sender):
                return
            actor = f"telegram:{self.chat_id}:{sender['id']}"
            target = None
        try:
            if callback is not None:
                await self._callback(callback, actor)
            else:
                await self._command(message, actor)
        except _CommandError as error:
            await self._render(str(error), self._keyboard("dashboard"), target)
        except TelegramError:
            raise
        except Exception:
            # Neither service exceptions nor aiohttp URLs may expose credentials.
            # A failed confirmation response may follow a committed change.
            await self._render(
                "⚠️ Unable to verify the request. Check /status and /risk before requesting a new preview.",
                self._keyboard("dashboard"),
                target,
            )

    async def _command(self, message: dict, actor: str) -> None:
        text = message.get("text")
        if not isinstance(text, str) or not text.startswith("/"):
            return
        if len(text) > 256:
            raise _CommandError(
                "Use a short command. /help lists the supported commands."
            )
        parts = text.split()
        command = parts[0][1:]
        if "@" in command:
            command, username = command.split("@", 1)
            if self._bot_username is None or username != self._bot_username:
                return
        args = parts[1:]
        if command == "start":
            # Telegram start payloads are navigation, never settings or resume.
            await self._view("dashboard", None, None)
        elif command in ("risk", "profile") and args:
            if len(args) != 1:
                raise _CommandError(
                    "Use /risk 0.25 or /profile balanced to request a preview."
                )
            self._fresh(message)
            if command == "risk":
                if not _PERCENT.fullmatch(args[0]):
                    raise _CommandError(
                        "Risk must be a percentage from 0.05 to 1, e.g. /risk 0.25."
                    )
                percent = Decimal(args[0].rstrip("%"))
                if not Decimal("0.05") <= percent <= Decimal("1"):
                    raise _CommandError("Risk must be a percentage from 0.05 to 1.")
                # Percent units: 0.25 means 0.25%, never 25% or a risk fraction.
                await self._propose("risk_pct", float(percent), actor, None)
            else:
                if args[0] not in PROFILES:
                    raise _CommandError("Choose a profile from /profile.")
                await self._propose("profile", args[0], actor, None)
        elif command in ("pause", "resume"):
            if args:
                raise _CommandError("Use /pause or /resume without arguments.")
            self._fresh(message)
            await self._propose("paused", command == "pause", actor, None)
        elif command in VIEWS:
            symbol = None
            if args:
                if command not in ("why", "waves") or len(args) != 1:
                    raise _CommandError("Unexpected arguments. See /help.")
                symbol = args[0].upper()
                if not _SYMBOL.fullmatch(symbol):
                    raise _CommandError("Use a valid symbol, e.g. /why BTCUSDT.")
            await self._view(command, symbol, None)
        else:
            raise _CommandError("Unknown command. /help lists the supported commands.")

    async def _callback(self, callback: dict, actor: str) -> None:
        data, message = callback.get("data"), callback["message"]
        if not isinstance(data, str) or len(data.encode("utf-8")) > 64:
            raise _CommandError("Invalid control. Open /dashboard for fresh controls.")
        parts = data.split(":")
        if parts[0] == "v" and len(parts) in (2, 3) and parts[1] in VIEWS:
            symbol = parts[2] if len(parts) == 3 else None
            if symbol is not None and (
                parts[1] not in ("why", "waves") or not _SYMBOL.fullmatch(symbol)
            ):
                raise _CommandError("Invalid symbol control. Open /dashboard.")
            await self._view(parts[1], symbol, message)
        elif data == "custom_risk":
            await self._render(
                "✏️ Enter /risk 0.25 to preview 0.25% risk per trade.\n"
                "Allowed range: 0.05–1%. The preview shows cash risk at current equity, "
                "caps and prospective effect before confirmation.",
                self._keyboard("risk"),
                message,
            )
        elif (
            len(parts) == 3
            and parts[0] == "p"
            and (
                (parts[1] == "profile" and parts[2] in PROFILES)
                or (parts[1] == "paused" and parts[2] in ("0", "1"))
            )
        ):
            self._fresh(message, callback=True)
            value = parts[2] if parts[1] == "profile" else parts[2] == "1"
            await self._propose(parts[1], value, actor, message)
        elif (
            len(parts) == 2
            and parts[0] in ("confirm", "cancel")
            and _TOKEN.fullmatch(parts[1])
        ):
            # Require a matching button on the actual Telegram preview message;
            # callback_data alone is not evidence that a preview was displayed.
            markup = message.get("reply_markup")
            rows = markup.get("inline_keyboard", []) if isinstance(markup, dict) else []
            buttons = {
                b.get("callback_data")
                for row in rows
                if isinstance(row, list)
                for b in row
                if isinstance(b, dict) and isinstance(b.get("callback_data"), str)
            }
            if not {f"confirm:{parts[1]}", f"cancel:{parts[1]}"} <= buttons:
                raise _CommandError(
                    "⌛ Preview unavailable. Request a new preview before confirming."
                )
            if parts[0] == "cancel":
                result = await asyncio.wait_for(
                    self._service.cancel_change(parts[1], actor),
                    timeout=self.SERVICE_TIMEOUT,
                )
                await self._render(result, self._keyboard("dashboard"), message)
            else:
                self._fresh(message, callback=True)
                result = await asyncio.wait_for(
                    self._service.confirm_change(parts[1], actor),
                    timeout=self.SERVICE_TIMEOUT,
                )
                await self._render(result, self._keyboard("dashboard"), message)
        else:
            raise _CommandError("Unknown control. Open /dashboard for fresh controls.")

    async def _request(self, method: str, payload: dict, *, get: bool = False):
        if get:
            return await self._http(method, payload, get=True)
        async with self._write_lock:
            await asyncio.sleep(max(0.0, self._next_write - time.monotonic()))
            self._next_write = time.monotonic() + self.WRITE_INTERVAL
            try:
                return await self._http(method, payload)
            except TelegramError as error:
                if error.retry_after is not None:
                    self._next_write = max(
                        self._next_write, time.monotonic() + error.retry_after
                    )
                raise

    async def _http(self, method: str, payload: dict, *, get: bool = False):
        # Never log the URL, response description, exception repr or request info.
        # Disable redirects so a credential-bearing request cannot follow one.
        kwargs = {
            "params" if get else "json": payload,
            "allow_redirects": False,
            "timeout": aiohttp.ClientTimeout(total=30, connect=5, sock_read=25),
        }
        try:
            async with self._session.request(
                "GET" if get else "POST",
                f"https://api.telegram.org/bot{self._token}/{method}",
                **kwargs,
            ) as response:
                status = response.status
                try:
                    body = await response.json()
                except (ValueError, aiohttp.ContentTypeError):
                    body = None
                if (
                    isinstance(body, dict)
                    and body.get("ok") is True
                    and 200 <= status < 300
                ):
                    return body.get("result")
                # An identical edit is successful from the user's perspective.
                if (
                    method == "editMessageText"
                    and status == 400
                    and isinstance(body, dict)
                    and "message is not modified"
                    in str(body.get("description", "")).lower()
                ):
                    return None
                parameters = body.get("parameters") if isinstance(body, dict) else None
                retry = (
                    parameters.get("retry_after")
                    if isinstance(parameters, dict)
                    else None
                )
                if retry is None:
                    retry = response.headers.get("Retry-After")
                try:
                    retry = float(retry)
                    if not math.isfinite(retry) or retry < 0:
                        retry = None
                except (TypeError, ValueError, OverflowError):
                    retry = None
                error_code = body.get("error_code") if isinstance(body, dict) else None
                code = error_code if type(error_code) is int else status
                raise TelegramError(
                    "Telegram API request rejected",
                    status=code,
                    retry_after=retry,
                    uncertain=(
                        status >= 500
                        or code >= 500
                        or not isinstance(body, dict)
                        or body.get("ok") is not False
                    ),
                )
        except TelegramError:
            raise
        except Exception:
            # Suppress exception chaining: aiohttp errors can contain the bot URL.
            raise TelegramError("Telegram transport failed", uncertain=True) from None
