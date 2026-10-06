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
PROFILE_LABELS = {name: name.replace("_", " ").capitalize() for name in PROFILES}
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
    "comparison",
    "research",
    "observer",
    "system",
    "telemetry",
    "settings",
    "guide",
    "help",
)
COMMANDS = {
    "start": "Open the home overview (does not resume trading)",
    "dashboard": "Home: activity, freshness and active alerts",
    "status": "Operational status and decision blockers",
    "opportunities": "Current setups and their next required conditions",
    "why": "Setup evidence and rejection reasons; optional SYMBOL",
    "waves": "Wave structure, trigger and invalidation; optional SYMBOL",
    "positions": "Waiting entries, simulated trades and confirmed positions",
    "universe": "Qualified symbols, rankings and rotation",
    "performance": "Rules, AI-approved shadow and executed results",
    "comparison": "Comparison methodology and limits of the evidence",
    "risk": "Risk budgets and caps; add a percent to preview a change",
    "profile": "Current profile; add a profile name to preview a change",
    "research": "Stored research results, availability and sources",
    "observer": "Independent outcomes of capacity-rejected setups",
    "system": "Component health, current errors and recovery",
    "telemetry": "Decision timing and evidence quality",
    "settings": "Controls: pause, resume and profile previews",
    "guide": "Glossary: WR, R, setup states and simulation assumptions",
    "help": "Command reference and navigation help",
    "pause": "Preview pausing new entries; confirmation required",
    "resume": "Preview resuming; confirmation and safety checks required",
}
HELP = (
    "🤖 Apex controls\n"
    "/start · /dashboard — home overview\n"
    "/status — operational status and blockers\n"
    "/positions — waiting entries, simulated trades and confirmed positions\n"
    "/universe — selected symbols and rotation status (read-only)\n"
    "/opportunities — current setups and next conditions\n"
    "/why SYMBOL — evidence and rejection reasons\n"
    "/waves SYMBOL — structure, trigger and invalidation\n"
    "/performance — Rules, AI-approved shadow and executed results\n"
    "/comparison — methodology and comparison limitations\n"
    "/research — stored results, availability and sources\n"
    "/observer — capacity-rejected shadow outcomes\n"
    "/system — component health and recovery\n"
    "/telemetry — timing and evidence quality\n"
    "/guide — WR, R, states and simulation glossary\n"
    "/help — this command reference\n\n"
    "/settings — pause/resume and profile controls\n"
    "/risk — current risk budgets and caps\n"
    "/risk 0.25 — preview 0.25% per trade (0.05–1%)\n"
    "/profile — current profile and profile picker\n"
    "/profile balanced — preview a profile by name\n"
    "/pause · /resume — preview, then confirm\n"
    "Profiles: Ultra cautious · Cautious · Balanced · Aggressive · Extreme.\n\n"
    "Use Previous/Next for more records; Refresh keeps your page and symbol.\n"
    "🔎 /start only opens the dashboard.\n"
    "Research shows stored offline analysis.\n"
    "Changes require confirmation; resume remains subject to service safety checks."
)
_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,56}\Z")
_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9._/-]{0,31}\Z")
_PERCENT = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)%?\Z")
_PAGE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_UNCHANGED_EDIT_DESCRIPTIONS = {
    "Bad Request: message is not modified",
    "Bad Request: message is not modified: specified new message content and reply "
    "markup are exactly the same as a current content and reply markup of the message",
}

# Submenus have at most four actions, leaving room for paging and Home/Refresh.
_VIEW_LINKS = {
    "dashboard": (
        ("🔎 Opportunities", "opportunities"),
        ("📂 Positions", "positions"),
        ("📈 Performance", "performance"),
        ("🌐 Universe", "universe"),
        ("⚙️ System", "system"),
        ("🎛 Settings", "settings"),
    ),
    "status": (("⚙️ System", "system"), ("📡 Telemetry", "telemetry")),
    "opportunities": (
        ("💡 Why", "why"),
        ("🌊 Waves", "waves"),
        ("🌐 Universe", "universe"),
        ("📖 Guide", "guide"),
    ),
    "why": (("🌊 Waves", "waves"), ("🔎 Opportunities", "opportunities")),
    "waves": (("💡 Why", "why"), ("🔎 Opportunities", "opportunities")),
    "positions": (("📈 Performance", "performance"), ("🛡 Risk", "risk")),
    "universe": (("🔎 Opportunities", "opportunities"), ("📖 Guide", "guide")),
    "performance": (
        ("⚖️ Comparison", "comparison"),
        ("🔬 Observer", "observer"),
        ("📂 Positions", "positions"),
        ("📖 Guide", "guide"),
    ),
    "comparison": (("📈 Performance", "performance"), ("🔬 Observer", "observer")),
    "risk": (("🎛 Settings", "settings"), ("📖 Guide", "guide")),
    "profile": (("🛡 Risk", "risk"), ("🎛 Settings", "settings")),
    "research": (("⚙️ System", "system"), ("📖 Guide", "guide")),
    "observer": (("📈 Performance", "performance"), ("⚖️ Comparison", "comparison")),
    "system": (
        ("📊 Status", "status"),
        ("📡 Telemetry", "telemetry"),
        ("🧪 Research", "research"),
        ("❓ Help", "help"),
    ),
    "telemetry": (("⚙️ System", "system"), ("📖 Guide", "guide")),
    "settings": (("🛡 Risk", "risk"), ("🎚 Profiles", "profile")),
    "guide": (("❓ Help", "help"), ("⚖️ Comparison", "comparison")),
    "help": (("📖 Guide", "guide"), ("🎛 Settings", "settings")),
}


class Proposal(TypedDict):
    token: str
    summary: str


class RenderedPage(TypedDict):
    text: str
    page: int
    pages: int


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


class PaginatedTelegramService(TelegramService, Protocol):
    async def render_page(
        self, view: str, symbol: str | None = None, page: int = 0
    ) -> RenderedPage:
        """Optional: text and clamped page metadata from ONE committed snapshot.

        Includes all page headings/guidance. The controller never appends content
        or reads page counts separately. Legacy services only need snapshot().
        """
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
    """Lossless UTF-16 chunks, preferring paragraphs then complete lines."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A nonempty plain-text message is required")
    chunks = []
    start = 0
    while start < len(text):
        end, units = start, 0
        while end < len(text):
            width = 2 if ord(text[end]) > 0xFFFF else 1
            if units + width > 3900:
                break
            units += width
            end += 1
        if end < len(text):
            # Keep the delimiters: stripping a chunk would lose preview content.
            boundaries = [
                index + len(separator)
                for separator in ("\n\n", "\r\n\r\n")
                if (index := text.rfind(separator, start, end)) >= start
            ]
            boundary = (
                max(boundaries) if boundaries else text.rfind("\n", start, end) + 1
            )
            # Do not turn leading blank lines into a separate empty message.
            if boundary > start and text[start:boundary].strip():
                end = boundary
        chunks.append(text[start:end])
        start = end
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

    @staticmethod
    def _page_callback(view: str, page: int = 0, symbol: str | None = None) -> str:
        return f"pg:{view}:{page}" + (f":{symbol}" if symbol else "")

    def _keyboard(
        self, view: str, symbol: str | None = None, page: int = 0, pages: int = 1
    ) -> dict:
        button = self._button
        actions = []
        if view in ("risk", "profile"):
            actions.extend(
                [
                    button("🎚 Choose profile", "profiles"),
                    button("✏️ Custom %", "custom_risk"),
                ]
            )
        elif view == "settings":
            actions.extend(
                [
                    button("⏸ Pause", "p:paused:1"),
                    button("▶️ Resume", "p:paused:0"),
                ]
            )
        actions.extend(
            button(
                label,
                self._page_callback(
                    destination,
                    symbol=symbol if destination in ("why", "waves") else None,
                ),
            )
            for label, destination in _VIEW_LINKS[view]
        )
        rows = [actions[index : index + 2] for index in range(0, len(actions), 2)]
        navigation = []
        if page > 0:
            navigation.append(
                button("← Previous", self._page_callback(view, page - 1, symbol))
            )
        if page + 1 < pages:
            navigation.append(
                button("Next →", self._page_callback(view, page + 1, symbol))
            )
        if navigation:
            rows.append(navigation)
        rows.append(
            [
                button("🔄 Refresh", self._page_callback(view, page, symbol)),
                button("🏠 Home", self._page_callback("dashboard")),
            ]
        )
        return {"inline_keyboard": rows}

    async def _profiles(self, message: dict | None) -> None:
        # Separate chooser keeps every profile accessible even on paged reports.
        buttons = [
            self._button(PROFILE_LABELS[name], f"p:profile:{name}") for name in PROFILES
        ]
        rows = [buttons[index : index + 2] for index in range(0, len(buttons), 2)]
        rows.append(
            [
                self._button("🔄 Refresh", "profiles"),
                self._button("🏠 Home", self._page_callback("dashboard")),
            ]
        )
        await self._render(
            "🎚 Choose a risk profile\n\n"
            "Select a profile to preview its risk budget, caps and prospective effect.\n"
            "Changes require confirmation and apply to future entries; existing positions "
            "keep their protection.",
            {"inline_keyboard": rows},
            message,
        )

    async def _view(
        self, view: str, symbol: str | None, message: dict | None, page: int = 0
    ) -> None:
        render_page = getattr(self._service, "render_page", None)
        pages = 1
        if callable(render_page):
            result = await asyncio.wait_for(
                render_page(view, symbol=symbol, page=page),
                timeout=self.SERVICE_TIMEOUT,
            )
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("text"), str)
                or not result["text"].strip()
                or type(result.get("page")) is not int
                or type(result.get("pages")) is not int
                or not 0 <= result["page"] < result["pages"]
            ):
                raise ValueError("Invalid service page")
            text, page, pages = result["text"], result["page"], result["pages"]
        elif view == "help":
            text = HELP
            page = 0
        else:
            # Keep the original two-argument API for older services/fakes. A stale
            # paged button simply reopens the full report, with no paging controls.
            text = await asyncio.wait_for(
                self._service.snapshot(view, symbol),
                timeout=self.SERVICE_TIMEOUT,
            )
            page = 0
        await self._render(text, self._keyboard(view, symbol, page, pages), message)

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
        from .service import ServiceRejection

        try:
            proposal = await asyncio.wait_for(
                self._service.propose_change(key, value, actor),
                timeout=self.SERVICE_TIMEOUT,
            )
        except ServiceRejection as error:
            # The service sanitizes known pre-proposal policy rejections. Never
            # apply this exception handling to a possibly committed confirmation.
            await self._render(str(error), self._keyboard("settings"), message)
            return
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
        if (
            (parts[0] == "v" and len(parts) in (2, 3))
            or (parts[0] == "pg" and len(parts) in (3, 4))
        ) and parts[1] in VIEWS:
            page = 0
            if parts[0] == "pg":
                if not _PAGE.fullmatch(parts[2]):
                    raise _CommandError("Invalid page control. Open /dashboard.")
                page = int(parts[2])
                symbol = parts[3] if len(parts) == 4 else None
            else:
                symbol = parts[2] if len(parts) == 3 else None
            if symbol is not None and (
                parts[1] not in ("why", "waves") or not _SYMBOL.fullmatch(symbol)
            ):
                raise _CommandError("Invalid symbol control. Open /dashboard.")
            await self._view(parts[1], symbol, message, page)
        elif data == "profiles":
            await self._profiles(message)
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
                    and body.get("ok") is False
                    and type(body.get("error_code")) is int
                    and body["error_code"] == 400
                    and isinstance(body.get("description"), str)
                    and body["description"] in _UNCHANGED_EDIT_DESCRIPTIONS
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
