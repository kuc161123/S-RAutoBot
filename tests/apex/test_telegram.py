"""Offline contract/security tests. No real session, credentials, or polling."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import time
import traceback
import unittest
from collections import defaultdict, deque
from unittest.mock import patch

import aiohttp

from apex_bot.telegram import (
    COMMANDS,
    PROFILES,
    VIEWS,
    TelegramController,
    TelegramError,
)


TOKEN = "123456:FAKE_SECRET_FOR_OFFLINE_TESTS"
CHAT = "-10042"
USER = 7
ACTOR = "telegram:-10042:7"


def message(text="/dashboard", *, chat=-10042, user=USER, age=0):
    return {
        "message_id": 90,
        "chat": {"id": chat},
        "from": {"id": user},
        "text": text,
        "date": int(time.time()) - age,
    }


def callback(data, *, chat=-10042, user=USER, age=0, markup=None):
    origin = message(chat=chat, age=age)
    origin["from"] = {"id": 123456, "is_bot": True}
    if markup is not None:
        origin["reply_markup"] = copy.deepcopy(markup)
    return {"id": "query-id", "from": {"id": user}, "message": origin, "data": data}


def confirmation_keyboard(token):
    return {
        "inline_keyboard": [
            [
                {"text": "Confirm", "callback_data": f"confirm:{token}"},
                {"text": "Cancel", "callback_data": f"cancel:{token}"},
            ]
        ]
    }


class FakeResponse:
    def __init__(self, body=None, status=200, headers=None, error=None):
        self.body = body
        self.status = status
        self.headers = headers or {}
        self.error = error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class FakeSession:
    def __init__(self):
        self.calls = []
        self.responses = defaultdict(deque)
        self.updates = []
        self.message_id = 100

    def request(self, verb, url, **kwargs):
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, verb, copy.deepcopy(kwargs)))
        if self.responses[method]:
            return self.responses[method].popleft()
        if method == "getUpdates":
            result = self.updates
            self.updates = []
        elif method == "getMe":
            result = {"id": 123456, "is_bot": True, "username": "AutotradingBot222_bot"}
        elif method in ("sendMessage", "editMessageText"):
            self.message_id += 1
            result = {"message_id": self.message_id}
        else:
            result = True
        return FakeResponse({"ok": True, "result": result})

    def payloads(self, method):
        return [
            call[2].get("json", call[2].get("params"))
            for call in self.calls
            if call[0] == method
        ]

    def text_payloads(self):
        return [
            call[2]["json"]
            for call in self.calls
            if call[0] in ("sendMessage", "editMessageText")
        ]


class FakeService:
    """Models persistence in the service across replacement controller instances."""

    def __init__(self):
        self.reads = []
        self.proposals = []
        self.confirms = []
        self.cancels = []
        self.applied = []
        self.tokens = {}
        self.summary = (
            "Per trade risk: 0.25% = $2.50 at current equity $1,000.00.\n"
            "Caps: gross heat 4% ($40), bucket 2% ($20), max 4 positions.\n"
            "Prospective effect: applies to future entries; existing stops stay in place."
        )
        self.fail_snapshot = False
        self.fail_propose = False
        self.fail_confirm = False
        self.proposal_override = None

    async def snapshot(self, view, symbol=None):
        self.reads.append((view, symbol))
        if self.fail_snapshot:
            raise RuntimeError(f"https://api.telegram.org/bot{TOKEN}/getUpdates")
        if view in ("risk", "profile"):
            return "🛡 Committed risk\n" + self.summary
        return f"📊 {view} {symbol or ''} <plain> _text_ [literal]"

    async def change_setting(self, key, value, actor):
        raise AssertionError("Telegram must never bypass confirmation")

    async def propose_change(self, key, value, actor):
        self.proposals.append((key, value, actor))
        if self.fail_propose:
            raise RuntimeError(TOKEN)
        if key == "risk_pct" and not 0.05 <= value <= 1:
            raise ValueError("Outside service bounds")
        if self.proposal_override is not None:
            return self.proposal_override
        token = f"proposal_{len(self.proposals)}"
        self.tokens[token] = {
            "actor": actor,
            "expires": time.time() + 120,
            "used": False,
            "key": key,
            "value": value,
        }
        return {"token": token, "summary": self.summary}

    async def confirm_change(self, token, actor):
        self.confirms.append((token, actor))
        record = self.tokens.get(token)
        if (
            record is None
            or record["used"]
            or record["expires"] <= time.time()
            or record["actor"] != actor
        ):
            raise ValueError("Expired, consumed, unknown or wrong-actor token")
        if self.fail_confirm:
            raise RuntimeError(TOKEN)
        record["used"] = True
        self.applied.append((record["key"], record["value"], actor))
        return "✅ Confirmed by service"

    async def cancel_change(self, token, actor):
        self.cancels.append((token, actor))
        record = self.tokens.get(token)
        if not record or record["actor"] != actor or record["used"]:
            return "Preview unavailable or already handled."
        del self.tokens[token]
        return "✖️ Preview revoked. No setting change."


class TelegramTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = FakeSession()
        self.service = FakeService()
        self.controller = self.make_controller()

    def make_controller(self, *, users=None, chat=CHAT):
        controller = TelegramController(
            self.session, TOKEN, chat, {USER} if users is None else users, self.service
        )
        controller.WRITE_INTERVAL = 0
        return controller

    async def dispatch(self, *, text=None, cb=None, **kwargs):
        update = (
            {"message": message(text, **kwargs)}
            if text is not None
            else {"callback_query": cb}
        )
        await self.controller._handle_update(update)

    def assert_no_mutation(self):
        self.assertEqual(self.service.proposals, [])
        self.assertEqual(self.service.confirms, [])
        self.assertEqual(self.service.applied, [])
        self.assertEqual(self.service.cancels, [])

    async def test_constructor_has_no_io_and_initialize_verifies_then_publishes_commands(
        self,
    ):
        self.assertEqual(self.session.calls, [])
        await self.controller.initialize()
        self.assertEqual([c[0] for c in self.session.calls], ["getMe", "setMyCommands"])
        payload = self.session.payloads("setMyCommands")[0]
        self.assertEqual({c["command"] for c in payload["commands"]}, set(COMMANDS))
        self.assertEqual(payload["scope"], {"type": "chat", "chat_id": CHAT})
        self.assertFalse(
            {"live", "reset", "resetstats", "resetlifetime"} & set(COMMANDS)
        )

    async def test_addressed_commands_require_verified_exact_bot_identity(self):
        await self.dispatch(text="/profile@AutotradingBot222_bot aggressive")
        self.assertEqual(self.service.proposals, [])
        await self.controller.initialize(publish_commands=False)
        self.assertEqual([c[0] for c in self.session.calls], ["getMe"])
        await self.dispatch(text="/start@AutotradingBot222_bot resume")
        self.assertEqual(self.service.reads[-1], ("dashboard", None))
        self.assertEqual(self.service.proposals, [])
        await self.dispatch(text="/profile@AutotradingBot222_bot aggressive")
        self.assertEqual(self.service.proposals, [("profile", "aggressive", ACTOR)])
        await self.dispatch(text="/telemetry@AutotradingBot222_bot")
        self.assertEqual(self.service.reads[-1], ("telemetry", None))
        for text in (
            "/resume@OtherBot",
            "/resume@autotradingbot222_bot",
            "/resume@AutotradingBot222_bot@OtherBot",
        ):
            await self.dispatch(text=text)
        await self.dispatch(text="/resume@AutotradingBot222_bot", user=999)
        self.assertEqual(len(self.service.proposals), 1)

    async def test_getme_identity_mismatch_fails_closed_without_publishing_or_polling(
        self,
    ):
        for identity in (
            {"id": 123456, "is_bot": True, "username": "OtherBot"},
            {"id": 987654, "is_bot": True, "username": "AutotradingBot222_bot"},
            {"id": 123456, "is_bot": False, "username": "AutotradingBot222_bot"},
            None,
            {},
        ):
            self.controller = self.make_controller()
            self.session.responses["getMe"].append(
                FakeResponse({"ok": True, "result": identity})
            )
            with self.assertRaises(TelegramError):
                await self.controller.initialize()
            await self.dispatch(text="/resume")
            self.assertEqual(await self.controller.poll_once(21), 21)
            with self.assertRaises(TelegramError):
                await self.controller.send("blocked")
        self.assertEqual({c[0] for c in self.session.calls}, {"getMe"})
        self.assert_no_mutation()
        # A subsequent successful verification can recover this instance.
        await self.controller.initialize(publish_commands=False)
        await self.dispatch(text="/status@AutotradingBot222_bot")
        self.assertEqual(self.service.reads[-1], ("status", None))

    async def test_constructor_username_is_only_an_expectation_not_verification(self):
        self.controller = TelegramController(
            self.session, TOKEN, CHAT, {USER}, self.service, bot_username="@OtherBot"
        )
        self.controller.WRITE_INTERVAL = 0
        await self.dispatch(text="/resume@OtherBot")
        self.assertEqual(self.service.proposals, [])
        self.session.responses["getMe"].append(
            FakeResponse(
                {
                    "ok": True,
                    "result": {"id": 123456, "is_bot": True, "username": "OtherBot"},
                }
            )
        )
        await self.controller.initialize(publish_commands=False)
        await self.dispatch(text="/pause@OtherBot")
        self.assertEqual(self.service.proposals, [("paused", True, ACTOR)])

    async def test_every_read_command_and_start_are_read_only(self):
        for view in VIEWS:
            with self.subTest(view=view):
                await self.dispatch(text=f"/{view}")
                if view != "help":
                    self.assertEqual(self.service.reads[-1], (view, None))
        for start in ("/start", "/start resume", "/start live"):
            await self.dispatch(text=start)
            self.assertEqual(self.service.reads[-1], ("dashboard", None))
        self.assert_no_mutation()

    async def test_symbol_read_views_and_refresh_preserve_symbol(self):
        for view in ("why", "waves"):
            await self.dispatch(text=f"/{view} btcusdt")
            self.assertEqual(self.service.reads[-1], (view, "BTCUSDT"))
            markup = self.session.text_payloads()[-1]["reply_markup"]
            data = [
                b["callback_data"] for row in markup["inline_keyboard"] for b in row
            ]
            self.assertIn(f"v:{view}:BTCUSDT", data)
            await self.dispatch(cb=callback(f"v:{view}:BTCUSDT"))
            self.assertEqual(self.service.reads[-1], (view, "BTCUSDT"))

    async def test_authorization_for_every_message_command(self):
        commands = [f"/{view}" for view in VIEWS] + [
            "/start",
            "/pause",
            "/resume",
            "/risk 0.25",
            "/profile extreme",
            "/nonsense",
        ]
        for command in commands:
            for chat, user in ((1, USER), (-10042, 8), (1, 8), (-10042, True)):
                with self.subTest(command=command, chat=chat, user=user):
                    await self.dispatch(text=command, chat=chat, user=user)
        self.assertEqual(self.session.calls, [])
        self.assertEqual(self.service.reads, [])
        self.assert_no_mutation()

    async def test_authorization_for_every_callback_family(self):
        data = [f"v:{view}" for view in VIEWS] + [
            *(f"p:profile:{name}" for name in PROFILES),
            "p:paused:0",
            "p:paused:1",
            "custom_risk",
            "confirm:proposal_1",
            "cancel:proposal_1",
            "garbage",
        ]
        for action in data:
            for chat, user in ((1, USER), (-10042, 8), (1, 8), (-10042, True)):
                await self.dispatch(
                    cb=callback(
                        action,
                        chat=chat,
                        user=user,
                        markup=confirmation_keyboard("proposal_1"),
                    )
                )
        self.assertTrue(self.session.calls)
        self.assertEqual({c[0] for c in self.session.calls}, {"answerCallbackQuery"})
        self.assertTrue(
            all(
                p["text"] == "Not authorized"
                for p in self.session.payloads("answerCallbackQuery")
            )
        )
        self.assertEqual(self.service.reads, [])
        self.assert_no_mutation()

    async def test_missing_identity_sender_chat_bot_and_inline_callback_fail_closed(
        self,
    ):
        messages = [message(), message(), message(), message(), message()]
        messages[0].pop("from")
        messages[1].pop("chat")
        messages[2]["sender_chat"] = {"id": -10042}
        messages[3]["from"]["is_bot"] = True
        messages[4]["chat"]["id"] = CHAT  # Incoming IDs must be actual integers.
        for msg in messages:
            await self.controller._handle_update({"message": msg})
        cb = callback("p:paused:0")
        cb.pop("message")
        cb["inline_message_id"] = "inline"
        await self.dispatch(cb=cb)
        self.assertEqual(self.service.reads, [])
        self.assert_no_mutation()
        self.assertEqual({c[0] for c in self.session.calls}, {"answerCallbackQuery"})

    async def test_empty_or_invalid_auth_disables_poll_send_and_initialize(self):
        for users, chat in (
            (set(), CHAT),
            ({True, 0, -1}, CHAT),
            ({USER}, ""),
            ({USER}, "@channel"),
            ({USER}, "0"),
        ):
            self.controller = self.make_controller(users=users, chat=chat)
            await self.dispatch(text="/risk 0.25")
            await self.dispatch(text="/status")
            self.assertEqual(await self.controller.poll_once(12), 12)
            with self.assertRaises(TelegramError):
                await self.controller.send("test")
            with self.assertRaises(TelegramError):
                await self.controller.initialize()
        self.assertEqual(self.session.calls, [])
        self.assertEqual(self.service.reads, [])
        self.assert_no_mutation()

    async def test_risk_menu_lists_every_profile_and_authoritative_values(self):
        for command in ("/risk", "/profile"):
            await self.dispatch(text=command)
            payload = self.session.text_payloads()[-1]
            self.assertIn(self.service.summary, payload["text"])
            for name in PROFILES:
                self.assertIn(name, payload["text"])
                self.assertIn(
                    {"text": name, "callback_data": f"p:profile:{name}"},
                    [
                        b
                        for row in payload["reply_markup"]["inline_keyboard"]
                        for b in row
                    ],
                )
        self.assert_no_mutation()

    async def test_all_mutations_preview_exact_service_summary_before_confirm(self):
        commands = [(f"/profile {name}", "profile", name) for name in PROFILES]
        commands += [
            ("/risk 0.25", "risk_pct", 0.25),
            ("/risk .05%", "risk_pct", 0.05),
            ("/risk 1%", "risk_pct", 1.0),
            ("/pause", "paused", True),
            ("/resume", "paused", False),
        ]
        for command, key, value in commands:
            await self.dispatch(text=command)
            self.assertEqual(self.service.proposals[-1], (key, value, ACTOR))
            payload = self.session.text_payloads()[-1]
            self.assertIn(self.service.summary, payload["text"])
            actions = [
                b["callback_data"]
                for row in payload["reply_markup"]["inline_keyboard"]
                for b in row
            ]
            self.assertEqual(
                actions,
                [
                    f"confirm:proposal_{len(self.service.proposals)}",
                    f"cancel:proposal_{len(self.service.proposals)}",
                ],
            )
        self.assertEqual(self.service.applied, [])
        self.assertEqual(self.service.confirms, [])

    async def test_confirm_uses_persistent_service_token_across_controller_restart(
        self,
    ):
        await self.dispatch(text="/profile aggressive")
        markup = self.session.text_payloads()[-1]["reply_markup"]
        self.controller = self.make_controller()
        await self.dispatch(cb=callback("confirm:proposal_1", markup=markup))
        self.assertEqual(self.service.applied, [("profile", "aggressive", ACTOR)])
        self.assertIn("Confirmed", self.session.text_payloads()[-1]["text"])
        # A replay with even the original markup cannot apply a second time.
        await self.dispatch(cb=callback("confirm:proposal_1", markup=markup))
        self.assertEqual(len(self.service.applied), 1)
        self.assertIn("Unable to verify", self.session.text_payloads()[-1]["text"])

    async def test_expired_unknown_wrong_actor_and_consumed_tokens_fail_closed(self):
        await self.dispatch(text="/resume")
        self.service.tokens["proposal_1"]["expires"] = 0
        for token in ("proposal_1", "unknown"):
            await self.dispatch(
                cb=callback(f"confirm:{token}", markup=confirmation_keyboard(token))
            )
        await self.dispatch(text="/pause")
        self.controller = self.make_controller(users={USER, 8})
        await self.dispatch(
            cb=callback(
                "confirm:proposal_2", user=8, markup=confirmation_keyboard("proposal_2")
            )
        )
        self.assertEqual(self.service.applied, [])
        self.assertEqual(self.service.confirms[-1], ("proposal_2", "telegram:-10042:8"))

    async def test_confirm_without_actual_matching_preview_never_calls_service(self):
        for markup in (
            None,
            {},
            {"inline_keyboard": []},
            confirmation_keyboard("different"),
        ):
            await self.dispatch(cb=callback("confirm:proposal_1", markup=markup))
        self.assert_no_mutation()

    async def test_cancel_revokes_token_and_removes_controls_without_setting_change(
        self,
    ):
        await self.dispatch(text="/pause")
        await self.dispatch(
            cb=callback("cancel:proposal_1", markup=confirmation_keyboard("proposal_1"))
        )
        payload = self.session.text_payloads()[-1]
        self.assertIn("revoked", payload["text"])
        self.assertEqual(self.service.cancels, [("proposal_1", ACTOR)])
        self.assertNotIn("proposal_1", self.service.tokens)
        self.assertNotIn("confirm:", json.dumps(payload["reply_markup"]))
        self.assertEqual(self.service.confirms, [])
        self.assertEqual(self.service.applied, [])
        # Telegram's updated message markup cannot be used to confirm the old token.
        await self.dispatch(
            cb=callback("confirm:proposal_1", markup=payload["reply_markup"])
        )
        self.assertEqual(self.service.confirms, [])
        await self.dispatch(
            cb=callback(
                "confirm:proposal_1", markup=confirmation_keyboard("proposal_1")
            )
        )
        self.assertEqual(self.service.applied, [])

    async def test_cancel_service_failure_never_claims_revocation(self):
        await self.dispatch(text="/pause")
        with patch.object(
            self.service, "cancel_change", side_effect=RuntimeError(TOKEN)
        ):
            await self.dispatch(
                cb=callback(
                    "cancel:proposal_1", markup=confirmation_keyboard("proposal_1")
                )
            )
        text = self.session.text_payloads()[-1]["text"]
        self.assertIn("Unable to verify", text)
        self.assertNotIn("revoked", text)
        self.assertIn("proposal_1", self.service.tokens)

    async def test_old_missing_future_and_forwarded_mutating_messages_rejected(self):
        for command in ("/pause", "/resume", "/risk 0.25", "/profile aggressive"):
            for age in (121, 10000, -60):
                await self.dispatch(text=command, age=age)
            msg = message(command)
            msg.pop("date")
            await self.controller._handle_update({"message": msg})
            msg["date"] = int(time.time())
            msg["forward_origin"] = {"type": "user"}
            await self.controller._handle_update({"message": msg})
        self.assert_no_mutation()
        await self.dispatch(text="/start", age=99999)
        self.assertEqual(self.service.reads[-1], ("dashboard", None))

    async def test_freshness_boundary_and_edited_callback_preview(self):
        with patch("apex_bot.telegram.time.time", return_value=1000):
            for age in (120, 121):
                await self.dispatch(text="/pause", age=age)
            self.assertEqual(len(self.service.proposals), 1)
            cb = callback("p:profile:extreme", age=500)
            cb["message"]["edit_date"] = 995
            await self.dispatch(cb=cb)
            self.assertEqual(self.service.proposals[-1], ("profile", "extreme", ACTOR))

    async def test_stale_mutating_callbacks_acknowledged_but_not_applied(self):
        for data in (
            "p:profile:extreme",
            "p:paused:0",
            "p:paused:1",
            "confirm:proposal_1",
        ):
            await self.dispatch(
                cb=callback(data, age=121, markup=confirmation_keyboard("proposal_1"))
            )
        self.assert_no_mutation()
        self.assertEqual(len(self.session.payloads("answerCallbackQuery")), 4)
        await self.dispatch(cb=callback("v:risk", age=90000))
        self.assertEqual(self.service.reads[-1], ("risk", None))

    async def test_malformed_commands_and_forbidden_controls_never_mutate(self):
        commands = [
            "/risk nan",
            "/risk inf",
            "/risk -1",
            "/risk 0",
            "/risk 0.04999",
            "/risk 1.0001",
            "/risk 1e-1",
            "/risk $5",
            "/risk 0.25 extra",
            "/risk 0.05%%",
            "/risk +0.1",
            "/profile wild",
            "/profile balanced extra",
            "/pause now",
            "/resume live",
            "/status extra",
            "/why BTC ETH",
            "/why <b>",
            "/dashboard x",
            "/setbalance 500",
            "/resetstats",
            "/resetlifetime confirm",
            "/live",
            "/mode live",
            "/setregime favorable",
            "/" + "x" * 300,
        ]
        for command in commands:
            with self.subTest(command=command):
                await self.dispatch(text=command)
        self.assert_no_mutation()
        self.assertEqual(self.service.reads, [])
        for text in ("ordinary text", "/resume@OtherBot", "/status@OtherBot"):
            await self.dispatch(text=text)
        self.assert_no_mutation()

    async def test_dashboard_menu_exposes_why_and_uses_read_only_callback(self):
        await self.dispatch(text="/dashboard")
        markup = self.session.text_payloads()[-1]["reply_markup"]
        data = [b["callback_data"] for row in markup["inline_keyboard"] for b in row]
        self.assertIn("v:why", data)
        before = len(self.session.payloads("answerCallbackQuery"))
        await self.dispatch(cb=callback("v:why", markup=markup))
        self.assertEqual(self.service.reads[-1], ("why", None))
        self.assertEqual(len(self.session.payloads("answerCallbackQuery")), before + 1)
        self.assert_no_mutation()

    async def test_universe_command_dashboard_button_help_and_refresh_are_read_only(
        self,
    ):
        self.assertIn("universe", COMMANDS)
        await self.dispatch(text="/dashboard")
        markup = self.session.text_payloads()[-1]["reply_markup"]
        buttons = [b for row in markup["inline_keyboard"] for b in row]
        self.assertTrue(
            any(
                b["callback_data"] == "v:universe" and "Universe" in b["text"]
                for b in buttons
            )
        )
        await self.dispatch(cb=callback("v:universe", age=90000, markup=markup))
        self.assertEqual(self.service.reads[-1], ("universe", None))
        self.assertEqual(self.session.calls[-2][0], "answerCallbackQuery")
        self.assertEqual(self.session.calls[-1][0], "editMessageText")
        markup = self.session.text_payloads()[-1]["reply_markup"]
        controls = [
            b["callback_data"] for row in markup["inline_keyboard"] for b in row
        ]
        self.assertEqual(controls, ["v:universe", "v:dashboard"])
        await self.dispatch(cb=callback("v:universe", markup=markup))
        await self.dispatch(text="/universe", age=90000)
        self.assertEqual(self.service.reads[-3:], [("universe", None)] * 3)
        await self.dispatch(text="/help")
        self.assertIn("/universe", self.session.text_payloads()[-1]["text"])
        self.assertIn("read-only", self.session.text_payloads()[-1]["text"])
        self.assertIn(
            "Research shows stored offline analysis",
            self.session.text_payloads()[-1]["text"],
        )
        self.assert_no_mutation()

    async def test_universe_rejects_manual_selection_commands_and_callbacks(self):
        for command in (
            "/universe rotate",
            "/universe BTCUSDT",
            "/universe add BTCUSDT",
        ):
            await self.dispatch(text=command)
        for data in ("v:universe:BTCUSDT", "p:universe:rotate", "p:universe:BTCUSDT"):
            await self.dispatch(cb=callback(data))
        self.assertEqual(self.service.reads, [])
        self.assert_no_mutation()

    async def test_universe_all_fifty_symbols_survive_command_and_long_callback_render(
        self,
    ):
        symbols = [f"COIN{i:02}USDT" for i in range(50)]
        listing = "\n".join(" · ".join(symbols[i : i + 2]) for i in range(0, 50, 2))
        short = "🌐 Universe · read-only\n50/50 active\n" + listing
        for long in (False, True):
            text = short + (
                "\nPolicy: " + "stored policy detail " * 250 if long else ""
            )
            self.session.calls.clear()
            with patch.object(self.service, "snapshot", return_value=text):
                if long:
                    await self.dispatch(cb=callback("v:universe"))
                else:
                    await self.dispatch(text="/universe")
            payloads = self.session.text_payloads()
            rendered = "".join(p["text"] for p in payloads)
            self.assertEqual(rendered, text)
            for symbol in symbols:
                self.assertIn(symbol, rendered)
            for payload in payloads:
                self.assertLessEqual(
                    len(payload["text"].encode("utf-16-le")) // 2, 3900
                )
                self.assertNotIn("parse_mode", payload)
            self.assertEqual(len(payloads) > 1, long)
            self.assertEqual(
                payloads[-1]["reply_markup"], self.controller._keyboard("universe")
            )
        self.assert_no_mutation()

    async def test_every_generated_callback_has_an_acknowledged_handler(self):
        # Collect all unique buttons exposed by every view and symbol refresh.
        controls = {}
        for view, symbol in [(v, None) for v in VIEWS] + [
            ("why", "BTCUSDT"),
            ("waves", "ETHUSDT"),
        ]:
            await self.controller._view(view, symbol, None)
            markup = self.session.text_payloads()[-1]["reply_markup"]
            for row in markup["inline_keyboard"]:
                for button in row:
                    controls[button["callback_data"]] = markup
        for data, markup in controls.items():
            with self.subTest(data=data):
                before = len(self.session.payloads("answerCallbackQuery"))
                await self.dispatch(cb=callback(data, markup=markup))
                self.assertEqual(
                    len(self.session.payloads("answerCallbackQuery")), before + 1
                )
                self.assertNotIn(
                    "Unknown control", self.session.text_payloads()[-1]["text"]
                )
                self.assertNotIn(
                    "Unable to verify", self.session.text_payloads()[-1]["text"]
                )
                self.assertLessEqual(len(data.encode("utf-8")), 64)
        # Confirm/cancel are dynamically generated and tested with separate tokens.
        for action in ("confirm", "cancel"):
            await self.dispatch(text="/profile cautious")
            markup = self.session.text_payloads()[-1]["reply_markup"]
            data = next(
                b["callback_data"]
                for row in markup["inline_keyboard"]
                for b in row
                if b["callback_data"].startswith(action + ":")
            )
            before = len(self.session.payloads("answerCallbackQuery"))
            await self.dispatch(cb=callback(data, markup=markup))
            self.assertEqual(
                len(self.session.payloads("answerCallbackQuery")), before + 1
            )
            self.assertNotIn(
                "Unable to verify", self.session.text_payloads()[-1]["text"]
            )

    async def test_malformed_callbacks_and_updates_do_not_crash_or_mutate(self):
        for data in (
            None,
            {},
            [],
            "",
            "unknown",
            "v:live",
            "v:risk:BTC",
            "v:why:<bad>",
            "p:profile:evil",
            "p:paused:2",
            "p:mode:live",
            "confirm:",
            "confirm:a:b",
            "cancel:",
            "confirm:" + "x" * 57,
            "🌊" * 40,
        ):
            await self.dispatch(cb=callback(data))
        for update in (
            {},
            {"message": None},
            {"message": []},
            {"callback_query": []},
            {"callback_query": {}},
            {"message": {"chat": [], "from": 4}},
            {"edited_message": message("/resume")},
        ):
            await self.controller._handle_update(update)
        self.assert_no_mutation()
        self.assertEqual(self.service.reads, [])

    async def test_invalid_service_proposal_never_emits_confirm_button(self):
        for proposal in (
            {},
            {"token": "bad token", "summary": "x"},
            {"token": "x" * 57, "summary": "x"},
            {"token": "abc", "summary": ""},
            {"token": "abc", "summary": 2},
        ):
            self.service.proposal_override = proposal
            await self.dispatch(text="/profile balanced")
            self.assertNotIn("confirm:", json.dumps(self.session.text_payloads()[-1]))
        self.assertEqual(self.service.applied, [])

    async def test_service_errors_are_sanitized_and_do_not_report_false_success(self):
        self.service.fail_snapshot = True
        await self.dispatch(text="/status")
        self.service.fail_propose = True
        await self.dispatch(text="/pause")
        self.service.fail_propose = False
        await self.dispatch(text="/pause")
        self.service.fail_confirm = True
        await self.dispatch(
            cb=callback(
                "confirm:proposal_2", markup=confirmation_keyboard("proposal_2")
            )
        )
        for payload in self.session.text_payloads():
            self.assertNotIn(TOKEN, payload["text"])
        self.assertIn("Unable to verify", self.session.text_payloads()[-1]["text"])
        self.assertEqual(self.service.applied, [])

    async def test_poll_get_timeout_allowed_types_cursor_order_and_duplicate_ids(self):
        self.session.updates = [
            {"update_id": 12, "message": message("/positions")},
            {"update_id": 9, "message": message("/resume")},
            {"update_id": 11, "message": message("/status")},
            {"update_id": 11, "message": message("/pause")},
            {"update_id": 13, "message": message("/resume", user=999)},
            {"update_id": 14, "edited_message": message("/resume")},
            {"update_id": True, "message": message("/resume")},
            None,
            {"update_id": "15"},
        ]
        self.assertEqual(await self.controller.poll_once(10), 15)
        self.assertEqual(self.service.reads, [("status", None), ("positions", None)])
        self.assert_no_mutation()
        method, verb, kwargs = self.session.calls[0]
        self.assertEqual((method, verb), ("getUpdates", "GET"))
        self.assertEqual(kwargs["params"]["offset"], 10)
        self.assertEqual(kwargs["params"]["timeout"], 20)
        self.assertEqual(
            json.loads(kwargs["params"]["allowed_updates"]),
            ["message", "callback_query"],
        )
        self.assertEqual(kwargs["timeout"].total, 30)
        self.assertEqual(kwargs["timeout"].connect, 5)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(await self.controller.poll_once(15), 15)

    async def test_poll_continues_after_bad_commands_and_service_failures(self):
        self.service.fail_snapshot = True
        self.session.updates = [
            {"update_id": i, "message": message(text)}
            for i, text in enumerate(("/risk NaN", "/status", "/help"), 1)
        ]
        self.assertEqual(await self.controller.poll_once(1), 4)
        self.assertEqual(len(self.session.payloads("sendMessage")), 3)
        self.assertIn("Apex controls", self.session.text_payloads()[-1]["text"])

    async def test_poll_failure_exposes_only_completed_cursor_for_parent(self):
        self.session.updates = [
            {"update_id": 5, "message": message("/status", user=888)},
            {"update_id": 6, "message": message("/status")},
        ]
        self.session.responses["sendMessage"].append(
            FakeResponse(error=TimeoutError(TOKEN))
        )
        with self.assertRaises(TelegramError) as caught:
            await self.controller.poll_once(5)
        self.assertEqual(caught.exception.next_offset, 6)
        self.assertTrue(caught.exception.uncertain)
        self.assertNotIn(TOKEN, str(caught.exception))

    async def test_poll_invalid_offsets_and_invalid_result(self):
        for offset in (-1, True, 0.5, "2"):
            with self.assertRaises(ValueError):
                await self.controller.poll_once(offset)
        self.session.responses["getUpdates"].append(
            FakeResponse({"ok": True, "result": {}})
        )
        with self.assertRaises(TelegramError):
            await self.controller.poll_once(0)

    async def test_poll_lock_prevents_concurrent_get_updates(self):
        active = maximum = 0

        async def request(method, payload, **kwargs):
            nonlocal active, maximum
            self.assertEqual(method, "getUpdates")
            active += 1
            maximum = max(active, maximum)
            await asyncio.sleep(0.001)
            active -= 1
            return []

        with patch.object(self.controller, "_request", side_effect=request):
            await asyncio.gather(
                self.controller.poll_once(0), self.controller.poll_once(0)
            )
        self.assertEqual(maximum, 1)

    async def test_plaintext_long_unicode_chunks_lossless_keyboard_only_last(self):
        text = "🌊" * 2000 + "\n\n" + "<b> & _markdown_ [literal] " * 400
        keyboard = {
            "inline_keyboard": [[{"text": "Home", "callback_data": "v:dashboard"}]]
        }
        last_id = await self.controller.send(text, keyboard)
        payloads = self.session.payloads("sendMessage")
        self.assertGreater(len(payloads), 2)
        self.assertEqual("".join(p["text"] for p in payloads), text)
        for payload in payloads:
            self.assertLessEqual(len(payload["text"].encode("utf-16-le")) // 2, 3900)
            self.assertNotIn("parse_mode", payload)
            self.assertNotIn("entities", payload)
            self.assertEqual(payload["chat_id"], CHAT)
            self.assertTrue(payload["link_preview_options"]["is_disabled"])
        self.assertTrue(all("reply_markup" not in p for p in payloads[:-1]))
        self.assertEqual(payloads[-1]["reply_markup"], keyboard)
        self.assertEqual(last_id, str(100 + len(payloads)))

    async def test_long_preview_entire_summary_precedes_confirmation(self):
        self.service.summary *= 50
        await self.dispatch(text="/risk 0.25")
        payloads = self.session.payloads("sendMessage")
        self.assertIn(self.service.summary, "".join(p["text"] for p in payloads))
        self.assertTrue(all("reply_markup" not in p for p in payloads[:-1]))
        self.assertIn("confirm:proposal_1", json.dumps(payloads[-1]["reply_markup"]))

    async def test_sends_serialized_chunks_and_rate_limits_all_writes(self):
        self.controller.WRITE_INTERVAL = 0.01
        waits = []
        real_sleep = asyncio.sleep

        async def sleep(delay):
            waits.append(delay)
            await real_sleep(0)

        with patch("apex_bot.telegram.asyncio.sleep", side_effect=sleep):
            await asyncio.gather(
                self.controller.send("A" * 8000), self.controller.send("B" * 5000)
            )
        payloads = self.session.payloads("sendMessage")
        self.assertEqual([p["text"][0] for p in payloads], ["A", "A", "A", "B", "B"])
        self.assertEqual(len(waits), 5)
        self.assertTrue(all(delay > 0 for delay in waits[1:]))

    async def test_callback_edits_view_in_place_and_acks_before_service_work(self):
        await self.dispatch(cb=callback("v:status"))
        self.assertEqual(
            [call[0] for call in self.session.calls],
            ["answerCallbackQuery", "editMessageText"],
        )
        self.assertEqual(self.session.payloads("editMessageText")[0]["message_id"], 90)
        self.assertEqual(self.session.payloads("sendMessage"), [])

    async def test_long_callback_edit_clears_old_keyboard_before_remaining_chunks(self):
        self.service.summary *= 40
        await self.dispatch(cb=callback("p:profile:extreme"))
        edit = self.session.payloads("editMessageText")[0]
        sends = self.session.payloads("sendMessage")
        self.assertEqual(edit["reply_markup"], {"inline_keyboard": []})
        self.assertIn(
            self.service.summary, edit["text"] + "".join(p["text"] for p in sends)
        )
        self.assertIn("confirm:proposal_1", json.dumps(sends[-1]["reply_markup"]))

    async def test_unchanged_edit_is_success_and_explicit_uneditable_falls_back(self):
        self.session.responses["editMessageText"].append(
            FakeResponse(
                {
                    "ok": False,
                    "description": "Bad Request: message is not modified",
                    "error_code": 400,
                },
                400,
            )
        )
        await self.dispatch(cb=callback("v:status"))
        self.assertEqual(self.session.payloads("sendMessage"), [])
        self.session.responses["editMessageText"].append(
            FakeResponse(
                {
                    "ok": False,
                    "description": "Bad Request: message can't be edited",
                    "error_code": 400,
                },
                400,
            )
        )
        await self.dispatch(cb=callback("v:status"))
        self.assertEqual(len(self.session.payloads("sendMessage")), 1)

    async def test_edit_timeout_does_not_fallback_or_retry(self):
        self.session.responses["editMessageText"].append(
            FakeResponse(error=TimeoutError(TOKEN))
        )
        with self.assertRaises(TelegramError):
            await self.dispatch(cb=callback("v:status"))
        self.assertEqual(self.session.payloads("sendMessage"), [])
        self.assertEqual(len(self.session.payloads("editMessageText")), 1)

    async def test_failed_callback_ack_prevents_service_execution(self):
        for status in (400, 429):
            self.session.responses["answerCallbackQuery"].append(
                FakeResponse(
                    {"ok": False, "error_code": status, "description": TOKEN}, status
                )
            )
            if status == 400:
                await self.dispatch(cb=callback("p:paused:0"))
            else:
                with self.assertRaises(TelegramError):
                    await self.dispatch(cb=callback("p:paused:0"))
        self.assert_no_mutation()

    async def test_retry_after_is_explicit_no_retry_and_retained_cooldown(self):
        self.session.responses["sendMessage"].append(
            FakeResponse(
                {
                    "ok": False,
                    "error_code": 429,
                    "parameters": {"retry_after": 17},
                    "description": TOKEN,
                },
                429,
            )
        )
        before = time.monotonic()
        with self.assertRaises(TelegramError) as caught:
            await self.controller.send("notification")
        self.assertEqual(caught.exception.retry_after, 17)
        self.assertEqual(caught.exception.status, 429)
        self.assertFalse(caught.exception.uncertain)
        self.assertGreaterEqual(self.controller._next_write, before + 17)
        self.assertEqual(len(self.session.payloads("sendMessage")), 1)
        self.assertNotIn(TOKEN, str(caught.exception))

    async def test_retry_after_header_and_invalid_json_error(self):
        self.session.responses["sendMessage"].append(
            FakeResponse(ValueError(TOKEN), 429, headers={"Retry-After": "3"})
        )
        with self.assertRaises(TelegramError) as caught:
            await self.controller.send("notification")
        self.assertEqual(caught.exception.retry_after, 3)
        self.assertEqual(caught.exception.status, 429)
        self.assertNotIn(TOKEN, str(caught.exception))

    async def test_transport_exception_traceback_and_body_never_expose_token(self):
        for response in (
            FakeResponse(error=aiohttp.ClientError(f"URL/bot{TOKEN}/sendMessage")),
            FakeResponse({"ok": False, "error_code": 500, "description": TOKEN}, 500),
            FakeResponse(ValueError(TOKEN), 502),
        ):
            self.session.responses["sendMessage"].append(response)
            try:
                await self.controller.send("notification")
            except TelegramError as error:
                rendered = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
                self.assertNotIn(TOKEN, rendered)
                self.assertNotIn(TOKEN, repr(error))
                self.assertTrue(error.uncertain)
            else:
                self.fail("Expected sanitized transport failure")
        await self.controller.send(f"Unexpected service text {TOKEN}")
        self.assertNotIn(TOKEN, self.session.payloads("sendMessage")[-1]["text"])

    async def test_partial_chunk_delivery_exposes_confirmed_ids_for_outbox(self):
        self.session.responses["sendMessage"].extend(
            [
                FakeResponse({"ok": True, "result": {"message_id": 111}}),
                FakeResponse(error=TimeoutError(TOKEN)),
            ]
        )
        with self.assertRaises(TelegramError) as caught:
            await self.controller.send("x" * 8000)
        self.assertEqual(caught.exception.sent_message_ids, ("111",))
        self.assertTrue(caught.exception.uncertain)
        self.assertEqual(len(self.session.payloads("sendMessage")), 2)

    async def test_invalid_message_id_response_is_uncertain_and_no_false_ack(self):
        self.session.responses["sendMessage"].append(
            FakeResponse({"ok": True, "result": {}})
        )
        with self.assertRaises(TelegramError) as caught:
            await self.controller.send("x")
        self.assertTrue(caught.exception.uncertain)
        self.assertEqual(caught.exception.sent_message_ids, ())

    async def test_invalid_api_envelope_is_uncertain_and_bad_retry_after_ignored(self):
        for body in ([], {"unexpected": "response"}, {"ok": "true"}):
            self.session.responses["sendMessage"].append(FakeResponse(body))
            with self.assertRaises(TelegramError) as caught:
                await self.controller.send("x")
            self.assertTrue(caught.exception.uncertain)
        for retry_after in ("nan", "inf", -1, {}, "invalid"):
            self.session.responses["sendMessage"].append(
                FakeResponse(
                    {
                        "ok": False,
                        "error_code": 429,
                        "parameters": {"retry_after": retry_after},
                    },
                    429,
                )
            )
            with self.assertRaises(TelegramError) as caught:
                await self.controller.send("x")
            self.assertIsNone(caught.exception.retry_after)

    async def test_missing_confirmation_api_has_no_direct_mutation_fallback(self):
        self.service.propose_change = None
        await self.dispatch(text="/risk 0.25")
        self.assertIn("Unable to verify", self.session.text_payloads()[-1]["text"])
        self.assertNotIn("confirm:", json.dumps(self.session.text_payloads()[-1]))
        self.assert_no_mutation()

    async def test_committed_change_with_failed_response_does_not_claim_no_change(self):
        await self.dispatch(text="/pause")
        confirm = self.service.confirm_change

        async def uncertain_confirm(token, actor):
            await confirm(token, actor)
            raise TimeoutError(TOKEN)

        with patch.object(
            self.service, "confirm_change", side_effect=uncertain_confirm
        ):
            await self.dispatch(
                cb=callback(
                    "confirm:proposal_1", markup=confirmation_keyboard("proposal_1")
                )
            )
        self.assertEqual(len(self.service.applied), 1)
        text = self.session.text_payloads()[-1]["text"]
        self.assertIn("Unable to verify", text)
        self.assertNotIn("No change", text)
        await self.dispatch(
            cb=callback(
                "confirm:proposal_1", markup=confirmation_keyboard("proposal_1")
            )
        )
        self.assertEqual(len(self.service.applied), 1)

    async def test_local_store_contract_persists_tokens_cursor_and_outbox(self):
        # Explicit isolated SQLite fixture, never ambient DATABASE_URL or real APIs.
        from apex_bot.storage import Store

        class StoredService(FakeService):
            def __init__(self, store):
                super().__init__()
                self.store = store

            async def propose_change(self, key, value, actor):
                def propose(tx):
                    tx.state["confirmations"]["durable_token"] = {
                        "key": key,
                        "value": value,
                        "actor": actor,
                        "expires": time.time() + 120,
                        "used": False,
                    }
                    return {"token": "durable_token", "summary": self.summary}

                return await self.store.update(propose)

            async def confirm_change(self, token, actor):
                def confirm(tx):
                    proposal = tx.state["confirmations"].get(token)
                    if (
                        not proposal
                        or proposal["actor"] != actor
                        or proposal["used"]
                        or proposal["expires"] <= time.time()
                    ):
                        raise ValueError("Invalid confirmation")
                    proposal["used"] = True
                    tx.state["settings"][proposal["key"]] = proposal["value"]
                    tx.state["settings_version"] += 1
                    tx.event(
                        "change:" + token,
                        "setting_changed",
                        {"actor": actor},
                        "Setting changed",
                    )
                    return "✅ Applied once"

                return await self.store.update(confirm)

        with tempfile.TemporaryDirectory(prefix="apex-telegram-test-") as directory:
            path = directory + "/isolated.sqlite"
            store = Store(database_url="", sqlite_path=path)
            await store.initialize()
            self.assertTrue(await store.lease())
            try:
                self.service = StoredService(store)
                self.controller = self.make_controller()
                await self.dispatch(text="/pause")
                self.assertFalse((await store.read())["settings"]["paused"])
            finally:
                await store.close()
            store = Store(database_url="", sqlite_path=path)
            await store.initialize()
            self.assertTrue(await store.lease())
            try:
                self.service = StoredService(store)
                self.controller = self.make_controller()
                self.session.updates = [
                    {
                        "update_id": 12,
                        "callback_query": callback(
                            "confirm:durable_token",
                            markup=confirmation_keyboard("durable_token"),
                        ),
                    }
                ]
                offset = await self.controller.poll_once(0)
                # This is the parent's checkpoint responsibility, not adapter I/O.
                await store.update(lambda tx: tx.state.update(telegram_offset=offset))
                state = await store.read()
                self.assertEqual(state["telegram_offset"], 13)
                self.assertTrue(state["settings"]["paused"])
                self.assertEqual(state["settings_version"], 1)
                await self.dispatch(
                    cb=callback(
                        "confirm:durable_token",
                        markup=confirmation_keyboard("durable_token"),
                    )
                )
                self.assertEqual((await store.read())["settings_version"], 1)
                pending = await store.pending_notifications()
                self.assertEqual(len(pending), 1)
                msg_id = await self.controller.send(pending[0]["text"])
                await store.notification_result(pending[0]["key"], message_id=msg_id)
                self.assertEqual(await store.pending_notifications(), [])
            finally:
                await store.close()

    async def test_service_timeout_returns_safe_error_and_cancellation_propagates(self):
        async def slow_snapshot(*args):
            await asyncio.sleep(10)

        self.controller.SERVICE_TIMEOUT = 0.001
        with patch.object(self.service, "snapshot", side_effect=slow_snapshot):
            await self.dispatch(text="/status")
        self.assertIn("Unable to verify", self.session.text_payloads()[-1]["text"])
        self.session.responses["sendMessage"].append(
            FakeResponse(error=asyncio.CancelledError())
        )
        with self.assertRaises(asyncio.CancelledError):
            await self.controller.send("test")

    async def test_real_service_controls_revocation_and_hard_halt_via_telegram(self):
        from apex_bot.config import Config
        from apex_bot.service import Service
        from apex_bot.storage import Store

        with tempfile.TemporaryDirectory(prefix="apex-telegram-service-") as directory:
            path = directory + "/isolated.sqlite"
            store = Store(database_url="", sqlite_path=path)
            await store.initialize()
            self.assertTrue(await store.lease(ttl=10000))
            config = Config(
                mode="live",
                telegram_token=TOKEN,
                telegram_chat_id=CHAT,
                telegram_user_ids=frozenset({USER}),
                live_enabled=True,
                migration_verified=True,
            )
            try:
                now = time.time()
                await store.update(
                    lambda tx: tx.state.update(
                        account={
                            "equity": 2000,
                            "as_of": now,
                            "positions": [],
                            "blockers": [],
                        },
                        context={
                            "as_of": now,
                            "expires_at": now + 600,
                            "data_complete": True,
                            "event_blackout": False,
                            "risk_state": "neutral",
                            "long_multiplier": 1,
                            "short_multiplier": 1,
                        },
                        health={"scan_at": now},
                        risk_circuits={
                            "live": {
                                "as_of": now,
                                "daily_loss_pct": 0,
                                "weekly_loss_pct": 0,
                                "drawdown_pct": 0,
                            }
                        },
                        risk_reference_equities={"live": 2000},
                    )
                )
                self.service = Service(config, store)
                self.controller = self.make_controller()
                await self.controller.initialize(publish_commands=False)
                await self.dispatch(text="/risk@AutotradingBot222_bot 0.75")
                preview = self.session.text_payloads()[-1]
                self.assertIn("0.25% (5.00 USDT) → 0.75% (15.00 USDT)", preview["text"])
                self.assertIn("Portfolio heat", preview["text"])
                self.assertIsNone((await store.read())["settings"]["risk_pct"])
                markup = preview["reply_markup"]
                token = markup["inline_keyboard"][0][0]["callback_data"].split(":", 1)[
                    1
                ]
                await self.dispatch(cb=callback("cancel:" + token, markup=markup))
                self.assertNotIn(token, (await store.read())["confirmations"])
                # Replacement Store + controller, with the original queued markup.
                await store.close()
                store = Store(database_url="", sqlite_path=path)
                await store.initialize()
                self.assertTrue(await store.lease(ttl=10000))
                self.service = Service(config, store)
                self.controller = self.make_controller()
                await self.dispatch(cb=callback("confirm:" + token, markup=markup))
                self.assertIsNone((await store.read())["settings"]["risk_pct"])
                await self.dispatch(text="/pause")
                markup = self.session.text_payloads()[-1]["reply_markup"]
                confirm = markup["inline_keyboard"][0][0]["callback_data"]
                await self.dispatch(cb=callback(confirm, markup=markup))
                self.assertTrue((await store.read())["settings"]["paused"])
                await self.dispatch(text="/start")
                self.assertTrue((await store.read())["settings"]["paused"])
                await self.dispatch(text="/resume")
                markup = self.session.text_payloads()[-1]["reply_markup"]
                confirm = markup["inline_keyboard"][0][0]["callback_data"]
                await store.update(lambda tx: tx.state.update(risk={"hard_halt": True}))
                await self.dispatch(cb=callback(confirm, markup=markup))
                self.assertTrue((await store.read())["settings"]["paused"])
                self.assertIn(
                    "Resume blocked", self.session.text_payloads()[-1]["text"]
                )
                self.assertEqual((await store.read())["settings_version"], 1)
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
