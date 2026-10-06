"""Offline navigation/page contract and transport regressions."""

import asyncio
import re
import unittest
from unittest.mock import patch

from apex_bot.service import ServiceRejection
from apex_bot.telegram import (
    COMMANDS,
    HELP,
    PROFILES,
    VIEWS,
    TelegramController,
    TelegramError,
    _chunks,
)

from .test_telegram import (
    ACTOR,
    CHAT,
    TOKEN,
    USER,
    FakeResponse,
    FakeService,
    FakeSession,
    callback,
    confirmation_keyboard,
    message,
)


PAGED_VIEWS = {
    "opportunities",
    "why",
    "waves",
    "positions",
    "universe",
    "performance",
    "research",
    "system",
    "telemetry",
    "comparison",
    "risk",
    "profile",
}


def buttons(markup):
    return [button for row in markup["inline_keyboard"] for button in row]


class PagedService(FakeService):
    def __init__(self):
        super().__init__()
        self.page_reads = []
        self.pages = 3

    async def render_page(self, view, symbol=None, page=0):
        self.page_reads.append((view, symbol, page))
        pages = self.pages if view in PAGED_VIEWS else 1
        index = min(max(0, page), pages - 1)
        return {
            "text": f"{view} {symbol or ''}\nRecord {index + 1}\nPage {index + 1} of {pages}",
            "page": index,
            "pages": pages,
        }


class NavigationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = FakeSession()
        self.service = PagedService()
        self.controller = TelegramController(
            self.session, TOKEN, CHAT, {USER}, self.service
        )
        self.controller.WRITE_INTERVAL = 0

    async def command(self, text):
        await self.controller._handle_update({"message": message(text)})

    async def click(self, data, **kwargs):
        await self.controller._handle_update(
            {"callback_query": callback(data, **kwargs)}
        )

    def markup(self):
        return self.session.text_payloads()[-1]["reply_markup"]

    def actions(self):
        return [button["callback_data"] for button in buttons(self.markup())]

    def assert_read_only(self):
        self.assertEqual(self.service.proposals, [])
        self.assertEqual(self.service.confirms, [])
        self.assertEqual(self.service.cancels, [])
        self.assertEqual(self.service.applied, [])

    async def test_all_read_commands_use_one_render_page_and_never_legacy_snapshot(
        self,
    ):
        for view in VIEWS:
            before = len(self.service.page_reads)
            await self.command("/" + view)
            self.assertEqual(self.service.page_reads[before:], [(view, None, 0)])
        await self.command("/start resume")
        self.assertEqual(self.service.page_reads[-1], ("dashboard", None, 0))
        self.assertEqual(self.service.reads, [])
        self.assert_read_only()

    async def test_all_views_are_reachable_from_home_and_menus_are_compact(self):
        pending, visited = ["pg:dashboard:0"], set()
        while pending:
            data = pending.pop()
            if data in visited:
                continue
            visited.add(data)
            await self.click(data)
            view = data.split(":")[1] if data.startswith("pg:") else data
            keys = buttons(self.markup())
            self.assertLessEqual(len(keys), 10 if view == "dashboard" else 8)
            self.assertTrue(
                all(1 <= len(row) <= 2 for row in self.markup()["inline_keyboard"])
            )
            self.assertTrue(any("Home" in key["text"] for key in keys))
            self.assertTrue(any("Refresh" in key["text"] for key in keys))
            for key in keys:
                action = key["callback_data"]
                self.assertLessEqual(len(action.encode("utf-8")), 64)
                if action.startswith("pg:") or action in ("profiles", "custom_risk"):
                    pending.append(action)
                if action.startswith("p:paused:"):
                    self.assertEqual(view, "settings")
                if action.startswith("p:profile:"):
                    self.assertEqual(view, "profiles")
        self.assertTrue({f"pg:{view}:0" for view in VIEWS} <= visited)
        self.assertIn("profiles", visited)
        self.assertEqual(self.service.reads, [])
        self.assert_read_only()

    async def test_every_page_is_reachable_and_refresh_preserves_context(self):
        for view in sorted(PAGED_VIEWS):
            symbol = "BTCUSDT" if view in ("why", "waves") else None
            suffix = ":BTCUSDT" if symbol else ""
            for page in range(3):
                await self.click(f"pg:{view}:{page}{suffix}", age=90000)
                self.assertEqual(self.service.page_reads[-1], (view, symbol, page))
                actions = self.actions()
                self.assertEqual(f"pg:{view}:{page - 1}{suffix}" in actions, page > 0)
                self.assertEqual(f"pg:{view}:{page + 1}{suffix}" in actions, page < 2)
                refresh = next(
                    b["callback_data"]
                    for b in buttons(self.markup())
                    if "Refresh" in b["text"]
                )
                self.assertEqual(refresh, f"pg:{view}:{page}{suffix}")
                await self.click(refresh)
                self.assertEqual(self.service.page_reads[-1], (view, symbol, page))
                self.assertIn(
                    f"Record {page + 1}", self.session.text_payloads()[-1]["text"]
                )
                self.assertLessEqual(len(buttons(self.markup())), 8)
        self.assert_read_only()

    async def test_clamped_page_metadata_drives_navigation_after_records_disappear(
        self,
    ):
        await self.click("pg:why:999999:BTCUSDT")
        self.assertEqual(self.service.page_reads[-1], ("why", "BTCUSDT", 999999))
        self.assertIn("pg:why:2:BTCUSDT", self.actions())
        self.assertIn("pg:why:1:BTCUSDT", self.actions())
        self.assertNotIn("pg:why:3:BTCUSDT", self.actions())
        self.service.pages = 1
        await self.click("pg:why:2:BTCUSDT")
        self.assertIn("pg:why:0:BTCUSDT", self.actions())
        self.assertFalse(
            any(
                "Previous" in b["text"] or "Next" in b["text"]
                for b in buttons(self.markup())
            )
        )

    async def test_symbol_evidence_links_preserve_symbol_and_new_view_starts_at_zero(
        self,
    ):
        for view, destination in (("why", "waves"), ("waves", "why")):
            symbol = "A" * 32
            await self.click(f"pg:{view}:1:{symbol}")
            self.assertIn(f"pg:{destination}:0:{symbol}", self.actions())
            self.assertIn("pg:opportunities:0", self.actions())
            for action in self.actions():
                self.assertLessEqual(len(action.encode("utf-8")), 64)
            await self.click(f"pg:{destination}:0:{symbol}")
            self.assertEqual(self.service.page_reads[-1], (destination, symbol, 0))

    async def test_legacy_callbacks_open_first_page_for_every_view(self):
        for view in VIEWS:
            await self.click("v:" + view)
            self.assertEqual(self.service.page_reads[-1], (view, None, 0))
        for view in ("why", "waves"):
            await self.click(f"v:{view}:BTCUSDT")
            self.assertEqual(self.service.page_reads[-1], (view, "BTCUSDT", 0))
        self.assert_read_only()

    async def test_legacy_service_takes_two_args_and_has_no_page_navigation(self):
        legacy = FakeService()
        self.controller._service = legacy
        for view in PAGED_VIEWS:
            await self.click(f"pg:{view}:2")
            self.assertEqual(legacy.reads[-1], (view, None))
            self.assertIn(f"pg:{view}:0", self.actions())
            self.assertFalse(
                any(
                    "Previous" in b["text"] or "Next" in b["text"]
                    for b in buttons(self.markup())
                )
            )
        await self.click("pg:waves:2:BTCUSDT")
        self.assertEqual(legacy.reads[-1], ("waves", "BTCUSDT"))
        await self.click("pg:help:2")
        self.assertEqual(self.session.text_payloads()[-1]["text"], HELP)
        self.assertIn("pg:help:0", self.actions())

    async def test_every_emitted_button_is_authorized_before_service_access(self):
        controls = set()
        for view in VIEWS:
            for page in range(3):
                for symbol in (
                    (None, "BTCUSDT") if view in ("why", "waves") else (None,)
                ):
                    controls.update(
                        b["callback_data"]
                        for b in buttons(
                            self.controller._keyboard(view, symbol, page, 3)
                        )
                    )
        await self.click("profiles")
        controls.update(self.actions())
        self.session.calls.clear()
        for data in sorted(controls):
            for chat, user in ((123, USER), (-10042, 999), (-10042, True)):
                await self.click(data, chat=chat, user=user)
        self.assertTrue(self.session.calls)
        self.assertEqual({c[0] for c in self.session.calls}, {"answerCallbackQuery"})
        self.assertTrue(
            all(
                p["text"] == "Not authorized"
                for p in self.session.payloads("answerCallbackQuery")
            )
        )
        self.assertEqual(self.service.page_reads, [])
        self.assert_read_only()

    async def test_new_callbacks_ack_before_render_and_rejected_ack_never_reads(self):
        render_page = self.service.render_page

        async def checked_render(*args, **kwargs):
            self.assertEqual(self.session.calls[-1][0], "answerCallbackQuery")
            return await render_page(*args, **kwargs)

        with patch.object(self.service, "render_page", side_effect=checked_render):
            await self.click("pg:positions:1")
        self.session.responses["answerCallbackQuery"].append(
            FakeResponse({"ok": False, "error_code": 400}, 400)
        )
        await self.click("pg:positions:2")
        self.assertEqual(self.service.page_reads, [("positions", None, 1)])

    async def test_malformed_page_controls_fail_safely_without_reads_or_mutations(self):
        for data in (
            "pg",
            "pg:",
            "pg:why",
            "pg:why:",
            "pg:why:-1",
            "pg:why:+1",
            "pg:why:1.0",
            "pg:why:01",
            "pg:why:١",
            "pg:why:1e3",
            "pg:why: 1",
            "pg:why:True",
            "pg:why:0:",
            "pg:why:0:btcusdt",
            "pg:why:0:<script>",
            "pg:waves:0:BTC:USDT",
            "pg:risk:0:BTCUSDT",
            "pg:system:1:BTCUSDT",
            "pg:live:0",
            "pg:why:0:" + "A" * 33,
            "pg:why:" + "9" * 60,
        ):
            with self.subTest(data=data):
                await self.click(data)
                self.assertNotIn(
                    "Unable to verify", self.session.text_payloads()[-1]["text"]
                )
        self.assertEqual(self.service.page_reads, [])
        self.assertEqual(self.service.reads, [])
        self.assert_read_only()

    async def test_invalid_page_results_fail_closed_without_legacy_fallback(self):
        for result in (
            None,
            {},
            "plain",
            {"text": "valid", "page": 0},
            {"text": "valid", "page": False, "pages": 1},
            {"text": "valid", "page": 0, "pages": True},
            {"text": "valid", "page": -1, "pages": 1},
            {"text": "valid", "page": 1, "pages": 1},
            {"text": "valid", "page": 0, "pages": 0},
            {"text": "valid", "page": "0", "pages": 2},
            {"text": " ", "page": 0, "pages": 1},
            {"text": 123, "page": 0, "pages": 1},
        ):
            with patch.object(self.service, "render_page", return_value=result):
                await self.click("pg:risk:0")
            self.assertIn("Unable to verify", self.session.text_payloads()[-1]["text"])
        self.assertEqual(self.service.reads, [])
        self.assert_read_only()

    async def test_rendered_risk_content_passes_through_without_appended_guidance(self):
        text = (
            "🛡 Risk\nAuthoritative complete guidance\n\n"
            + "🌊 <plain> _literal_\n" * 400
        )
        for view in ("risk", "profile"):
            self.session.calls.clear()
            with patch.object(
                self.service,
                "render_page",
                return_value={"text": text, "page": 1, "pages": 3},
            ):
                await self.click(f"pg:{view}:1")
            payloads = self.session.text_payloads()
            self.assertEqual("".join(p["text"] for p in payloads), text)
            self.assertEqual(payloads[0]["reply_markup"], {"inline_keyboard": []})
            self.assertTrue(all("reply_markup" not in p for p in payloads[1:-1]))
            self.assertTrue(all("parse_mode" not in p for p in payloads))
            self.assertIn(f"pg:{view}:1", self.actions())

    async def test_page_timeout_and_errors_are_sanitized_without_second_snapshot(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        self.controller.SERVICE_TIMEOUT = 0.001
        for failure in (slow, RuntimeError(TOKEN)):
            with patch.object(self.service, "render_page", side_effect=failure):
                await self.command("/research")
            text = self.session.text_payloads()[-1]["text"]
            self.assertIn("Unable to verify", text)
            self.assertNotIn(TOKEN, text)
        self.assertEqual(self.service.reads, [])

    async def test_safe_proposal_rejections_are_visible_but_unknown_errors_are_not(
        self,
    ):
        reason = (
            "Resume blocked: risk data is stale. Refresh the risk assessment first."
        )
        for command, action in (("/resume", None), (None, "p:paused:0")):
            with patch.object(
                self.service, "propose_change", side_effect=ServiceRejection(reason)
            ):
                if command:
                    await self.command(command)
                else:
                    await self.click(action)
            self.assertEqual(self.session.text_payloads()[-1]["text"], reason)
            self.assertFalse(
                any(action.startswith("confirm:") for action in self.actions())
            )
        for error in (ValueError(TOKEN), RuntimeError(TOKEN)):
            with patch.object(self.service, "propose_change", side_effect=error):
                await self.command("/resume")
            self.assertIn("Unable to verify", self.session.text_payloads()[-1]["text"])
            self.assertNotIn(TOKEN, self.session.text_payloads()[-1]["text"])
        self.assert_read_only()

    async def test_confirmation_rejection_keeps_uncertainty_even_for_safe_class(self):
        await self.command("/pause")
        confirm = self.service.confirm_change

        async def committed_then_failed(token, actor):
            await confirm(token, actor)
            raise ServiceRejection("Do not claim the change was rejected")

        with patch.object(
            self.service, "confirm_change", side_effect=committed_then_failed
        ):
            await self.click(
                "confirm:proposal_1", markup=confirmation_keyboard("proposal_1")
            )
        self.assertEqual(self.service.applied, [("paused", True, ACTOR)])
        self.assertIn("Unable to verify", self.session.text_payloads()[-1]["text"])

    async def test_all_profile_choices_preview_using_canonical_ids(self):
        await self.click("profiles")
        chooser = self.markup()
        choices = [
            b for b in buttons(chooser) if b["callback_data"].startswith("p:profile:")
        ]
        self.assertEqual(len(choices), len(PROFILES))
        self.assertLessEqual(len(buttons(chooser)), 8)
        for choice, profile in zip(choices, PROFILES):
            self.assertNotIn("_", choice["text"])
            await self.click(choice["callback_data"], markup=chooser)
            self.assertEqual(self.service.proposals[-1], ("profile", profile, ACTOR))
            self.assertIn(
                self.service.summary, self.session.text_payloads()[-1]["text"]
            )
        self.assertEqual(self.service.applied, [])

    def test_descriptive_commands_help_and_documented_commands_match(self):
        self.assertEqual(set(COMMANDS), set(VIEWS) | {"start", "pause", "resume"})
        self.assertTrue(
            all(
                1 <= len(description) <= 256 and not description.startswith("Show ")
                for description in COMMANDS.values()
            )
        )
        documented = set(re.findall(r"/([a-z_]+)", HELP))
        self.assertEqual(documented, set(COMMANDS))
        self.assertIn("Refresh keeps your page and symbol", HELP)


class ChunkTests(unittest.TestCase):
    def test_paragraph_boundaries_precede_line_boundaries(self):
        first = "A" * 1800 + "\n\n"
        text = first + "B" * 1600 + "\n" + "C" * 1800
        self.assertEqual(_chunks(text), [first, text[len(first) :]])

    def test_line_boundaries_and_windows_paragraphs_are_lossless(self):
        for separator in ("\n", "\r\n\r\n"):
            first = "A" * 2500 + separator
            second = "B" * 2500 + separator
            self.assertEqual(_chunks(first + second), [first, second])

    def test_utf16_limit_hard_splits_and_whitespace_are_lossless(self):
        samples = [
            "🌊" * 5000,
            "A" * 3900 + "🌊",
            "A" * 3899 + "🌊\n\nZ",
            " \n" + ("🌊" * 900 + "\n\n") * 10 + "  \n",
            ("Short paragraph\n\n" + "X" * 6000 + "\n") * 4,
            "A" * 3899 + "\n\n" + "B" * 3900,
        ]
        for text in samples:
            with self.subTest(length=len(text)):
                chunks = _chunks(text)
                self.assertEqual("".join(chunks), text)
                self.assertTrue(
                    all(
                        0 < len(chunk.encode("utf-16-le")) // 2 <= 3900
                        for chunk in chunks
                    )
                )
        self.assertEqual(_chunks("🌊" * 1950), ["🌊" * 1950])
        self.assertEqual(_chunks("a" * 3900), ["a" * 3900])

    def test_leading_blank_lines_do_not_create_empty_messages(self):
        for prefix in ("\n\n", " \n", "\r\n\r\n"):
            text = prefix + "A" * 5000
            chunks = _chunks(text)
            self.assertEqual("".join(chunks), text)
            self.assertTrue(all(chunk.strip() for chunk in chunks))


class EditResponseTests(unittest.IsolatedAsyncioTestCase):
    async def render_response(self, body, status):
        session = FakeSession()
        service = PagedService()
        controller = TelegramController(session, TOKEN, CHAT, {USER}, service)
        controller.WRITE_INTERVAL = 0
        session.responses["editMessageText"].append(FakeResponse(body, status))
        error = None
        try:
            await controller._handle_update(
                {"callback_query": callback("pg:positions:1")}
            )
        except TelegramError as caught:
            error = caught
        return session, error

    async def test_exact_unchanged_edits_succeed_without_duplicate_sends(self):
        for description in (
            "Bad Request: message is not modified",
            "Bad Request: message is not modified: specified new message content and reply "
            "markup are exactly the same as a current content and reply markup of the message",
        ):
            session, error = await self.render_response(
                {"ok": False, "error_code": 400, "description": description}, 400
            )
            self.assertIsNone(error)
            self.assertEqual(
                [c[0] for c in session.calls],
                ["answerCallbackQuery", "editMessageText"],
            )
            self.assertEqual(session.payloads("sendMessage"), [])

    async def test_other_explicit_edit_rejections_retain_replacement_behavior(self):
        for status, description in (
            (400, "Bad Request: message to edit not found"),
            (400, "Bad Request: message is not modified but another error occurred"),
            (400, "Not the exact message is not modified error"),
            (403, "Forbidden"),
        ):
            session, error = await self.render_response(
                {"ok": False, "error_code": status, "description": description}, status
            )
            self.assertIsNone(error)
            self.assertEqual(len(session.payloads("sendMessage")), 1)
            self.assertEqual(
                session.payloads("sendMessage")[0]["text"],
                session.payloads("editMessageText")[0]["text"],
            )

    async def test_uncertain_rate_limited_and_server_edits_never_duplicate(self):
        for status, body in (
            (
                400,
                {
                    "ok": "false",
                    "error_code": 400,
                    "description": "Bad Request: message is not modified",
                },
            ),
            (400, {"description": "Bad Request: message is not modified"}),
            (429, {"ok": False, "error_code": 429}),
            (500, {"ok": False, "error_code": 500}),
            (
                400,
                {
                    "ok": False,
                    "error_code": 500,
                    "description": "Bad Request: message is not modified",
                },
            ),
        ):
            session, error = await self.render_response(body, status)
            self.assertIsInstance(error, TelegramError)
            self.assertEqual(session.payloads("sendMessage"), [])
            self.assertEqual(len(session.payloads("editMessageText")), 1)
