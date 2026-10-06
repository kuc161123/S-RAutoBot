"""Review regressions through real Dashboard/Service and offline Telegram callbacks."""

import re
import time
import unittest
from unittest.mock import patch

from apex_bot.config import Config
from apex_bot.service import Service
from apex_bot.storage import Store
from apex_bot.telegram import TelegramController

from .test_telegram import CHAT, TOKEN, FakeSession, callback


USER = 987654321
ARM_LABELS = {
    "baseline_shadow": "Rules baseline",
    "ai_shadow": "AI-approved shadow",
}


class DashboardNavigationReviewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(database_url="", sqlite_path=":memory:")
        await self.store.initialize()
        self.addAsyncCleanup(self.store.close)
        self.assertTrue(await self.store.lease(ttl=10000))
        self.now = time.time()
        self.config = Config(
            mode="shadow",
            universe_mode="static",
            symbols=("BTCUSDT", "ETHUSDT"),
            telegram_token=TOKEN,
            telegram_chat_id=CHAT,
            telegram_user_ids=frozenset({USER}),
            openai_key="sk-FAKE_UNDERSCORE_REVIEW_TEST_ONLY",
            openai_model="fixture-model",
            bybit_secret="FAKE_BYBIT_UNDERSCORE_REVIEW_TEST_ONLY",
        )
        self.service = Service(self.config, self.store)
        self.session = FakeSession()
        self.controller = TelegramController(
            self.session, TOKEN, CHAT, {USER}, self.service
        )
        self.controller.WRITE_INTERVAL = 0

    async def callback_pages(self, view):
        """Follow emitted Next buttons with an authorized actor, never a poller."""
        before = await self.store.read()
        action, markup = f"pg:{view}:0", None
        output, seen = [], set()
        with patch.object(
            self.store, "update", side_effect=AssertionError("Read view mutated state")
        ):
            while action is not None:
                self.assertNotIn(action, seen, "Next button loops to a visited page")
                self.assertLess(len(seen), 50, "Unexpected unbounded pagination")
                seen.add(action)
                self.session.calls.clear()
                await self.controller._handle_update(
                    {
                        "callback_query": callback(
                            action, user=USER, chat=int(CHAT), markup=markup
                        )
                    }
                )
                self.assertEqual(
                    [call[0] for call in self.session.calls],
                    ["answerCallbackQuery", "editMessageText"],
                    "Authorized pages should acknowledge and edit one message",
                )
                ack = self.session.payloads("answerCallbackQuery")[0]
                self.assertEqual(ack["text"], "Opening…")
                payload = self.session.payloads("editMessageText")[0]
                text = payload["text"]
                self.assertNotIn("Unable to verify", text)
                self.assertNotIn("Unknown view", text)
                self.assertLessEqual(len(text.encode("utf-16-le")) // 2, 3900)
                self.assertNotIn("parse_mode", payload)
                markup = payload["reply_markup"]
                keys = [b for row in markup["inline_keyboard"] for b in row]
                self.assertLessEqual(len(keys), 8)
                refresh = next(b for b in keys if "Refresh" in b["text"])
                self.assertEqual(refresh["callback_data"], action)
                output.append(text)
                next_buttons = [b for b in keys if "Next" in b["text"]]
                self.assertLessEqual(len(next_buttons), 1)
                action = next_buttons[0]["callback_data"] if next_buttons else None
        self.assertEqual(await self.store.read(), before)
        return output

    def assert_secrets_redacted(self, pages):
        text = "\n".join(pages)
        self.assertIn("Fixture failure", text)
        self.assertIn("[redacted]", text)
        for secret in (
            self.config.telegram_token,
            self.config.openai_key,
            self.config.bybit_secret,
        ):
            self.assertNotIn(secret, text)
            self.assertNotIn(
                secret.replace("_", " "),
                text,
                "Underscore-to-space formatting must not bypass secret redaction",
            )

    def failure_text(self):
        return "Fixture failure: " + " / ".join(
            (
                self.config.telegram_token,
                self.config.openai_key,
                self.config.bybit_secret,
            )
        )

    async def test_component_errors_redact_before_formatting_through_callbacks(self):
        await self.store.update(
            lambda tx: tx.state.update(
                health={"scan_at": self.now, "scan_error": self.failure_text()}
            )
        )
        for view in ("system", "telemetry"):
            with self.subTest(view=view):
                self.assert_secrets_redacted(await self.callback_pages(view))

    async def test_research_failure_redacts_before_formatting_through_callbacks(self):
        await self.store.update(
            lambda tx: tx.state.update(
                research={
                    "status": "UNAVAILABLE",
                    "created_at": self.now,
                    "reason": self.failure_text(),
                }
            )
        )
        for view in ("research", "system", "telemetry"):
            with self.subTest(view=view):
                self.assert_secrets_redacted(await self.callback_pages(view))

    async def test_each_position_card_keeps_its_arm_on_every_callback_page(self):
        trades, expected = {}, {}
        for arm, label in ARM_LABELS.items():
            prefix = "RULES" if arm == "baseline_shadow" else "AI"
            for index in range(6):
                symbol = f"FIXTURE{prefix}{index:02}USDT"
                identifier = f"{arm}-{index}"
                expected[symbol] = label
                trades[identifier] = {
                    "id": identifier,
                    "arm": arm,
                    "symbol": symbol,
                    "side": "Buy",
                    "status": "OPEN",
                    "created_at": self.now - index,
                    "opened_at": self.now - index,
                    "entry": 100,
                    "limit": 100,
                    "stop": 90,
                    "target1": 120,
                    "target2": 140,
                    "qty": 1,
                    "remaining": 1,
                    "mark_price": 110,
                    "mark_at": self.now,
                }
        await self.store.update(lambda tx: tx.state.update(trades=trades))
        pages = await self.callback_pages("positions")
        self.assertGreater(len(pages), 1, "Exercise real continuation pages")
        found = []
        for page_number, text in enumerate(pages, 1):
            for card in text.split("\n\n"):
                symbols = re.findall(r"\bFIXTURE(?:RULES|AI)[0-9]{2}USDT\b", card)
                if not symbols:
                    continue
                with self.subTest(page=page_number, symbols=symbols):
                    self.assertEqual(len(symbols), 1)
                    symbol = symbols[0]
                    self.assertIn("Simulated position OPEN", card)
                    self.assertIn(
                        expected[symbol],
                        card,
                        "Each card needs its own portfolio label; a section/page heading is insufficient",
                    )
                    for other_label in set(ARM_LABELS.values()) - {expected[symbol]}:
                        self.assertNotIn(other_label, card)
                    found.append(symbol)
        self.assertCountEqual(
            found, expected.keys(), "All twelve positions appear once"
        )
