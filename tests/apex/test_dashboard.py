"""Presentation regressions use isolated ledgers, synthetic data and no network."""

import copy
import json
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from apex_bot.config import Config
from apex_bot.dashboard import age, pages, price, words
from apex_bot.service import Service
from apex_bot.storage import Store


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(sqlite_path=":memory:")
        await self.store.initialize()
        await self.store.lease(ttl=10000)
        self.now = time.time()
        self.config = Config(
            universe_mode="static",
            symbols=("BTCUSDT", "ETHUSDT"),
            openai_key="sk-fixture-only-never-real",
            openai_model="fixture-model",
        )
        self.service = Service(self.config, self.store)
        await self.write(
            health={
                key + "_at": self.now
                for key in (
                    "scan",
                    "risk",
                    "context",
                    "simulation",
                    "shadow_observer",
                    "funding",
                    "telegram",
                    "outbox",
                )
            },
            context={
                "data_complete": True,
                "as_of": self.now,
                "expires_at": self.now + 600,
                "event_blackout": False,
                "risk_state": "neutral",
                "long_multiplier": 1,
                "short_multiplier": 1,
            },
            risk_circuits={
                k: {
                    "as_of": self.now,
                    "daily_loss_pct": 0,
                    "weekly_loss_pct": 0,
                    "drawdown_pct": 0,
                }
                for k in ("baseline_shadow", "ai_shadow")
            },
            risk_reference_equities={
                k: 10000 for k in ("baseline_shadow", "ai_shadow")
            },
        )

    async def asyncTearDown(self):
        await self.store.close()

    async def write(self, **values):
        await self.store.update(lambda tx: tx.state.update(copy.deepcopy(values)))

    def candidate(self, identifier, **changes):
        op = dict(
            id=identifier,
            symbol="BTCUSDT",
            side="Buy",
            setup="2",
            state="WAIT",
            entry=100,
            stop=90,
            target1=120,
            target2=140,
            invalidation=92,
            expires_at=self.now + 600,
            reason="AWAIT_CLOSED_4H_TRIGGER",
            evidence={},
        )
        op.update(changes)
        return {"opportunity": op, "updated_at": self.now}

    async def test_home_separates_496_stored_records_from_current_candidates(self):
        records = {
            f"old-{i}": self.candidate(str(i), expires_at=self.now - 1)
            for i in range(494)
        }
        records.update(
            long=self.candidate("long", state="READY"),
            short=self.candidate("short", side="Sell"),
        )
        await self.write(opportunities=records)
        before = await self.store.read()
        home = await self.service.render_page("dashboard")
        self.assertEqual(home["pages"], 1)
        self.assertIn("Candidates now: 2", home["text"])
        self.assertIn("1 long / 1 short", home["text"])
        self.assertIn("Ready for checks: 1", home["text"])
        self.assertNotIn("496", home["text"])
        self.assertNotIn("Paired AI", home["text"])
        self.assertIn("Exchange execution: OFF", home["text"])
        self.assertEqual(await self.store.read(), before)
        status = await self.service.snapshot("status")
        self.assertIn("Historical / inactive records: 494", status)

    async def test_stale_future_retired_and_expired_candidates_not_current(self):
        records = {
            "stale": self.candidate("stale", state="READY"),
            "future": self.candidate("future"),
            "retired": self.candidate("retired", symbol="RETIREDUSDT"),
            "expired": self.candidate("expired", expires_at=self.now - 1),
        }
        records["stale"]["updated_at"] = self.now - 400
        records["future"]["updated_at"] = self.now + 400
        await self.write(opportunities=records)
        self.assertIn("Candidates now: 0", await self.service.snapshot("dashboard"))
        radar = await self.service.snapshot("opportunities")
        self.assertIn("No current qualifying setup", radar)
        self.assertNotIn("Planned entry 100", radar)
        self.assertIn(
            "Historical / inactive", await self.service.snapshot("why", "BTC")
        )

    async def test_historical_error_is_not_active_and_stale_loop_not_green(self):
        health = (await self.store.read())["health"]
        health.update(error="telegram: TelegramError", simulation_at=self.now - 900)
        await self.write(health=health)
        text = await self.service.snapshot("system")
        self.assertIn("Active component errors: 0", text)
        self.assertIn("Last recorded issue (historical)", text)
        self.assertIn("Shadow positions: ⚠️ Stale", text)
        self.assertIn("Private exchange: not used in shadow", text)
        self.assertNotIn("Issue recorded: yes", text)
        health["scan_error"] = "TimeoutError"
        await self.write(health=health)
        self.assertIn("Scanner: ⚠️ Error", await self.service.snapshot("system"))

    async def test_research_unavailable_is_not_empty_valid_research(self):
        await self.write(
            research={
                "status": "UNAVAILABLE",
                "reason": "Research response could not be validated",
                "created_at": self.now,
                "facts": [{"fact": "UNVALIDATED CLAIM"}],
            }
        )
        text = await self.service.snapshot("research")
        self.assertIn("Result: Unavailable", text)
        self.assertIn("No validated conclusion", text)
        self.assertNotIn("UNVALIDATED CLAIM", text)
        self.assertNotIn("No validated facts returned", text)

    async def test_context_failure_reason_is_readable_and_redacted(self):
        await self.write(
            context={
                "data_complete": False,
                "reason": "calendar_unavailable " + self.config.openai_key,
            }
        )
        text = await self.service.snapshot("system")
        self.assertIn("Context detail: calendar unavailable [redacted]", text)
        self.assertNotIn(self.config.openai_key, text)

    async def test_valid_research_renders_facts_sources_and_freshness(self):
        await self.write(
            research={
                "status": "COMPLETE",
                "asof": "2026-10-06",
                "created_at": self.now,
                "facts": [
                    {
                        "symbol": "MACRO",
                        "fact": "Synthetic official event",
                        "source_url": "https://www.federalreserve.gov/example",
                        "published": "2026-10-06",
                        "asof": "2026-10-06",
                        "uncertainty": "Timing may change",
                    }
                ],
            }
        )
        text = await self.service.snapshot("research")
        self.assertIn("Result: Available", text)
        self.assertIn("Synthetic official event", text)
        self.assertIn("https://www.federalreserve.gov/example", text)
        self.assertIn("Timing may change", text)
        data = (await self.store.read())["research"]
        data["created_at"] = self.now - 90000
        await self.write(research=data)
        self.assertIn("Result: Stale", await self.service.snapshot("research"))

    async def test_redaction_precedes_free_text_truncation(self):
        secret = self.config.openai_key
        await self.write(
            research={"summary": "x" * 1995 + secret, "status": "UNAVAILABLE"}
        )
        text = await self.service.snapshot("research")
        self.assertNotIn(secret[:5], text)
        rec = self.candidate("secret")
        rec["decision"] = "x" * 446 + secret
        await self.write(opportunities={"secret": rec})
        text = await self.service.snapshot("why", "BTC")
        self.assertNotIn(secret[:5], text)

    async def test_all_open_and_pending_positions_reachable_without_fake_fills(self):
        trades = {}
        for i in range(31):
            trades[str(i)] = dict(
                id=str(i),
                arm="baseline_shadow",
                symbol=f"FIXTURE{i:02}USDT",
                side="Sell",
                status="PENDING",
                created_at=self.now - i,
                limit=100,
                stop=110,
                target1=80,
                target2=70,
                qty=1,
                remaining=1,
                entry=None,
            )
        await self.write(trades=trades)
        first = await self.service.render_page("positions")
        self.assertGreater(first["pages"], 1)
        rendered = [
            await self.service.render_page("positions", page=i)
            for i in range(first["pages"])
        ]
        combined = "\n".join(p["text"] for p in rendered)
        for i in range(31):
            self.assertEqual(combined.count(f"FIXTURE{i:02}USDT"), 1)
        self.assertNotIn("Filled entry", combined)
        self.assertIn("Waiting for simulated fill", combined)
        self.assertIn("🔴 SHORT", combined)
        for p in rendered:
            self.assertLessEqual(len(p["text"].encode("utf-16-le")) // 2, 3900)
        self.assertEqual(
            (await self.service.render_page("positions", page=9999))["page"],
            first["pages"] - 1,
        )

    async def test_symbol_history_pagination_keeps_every_record(self):
        records = {
            str(i): self.candidate(str(i), entry=100 + i, state="EXPIRED")
            for i in range(30)
        }
        await self.write(opportunities=records)
        first = await self.service.render_page("why", "BTC")
        outputs = [
            await self.service.render_page("why", "BTC", i)
            for i in range(first["pages"])
        ]
        combined = "\n".join(p["text"] for p in outputs)
        for i in range(30):
            self.assertIn(f"Planned entry {100+i}\n", combined)
        self.assertNotIn("use /why SYMBOL", combined)

    async def test_zero_closed_shows_unmeasured_wr(self):
        text = await self.service.snapshot("dashboard")
        self.assertIn("no valid closed trades", text)
        self.assertNotIn("WR 0.0%", text)

    async def test_unknown_health_timestamp_not_reported_current(self):
        await self.write(
            health={"scan_at": self.now + 100, "telegram_at": 0, "risk_at": None}
        )
        text = await self.service.snapshot("system")
        self.assertNotIn("✅ Current", text)
        self.assertIn("unavailable", text)

    async def test_legacy_archive_exposes_only_safe_billing_diagnosis(self):
        data = {
            "status": "UNAVAILABLE",
            "evidence_hash": "fixture-hash",
            "created_at": self.now,
            "reason": "Research response could not be validated",
        }
        attempt = {
            "http_status": 429,
            "error": "HTTP_ERROR",
            "valid": False,
            "status": "completed",
        }
        record = {
            "kind": "research",
            "result": data,
            "archive_key": "test-archive",
            "attempts": [attempt],
        }
        raw = json.dumps(
            {
                "error": {
                    "code": "billing_not_active",
                    "message": "SECRET_RESPONSE_SHOULD_NOT_APPEAR",
                }
            }
        )
        await self.store.update(
            lambda tx: tx.event(
                "test-archive",
                "ai_archive",
                {"record": {"attempts": [dict(attempt, raw_response=raw)]}},
            )
        )
        await self.write(research=data, reviews={"fixture": record})
        before = await self.store.read()
        text = await self.service.snapshot("research")
        self.assertIn("API billing is not active", text)
        self.assertNotIn("could not be validated", text)
        self.assertNotIn("SECRET_RESPONSE", text)
        self.assertEqual(await self.store.read(), before)
        self.assertEqual(
            await self.store.ai_failure_diagnostic("test-archive"),
            {"http_status": 429, "error_code": "billing_not_active"},
        )

    async def test_arm_decisions_keep_their_own_reasons_and_timestamps(self):
        rec = self.candidate("separate")
        rec["per_arm_decisions"] = {
            "baseline_shadow": {
                "decision": "Risk checks passed",
                "updated_at": self.now - 30,
            },
            "ai_shadow": {
                "decision": "Portfolio capacity limit",
                "updated_at": self.now - 300,
            },
        }
        await self.write(opportunities={"separate": rec})
        text = await self.service.snapshot("why", "BTC")
        self.assertIn("Rules baseline: Risk checks passed", text)
        self.assertIn("AI-approved shadow: Portfolio capacity limit", text)
        self.assertIn("Decision recorded: 30s ago", text)
        self.assertIn("Decision recorded: 5m ago", text)


class FormattingTests(unittest.TestCase):
    def test_unicode_pages_lossless_and_bounded(self):
        text = "Header\n\n" + "😀" * 3100 + "\nnext\n\n" + "long line " * 1000
        chunks = pages(text)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(p.encode("utf-16-le")) // 2 <= 3000 for p in chunks))

    def test_numbers_and_ages_preserve_unknown_and_false(self):
        self.assertEqual(words(False), "False")
        self.assertEqual(price(0), "0")
        self.assertEqual(price(0.16863199999999998), "0.168632")
        self.assertEqual(price(4e-9), "0.000000004")
        self.assertEqual(price(float("nan")), "unavailable")
        self.assertEqual(age(1100, 1000), "unavailable")
        self.assertEqual(age(900, 1000), "1m ago")
