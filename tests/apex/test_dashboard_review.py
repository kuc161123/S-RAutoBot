"""Independent dashboard correctness regressions; isolated state and no network."""

import copy
import unittest
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

from apex_bot.dashboard import price
from tests.apex import test_dashboard as dashboard_fixtures


class DashboardReviewTests(unittest.IsolatedAsyncioTestCase):
    # Reuse the isolated ledger and valid payload fixtures without inheriting and
    # collecting the original test methods a second time.
    asyncSetUp = dashboard_fixtures.DashboardTests.asyncSetUp
    asyncTearDown = dashboard_fixtures.DashboardTests.asyncTearDown
    write = dashboard_fixtures.DashboardTests.write
    candidate = dashboard_fixtures.DashboardTests.candidate

    async def render_read_only(self, view, symbol=None, *, paged=False):
        before = await self.store.read()
        outbox_before = await self.store.outbox_health()
        with patch("apex_bot.service.time.time", return_value=self.now):
            if paged:
                first = await self.service.render_page(view, symbol)
                rendered = [first]
                for index in range(1, first["pages"]):
                    rendered.append(
                        await self.service.render_page(view, symbol, page=index)
                    )
                text = "\n".join(page["text"] for page in rendered)
            else:
                text = await self.service.snapshot(view, symbol)
        self.assertEqual(await self.store.read(), before)
        self.assertEqual(await self.store.outbox_health(), outbox_before)
        return text

    async def test_loop_current_does_not_imply_valid_context_or_risk_data(self):
        baseline = await self.store.read()
        context = baseline["context"]
        circuits = baseline["risk_circuits"]
        cases = (
            ("valid", context, circuits, True, True),
            (
                "incomplete context",
                dict(context, data_complete=False),
                circuits,
                False,
                True,
            ),
            (
                "expired context",
                dict(context, expires_at=self.now - 1),
                circuits,
                False,
                True,
            ),
            (
                "unavailable risk",
                context,
                {
                    arm: dict(circuit, as_of=0, error="Valuation evidence incomplete")
                    for arm, circuit in circuits.items()
                },
                True,
                False,
            ),
            (
                "stale risk",
                context,
                {
                    arm: dict(circuit, as_of=self.now - 600)
                    for arm, circuit in circuits.items()
                },
                True,
                False,
            ),
            ("missing payloads", {}, {}, False, False),
            (
                "future payloads",
                dict(context, as_of=self.now + 60),
                {
                    arm: dict(circuit, as_of=self.now + 60)
                    for arm, circuit in circuits.items()
                },
                False,
                False,
            ),
        )
        for name, context_data, risk_data, context_ok, risk_ok in cases:
            with self.subTest(case=name):
                await self.write(context=context_data, risk_circuits=risk_data)
                text = await self.render_read_only("system")
                loops, separator, validity = text.partition("DATA VALIDITY")
                self.assertTrue(
                    separator, "Payload validity must be separately labeled"
                )
                self.assertIn("LOOP FRESHNESS", loops)
                self.assertRegex(loops, r"Market context: [^\n]*Current")
                self.assertRegex(loops, r"Risk circuits: [^\n]*Current")
                self.assertIn("Active component errors: 0", text)
                context_lines = [
                    line
                    for line in validity.splitlines()
                    if "context data" in line.lower()
                ]
                risk_lines = [
                    line
                    for line in validity.splitlines()
                    if "risk data" in line.lower()
                ]
                self.assertEqual(len(context_lines), 1)
                self.assertEqual(len(risk_lines), 2)
                for lines, available in (
                    (context_lines, context_ok),
                    (risk_lines, risk_ok),
                ):
                    for line in lines:
                        if available:
                            self.assertRegex(line, r"(?i):\s*(?:available|valid)\b")
                        else:
                            self.assertRegex(
                                line, r"(?i)\b(?:unavailable|stale|incomplete)\b"
                            )
                            self.assertNotRegex(line, r"(?i):\s*(?:available|valid)\b")
                home = await self.render_read_only("dashboard")
                if context_ok and risk_ok:
                    self.assertIn("🟢 Running", home)
                else:
                    self.assertNotIn("🟢 Running", home)
                    self.assertRegex(home, r"(?i)needs attention")

    async def test_ai_age_uses_review_timestamp_and_keeps_candidate_current(self):
        cases = (
            ("fresh", self.now - 60, r"reviewed\s+1m\b", False),
            ("six-hour boundary", self.now - 21600, r"reviewed\s+6h\b", False),
            ("older than six hours", self.now - 21601, r"reviewed\s+6h\b", True),
            ("day-old approval", self.now - 86400, r"reviewed\s+1d\b", True),
            ("missing review time", None, r"reviewed\s+unavailable\b", True),
            ("zero review time", 0, r"reviewed\s+unavailable\b", True),
            ("future review time", self.now + 60, r"reviewed\s+unavailable\b", True),
        )
        for name, stamp, expected_age, historical in cases:
            with self.subTest(case=name):
                rec = self.candidate("review-age", state="READY")
                rec["ai"] = {"verdict": "APPROVE"}
                if stamp is not None:
                    rec["ai"]["created_at"] = stamp
                await self.write(opportunities={"review-age": rec})
                text = await self.render_read_only("why", "BTC")
                ai_line = next(
                    line for line in text.splitlines() if line.startswith("AI:")
                )
                self.assertIn("APPROVE", ai_line)
                self.assertRegex(ai_line, expected_age)
                self.assertNotRegex(ai_line, r"(?:checked|reviewed)\s+0s\b")
                if historical:
                    self.assertRegex(ai_line, r"(?i)historical|age unavailable")
                else:
                    self.assertNotRegex(ai_line, r"(?i)historical|age unavailable")
                self.assertIn("Current: 1", text)
                self.assertIn("Ready for risk checks", text)

        await self.write(opportunities={"unreviewed": self.candidate("unreviewed")})
        text = await self.render_read_only("why", "BTC")
        ai_line = next(line for line in text.splitlines() if line.startswith("AI:"))
        self.assertRegex(ai_line, r"(?i)not reviewed")
        self.assertNotIn("ago", ai_line)

    async def test_home_warns_for_incomplete_closes_not_unfinished_open_records(self):
        for incomplete_close in (True, False):
            with self.subTest(incomplete_close=incomplete_close):
                closed = {
                    "id": "closed",
                    "arm": "baseline_shadow",
                    "status": "CLOSED",
                    "net_pnl": 10,
                    "net_r": 1,
                    "funding_complete": True,
                    "management_complete": not incomplete_close,
                }
                trades = {"closed": closed}
                for status in ("OPEN", "PENDING", "EXPIRED"):
                    trades[status] = dict(
                        closed, id=status, status=status, management_complete=False
                    )
                await self.write(trades=trades)
                text = await self.render_read_only("dashboard", paged=True)
                self.assertIn("Closed 1", text)
                self.assertIn("WR 100.0%", text)
                if incomplete_close:
                    self.assertRegex(text, r"(?i)provisional[^\n]*performance")
                else:
                    self.assertNotRegex(text, r"(?i)provisional|incomplete outcome")

    async def test_empty_radar_distinguishes_missing_stale_and_failed_scans(self):
        baseline_health = (await self.store.read())["health"]
        cases = (
            ("missing", None, None),
            ("zero", 0, None),
            ("stale", self.now - 3600, None),
            ("future", self.now + 60, None),
            ("failed", self.now, "TimeoutError"),
        )
        for name, stamp, error in cases:
            with self.subTest(case=name):
                health = copy.deepcopy(baseline_health)
                health.pop("scan_at", None)
                if stamp is not None:
                    health["scan_at"] = stamp
                if error:
                    health["scan_error"] = error
                await self.write(health=health, opportunities={})
                text = await self.render_read_only("opportunities", paged=True)
                self.assertIn("Current: 0", text)
                self.assertRegex(text, r"(?i)scanner[^\n]*(?:stale|unavailable)")
                self.assertNotRegex(text, r"(?i)waiting for confirmed structure")

        await self.write(health=baseline_health)
        fresh = await self.render_read_only("opportunities")
        self.assertIn("Current: 0", fresh)
        self.assertRegex(fresh, r"(?i)no current qualifying setup")
        self.assertNotRegex(fresh, r"(?i)scanner[^\n]*(?:stale|unavailable)")

    async def test_secrets_are_redacted_before_humanization_in_all_output_paths(self):
        secrets = (
            "sk-proj-REVIEW_ONLY_under_score12345",
            "BYBIT_REVIEW_ONLY_under_score67890",
        )
        self.config = replace(
            self.config, openai_key=secrets[0], bybit_secret=secrets[1]
        )
        self.service.config = self.config
        diagnostic = "Fixture diagnostic " + " / ".join(secrets)
        health = (await self.store.read())["health"]
        health["scan_error"] = diagnostic
        await self.write(
            health=health,
            research={
                "status": "UNAVAILABLE",
                "created_at": self.now,
                "reason": diagnostic,
            },
        )
        for view in ("research", "system", "telemetry"):
            for paged in (False, True):
                with self.subTest(view=view, paged=paged):
                    text = await self.render_read_only(view, paged=paged)
                    self.assertIn("Fixture diagnostic", text)
                    self.assertIn("[redacted]", text)
                    for secret in secrets:
                        self.assertNotIn(secret, text)
                        self.assertNotIn(secret.replace("_", " "), text)


class DashboardReviewPriceTests(unittest.TestCase):
    def test_tiny_nonzero_prices_preserve_value_and_zero_stays_zero(self):
        for value in (4e-9, 1e-12, -4e-9, 1.23456789e-8):
            with self.subTest(value=value):
                rendered = price(value)
                self.assertNotEqual(Decimal(rendered), 0)
                self.assertEqual(Decimal(rendered), Decimal(str(value)))
        self.assertEqual(price(0), "0")
