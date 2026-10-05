"""Real Store integration; isolated SQLite and explicit config, no network."""

from __future__ import annotations

import asyncio
import copy
import tempfile
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from apex_bot.config import Config
from apex_bot import risk
from apex_bot.accounting import loss_metrics
from apex_bot.models import Instrument
from apex_bot.service import Service, CAP_LABELS
from apex_bot.simulation import comparison
from apex_bot.storage import Store

USER, OTHER = 987654321, 987654322
CHAT = "-100123456789"
ACTOR = f"telegram:{CHAT}:{USER}"
OTHER_ACTOR = f"telegram:{CHAT}:{OTHER}"


def shadow_trade(
    arm, opportunity, pnl, side="Buy", setup="1", status="CLOSED", **changes
):
    record = {
        "id": arm + ":" + opportunity,
        "arm": arm,
        "opportunity_id": opportunity,
        "status": status,
        "net_pnl": pnl,
        "side": side,
        "setup": setup,
        "funding_complete": True,
        "management_complete": True,
        "closed_at": 1000,
        "created_at": 100,
        "entry_eligible_at": 180,
        "qty": 2,
        "limit": 100,
        "original_stop": 90,
        "target1": 120,
        "target2": 140,
        "cost_model": "fixture",
        "management_mode": "fixture",
    }
    record.update(changes)
    return record


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="apex-service-test-")
        self.path = self.directory.name + "/isolated.sqlite"
        self.store = Store(database_url="", sqlite_path=self.path)
        await self.store.initialize()
        self.assertTrue(await self.store.lease(ttl=10000))
        self.config = Config(
            telegram_chat_id=CHAT,
            telegram_user_ids=frozenset({USER, OTHER}),
            telegram_token="123456:FAKE_SECRET",
            shadow_equity=10000,
        )
        self.service = Service(self.config, self.store)
        self.now = time.time()
        self.context = {
            "data_complete": True,
            "as_of": self.now,
            "expires_at": self.now + 600,
            "risk_state": "neutral",
            "event_blackout": False,
            "long_multiplier": 1.0,
            "short_multiplier": 0.5,
        }
        self.account = {
            "as_of": self.now,
            "equity": 10000.0,
            "positions": [],
            "open_orders": [],
            "blockers": [],
            "currency": "USDT",
        }
        self.circuits = {
            key: {
                "as_of": self.now,
                "daily_loss_pct": 0.0,
                "weekly_loss_pct": 0.0,
                "drawdown_pct": 0.0,
            }
            for key in ("live", "testnet", "baseline_shadow", "ai_shadow")
        }
        await self.write(
            context=self.context,
            account=self.account,
            health={"scan_at": self.now},
            risk_circuits=self.circuits,
            risk_reference_equities={key: 10000.0 for key in self.circuits},
        )

    async def asyncTearDown(self):
        await self.store.close()
        self.directory.cleanup()

    async def write(self, **values):
        await self.store.update(lambda tx: tx.state.update(copy.deepcopy(values)))

    def live(self):
        self.config = replace(
            self.config, mode="live", live_enabled=True, migration_verified=True
        )
        self.service = Service(self.config, self.store)

    async def test_all_views_render_empty_state_and_no_external_reads(self):
        for view in (
            "dashboard",
            "status",
            "universe",
            "risk",
            "profile",
            "positions",
            "opportunities",
            "why",
            "waves",
            "performance",
            "research",
            "system",
            "telemetry",
        ):
            with self.subTest(view=view):
                text = await self.service.snapshot(view)
                self.assertIsInstance(text, str)
                self.assertTrue(text)
        self.assertIn(
            "No closed trades = no measured win rate",
            await self.service.snapshot("performance"),
        )
        self.assertEqual((await self.store.read())["settings_version"], 0)

    async def test_universe_lists_all_fifty_persisted_members_and_liquidity(self):
        symbols = [f"COIN{i:02}USDT" for i in range(50)]
        self.service.config = replace(
            self.config, universe_mode="dynamic", symbols=("MANUALUSDT",)
        )
        members = {
            symbol: {
                "status": "eligible",
                "rank": i + 1,
                "score": 80_000_000 - i * 1_000_000,
                "turnover24h": 100_000_000 - i * 1_000_000,
                "spread_bps": 2.5,
                "bid_depth_usdt": 50_000,
                "ask_depth_usdt": 75_000,
                "sample_count": 28,
            }
            for i, symbol in enumerate(symbols)
        }
        members["OUTSIDEUSDT"] = {"score": 1, "turnover24h": 1}
        await self.write(
            universe={
                "mode": "dynamic",
                "active_symbols": symbols,
                "target": 50,
                "as_of": self.now - 3600,
                "status": "ready",
                "members": members,
                "blocked": {"THINUSDT": ["spread above limit", "insufficient depth"]},
                "basis": "7d median daily turnover",
                "policy": {
                    "target": 50,
                    "min_turnover_usdt": 20_000_000,
                    "max_spread_bps": 10,
                    "min_depth_usdt": 25_000,
                    "review_hours": 6,
                    "daily_max_replacements": 5,
                },
                "day_replacements": 2,
                "added": [symbols[-1]],
                "removed": ["OLDUSDT"],
                "draining_symbols": ["OLDUSDT", "OTHERUSDT"],
                "candidate_count": 120,
                "book_checked": 75,
                "snapshot_time": self.now,
            },
            orders={
                "old": {"symbol": "OLDUSDT", "status": "OPEN"},
                "other": {"symbol": "OTHERUSDT", "status": "PENDING"},
            },
        )
        before = await self.store.read()
        notifications = await self.store.pending_notifications()
        with patch.object(
            self.store, "update", side_effect=AssertionError("read-only")
        ):
            with patch("apex_bot.service.time.time", return_value=self.now):
                text = await self.service.snapshot("universe")
                for view in ("dashboard", "status", "start"):
                    dashboard = await self.service.snapshot(view)
                    self.assertIn("Universe: 50/50 active · dynamic", dashboard)
                    self.assertIn("1h 0m ago · fresh", dashboard)
                    self.assertIn("Draining: 2", dashboard)
                    self.assertIn("Eligible for entry: 50/50", dashboard)
        self.assertEqual(await self.store.read(), before)
        self.assertEqual(await self.store.pending_notifications(), notifications)
        listing = text.split("Active symbols (stored rank when available):\n")[1].split(
            "\n\n"
        )[0]
        self.assertEqual(len(listing.splitlines()), 25)
        for i, symbol in enumerate(symbols):
            self.assertIn(f"{i + 1}. {symbol}", listing)
        for excluded in ("MANUALUSDT", "OUTSIDEUSDT", "OLDUSDT", "OTHERUSDT"):
            self.assertNotIn(excluded, listing)
        for expected in (
            "Universe · read-only",
            "Selection status: ready",
            " UTC ·",
            "Ranking basis: 7d median daily turnover",
            "Review every 6h",
            "Healthy replacements ≤5/day",
            "Replacements today: 2",
            "Added: COIN49USDT",
            "Removed: OLDUSDT",
            "Candidates checked: 120",
            "Order books checked: 75",
            "Score: 31M–80M (50/50)",
            "24h turnover: 51M–100M USDT (50/50)",
            "Spread: 2.5–2.5 bps (50/50)",
            "Bid depth: 50K–50K USDT (50/50)",
            "Ask depth: 75K–75K USDT (50/50)",
            "History samples: 28–28 per member (50/50)",
            "Market snapshot:",
            "Blocked: 1",
            "THINUSDT: spread above limit, insufficient depth",
        ):
            self.assertIn(expected, text)
        self.assertLess(len(text.encode("utf-16-le")) // 2, 3900)

    async def test_dynamic_universe_missing_state_awaits_selection_without_manual_fallback(
        self,
    ):
        self.service.config = replace(
            self.config,
            universe_mode="dynamic",
            universe_size=40,
            symbols=("MANUALUSDT",),
        )
        for universe in (None, {}, "invalid", {"active_symbols": None}):
            await self.write(universe=universe)
            for view in ("universe", "dashboard", "status"):
                with self.subTest(universe=universe, view=view):
                    text = await self.service.snapshot(view)
                    self.assertIn("Universe: 0/40 active · dynamic", text)
                    self.assertIn("Awaiting selection · entry wait", text)
                    self.assertIn("freshness unknown", text)
                    self.assertNotIn("MANUALUSDT", text)
                    self.assertNotIn("configured · manual", text)

    async def test_universe_static_and_legacy_config_are_explicitly_manual(self):
        await self.write(universe={"active_symbols": ["DYNAMICUSDT"], "target": 50})
        static = replace(
            self.config, universe_mode="static", symbols=("BTCUSDT", "ETHUSDT")
        )
        legacy = SimpleNamespace(
            **{
                key: value
                for key, value in vars(static).items()
                if key != "universe_mode"
            }
        )
        for config in (static, legacy):
            self.service.config = config
            text = await self.service.snapshot("universe")
            self.assertIn("Universe: 2 configured · manual (static)", text)
            self.assertIn("BTCUSDT · ETHUSDT", text)
            self.assertNotIn("DYNAMICUSDT", text)
            self.assertNotIn("Awaiting selection", text)
            self.assertNotIn("Policy:", text)
            self.assertIn(
                "Universe: 2 configured · manual (static)",
                await self.service.snapshot("dashboard"),
            )

    async def test_universe_draining_is_derived_from_current_ledgers(self):
        self.service.config = replace(self.config, universe_mode="dynamic")
        await self.write(
            universe={
                "active_symbols": ["BTCUSDT"],
                "as_of": self.now,
                "status": "degraded",
                "members": {"BTCUSDT": {"status": "blocked_pending_fresh"}},
                "draining_symbols": ["STALEUSDT"],
            },
            trades={
                "active": {"symbol": "BTCUSDT", "status": "OPEN"},
                "open": {"symbol": "ETHUSDT", "status": "OPEN"},
                "pending": {"symbol": "SOLUSDT", "status": "PENDING"},
                "expired": {"symbol": "EXPIREDUSDT", "status": "EXPIRED"},
                "closed": {"symbol": "CLOSEDUSDT", "status": "CLOSED"},
            },
            orders={
                "duplicate": {"symbol": "ETHUSDT", "status": "OPEN"},
                "closing": {"symbol": "LINKUSDT", "status": "CLOSING"},
                "cancelled": {"symbol": "CANCELLEDUSDT", "status": "CANCELLED"},
                "rejected": {"symbol": "REJECTEDUSDT", "status": "REJECTED"},
                "closed": {"symbol": "CLOSEDUSDT", "status": "CLOSED"},
            },
        )
        before = await self.store.read()
        text = await self.service.snapshot("universe")
        self.assertIn("Draining: 3", text)
        self.assertIn("Eligible for entry: 0/1", text)
        self.assertEqual(await self.store.read(), before)
        await self.store.update(
            lambda tx: (
                [
                    trade.update(status="CLOSED")
                    for trade in tx.state["trades"].values()
                ],
                [
                    order.update(status="CLOSED")
                    for order in tx.state["orders"].values()
                ],
            )
        )
        self.assertIn("Draining: 0", await self.service.snapshot("universe"))
        self.assertEqual((await self.store.read())["universe"], before["universe"])

    async def test_dashboard_and_opportunities_use_current_eligible_unexpired_membership(
        self,
    ):
        from .test_risk import opportunity

        self.service.config = replace(self.config, universe_mode="dynamic")
        base = opportunity()
        records = {}
        for name, symbol, expiry in (
            ("current", "BTCUSDT", self.now + 600),
            ("blocked", "ETHUSDT", self.now + 600),
            ("retired", "SOLUSDT", self.now + 600),
            ("expired", "BTCUSDT", self.now),
        ):
            op = replace(base, id=name, symbol=symbol, expires_at=expiry, state="READY")
            records[name] = {
                "opportunity": op.to_dict(),
                "updated_at": self.now,
                "decision": "Historical approval",
            }
        universe = {
            "mode": "dynamic",
            "active_symbols": ["BTCUSDT", "ETHUSDT"],
            "as_of": self.now,
            "status": "degraded",
            "members": {
                "BTCUSDT": {"status": "eligible"},
                "ETHUSDT": {"status": "blocked_pending_fresh"},
            },
            "blocked": {"ETHUSDT": ["book_missing"]},
            "policy": {
                "target": 50,
                "min_turnover_usdt": 20_000_000,
                "max_spread_bps": 10,
                "min_depth_usdt": 25_000,
            },
        }
        await self.write(universe=universe, opportunities=records)
        before = await self.store.read()
        with patch("apex_bot.service.time.time", return_value=self.now):
            for view in ("dashboard", "status", "start"):
                text = await self.service.snapshot(view)
                self.assertIn("Candidates: 4 · Ready: 1", text)
                self.assertIn("Eligible for entry: 1/2", text)
            for view in ("opportunities", "why", "waves"):
                text = await self.service.snapshot(view)
                self.assertEqual(text.count("WAIT (stored READY)"), 3)
                self.assertEqual(text.count("outside the current eligible universe"), 2)
                self.assertIn("WAIT: candidate expired", text)
                for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
                    self.assertIn(symbol, text)
        self.assertEqual(await self.store.read(), before)
        universe["as_of"] = self.now - 86400
        await self.write(universe=universe)
        self.assertIn("Ready: 0", await self.service.snapshot("dashboard"))
        self.assertIn(
            "Eligible for entry: 0/2", await self.service.snapshot("universe")
        )
        text = await self.service.snapshot("opportunities")
        self.assertEqual(text.count("WAIT (stored READY)"), 4)

    async def test_universe_stale_boundary_retains_members_and_waits_for_entries(self):
        self.service.config = replace(self.config, universe_mode="dynamic")
        for age, stale in ((86399, False), (86400, True), (172800, True)):
            await self.write(
                universe={
                    "active_symbols": ["BTCUSDT", "ETHUSDT"],
                    "target": 50,
                    "as_of": self.now - age,
                    "status": "ready",
                    "draining_symbols": [],
                    "snapshot_time": self.now,
                }
            )
            before = await self.store.read()
            with patch("apex_bot.service.time.time", return_value=self.now):
                for view in ("universe", "dashboard", "status"):
                    with self.subTest(age=age, view=view):
                        text = await self.service.snapshot(view)
                        self.assertIn("Universe: 2/50 active", text)
                        self.assertEqual("STALE · entry wait" in text, stale)
                        self.assertEqual("ago · fresh" in text, not stale)
                        self.assertIn("Draining: 0", text)
                self.assertIn(
                    "BTCUSDT · ETHUSDT", await self.service.snapshot("universe")
                )
            self.assertEqual(await self.store.read(), before)

    async def test_universe_policy_change_shows_wait_until_selection_matches_config(
        self,
    ):
        from .test_risk import opportunity

        op = replace(
            opportunity(), symbol="BTCUSDT", state="READY", expires_at=self.now + 600
        )
        original_policy = {
            "target": 50,
            "min_turnover_usdt": 20_000_000,
            "max_spread_bps": 10,
            "min_depth_usdt": 25_000,
        }
        universe = {
            "mode": "dynamic",
            "status": "ready",
            "target": 50,
            "active_symbols": ["BTCUSDT"],
            "as_of": self.now,
            "members": {"BTCUSDT": {"status": "eligible"}},
            "policy": original_policy,
        }
        for field, key, value in (
            ("universe_min_turnover", "min_turnover_usdt", 25_000_000),
            ("universe_max_spread_bps", "max_spread_bps", 5),
            ("universe_min_depth", "min_depth_usdt", 50_000),
            ("universe_size", "target", 40),
        ):
            self.service.config = replace(
                self.config, universe_mode="dynamic", **{field: value}
            )
            await self.write(
                universe=universe,
                universe_policy=original_policy,
                opportunities={op.id: {"opportunity": op.to_dict()}},
            )
            before = await self.store.read()
            self.assertIn(
                "Eligible for entry: 0/1", await self.service.snapshot("universe")
            )
            self.assertIn("Ready: 0", await self.service.snapshot("dashboard"))
            text = await self.service.snapshot("opportunities")
            self.assertIn("WAIT (stored READY)", text)
            self.assertIn("outside the current eligible universe", text)
            self.assertEqual(await self.store.read(), before)
            matching = {**universe, "policy": {**original_policy, key: value}}
            if key == "target":
                matching["target"] = value
            await self.write(universe=matching)
            self.assertIn(
                "Eligible for entry: 1/1", await self.service.snapshot("universe")
            )
            self.assertIn("Ready: 1", await self.service.snapshot("dashboard"))
            self.assertNotIn(
                "WAIT (stored READY)", await self.service.snapshot("opportunities")
            )

    async def test_universe_invalid_time_partial_metrics_and_unknown_policy(self):
        self.service.config = replace(self.config, universe_mode="dynamic")
        for stamp in (None, 0, -1, True, "yesterday", self.now + 60):
            await self.write(
                universe={
                    "active_symbols": ["BTCUSDT", "ETHUSDT"],
                    "as_of": stamp,
                    "members": {"BTCUSDT": {"spread_bps": 3}, "ETHUSDT": None},
                    "added": [],
                    "removed": [],
                    "day_replacements": 0,
                }
            )
            text = await self.service.snapshot("universe")
            self.assertIn("freshness unknown · entry wait", text)
            self.assertIn("Policy: unavailable (not persisted)", text)
            self.assertIn("Ranking basis: unavailable (not persisted)", text)
            self.assertIn("Draining: 0", text)
            self.assertIn("Spread: 3–3 bps (1/2)", text)
            self.assertIn("24h turnover: unavailable", text)
            self.assertIn("Replacements today: 0", text)
            self.assertIn("Added: none", text)
            self.assertIn("Removed: none", text)
            for invented in ("7d median", "20M", "25K", "1.5x", "6h", "30d"):
                self.assertNotIn(invented, text)

    async def test_universe_refresh_error_preserves_last_successful_selection_age(self):
        self.service.config = replace(self.config, universe_mode="dynamic")
        for age in (21600, 86400):
            await self.write(
                universe={
                    "mode": "dynamic",
                    "active_symbols": ["BTCUSDT"],
                    "target": 50,
                    "as_of": self.now - age,
                    "status": "ready",
                    "draining_symbols": ["OLDUSDT"],
                },
                health={
                    "scan_at": self.now,
                    "universe_at": self.now - age,
                    "universe_attempt_at": self.now,
                    "universe_error": "TimeoutError " + self.config.telegram_token,
                },
            )
            before = await self.store.read()
            with patch.object(
                self.store, "update", side_effect=AssertionError("read-only")
            ):
                with patch("apex_bot.service.time.time", return_value=self.now):
                    for view in ("universe", "dashboard", "status", "start"):
                        text = await self.service.snapshot(view)
                        self.assertIn(
                            "Universe refresh error: TimeoutError [redacted]", text
                        )
                        self.assertNotIn(self.config.telegram_token, text)
                        self.assertIn("Universe: 1/50 active", text)
                        self.assertIn(f"{age // 3600}h 0m ago", text)
                        self.assertEqual("STALE · entry wait" in text, age >= 86400)
                        self.assertIn("Draining: 0", text)
                        self.assertNotIn("Awaiting selection", text)
            self.assertEqual(await self.store.read(), before)
        await self.write(health={"scan_at": self.now, "universe_attempt_at": self.now})
        self.assertNotIn(
            "Universe refresh error", await self.service.snapshot("universe")
        )

    async def test_universe_failed_initial_selection_does_not_invent_snapshot(self):
        self.service.config = replace(self.config, universe_mode="dynamic")
        for universe in (None, {"active_symbols": [], "as_of": 0, "snapshot_time": 0}):
            await self.write(
                universe=universe,
                health={
                    "scan_at": self.now,
                    "universe_at": self.now,
                    "universe_attempt_at": self.now,
                    "universe_error": "ValueError",
                },
            )
            for view in ("universe", "dashboard", "status"):
                text = await self.service.snapshot(view)
                self.assertIn("Universe: 0/50 active", text)
                self.assertIn("Awaiting selection · entry wait", text)
                self.assertIn("Universe refresh error: ValueError", text)
                self.assertIn("Selection as of: unavailable", text)
                self.assertNotIn("1970", text)
                self.assertNotIn("ago · fresh", text)
                if view == "universe" and universe is not None:
                    self.assertIn("Market snapshot: unavailable", text)

    async def test_universe_history_counts_and_blocked_reasons_are_bounded_and_redacted(
        self,
    ):
        self.service.config = replace(self.config, universe_mode="dynamic")
        private = f"{CHAT} {USER} {ACTOR} {self.config.telegram_token}"
        await self.write(
            universe={
                "active_symbols": ["BTCUSDT"],
                "as_of": self.now,
                "basis": {"daily_samples": 7, "snapshot_samples": 28},
                "policy": {"description": private},
                "blocked": {f"BLOCK{i}USDT": private for i in range(20)},
            }
        )
        text = await self.service.snapshot("universe")
        self.assertIn("daily samples: 7; snapshot samples: 28", text)
        self.assertIn("Blocked: 20", text)
        self.assertIn("+15 more blocked symbols", text)
        for value in (CHAT, str(USER), ACTOR, self.config.telegram_token):
            self.assertNotIn(value, text)

    async def test_why_explains_rr_bound_without_calling_it_permission(self):
        from .test_risk import opportunity

        op = opportunity(target1=108, target2=112)
        sizing = risk.assess(
            op,
            Instrument("TESTUSDT", 0.01, 0.01, 100000, 0.01, 5),
            10000,
            exposures=[],
            funding_rate_8h=0,
            spread_pct=0.01,
            now=op.evidence["as_of"],
        )
        self.assertFalse(sizing["allowed"])
        await self.write(
            opportunities={
                op.id: {
                    "opportunity": op.to_dict(),
                    "updated_at": self.now,
                    "decision": "; ".join(sizing["reasons"]),
                    "last_risk_review": {
                        "arm": "baseline_shadow",
                        "assessed_at": self.now,
                        "assessment": sizing,
                    },
                }
            }
        )
        text = await self.service.snapshot("why", symbol="TESTUSDT")
        self.assertIn("Net R:R", text)
        self.assertIn("need 1.5R", text)
        self.assertIn("need 2.5R", text)
        self.assertIn("R:R-only entry bound: ≤", text)
        self.assertIn("Diagnostic only", text)
        self.assertIn("every risk check", text)

    async def test_risk_menu_profiles_cash_full_caps_and_hypothetical_label(self):
        text = await self.service.snapshot("risk")
        self.assertIn("Hypothetical starting capital", text)
        self.assertIn("not current account equity", text)
        self.assertIn("10,000.00 USDT", text)
        for name, policy in risk.PROFILES.items():
            self.assertIn(name, text)
            self.assertIn(f"{10000 * policy['risk_pct'] / 100:,.2f} USDT", text)
        for key in risk.HARD_CAPS:
            self.assertIn(CAP_LABELS[key], text)
        self.assertNotIn("Reconciled exchange equity", text)

    async def test_every_profile_preview_has_old_new_caps_and_never_applies(self):
        await self.store.update(lambda tx: tx.state["settings"].update(risk_pct=0.37))
        old = copy.deepcopy((await self.store.read())["settings"])
        for name, policy in risk.PROFILES.items():
            preview = await self.service.propose_change("profile", name, ACTOR)
            text = preview["summary"]
            self.assertIn(f"Profile: cautious → {name}", text)
            self.assertIn("Custom override: 0.37 → None", text)
            self.assertIn("0.37% (37.00 USDT) →", text)
            self.assertIn(f"{10000 * policy['risk_pct'] / 100:,.2f} USDT", text)
            for label in CAP_LABELS.values():
                self.assertIn(label + ":", text)
            self.assertIn("future entries", text)
            self.assertIn("existing exchange stops/protection", text)
            self.assertLessEqual(len(("confirm:" + preview["token"]).encode()), 64)
            self.assertEqual((await self.store.read())["settings"], old)
        self.assertEqual(await self.store.pending_notifications(), [])

    async def test_caps_clamp_to_hard_limits_and_policy_change_invalidates_preview(
        self,
    ):
        self.live()
        policies = copy.deepcopy(risk.PROFILES)
        policies["extreme"]["heat_pct"] = 999
        with patch.dict(risk.PROFILES, policies, clear=True):
            preview = await self.service.propose_change("profile", "extreme", ACTOR)
            self.assertIn(
                "Portfolio heat: 4% (400.00 USDT) → 4% (400.00 USDT)",
                preview["summary"],
            )
        result = await self.service.confirm_change(preview["token"], ACTOR)
        self.assertIn("changed", result)
        self.assertEqual((await self.store.read())["settings_version"], 0)

    async def test_service_validates_all_keys_and_percent_units_independently(self):
        for value in (0.05, 0.25, 1):
            proposal = await self.service.propose_change("risk_pct", value, ACTOR)
            await self.service.confirm_change(proposal["token"], ACTOR)
            self.assertEqual((await self.store.read())["settings"]["risk_pct"], value)
        for value in (
            True,
            False,
            "0.25",
            0.04999,
            1.0001,
            -1,
            float("nan"),
            float("inf"),
            None,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await self.service.propose_change("risk_pct", value, ACTOR)
        for key, value in (
            ("mode", "live"),
            ("reset", True),
            ("paused", "false"),
            ("profile", "wild"),
        ):
            with self.assertRaises(ValueError):
                await self.service.propose_change(key, value, ACTOR)
        for key, value in (
            ("paused", False),
            ("profile", "extreme"),
            ("risk_pct", 1),
            ("reset", True),
        ):
            with self.assertRaises(ValueError):
                await self.service.change_setting(key, value, ACTOR)

    async def test_configured_chat_sender_required_for_all_service_mutations(self):
        for actor in ("system", "7", f"telegram:{CHAT}:42", f"telegram:42:{USER}"):
            for method, args in (
                ("propose_change", ("paused", True)),
                ("confirm_change", ("unknown",)),
                ("cancel_change", ("unknown",)),
                ("change_setting", ("paused", True)),
            ):
                with self.subTest(actor=actor, method=method), self.assertRaises(
                    ValueError
                ):
                    await getattr(self.service, method)(*args, actor)
        disabled = Service(
            replace(self.config, telegram_user_ids=frozenset()), self.store
        )
        with self.assertRaises(ValueError):
            await disabled.propose_change("paused", True, ACTOR)
        disabled = Service(
            replace(self.config, telegram_chat_id="OWNER_CHAT_ID"), self.store
        )
        with self.assertRaises(ValueError):
            await disabled.change_setting("paused", True, ACTOR)

    async def test_live_uses_current_account_equity_and_time_without_fallback(self):
        self.live()
        await self.write(account=dict(self.account, equity=4321.0))
        text = await self.service.snapshot("risk")
        self.assertIn(
            "Reconciled USDT coin equity (not account-total USD): 4,321.00 USDT", text
        )
        self.assertIn("Account as of:", text)
        self.assertIn("10.80 USDT", text)
        self.assertNotIn("Hypothetical", text)
        preview = await self.service.propose_change("risk_pct", 0.5, ACTOR)
        self.assertIn("0.25% (10.80 USDT) → 0.5% (21.60 USDT)", preview["summary"])

    async def test_live_stale_missing_invalid_equity_never_uses_shadow_capital(self):
        self.live()
        for account in (
            {},
            dict(self.account, as_of=self.now - 31),
            dict(self.account, as_of=self.now + 30),
            dict(self.account, equity=0),
            dict(self.account, equity=-1),
            dict(self.account, equity=True),
            dict(self.account, equity="10000"),
            dict(self.account, currency="USD"),
        ):
            with self.subTest(account=account):
                await self.write(account=account)
                for view in ("status", "risk", "profile"):
                    text = await self.service.snapshot(view)
                    self.assertIn("no fallback", text)
                    self.assertIn("cash unavailable", text)
                    self.assertNotIn("10,000.00", text)
                self.assertIn(
                    "do not interpret this as flat",
                    await self.service.snapshot("positions"),
                )
                with self.assertRaises(ValueError):
                    await self.service.propose_change("profile", "extreme", ACTOR)

    async def test_equity_changed_or_staled_after_preview_cannot_confirm(self):
        self.live()
        for update in (
            dict(self.account, equity=11000),
            dict(self.account, as_of=self.now - 31),
        ):
            await self.write(account=self.account)
            preview = await self.service.propose_change("risk_pct", 0.75, ACTOR)
            await self.write(account=update)
            result = await self.service.confirm_change(preview["token"], ACTOR)
            self.assertIn("fresh preview", result)
            self.assertEqual((await self.store.read())["settings_version"], 0)
            self.assertNotIn(
                preview["token"], (await self.store.read())["confirmations"]
            )

    async def test_account_refresh_same_equity_does_not_invalidate_preview(self):
        self.live()
        preview = await self.service.propose_change("risk_pct", 0.75, ACTOR)
        await self.write(account=dict(self.account, as_of=time.time()))
        self.assertIn(
            "Saved", await self.service.confirm_change(preview["token"], ACTOR)
        )

    async def test_pause_during_outage_preserves_all_existing_records(self):
        self.live()
        trades = {
            "sim": {"status": "OPEN", "arm": "baseline_shadow", "stop": 90},
            "pending": {"status": "PENDING", "arm": "ai_shadow", "stop": 91},
        }
        orders = {"real": {"status": "OPEN", "stop": 80}}
        await self.write(
            account={}, context={}, health={}, trades=trades, orders=orders
        )
        proposal = await self.service.propose_change("paused", True, ACTOR)
        self.assertIn("cash unavailable", proposal["summary"])
        await self.service.confirm_change(proposal["token"], ACTOR)
        state = await self.store.read()
        self.assertTrue(state["settings"]["paused"])
        self.assertEqual(state["trades"], trades)
        self.assertEqual(state["orders"], orders)
        with self.assertRaises(ValueError):
            await self.service.propose_change("paused", False, ACTOR)

    async def test_resume_rechecks_all_current_dependencies_in_transaction(self):
        self.live()
        await self.service.change_setting("paused", True, ACTOR)
        bad_states = [
            {"context": {}},
            {"context": dict(self.context, as_of=self.now - 21601)},
            {"context": dict(self.context, as_of=self.now + 20)},
            {"context": dict(self.context, expires_at=self.now - 1)},
            {"context": dict(self.context, data_complete=False)},
            {"context": dict(self.context, event_blackout=True)},
            {"context": dict(self.context, policy_authority=False)},
            {"context": dict(self.context, long_multiplier=0, short_multiplier=0)},
            {"context": dict(self.context, long_multiplier="1")},
            {"account": dict(self.account, blockers=["Unprotected position"])},
            {"account": dict(self.account, as_of=self.now - 31)},
            {"account": dict(self.account, hard_halt=True)},
            {"health": {"scan_at": self.now - 1000}},
            {"health": {"scan_at": self.now, "blockers": ["Critical"]}},
            *(
                {"health": {"scan_at": self.now, component + "_error": "Unavailable"}}
                for component in ("scan", "context", "risk", "reconcile")
            ),
            {"risk": {"hard_halt": True}},
            {"hard_halt": True},
            {"risk": {"daily_loss_pct": 2}},
            {"risk": {"weekly_loss_pct": 4}},
            {"risk": {"drawdown_pct": 12}},
            {"risk": {"daily_loss_pct": "unknown"}},
        ]
        for change in bad_states:
            with self.subTest(change=change):
                await self.write(
                    context=self.context,
                    account=dict(self.account, as_of=time.time()),
                    health={"scan_at": self.now},
                    risk={},
                    hard_halt=False,
                )
                preview = await self.service.propose_change("paused", False, ACTOR)
                await self.write(**change)
                result = await self.service.confirm_change(preview["token"], ACTOR)
                self.assertNotIn("✅", result)
                self.assertTrue((await self.store.read())["settings"]["paused"])
                with self.assertRaises(ValueError):
                    await self.service.propose_change("paused", False, ACTOR)

    async def test_resume_accepts_verified_context_asof_contract_and_never_changes_mode(
        self,
    ):
        self.live()
        await self.service.change_setting("paused", True, ACTOR)
        context = dict(self.context)
        context["asof"] = context.pop("as_of")
        context.pop("data_complete")
        context.update(
            fresh=True,
            live_blocked=False,
            policy_authority=True,
            fresh_until=self.now + 120,
        )
        await self.write(context=context)
        preview = await self.service.propose_change("paused", False, ACTOR)
        self.assertIn(
            "Saved", await self.service.confirm_change(preview["token"], ACTOR)
        )
        self.assertFalse((await self.store.read())["settings"]["paused"])
        self.assertEqual(self.config.mode, "live")
        denied = Service(replace(self.config, live_enabled=False), self.store)
        with self.assertRaises(ValueError):
            await denied.propose_change("paused", False, ACTOR)

    async def test_shadow_resume_needs_context_but_not_exchange_account(self):
        await self.write(account={})
        await self.service.change_setting("paused", True, ACTOR)
        preview = await self.service.propose_change("paused", False, ACTOR)
        self.assertIn(
            "Saved", await self.service.confirm_change(preview["token"], ACTOR)
        )
        self.assertEqual(self.config.mode, "shadow")
        self.assertFalse(self.config.live_enabled)

    async def test_realized_strategy_losses_enforce_hard_halts(self):
        self.live()
        await self.write(
            orders={
                "loss": {
                    "status": "CLOSED",
                    "closed_at": self.now - 1,
                    "net_pnl_before_funding": -200,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "daily_loss_pct"):
            await self.service.propose_change("paused", False, ACTOR)
        self.config = replace(self.config, mode="shadow")
        self.service = Service(self.config, self.store)
        await self.write(
            trades={
                "loss": {
                    "status": "CLOSED",
                    "closed_at": self.now - 1,
                    "net_pnl": -200,
                    "arm": "ai_shadow",
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "ai_shadow daily_loss_pct"):
            await self.service.propose_change("paused", False, ACTOR)

    async def test_missing_stale_future_invalid_or_halted_circuits_block_resume(self):
        self.live()
        for circuit in (
            {},
            dict(self.circuits["live"], as_of=self.now - 121),
            dict(self.circuits["live"], as_of=self.now + 1),
            dict(self.circuits["live"], daily_loss_pct=True),
            dict(self.circuits["live"], weekly_loss_pct=-1),
            dict(self.circuits["live"], hard_halt=True),
        ):
            await self.write(risk_circuits={"live": circuit})
            with self.assertRaises(ValueError):
                await self.service.propose_change("paused", False, ACTOR)
        await self.write(risk_circuits=self.circuits)
        proposal = await self.service.propose_change("paused", False, ACTOR)
        await self.write(
            risk_circuits={"live": dict(self.circuits["live"], as_of=self.now - 121)}
        )
        result = await self.service.confirm_change(proposal["token"], ACTOR)
        self.assertNotIn("✅", result)
        self.assertEqual((await self.store.read())["settings_version"], 0)

    async def test_fresh_circuit_halt_is_honored_without_clearing_on_read(self):
        self.live()
        for metric, limit in (
            ("daily_loss_pct", 2),
            ("weekly_loss_pct", 4),
            ("drawdown_pct", 12),
        ):
            circuits = {"live": dict(self.circuits["live"], **{metric: limit})}
            await self.write(risk_circuits=circuits)
            with self.assertRaises(ValueError):
                await self.service.propose_change("paused", False, ACTOR)
            for view in ("risk", "dashboard", "telemetry"):
                text = await self.service.snapshot(view)
                self.assertIn("Strategy MTM risk gates", text)
                self.assertIn("not full-account performance", text)
                self.assertIn("HALTED", text)
                self.assertIn("entire current open loss", text)
                self.assertIn("not exact calendar-period returns", text)
            self.assertEqual((await self.store.read())["risk_circuits"], circuits)

    async def test_shared_accounting_open_exchange_loss_blocks_even_with_zero_circuit(
        self,
    ):
        self.live()
        await self.write(
            account=dict(
                self.account, positions=[{"size": "1", "unrealisedPnl": "-250"}]
            )
        )
        with patch("apex_bot.service.loss_metrics", wraps=loss_metrics) as shared:
            with self.assertRaisesRegex(ValueError, "daily_loss_pct"):
                await self.service.propose_change("paused", False, ACTOR)
            self.assertTrue(shared.called)
            self.assertEqual(shared.call_args.args[-1], "live")
        await self.write(account=dict(self.account, positions=[{"size": "1"}]))
        with self.assertRaisesRegex(ValueError, "accounting"):
            await self.service.propose_change("paused", False, ACTOR)

    async def test_shadow_open_mtm_freshness_and_loss_checks_use_shared_helper(self):
        trade = {
            "arm": "baseline_shadow",
            "status": "OPEN",
            "side": "Buy",
            "entry": 100.0,
            "remaining": 10.0,
            "net_pnl": 0.0,
            "mark_price": 70.0,
            "mark_at": self.now,
        }
        await self.write(trades={"open": trade})
        with self.assertRaisesRegex(ValueError, "baseline_shadow daily_loss_pct"):
            await self.service.propose_change("paused", False, ACTOR)
        await self.write(
            trades={"open": dict(trade, mark_price=100, mark_at=self.now - 601)}
        )
        with self.assertRaisesRegex(ValueError, "accounting"):
            await self.service.propose_change("paused", False, ACTOR)
        await self.write(trades={"open": dict(trade, mark_price=100, mark_at=self.now)})
        proposal = await self.service.propose_change("paused", False, ACTOR)
        self.assertIn(
            "Saved", await self.service.confirm_change(proposal["token"], ACTOR)
        )

    async def test_context_and_drawdown_effective_budgets_and_reference_bases(self):
        self.live()
        await self.write(
            risk_circuits={"live": dict(self.circuits["live"], drawdown_pct=8)},
            risk_reference_equities={"live": 15000},
        )
        preview = await self.service.propose_change("profile", "aggressive", ACTOR)
        self.assertIn(
            "Long tier-1 budget after context/drawdown: 0.375% (37.50 USDT)",
            preview["summary"],
        )
        self.assertIn(
            "Short tier-1 budget after context/drawdown: 0.1875% (18.75 USDT)",
            preview["summary"],
        )
        self.assertEqual(
            (await self.store.read())["risk_reference_equities"], {"live": 15000}
        )

    async def test_pause_survives_store_restart_and_confirm_applies_once_outbox_once(
        self,
    ):
        preview = await self.service.propose_change("paused", True, ACTOR)
        await self.store.close()
        self.store = Store(database_url="", sqlite_path=self.path)
        await self.store.initialize()
        self.assertTrue(await self.store.lease(ttl=10000))
        self.service = Service(self.config, self.store)
        await asyncio.gather(
            *(self.service.confirm_change(preview["token"], ACTOR) for _ in range(3))
        )
        state = await self.store.read()
        self.assertTrue(state["settings"]["paused"])
        self.assertEqual(state["settings_version"], 1)
        self.assertEqual(len(await self.store.pending_notifications()), 1)

    async def test_cancel_is_durable_actor_bound_and_preserves_settings(self):
        preview = await self.service.propose_change("profile", "extreme", ACTOR)
        before = (await self.store.read())["settings"]
        result = await self.service.cancel_change(preview["token"], OTHER_ACTOR)
        self.assertIn("unavailable", result)
        self.assertIn(preview["token"], (await self.store.read())["confirmations"])
        self.assertIn(
            "revoked", await self.service.cancel_change(preview["token"], ACTOR)
        )
        await self.store.close()
        self.store = Store(database_url="", sqlite_path=self.path)
        await self.store.initialize()
        self.assertTrue(await self.store.lease(ttl=10000))
        self.service = Service(self.config, self.store)
        await self.service.confirm_change(preview["token"], ACTOR)
        self.assertEqual((await self.store.read())["settings"], before)
        self.assertEqual((await self.store.read())["settings_version"], 0)
        self.assertEqual(await self.store.pending_notifications(), [])

    async def test_confirm_cancel_race_has_one_outcome(self):
        preview = await self.service.propose_change("profile", "aggressive", ACTOR)
        results = await asyncio.gather(
            self.service.cancel_change(preview["token"], ACTOR),
            self.service.confirm_change(preview["token"], ACTOR),
        )
        state = await self.store.read()
        self.assertNotIn(preview["token"], state["confirmations"])
        self.assertIn(state["settings_version"], (0, 1))
        if state["settings_version"] == 0:
            self.assertEqual(state["settings"]["profile"], "cautious")
            self.assertTrue(any("revoked" in r for r in results))
        else:
            self.assertTrue(any("Saved" in r for r in results))
            self.assertFalse(any("revoked" in r for r in results))

    async def test_expiry_wrong_actor_superseded_settings_and_unknown_tokens(self):
        preview = await self.service.propose_change("profile", "extreme", ACTOR)
        self.assertIn(
            "unavailable",
            await self.service.confirm_change(preview["token"], OTHER_ACTOR),
        )
        self.assertIn(preview["token"], (await self.store.read())["confirmations"])
        expires = (await self.store.read())["confirmations"][preview["token"]][
            "expires"
        ]
        with patch("apex_bot.service.time.time", return_value=expires):
            self.assertIn(
                "expired", await self.service.confirm_change(preview["token"], ACTOR)
            )
        old = await self.service.propose_change("profile", "extreme", ACTOR)
        newer = await self.service.propose_change("profile", "balanced", ACTOR)
        self.assertIn(
            "unavailable", await self.service.confirm_change(old["token"], ACTOR)
        )
        await self.service.change_setting("paused", True, OTHER_ACTOR)
        self.assertIn(
            "settings changed", await self.service.confirm_change(newer["token"], ACTOR)
        )
        self.assertIn(
            "unavailable", await self.service.confirm_change("unknown", ACTOR)
        )
        self.assertEqual((await self.store.read())["settings"]["profile"], "cautious")

    async def test_read_snapshots_exclude_personal_ids_and_credentials(self):
        private = f"{CHAT} {USER} {OTHER} {ACTOR} {self.config.telegram_token}"
        await self.write(
            research={"summary": private},
            health={"scan_at": self.now, "error": private},
            reviews={private: {}},
            references={private: {}},
            ai_budget={private: 2},
        )
        for view in (
            "dashboard",
            "risk",
            "performance",
            "research",
            "system",
            "telemetry",
        ):
            text = await self.service.snapshot(view)
            for value in (
                CHAT,
                str(USER),
                str(OTHER),
                ACTOR,
                self.config.telegram_token,
            ):
                self.assertNotIn(value, text)

    async def test_paired_performance_direction_setup_counts_and_no_unfilled_wins(self):
        records = [
            shadow_trade("baseline_shadow", "pair", 100),
            shadow_trade("ai_shadow", "pair", 40),
            shadow_trade("baseline_shadow", "unmatched", -20, "Sell", "1S"),
            shadow_trade("baseline_shadow", "expired", 9999, status="EXPIRED"),
            shadow_trade("ai_shadow", "open", 9999, status="OPEN"),
        ]
        await self.write(trades={str(i): r for i, r in enumerate(records)})
        text = await self.service.snapshot("performance")
        self.assertIn("Estimated net 80.00 USDT", text)
        self.assertIn("Estimated net 40.00 USDT", text)
        self.assertIn("WR 50.0% (valid closed)", text)
        self.assertIn("Matched opportunities closed in both arms: 1", text)
        self.assertIn("Paired AI − baseline net: -60.00 USDT", text)
        self.assertIn("Unmatched candidates: baseline 2 · AI 1", text)
        self.assertIn("Long: 1 closed · 1W/0L/0BE", text)
        self.assertIn("Short: 1 closed · 0W/1L/0BE", text)
        self.assertIn("Setup 1S: 1 closed · long 0 / short 1", text)
        telemetry = await self.service.snapshot("telemetry")
        self.assertIn("Telegram pending", telemetry)
        self.assertIn("Matched opportunities", telemetry)

    async def test_shared_comparison_reports_actual_latency_sizing_and_incomplete_funding(
        self,
    ):
        records = [
            shadow_trade("baseline_shadow", "pair", 100),
            shadow_trade(
                "ai_shadow",
                "pair",
                40,
                created_at=220,
                entry_eligible_at=360,
                qty=1,
                funding_complete=False,
                management_complete=False,
                management_mode="changed",
                data_gap={"reason": "missing candle"},
            ),
        ]
        await self.write(trades={r["id"]: r for r in records})
        before = await self.store.read()
        with patch("apex_bot.service.comparison", wraps=comparison) as shared:
            for view in ("performance", "telemetry", "dashboard"):
                text = await self.service.snapshot(view)
                self.assertIn("AI decision delay vs baseline: 120–120s", text)
                self.assertIn("Same entry window: 0/1 pairs", text)
                self.assertIn(
                    "entry timing 1 · sizing/levels 1 · simulation assumptions 1", text
                )
                self.assertIn("Funding incomplete on 1 closed pairs", text)
                self.assertIn("Management incomplete: 1 · Data gaps/errors: 1", text)
                self.assertIn("No causal AI uplift claim", text)
            self.assertEqual(shared.call_count, 3)
        self.assertEqual(await self.store.read(), before)

    async def test_universe_prioritizes_blocked_active_members_and_persisted_policy(
        self,
    ):
        self.service.config = replace(self.config, universe_mode="dynamic")
        await self.write(
            universe={
                "active_symbols": ["BTCUSDT"],
                "as_of": self.now,
                "blocked": {
                    **{f"THIN{i}USDT": ["wide_spread"] for i in range(10)},
                    "BTCUSDT": ["book_missing"],
                },
                "policy": {
                    "min_listing_days": 30,
                    "min_turnover_usdt": 20_000_000,
                    "max_spread_bps": 10,
                    "min_depth_usdt": 25_000,
                    "depth_band_bps": 25,
                    "replacement_ratio": 1.5,
                    "confirmation_observations": 2,
                    "max_daily_replacements": 5,
                },
            }
        )
        text = await self.service.snapshot("universe")
        for expected in (
            "Blocked: 11 total · 1 active",
            "Entry wait for blocked active members.",
            "BTCUSDT: book_missing",
            "24h turnover ≥20M USDT",
            "Depth ≥25K USDT/side",
            "Listing age ≥30 days",
            "Spread ≤10 bps",
            "Depth band: 25 bps",
            "Challenger ≥1.5× incumbent",
            "Confirm across 2 observations",
            "Healthy replacements ≤5/day",
        ):
            self.assertIn(expected, text)
        self.assertLess(text.index("BTCUSDT: book_missing"), text.index("THIN0USDT:"))

    async def test_invalid_closed_outcomes_never_become_wins_breakevens_or_paired_uplift(
        self,
    ):
        invalid = shadow_trade("ai_shadow", "bad", 99999, data_error="invalid history")
        missing = shadow_trade("ai_shadow", "missing", 1, side="Sell", setup="2S")
        missing.pop("net_pnl")
        records = [
            shadow_trade("baseline_shadow", "bad", 100),
            invalid,
            missing,
            shadow_trade("ai_shadow", "valid", -10),
        ]
        await self.write(trades={r["id"]: r for r in records})
        text = await self.service.snapshot("performance")
        ai_text = text.split("👻 AI-approved shadow")[1].split("🔬")[0]
        self.assertIn("0 wins · 1 losses · 0 breakeven", ai_text)
        self.assertIn("WR 0.0% (valid closed)", ai_text)
        self.assertIn("Invalid closed outcomes excluded: 2", ai_text)
        self.assertIn("Long: 1 closed · 0W/1L/0BE", ai_text)
        self.assertIn("Short: 0 closed", ai_text)
        self.assertNotIn("Setup 2S", ai_text)
        self.assertIn(
            "Paired results unavailable: 1 matched opportunities have invalid outcomes",
            text,
        )
        self.assertNotIn("Paired AI − baseline net:", text)
        self.assertNotIn("99,999", text)

    async def test_incomplete_pair_metadata_is_unavailable_without_inventing_decision_times(
        self,
    ):
        records = [
            shadow_trade("baseline_shadow", "pair", 10),
            shadow_trade("ai_shadow", "pair", 5),
        ]
        records[1].pop("created_at")
        await self.write(trades={r["id"]: r for r in records})
        text = await self.service.snapshot("performance")
        self.assertIn("Estimated net 5.00 USDT", text)
        self.assertIn(
            "Paired comparison unavailable: decision/identity metadata incomplete", text
        )
        self.assertNotIn("Paired AI − baseline net:", text)
        self.assertNotIn("AI decision delay", text)

    async def test_matched_pending_candidates_do_not_report_zero_measured_uplift(self):
        records = [
            shadow_trade(arm, "pair", 9999, status="PENDING")
            for arm, _ in (("baseline_shadow", ""), ("ai_shadow", ""))
        ]
        await self.write(trades={r["id"]: r for r in records})
        text = await self.service.snapshot("performance")
        self.assertIn("Matched opportunities: 1", text)
        self.assertIn("Matched opportunities closed in both arms: 0", text)
        self.assertIn("Paired AI − baseline net: — (no closed pairs)", text)
        self.assertNotIn("9,999", text)


if __name__ == "__main__":
    unittest.main()
