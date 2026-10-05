import json
import math
import unittest
from dataclasses import replace
from decimal import Decimal

from apex_bot.engine import DATA_GRACE, DAY, H4, PLAN_LIFETIME, VERSION
from apex_bot.models import Instrument, Opportunity
from apex_bot.risk import HARD_CAPS, PROFILES, assess

NOW = 100 * DAY + H4


def opportunity(**changes):
    evidence = dict(
        engine_version=VERSION,
        evidence_id="frozen",
        trigger_evidence_id="trigger",
        structural_valid=True,
        data_valid=True,
        terminal_status=None,
        trigger_kind="SWING_BREAK",
        trigger_closed_at=NOW,
        daily_closed_at=100 * DAY,
        execution_closed_at=NOW,
        as_of=NOW,
        zone_low=98,
        zone_high=101,
        entry_zone_low=98,
        entry_zone_high=101,
        trigger_price=100,
        risk_multiplier=1,
    )
    op = Opportunity(
        "frozen",
        "TESTUSDT",
        "Buy",
        "1",
        "READY",
        100,
        94,
        112,
        130,
        96,
        99,
        NOW - H4,
        NOW + 2 * H4,
        "SWING_BREAK",
        evidence,
    )
    return replace(op, **changes)


def monitored_opportunity(side="Buy", **changes):
    op = opportunity(
        side=side,
        setup="1" if side == "Buy" else "1S",
        entry=100.05 if side == "Buy" else 99.95,
        stop=94 if side == "Buy" else 106,
        target1=112 if side == "Buy" else 88,
        target2=130 if side == "Buy" else 70,
        invalidation=96 if side == "Buy" else 104,
        confirmation=0,
        created_at=NOW - 10 * DAY,
        expires_at=NOW - 10 * DAY + PLAN_LIFETIME,
        reason="ZONE_ARRIVAL",
    )
    op = replace(
        op,
        evidence=dict(
            op.evidence,
            entry_style="monitored_zone",
            trigger_kind="ZONE_ARRIVAL",
            trigger_closed_at=None,
            trigger_price=None,
            trigger_evidence_id=None,
            plan_evidence_id="parent-plan",
            monitored_started_at=op.created_at,
            observation_at=NOW,
            observation_price=100,
            quote_limit=op.entry,
        ),
    )
    return replace(op, **changes)


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument("TESTUSDT", 0.01, 0.01, 100000, 0.01, 5)

    def check(self, op=None, **kwargs):
        args = dict(exposures=[], funding_rate_8h=0, spread_pct=0.01, now=NOW)
        args.update(kwargs)
        return assess(
            op or opportunity(),
            args.pop("instrument", self.instrument),
            args.pop("equity", 10000),
            **args
        )

    def reject(self, reason, op=None, **kwargs):
        result = self.check(op, **kwargs)
        self.assertFalse(result["allowed"], result)
        self.assertIn(reason, result["reasons"])
        self.assertEqual(
            (result["qty"], result["risk_cash"], result["notional"]), (0, 0, 0)
        )
        return result

    def test_profiles_and_budget_include_cost_allowance(self):
        sizes = []
        for name, expected in [
            ("ultra_cautious", 0.1),
            ("cautious", 0.25),
            ("balanced", 0.5),
            ("aggressive", 0.75),
            ("extreme", 1),
        ]:
            result = self.check(profile=name)
            self.assertTrue(result["allowed"], result)
            self.assertEqual(result["risk_pct"], expected)
            self.assertLessEqual(result["risk_cash"], 10000 * expected / 100)
            self.assertGreater(result["risk_cash"], result["qty"] * 6)
            sizes.append(result["qty"])
            for key, cap in HARD_CAPS.items():
                self.assertLessEqual(PROFILES[name][key], cap)
        self.assertEqual(sizes, sorted(set(sizes)))
        self.assertEqual(self.check()["qty"], 4.05)

    def test_resting_research_does_not_claim_confirmation_or_relax_rr(self):
        op = opportunity(confirmation=0)
        op = replace(
            op,
            evidence={
                **op.evidence,
                "entry_style": "resting_limit",
                "trigger_kind": "ZONE_LIMIT",
                "trigger_closed_at": None,
                "trigger_price": None,
                "trigger_evidence_id": None,
                "plan_evidence_id": "parent-plan",
                "order_armed_at": op.created_at,
            },
        )
        self.reject("RESTING_RESEARCH_ONLY", op)
        self.assertTrue(self.check(op, allow_resting=True)["allowed"])
        self.reject("RESTING_TIER1_ONLY", replace(op, tier=2), allow_resting=True)
        self.reject("RR_TARGET1", replace(op, target1=108), allow_resting=True)
        self.reject(
            "INVALID_RESTING_EVIDENCE",
            replace(
                op,
                evidence={**op.evidence, "trigger_evidence_id": "forged-confirmation"},
            ),
            allow_resting=True,
        )
        self.reject(
            "INVALID_EVIDENCE_TIME",
            replace(op, expires_at=op.created_at + 2 * DAY + 1),
            allow_resting=True,
        )
        self.reject(
            "UNVERIFIED_EVIDENCE",
            replace(op, evidence={**op.evidence, "plan_evidence_id": None}),
            allow_resting=True,
        )

    def test_monitored_requires_separate_explicit_opt_in_and_tier1(self):
        for side in ("Buy", "Sell"):
            op = monitored_opportunity(side)
            with self.subTest(side=side):
                self.reject("MONITORED_RESEARCH_ONLY", op)
                self.reject("MONITORED_RESEARCH_ONLY", op, allow_resting=True)
                for flag in (False, None, 1, "true"):
                    self.reject("MONITORED_RESEARCH_ONLY", op, allow_monitored=flag)
                for tier in (2, 3, True):
                    self.reject(
                        "MONITORED_TIER1_ONLY",
                        replace(op, tier=tier),
                        allow_monitored=True,
                    )
                before = op.to_dict()
                result = self.check(op, allow_monitored=True)
                self.assertTrue(result["allowed"], result)
                self.assertEqual(result["rr_required_target1"], 1.5)
                self.assertEqual(result["rr_required_blended"], 2.5)
                self.assertEqual(op.to_dict(), before)
                self.assertEqual(op.confirmation, 0)
                for key in (
                    "trigger_closed_at",
                    "trigger_price",
                    "trigger_evidence_id",
                ):
                    self.assertIsNone(op.evidence[key])
                for key in ("stop", "target1", "target2"):
                    self.assertEqual(result[key], getattr(op, key))

    def test_monitored_rejects_mixed_styles_and_fake_confirmation(self):
        for side in ("Buy", "Sell"):
            op = monitored_opportunity(side)
            variants = [
                {key: value}
                for key, value in (
                    ("trigger_closed_at", NOW),
                    ("trigger_price", 100),
                    ("trigger_evidence_id", "forged-confirmation"),
                    ("trigger_closed_at", 0),
                    ("trigger_price", False),
                    ("trigger_evidence_id", ""),
                    ("trigger_kind", "SWING_BREAK"),
                    ("trigger_kind", "BREAKOUT_RETEST"),
                    ("trigger_kind", "ZONE_LIMIT"),
                    ("trigger_kind", "ZONE_WATCH"),
                    ("entry_style", "resting_limit"),
                    ("entry_style", "confirmed"),
                    ("entry_style", None),
                )
            ]
            for changes in variants:
                with self.subTest(side=side, changes=changes):
                    self.reject(
                        "INVALID_MONITORED_EVIDENCE",
                        replace(op, evidence=dict(op.evidence, **changes)),
                        allow_monitored=True,
                        allow_resting=True,
                    )
            for kind in ("ZONE_ARRIVAL", "ZONE_WATCH"):
                # Reserved observation markers cannot masquerade as confirmed
                # entries just by omitting their entry_style.
                fake = opportunity()
                fake = replace(fake, evidence=dict(fake.evidence, trigger_kind=kind))
                self.reject("MONITORED_RESEARCH_ONLY", fake)

    def test_monitored_requires_matching_identity_and_frozen_zone(self):
        for side in ("Buy", "Sell"):
            op = monitored_opportunity(side)
            self.reject(
                "EVIDENCE_ID_MISMATCH",
                replace(op, id="changed"),
                allow_monitored=True,
            )
            for plan_id in (None, "", " ", False, 123):
                self.reject(
                    "INVALID_MONITORED_EVIDENCE",
                    replace(op, evidence=dict(op.evidence, plan_evidence_id=plan_id)),
                    allow_monitored=True,
                )
            for key in ("entry_zone_low", "entry_zone_high", "zone_low", "zone_high"):
                with self.subTest(side=side, key=key):
                    self.reject(
                        "INVALID_MONITORED_EVIDENCE",
                        replace(op, evidence=dict(op.evidence, **{key: 100})),
                        allow_monitored=True,
                    )
            self.reject(
                "INVALID_MONITORED_EVIDENCE",
                replace(op, entry=100),
                allow_monitored=True,
            )

    def test_monitored_observation_and_assessment_freshness_are_independent(self):
        for side in ("Buy", "Sell"):
            op = monitored_opportunity(side)
            # Quotes expire after 60s even while daily/4H bars remain fresh.
            result = self.check(op, allow_monitored=True, now=NOW + 60)
            self.assertTrue(result["allowed"], result)
            stale = self.reject(
                "STALE_OBSERVATION",
                op,
                allow_monitored=True,
                now=NOW + 61,
            )
            self.assertNotIn("STALE_ASSESSMENT", stale["reasons"])
            stale = self.reject(
                "STALE_OBSERVATION",
                op,
                allow_monitored=True,
                now=NOW + DATA_GRACE + 1,
            )
            self.assertIn("STALE_ASSESSMENT", stale["reasons"])
            # Refreshing as_of must not refresh an old observation.
            old = replace(
                op,
                evidence=dict(
                    op.evidence,
                    observation_at=NOW - DATA_GRACE - 1,
                    execution_closed_at=NOW - H4,
                ),
            )
            stale = self.reject("STALE_OBSERVATION", old, allow_monitored=True)
            self.assertNotIn("STALE_ASSESSMENT", stale["reasons"])
            for key, age in (("daily_closed_at", DAY), ("execution_closed_at", H4)):
                with self.subTest(side=side, key=key):
                    boundary = replace(
                        op, evidence=dict(op.evidence, **{key: NOW - age - DATA_GRACE})
                    )
                    self.assertTrue(
                        self.check(boundary, allow_monitored=True)["allowed"]
                    )
                    self.reject(
                        "STALE_DATA",
                        replace(
                            op,
                            evidence=dict(
                                op.evidence, **{key: NOW - age - DATA_GRACE - 1}
                            ),
                        ),
                        allow_monitored=True,
                    )

    def test_monitored_timestamp_ordering_and_nonfinite_times_fail_closed(self):
        keys = (
            "monitored_started_at",
            "observation_at",
            "as_of",
            "daily_closed_at",
            "execution_closed_at",
        )
        for side in ("Buy", "Sell"):
            op = monitored_opportunity(side)
            for key in keys:
                for value in (None, math.nan, math.inf, -1, True, "123"):
                    with self.subTest(side=side, key=key, value=value):
                        self.reject(
                            "MISSING_EVIDENCE_TIME",
                            replace(op, evidence=dict(op.evidence, **{key: value})),
                            allow_monitored=True,
                        )
            for changes in (
                {"monitored_started_at": op.created_at + 1},
                {"monitored_started_at": op.created_at - 1},
                {"observation_at": op.created_at - 1},
                {"observation_at": NOW + 1},
                {"as_of": NOW - 1},
                {"as_of": NOW + 1},
                {"execution_closed_at": NOW + 1},
                {"daily_closed_at": NOW + 1},
                {"observation_at": NOW - 1},  # execution must precede observation
            ):
                with self.subTest(side=side, changes=changes):
                    self.reject(
                        "INVALID_EVIDENCE_TIME",
                        replace(op, evidence=dict(op.evidence, **changes)),
                        allow_monitored=True,
                    )
            for key in ("created_at", "expires_at"):
                for value in (None, math.nan, math.inf, True):
                    self.reject(
                        "MISSING_EVIDENCE_TIME",
                        replace(op, **{key: value}),
                        allow_monitored=True,
                    )
            self.reject("INVALID_TIME", op, allow_monitored=True, now=math.nan)
            # Creation, observation, assessment and execution can coincide.
            same_time = replace(
                op,
                created_at=NOW,
                evidence=dict(
                    op.evidence,
                    monitored_started_at=NOW,
                ),
            )
            self.assertTrue(self.check(same_time, allow_monitored=True)["allowed"])

    def test_monitored_plan_expiry_is_not_rearmed_by_fresh_observation(self):
        self.assertEqual(PLAN_LIFETIME, 30 * DAY)
        for side in ("Buy", "Sell"):
            op = monitored_opportunity(side)
            self.assertTrue(self.check(op, allow_monitored=True)["allowed"])
            self.reject(
                "INVALID_EVIDENCE_TIME",
                replace(op, expires_at=op.created_at + PLAN_LIFETIME + 1),
                allow_monitored=True,
            )
            self.reject("EXPIRED", replace(op, expires_at=NOW), allow_monitored=True)
            self.reject(
                "INVALID_EVIDENCE_TIME",
                replace(op, expires_at=op.created_at),
                allow_monitored=True,
            )
            old_created = NOW - PLAN_LIFETIME
            expired = replace(
                op,
                created_at=old_created,
                expires_at=NOW,
                evidence=dict(op.evidence, monitored_started_at=old_created),
            )
            self.reject("EXPIRED", expired, allow_monitored=True)
            self.reject(
                "INVALID_EVIDENCE_TIME",
                replace(expired, expires_at=NOW + 1),
                allow_monitored=True,
            )
            control_created = NOW - DAY
            control = replace(
                op,
                created_at=control_created,
                expires_at=control_created + 2 * DAY,
                evidence=dict(op.evidence, monitored_started_at=control_created),
            )
            self.assertTrue(self.check(control, allow_monitored=True)["allowed"])

    def test_monitored_positive_quotes_and_zero_to_five_bp_cap(self):
        for side, direction in (("Buy", 1), ("Sell", -1)):
            op = monitored_opportunity(side)
            for key in ("observation_price", "quote_limit"):
                for value in (None, 0, -1, math.nan, math.inf, True, "100"):
                    with self.subTest(side=side, key=key, value=value):
                        self.reject(
                            "INVALID_MONITORED_EVIDENCE",
                            replace(op, evidence=dict(op.evidence, **{key: value})),
                            allow_monitored=True,
                        )
            for adverse in (0, 0.025, 0.05, -0.000001, 0.050001):
                quote = float(Decimal(100) + direction * Decimal(str(adverse)))
                candidate = replace(
                    op,
                    entry=quote,
                    evidence=dict(
                        op.evidence,
                        quote_limit=quote,
                    ),
                )
                with self.subTest(side=side, adverse=adverse):
                    if 0 <= adverse <= 0.05:
                        result = self.check(candidate, allow_monitored=True)
                        self.assertTrue(result["allowed"], result)
                        self.assertLessEqual(direction * (result["entry"] - quote), 0)
                    else:
                        self.reject(
                            "INVALID_MONITORED_EVIDENCE",
                            candidate,
                            allow_monitored=True,
                        )

    def test_monitored_float_formula_caps_pass_but_larger_caps_fail(self):
        # These analyzer-style multiplications straddle the exact decimal 5bp
        # boundary by float representation alone. No broader epsilon is allowed.
        for side, direction in (("Buy", 1), ("Sell", -1)):
            for price in (100.076, 100.177, 100.278, 100.323):
                with self.subTest(side=side, price=price):
                    op = monitored_opportunity(side)
                    cap = price * (1 + direction * 0.0005)
                    op = replace(
                        op,
                        entry=cap,
                        evidence=dict(
                            op.evidence,
                            observation_price=price,
                            quote_limit=cap,
                        ),
                    )
                    result = self.check(
                        op,
                        allow_monitored=True,
                        instrument=replace(self.instrument, tick_size=0.0001),
                    )
                    self.assertTrue(result["allowed"], result)
                    self.assertLessEqual(direction * (result["entry"] - cap), 0)
                    exact = Decimal(str(price)) * (1 + direction * Decimal("0.0005"))
                    beyond = math.nextafter(
                        (
                            max(cap, float(exact))
                            if direction > 0
                            else min(cap, float(exact))
                        ),
                        math.inf if direction > 0 else -math.inf,
                    )
                    self.reject(
                        "INVALID_MONITORED_EVIDENCE",
                        replace(
                            op,
                            entry=beyond,
                            evidence=dict(op.evidence, quote_limit=beyond),
                        ),
                        allow_monitored=True,
                    )

    def test_monitored_tick_rounding_respects_quote_cap_and_exact_zone(self):
        for side, expected in (("Buy", 100), ("Sell", 100)):
            op = monitored_opportunity(side)
            result = self.check(
                op,
                allow_monitored=True,
                instrument=replace(self.instrument, tick_size=0.1),
            )
            self.assertTrue(result["allowed"], result)
            self.assertEqual(result["entry"], expected)
            # Rounding a narrow point zone toward the quote cap cannot escape
            # the frozen zone, even by less than the confirmed-path tolerance.
            price = 100.005 if side == "Buy" else 99.995
            narrow = replace(
                op,
                entry=price,
                evidence=dict(
                    op.evidence,
                    observation_price=price,
                    quote_limit=price,
                    zone_low=price,
                    zone_high=price,
                    entry_zone_low=price,
                    entry_zone_high=price,
                ),
            )
            self.reject("ENTRY_OUTSIDE_ZONE", narrow, allow_monitored=True)
            for observation, quote, reason in (
                (101.001, 101, "OBSERVATION_OUTSIDE_ZONE"),
                (97.999, 98, "OBSERVATION_OUTSIDE_ZONE"),
                (101, 101.05, "ENTRY_OUTSIDE_ZONE"),
                (98, 97.95, "ENTRY_OUTSIDE_ZONE"),
            ):
                candidate = replace(
                    op,
                    entry=quote,
                    evidence=dict(
                        op.evidence,
                        observation_price=observation,
                        quote_limit=quote,
                    ),
                )
                self.reject(reason, candidate, allow_monitored=True)
            for boundary in (98, 101):
                direction = 1 if side == "Buy" else -1
                candidate = replace(
                    op,
                    entry=boundary,
                    target1=100 + direction * 20,
                    target2=100 + direction * 40,
                    evidence=dict(
                        op.evidence,
                        observation_price=boundary,
                        quote_limit=boundary,
                    ),
                )
                result = self.check(candidate, allow_monitored=True)
                self.assertTrue(result["allowed"], result)

    def test_monitored_preserves_costs_rr_stops_and_portfolio_limits(self):
        for side, direction in (("Buy", 1), ("Sell", -1)):
            op = monitored_opportunity(side, entry=100)
            op = replace(op, evidence=dict(op.evidence, quote_limit=100))
            confirmed = opportunity(
                side=side,
                setup=op.setup,
                stop=op.stop,
                target1=op.target1,
                target2=op.target2,
                invalidation=op.invalidation,
                confirmation=100 - direction,
            )
            # Exact same prices give exactly the same assessment and budgets.
            for profile in PROFILES:
                self.assertEqual(
                    self.check(op, allow_monitored=True, profile=profile),
                    self.check(confirmed, profile=profile),
                )
            for reason, candidate, kwargs in (
                ("STOP_BUFFER_TOO_SMALL", replace(op, stop=100 - direction * 4.5), {}),
                ("RR_TARGET1", replace(op, target1=100 + direction * 8), {}),
                ("RR_BLENDED", replace(op, target2=100 + direction * 14), {}),
                ("ADVERSE_FUNDING", op, {"funding_rate_8h": direction * 0.00031}),
                ("FUNDING_UNKNOWN", op, {"funding_rate_8h": None}),
                ("SPREAD_TOO_WIDE", op, {"spread_pct": 0.20001}),
                ("EXPOSURES_UNKNOWN", op, {"exposures": None}),
                ("OPPORTUNITY_NOT_READY", replace(op, state="WAIT"), {}),
                ("DAILY_LOSS_HALT", op, {"daily_loss_pct": 2}),
                ("WEEKLY_LOSS_HALT", op, {"weekly_loss_pct": 4}),
                ("DRAWDOWN_HALT", op, {"drawdown_pct": 12}),
                ("REGIME_BLOCKED", op, {"regime_multiplier": 0}),
                (
                    "HEAT_CAP",
                    op,
                    {
                        "exposures": [
                            dict(
                                symbol="OTHER",
                                bucket="other",
                                risk_cash=376,
                                notional=500,
                            )
                        ]
                    },
                ),
                (
                    "BUCKET_HEAT_CAP",
                    op,
                    {
                        "exposures": [
                            dict(
                                symbol="OTHER",
                                bucket=op.bucket,
                                risk_cash=176,
                                notional=500,
                            )
                        ]
                    },
                ),
                (
                    "GROSS_NOTIONAL_CAP",
                    op,
                    {
                        "exposures": [
                            dict(
                                symbol="OTHER",
                                bucket="other",
                                risk_cash=1,
                                notional=9800,
                            )
                        ]
                    },
                ),
            ):
                with self.subTest(side=side, reason=reason):
                    self.reject(reason, candidate, allow_monitored=True, **kwargs)
            odd = replace(
                op, target1=100 + direction * 10, target2=100 + direction * 23
            )
            self.reject(
                "RR_ROUNDED_SPLIT",
                odd,
                allow_monitored=True,
                equity=7500,
                instrument=replace(self.instrument, qty_step=1, min_qty=1),
            )

    def test_custom_risk_bounds_and_tier_reduction(self):
        for pct in (0.05, 1):
            self.assertTrue(self.check(risk_pct=pct)["allowed"])
        for pct in (0.0499, 1.00001, 0, -1, math.nan, math.inf, True):
            self.reject("INVALID_RISK_OVERRIDE", risk_pct=pct)
        self.reject("UNKNOWN_PROFILE", profile="unknown")
        result = self.check(opportunity(tier=2), risk_pct=0.8)
        self.assertTrue(result["allowed"], result)
        self.assertEqual(result["risk_pct"], 0.4)

    def test_profiles_never_relax_quality_data_or_stop(self):
        for profile in PROFILES:
            self.reject(
                "OPPORTUNITY_NOT_READY", opportunity(state="WAIT"), profile=profile
            )
            self.reject(
                "STOP_BUFFER_TOO_SMALL", opportunity(stop=95.5), profile=profile
            )
            self.reject("RR_TARGET1", opportunity(target1=108), profile=profile)
            self.reject(
                "RR_BLENDED", opportunity(target1=112, target2=114), profile=profile
            )
            self.reject("FUNDING_UNKNOWN", profile=profile, funding_rate_8h=None)
            self.reject("SPREAD_UNKNOWN", profile=profile, spread_pct=None)
            self.reject("EXPOSURES_UNKNOWN", profile=profile, exposures=None)

    def test_funding_signed_thresholds_and_short_mirror(self):
        self.assertTrue(self.check(funding_rate_8h=0.0003)["allowed"])
        self.assertTrue(self.check(funding_rate_8h=-0.001)["allowed"])
        self.reject("ADVERSE_FUNDING", funding_rate_8h=0.00030001)
        op = opportunity(
            side="Sell",
            setup="1S",
            stop=106,
            target1=88,
            target2=70,
            invalidation=104,
            confirmation=101,
        )
        self.assertTrue(self.check(op, funding_rate_8h=-0.0003)["allowed"])
        self.assertTrue(self.check(op, funding_rate_8h=0.001)["allowed"])
        self.reject("ADVERSE_FUNDING", op, funding_rate_8h=-0.00030001)
        self.assertEqual(self.check(op)["qty"], self.check()["qty"])

    def test_loss_halts_boundaries_and_drawdown_halves(self):
        self.reject("DAILY_LOSS_HALT", daily_loss_pct=2)
        self.reject("WEEKLY_LOSS_HALT", weekly_loss_pct=4)
        self.reject("DRAWDOWN_HALT", drawdown_pct=12)
        self.assertTrue(
            self.check(daily_loss_pct=1.999, weekly_loss_pct=3.999)["allowed"]
        )
        reduced = self.check(drawdown_pct=8)
        self.assertTrue(reduced["allowed"])
        self.assertEqual(reduced["risk_pct"], 0.125)
        tier2 = self.check(opportunity(tier=2), drawdown_pct=8)
        self.assertEqual(tier2["risk_pct"], 0.0625)
        self.assertLessEqual(tier2["risk_cash"], 6.25)

    def test_exposures_include_owned_external_and_unknown_risk(self):
        for owner in ("OWN", "bot", "external"):
            p = dict(
                symbol="testusdt",
                bucket="crypto",
                risk_cash=1,
                notional=10,
                owner=owner,
            )
            self.reject("SYMBOL_ALREADY_EXPOSED", exposures=[p])
        for risk in (None, math.nan, math.inf, -1):
            self.reject(
                "EXPOSURE_RISK_OR_NOTIONAL_UNKNOWN",
                exposures=[
                    dict(symbol="OTHER", bucket="other", risk_cash=risk, notional=10)
                ],
            )
        self.reject(
            "EXPOSURE_RISK_OR_NOTIONAL_UNKNOWN",
            exposures=[dict(symbol="OTHER", notional=10)],
        )

    def test_heat_and_bucket_caps(self):
        p = dict(symbol="OTHER", bucket="other", risk_cash=376, notional=500)
        self.reject("HEAT_CAP", exposures=[p])
        self.assertTrue(self.check(exposures=[dict(p, risk_cash=375)])["allowed"])
        self.reject(
            "BUCKET_HEAT_CAP", exposures=[dict(p, bucket="crypto", risk_cash=176)]
        )
        self.assertTrue(
            self.check(exposures=[dict(p, bucket="crypto", risk_cash=175)])["allowed"]
        )

    def test_position_caps_and_gross_notional(self):
        positions = [
            dict(symbol=str(i), bucket=str(i), risk_cash=1, notional=10)
            for i in range(6)
        ]
        self.reject("POSITION_CAP", exposures=positions)
        self.reject(
            "BUCKET_POSITION_CAP",
            exposures=[dict(p, bucket="crypto") for p in positions[:2]],
        )
        self.reject("GROSS_NOTIONAL_CAP", exposures=[dict(positions[0], notional=9800)])
        self.reject(
            "SINGLE_NOTIONAL_CAP",
            opportunity(stop=98, invalidation=99.6, target1=112, target2=130),
            profile="extreme",
        )

    def test_zero_nonfinite_negative_and_invalid_instrument(self):
        for value in (0, -1, math.nan, math.inf):
            self.reject("INVALID_EQUITY", equity=value)
            self.reject("INVALID_PRICE", opportunity(entry=value))
            self.reject(
                "INVALID_INSTRUMENT",
                instrument=replace(self.instrument, qty_step=value),
            )
        for value in (math.nan, math.inf, -1):
            self.reject("INVALID_LOSS_DATA", drawdown_pct=value)
        self.reject(
            "SYMBOL_MISMATCH", instrument=replace(self.instrument, symbol="OTHER")
        )
        self.reject("INVALID_SIDE", opportunity(side="Long"))

    def test_tiny_qty_and_minimum_notional_never_round_up(self):
        self.reject("QUANTITY_BELOW_MINIMUM", equity=0.1)
        self.reject(
            "QUANTITY_BELOW_MINIMUM", instrument=replace(self.instrument, min_qty=10)
        )
        self.reject(
            "NOTIONAL_BELOW_MINIMUM",
            instrument=replace(self.instrument, min_notional=500),
        )
        self.reject(
            "QUANTITY_ABOVE_MAXIMUM", instrument=replace(self.instrument, max_qty=1)
        )

    def test_price_rounding_and_qty_floor(self):
        instrument = replace(self.instrument, tick_size=0.1, qty_step=0.3)
        op = opportunity(entry=100.01, stop=93.99, target1=112.09, target2=130.09)
        result = self.check(op, instrument=instrument)
        self.assertTrue(result["allowed"], result)
        self.assertEqual(
            [result[k] for k in ("entry", "stop", "target1", "target2")],
            [100.1, 93.9, 112, 130],
        )
        self.assertEqual(Decimal(str(result["qty"])) % Decimal(".3"), 0)
        self.assertLessEqual(result["risk_cash"], 25)
        self.reject(
            "INVALID_ROUNDED_PRICE_ORDER",
            opportunity(stop=99.9, target1=100.01, target2=100.02),
            instrument=replace(self.instrument, tick_size=1),
        )

    def test_rr_is_net_cost_and_actual_odd_lot_split(self):
        self.reject(
            "RR_TARGET1", opportunity(target1=109, target2=130)
        )  # gross 1.5R fails net
        instrument = replace(self.instrument, qty_step=1, min_qty=1)
        op = opportunity(target1=110, target2=123)
        odd = self.reject("RR_ROUNDED_SPLIT", op, instrument=instrument, equity=7500)
        # Three lots: two exit at T1, one at T2. Net rewards are 9.84/22.84
        # against 6.16 risk, so the nominal blend passes but the real split fails.
        self.assertEqual(odd["reasons"], ["RR_ROUNDED_SPLIT"])
        self.assertAlmostEqual(odd["rr_target1"], 9.84 / 6.16)
        self.assertAlmostEqual(odd["rr_blended"], 16.34 / 6.16)
        self.assertAlmostEqual(odd["rr_actual_split"], (2 * 9.84 + 22.84) / 18.48)
        self.assertGreater(odd["rr_blended"], odd["rr_required_blended"])
        self.assertLess(odd["rr_actual_split"], odd["rr_required_blended"])
        self.assertGreater(odd["rr_entry_bound"], op.entry)
        self.assertTrue(odd["rr_zone_compatible"])
        self.assertTrue(self.check(op, instrument=instrument, equity=5000)["allowed"])
        single = self.check(op, instrument=instrument, equity=2500)
        self.assertTrue(single["allowed"], single)
        self.assertEqual(single["qty"], 1)
        self.assertEqual(single["rr_actual_split"], single["rr_blended"])

    def test_rr_long_bound_uses_rounded_prices_and_entry_dependent_costs(self):
        instrument = replace(self.instrument, tick_size=0.1)
        op = opportunity(entry=100.01, stop=93.99, target1=112.09, target2=130.09)
        op = replace(op, evidence=dict(op.evidence, trigger_price=101))
        before = op.to_dict()
        args = dict(instrument=instrument, spread_pct=0.02, funding_rate_8h=0.0003)
        result = self.check(op, **args)
        self.assertTrue(result["allowed"], result)
        # Rounded E/S/T1/T2 = 100.1/93.9/112/130. Cost rate is .002:
        # .0015 base + .0002 spread + .0003 paid funding. Cost = .2002.
        self.assertAlmostEqual(result["rr_target1"], 11.6998 / 6.4002)
        self.assertAlmostEqual(result["rr_blended"], 20.6998 / 6.4002)
        self.assertEqual(result["rr_entry_bound"], 100.9)
        self.assertEqual(result["rr_entry_relation"], "at_or_below")
        self.assertTrue(result["rr_zone_compatible"])
        self.assertEqual(result["entry"], 100.1)
        self.assertEqual(result["qty"], 3.9)
        self.assertEqual(op.to_dict(), before)
        # At 100.9, net T1/risk = 10.8982/7.2018 > 1.5. The next tick
        # gives 10.798/7.302 < 1.5. Assess both prices through the real gates.
        self.assertTrue(self.check(replace(op, entry=100.9), **args)["allowed"])
        rejected = self.reject("RR_TARGET1", replace(op, entry=101), **args)
        self.assertAlmostEqual(rejected["rr_target1"], 10.798 / 7.302)
        self.assertEqual(rejected["rr_entry_bound"], 100.9)
        self.assertEqual(rejected["entry"], 101)

    def test_rr_short_bound_uses_rounded_prices_and_entry_dependent_costs(self):
        instrument = replace(self.instrument, tick_size=0.1)
        op = opportunity(
            side="Sell",
            setup="1S",
            entry=99.99,
            stop=106.01,
            target1=87.91,
            target2=69.91,
            invalidation=104,
            confirmation=101,
        )
        op = replace(op, evidence=dict(op.evidence, trigger_price=99))
        before = op.to_dict()
        args = dict(instrument=instrument, spread_pct=0.02, funding_rate_8h=-0.0003)
        result = self.check(op, **args)
        self.assertTrue(result["allowed"], result)
        # Rounded E/S/T1/T2 = 99.9/106.1/88/70; same .002 cost rate.
        self.assertAlmostEqual(result["rr_target1"], 11.7002 / 6.3998)
        self.assertAlmostEqual(result["rr_blended"], 20.7002 / 6.3998)
        self.assertEqual(result["rr_entry_bound"], 99.1)
        self.assertEqual(result["rr_entry_relation"], "at_or_above")
        self.assertTrue(result["rr_zone_compatible"])
        self.assertEqual(result["entry"], 99.9)
        self.assertEqual(result["qty"], 3.9)
        self.assertEqual(op.to_dict(), before)
        # At 99.1, net T1/risk = 10.9018/7.1982 > 1.5. At 99.0 it
        # becomes 10.802/7.298 < 1.5, making 99.1 the first eligible tick.
        self.assertTrue(self.check(replace(op, entry=99.1), **args)["allowed"])
        rejected = self.reject("RR_TARGET1", replace(op, entry=99), **args)
        self.assertAlmostEqual(rejected["rr_target1"], 10.802 / 7.298)
        self.assertEqual(rejected["rr_entry_bound"], 99.1)
        self.assertEqual(rejected["entry"], 99)

    def test_rr_blend_can_limit_bound_and_exact_boundary_is_inclusive(self):
        instrument = replace(self.instrument, qty_step=1, min_qty=1)
        # At E=100, cost=.2 and risk=6.2 for both directions. The tier-1
        # average reward is 15.5 (2.5R); tier-2 is 18.6 (3R). Two lots
        # avoid an odd-lot rejection at these exact nominal boundaries.
        for tier, target2_long, target2_short, equity, required in (
            (1, 119.4, 80.6, 4960, (1.5, 2.5)),
            (2, 125.6, 74.4, 9920, (1.8, 3.0)),
        ):
            for side, target2 in (("Buy", target2_long), ("Sell", target2_short)):
                with self.subTest(tier=tier, side=side):
                    long = side == "Buy"
                    op = opportunity(
                        side=side,
                        setup="1" if long else "1S",
                        tier=tier,
                        stop=94 if long else 106,
                        target1=112 if long else 88,
                        target2=target2,
                        invalidation=96 if long else 104,
                        confirmation=99 if long else 101,
                    )
                    args = dict(
                        instrument=instrument,
                        equity=equity,
                        spread_pct=0.02,
                        funding_rate_8h=0.0003 if long else -0.0003,
                    )
                    result = self.check(op, **args)
                    self.assertTrue(result["allowed"], result)
                    self.assertEqual(result["qty"], 2)
                    self.assertEqual(result["rr_required_target1"], required[0])
                    self.assertEqual(result["rr_required_blended"], required[1])
                    self.assertAlmostEqual(result["rr_target1"], 11.8 / 6.2)
                    self.assertEqual(result["rr_blended"], required[1])
                    self.assertEqual(result["rr_actual_split"], required[1])
                    self.assertEqual(result["rr_entry_bound"], 100)
                    worse = replace(op, entry=100.01 if long else 99.99)
                    rejected = self.reject("RR_BLENDED", worse, **args)
                    self.assertNotIn("RR_TARGET1", rejected["reasons"])
                    better = replace(op, entry=99.99 if long else 100.01)
                    self.assertTrue(self.check(better, **args)["allowed"])

    def test_rr_zone_compatibility_is_intersection_with_allowed_zone(self):
        for side, stop, t1, t2, invalidation, confirmation, bound in (
            ("Buy", 94, 108, 112, 96, 99, 98.37),
            ("Sell", 106, 92, 88, 104, 101, 101.64),
        ):
            with self.subTest(side=side):
                op = opportunity(
                    side=side,
                    setup="1" if side == "Buy" else "1S",
                    stop=stop,
                    target1=t1,
                    target2=t2,
                    invalidation=invalidation,
                    confirmation=confirmation,
                )
                op = replace(op, evidence=dict(op.evidence, entry_zone_low=99))
                args = dict(
                    spread_pct=0.02,
                    funding_rate_8h=0.0003 if side == "Buy" else -0.0003,
                )
                result = self.reject("RR_TARGET1", op, **args)
                self.assertIn("RR_BLENDED", result["reasons"])
                self.assertEqual(result["rr_entry_bound"], bound)
                self.assertFalse(result["rr_zone_compatible"])
                # Touching the bound counts as overlap; entry remains 100 and
                # the failed trade remains rejected with zero executable size.
                ev = dict(op.evidence)
                ev["entry_zone_low" if side == "Buy" else "entry_zone_high"] = bound
                touching = self.reject("RR_TARGET1", replace(op, evidence=ev), **args)
                self.assertTrue(touching["rr_zone_compatible"])
                self.assertEqual(touching["entry"], 100)

        # The bound need not itself be inside the zone: the permitted side can
        # cover the whole zone. These entries rely on the existing 1% tolerance.
        op = opportunity()
        op = replace(op, evidence=dict(op.evidence, entry_zone_high=99.5))
        result = self.check(op)
        self.assertTrue(result["allowed"], result)
        self.assertGreater(result["rr_entry_bound"], 99.5 * 1.01)
        self.assertTrue(result["rr_zone_compatible"])
        short = opportunity(
            side="Sell",
            setup="1S",
            stop=106,
            target1=88,
            target2=70,
            invalidation=104,
            confirmation=101,
        )
        short = replace(short, evidence=dict(short.evidence, entry_zone_low=100.5))
        result = self.check(short)
        self.assertTrue(result["allowed"], result)
        self.assertLess(result["rr_entry_bound"], 100.5 * 0.99)
        self.assertTrue(result["rr_zone_compatible"])

    def test_rr_diagnostics_do_not_grant_size_or_bypass_account_caps(self):
        tiny = self.reject("QUANTITY_BELOW_MINIMUM", equity=0.1)
        self.assertEqual(
            tiny["reasons"], ["QUANTITY_BELOW_MINIMUM", "NOTIONAL_BELOW_MINIMUM"]
        )
        self.assertIsNotNone(tiny["rr_target1"])
        self.assertIsNotNone(tiny["rr_blended"])
        self.assertIsNone(tiny["rr_actual_split"])
        self.assertTrue(tiny["rr_zone_compatible"])
        self.assertEqual(tiny["entry"], 100)
        capped = self.reject(
            "HEAT_CAP",
            exposures=[
                dict(symbol="OTHER", bucket="other", risk_cash=376, notional=500)
            ],
        )
        self.assertTrue(capped["rr_zone_compatible"])
        self.assertIsNotNone(capped["rr_actual_split"])
        self.assertEqual(capped["rr_entry_bound"], tiny["rr_entry_bound"])

    def test_rr_favorable_funding_does_not_discount_bound_costs(self):
        for side in ("Buy", "Sell"):
            with self.subTest(side=side):
                op = (
                    opportunity()
                    if side == "Buy"
                    else opportunity(
                        side="Sell",
                        setup="1S",
                        stop=106,
                        target1=88,
                        target2=70,
                        invalidation=104,
                        confirmation=101,
                    )
                )
                zero = self.check(op, funding_rate_8h=0)
                credit = self.check(
                    op, funding_rate_8h=-0.001 if side == "Buy" else 0.001
                )
                self.assertEqual(credit, zero)

    def test_rr_diagnostic_contract_on_early_rejection(self):
        unavailable = (
            "rr_target1",
            "rr_blended",
            "rr_actual_split",
            "rr_entry_bound",
            "rr_entry_relation",
            "rr_zone_compatible",
        )
        for reason, op, kwargs in (
            ("INVALID_PRICE", opportunity(entry=math.nan), {}),
            ("INVALID_ROUNDED_PRICE_ORDER", opportunity(target1=100), {}),
            ("STOP_BUFFER_TOO_SMALL", opportunity(stop=95.5), {}),
            ("FUNDING_UNKNOWN", opportunity(), dict(funding_rate_8h=None)),
            ("ADVERSE_FUNDING", opportunity(), dict(funding_rate_8h=0.00031)),
            ("CHASING", opportunity(entry=100.51), {}),
        ):
            with self.subTest(reason=reason):
                result = self.reject(reason, op, **kwargs)
                for key in unavailable:
                    self.assertIsNone(result[key], key)
                self.assertEqual(result["rr_required_target1"], 1.5)
                self.assertEqual(result["rr_required_blended"], 2.5)
                json.dumps(result, allow_nan=False)
        invalid = self.reject("TIER_NOT_ELIGIBLE", opportunity(tier=3))
        self.assertIsNone(invalid["rr_required_target1"])
        self.assertIsNone(invalid["rr_required_blended"])
        invalid = assess(None, self.instrument, 10000)
        self.assertEqual(invalid["reasons"], ["INVALID_CONTRACT"])
        for key in (*unavailable, "rr_required_target1", "rr_required_blended"):
            self.assertIsNone(invalid[key], key)

    def test_rr_no_bound_without_a_mathematically_valid_tick(self):
        for side, entry, stop, t1, t2, invalidation, confirmation in (
            ("Buy", 2, 1, 3, 4, 1.5, 1.9),
            ("Sell", 3, 4, 2, 1, 3.5, 3.1),
        ):
            with self.subTest(side=side):
                op = opportunity(
                    side=side,
                    setup="1" if side == "Buy" else "1S",
                    entry=entry,
                    stop=stop,
                    target1=t1,
                    target2=t2,
                    invalidation=invalidation,
                    confirmation=confirmation,
                )
                op = replace(
                    op,
                    evidence=dict(
                        op.evidence,
                        entry_zone_low=entry - 0.1,
                        entry_zone_high=entry + 0.1,
                        trigger_price=entry,
                    ),
                )
                result = self.reject(
                    "RR_TARGET1", op, instrument=replace(self.instrument, tick_size=1)
                )
                # Conservative rounding lands the solved bound on the stop.
                # It is not a valid hypothetical entry and must not be offered.
                self.assertIsNone(result["rr_entry_bound"])
                self.assertIsNone(result["rr_entry_relation"])
                self.assertIsNone(result["rr_zone_compatible"])
                self.assertTrue(math.isfinite(result["rr_target1"]))

    def test_rr_unrepresentable_ratios_remain_json_safe(self):
        op = opportunity(
            entry=1e-300,
            stop=9e-301,
            target1=1e308,
            target2=1.7e308,
            invalidation=9.5e-301,
            confirmation=9.9e-301,
        )
        op = replace(
            op,
            evidence=dict(
                op.evidence,
                entry_zone_low=9.8e-301,
                entry_zone_high=1.01e-300,
                trigger_price=1e-300,
            ),
        )
        result = self.reject(
            "QUANTITY_ABOVE_MAXIMUM",
            op,
            instrument=replace(self.instrument, tick_size=1e-303),
        )
        for key in ("rr_target1", "rr_blended", "rr_actual_split"):
            self.assertIsNone(result[key], key)
        self.assertTrue(math.isfinite(result["rr_entry_bound"]))
        json.dumps(result, allow_nan=False)

    def test_entry_expiry_stale_and_frozen_evidence_gates(self):
        self.reject("EXPIRED", now=NOW + 2 * H4)
        self.reject("STALE_ASSESSMENT", now=NOW + 301)
        op = opportunity()
        for key in ("trigger_closed_at", "daily_closed_at", "execution_closed_at"):
            ev = dict(op.evidence)
            ev.pop(key)
            self.reject("MISSING_EVIDENCE_TIME", replace(op, evidence=ev))
        self.reject(
            "UNVERIFIED_EVIDENCE",
            replace(op, evidence=dict(op.evidence, structural_valid=False)),
        )
        self.reject("EVIDENCE_ID_MISMATCH", replace(op, id="changed"))
        self.reject("ENTRY_OUTSIDE_ZONE", opportunity(entry=105))
        self.reject("CHASING", opportunity(entry=100.51))
        self.reject("SPREAD_TOO_WIDE", spread_pct=0.20001)

    def test_tick_rounding_cannot_cross_chase_zone_or_rr_gates(self):
        # Raw price is within 0.5%, but the adverse entry tick is outside it.
        self.reject(
            "CHASING",
            opportunity(entry=100.49),
            instrument=replace(self.instrument, tick_size=0.2),
        )
        op = opportunity(entry=102.009)
        self.reject(
            "ENTRY_OUTSIDE_ZONE", op, instrument=replace(self.instrument, tick_size=0.1)
        )
        self.reject("TRIGGER_NOT_BEYOND_CONFIRMATION", opportunity(confirmation=100))

    def test_stop_buffers_core_memes_and_short_prices(self):
        for symbol, bucket, stop, allowed in [
            ("BTCUSDT", "crypto", 95, True),
            ("TESTUSDT", "crypto", 95, False),
            ("TESTUSDT", "memes", 94, False),
            ("TESTUSDT", "memes", 93.6, True),
        ]:
            op = opportunity(symbol=symbol, bucket=bucket, stop=stop)
            result = self.check(op, instrument=replace(self.instrument, symbol=symbol))
            self.assertEqual(result["allowed"], allowed, result)
            if not allowed:
                self.assertIn("STOP_BUFFER_TOO_SMALL", result["reasons"])
        short = opportunity(
            side="Sell",
            setup="1S",
            entry=99.99,
            stop=106.01,
            target1=87.91,
            target2=69.91,
            invalidation=104,
            confirmation=101,
        )
        result = self.check(short, instrument=replace(self.instrument, tick_size=0.1))
        self.assertTrue(result["allowed"], result)
        self.assertEqual(
            [result[k] for k in ("entry", "stop", "target1", "target2")],
            [99.9, 106.1, 88, 70],
        )

    def test_higher_tier_rr_and_extended_risk_are_enforced(self):
        op = opportunity(target1=111, target2=128)
        self.assertTrue(self.check(op)["allowed"])
        self.reject("RR_TARGET1", replace(op, tier=2))
        extended = opportunity(setup="2X")
        result = self.check(extended, risk_pct=1)
        self.assertTrue(result["allowed"], result)
        self.assertEqual(result["risk_pct"], 0.5)
        self.reject("TIER_NOT_ELIGIBLE", replace(extended, tier=2))

    def test_missing_invalid_and_future_evidence_never_sizes(self):
        op = opportunity()
        self.reject("MISSING_EVIDENCE", replace(op, evidence=None))
        self.reject(
            "MISSING_TRIGGER_KIND",
            replace(op, evidence=dict(op.evidence, trigger_kind=None)),
        )
        self.reject(
            "INVALID_EVIDENCE_TIME",
            replace(op, evidence=dict(op.evidence, trigger_closed_at=NOW + 1)),
        )
        self.reject(
            "INVALID_RISK_MULTIPLIER",
            replace(op, evidence=dict(op.evidence, risk_multiplier=2)),
        )
        self.reject("INVALID_TIME", now=math.nan)
        self.reject("INVALID_EQUITY", equity=10**1000)

    def test_runtime_profile_keys_qty_step_and_context_multiplier(self):
        keys = {
            "risk_pct",
            "heat_pct",
            "bucket_pct",
            "max_positions",
            "max_per_bucket",
            "single_notional_pct",
            "gross_notional_pct",
        }
        for profile in PROFILES.values():
            self.assertIsInstance(profile, dict)
            self.assertEqual(set(profile), keys)
        result = self.check(
            opportunity(tier=2),
            profile="ultra_cautious",
            regime_multiplier=0.5,
            drawdown_pct=8,
        )
        self.assertTrue(result["allowed"], result)
        self.assertEqual(result["risk_pct"], 0.0125)
        self.assertEqual(result["qty_step"], self.instrument.qty_step)
        self.reject("REGIME_BLOCKED", regime_multiplier=0)
        for value in (math.nan, -1, 1.01):
            self.reject("INVALID_REGIME_MULTIPLIER", regime_multiplier=value)


if __name__ == "__main__":
    unittest.main()
