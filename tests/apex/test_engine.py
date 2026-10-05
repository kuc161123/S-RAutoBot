import math
import unittest
from dataclasses import replace

from apex_bot.engine import (
    DAY,
    H4,
    Pivot,
    _plans,
    _evaluate,
    analyze,
    atr,
    confirmed_zigzag,
    validate_impulse,
)
from apex_bot.models import Candle, Instrument
from apex_bot.risk import assess


def candles(prices, seconds=DAY, start=0):
    return [
        Candle(int((start + i * seconds) * 1000), p, p + 0.2, p - 0.2, p, 10)
        for i, p in enumerate(prices)
    ]


def daily_history(endpoints=(95, 130, 110, 165, 140, 180, 150)):
    prices = [100] * 16
    for end in endpoints:
        start = prices[-1]
        prices.extend(start + (end - start) * i / 8 for i in range(1, 9))
    return candles(prices)


def execution_history():
    return candles(
        [151] * 16
        + [145, 143, 141, 143, 145, 144, 142, 139, 140, 141, 140, 139, 138, 140, 142],
        H4,
        67 * DAY,
    )


def mirror(bars, center=400):
    return [
        replace(
            b,
            open=center - b.open,
            high=center - b.low,
            low=center - b.high,
            close=center - b.close,
        )
        for b in bars
    ]


def pivots(prices):
    return [
        Pivot(
            str(i),
            "low" if i % 2 == 0 else "high",
            value,
            i * DAY * 1000,
            (i + 2) * DAY,
            i,
            1,
        )
        for i, value in enumerate(prices)
    ]


class ZigzagTests(unittest.TestCase):
    def test_confirmation_lag_preserves_opposite_extreme_and_prefix_causality(self):
        bars = candles([100, 100, 100, 120])
        # High 125 is only confirmed on bar7. Low110 on bar5 is part of
        # the next leg, and must not be replaced by confirmation-bar low114.
        for values in (
            (124, 125, 120, 124),
            (123, 124, 110, 123),
            (120, 123, 115, 116),
            (116, 120, 114, 115),
            (117, 124, 117, 122),
        ):
            bars.append(Candle(len(bars) * DAY * 1000, *values, 10))
        for series, price, kind in ((bars, 110, "low"), (mirror(bars), 290, "high")):
            full = confirmed_zigzag(series, atr_period=2, atr_multiple=1)
            self.assertEqual(
                (full[-1].kind, full[-1].price, full[-1].index), (kind, price, 5)
            )
            self.assertEqual(full[-1].available_at, 9 * DAY)
            for n in range(len(series) + 1):
                self.assertEqual(
                    confirmed_zigzag(
                        series[:n], now=n * DAY, atr_period=2, atr_multiple=1
                    ),
                    [p for p in full if p.available_at <= n * DAY],
                )

    def test_wilder_atr_uses_previous_close(self):
        bars = [
            Candle(0, 10, 12, 9, 11),
            Candle(DAY * 1000, 15, 16, 14, 15),
            Candle(2 * DAY * 1000, 15, 17, 14, 16),
        ]
        self.assertEqual(atr(bars, 2), [None, 4, 3.5])

    def test_every_prefix_has_identical_historical_availability(self):
        bars = daily_history((95, 130, 110, 175, 150, 190, 120, 160, 140))
        full = confirmed_zigzag(bars)
        self.assertGreater(len(full), 6)
        for n in range(len(bars) + 1):
            cutoff = n * DAY
            prefix = confirmed_zigzag(bars[:n], now=cutoff)
            self.assertEqual(prefix, [p for p in full if p.available_at <= cutoff], n)
            self.assertEqual(prefix, confirmed_zigzag(bars, now=cutoff), n)
        for p in full:
            self.assertGreater(p.available_at, p.open_time / 1000 + DAY)

    def test_execution_prefixes_and_mirror(self):
        bars = execution_history()
        full = confirmed_zigzag(bars, H4, atr_multiple=1)
        reverse = confirmed_zigzag(mirror(bars), H4, atr_multiple=1)
        for a, b in zip(full, reverse):
            self.assertNotEqual(a.kind, b.kind)
            self.assertAlmostEqual(a.price + b.price, 400)
            self.assertEqual(a.available_at, b.available_at)
        for n in range(len(bars) + 1):
            now = 67 * DAY + n * H4
            self.assertEqual(
                confirmed_zigzag(bars[:n], H4, now, atr_multiple=1),
                [p for p in full if p.available_at <= now],
            )

    def test_flat_zero_atr_has_no_pivots(self):
        bars = [Candle(i * DAY * 1000, 100, 100, 100, 100) for i in range(40)]
        self.assertEqual(confirmed_zigzag(bars), [])

    def test_same_bar_extreme_is_never_its_own_confirmation(self):
        bars = candles([100] * 16 + [95, 110])
        bars.append(Candle(len(bars) * DAY * 1000, 110, 150, 90, 91))
        self.assertFalse(
            any(p.open_time == bars[-1].open_time for p in confirmed_zigzag(bars))
        )

    def test_reject_bad_closed_data_ignore_future_ohlc(self):
        bars = daily_history()
        now = len(bars) * DAY
        future = Candle(now * 1000, math.nan, math.inf, -1, 0)
        self.assertEqual(
            confirmed_zigzag(bars, now=now), confirmed_zigzag(bars + [future], now=now)
        )
        for invalid in [
            replace(bars[-1], close=math.nan),
            replace(bars[-1], low=0),
            replace(bars[-1], high=1),
            replace(bars[-1], volume=-1),
        ]:
            with self.assertRaises(ValueError):
                confirmed_zigzag(bars[:-1] + [invalid])
        with self.assertRaises(ValueError):
            confirmed_zigzag(bars[:20] + bars[21:])
        with self.assertRaises(ValueError):
            confirmed_zigzag(bars + [bars[-1]])


class ImpulseTests(unittest.TestCase):
    def test_standard_impulse_and_short_mirror(self):
        p = pivots([100, 130, 110, 170, 140, 185])
        self.assertEqual(validate_impulse(p), [])
        reverse = [
            replace(v, kind="high" if v.kind == "low" else "low", price=400 - v.price)
            for v in p
        ]
        self.assertEqual(validate_impulse(reverse), [])
        for size in (3, 4, 5):
            self.assertEqual(validate_impulse(p[:size], complete=False), [])

    def test_hard_elliott_rules(self):
        cases = [
            ([100, 130, 100, 170, 140, 185], "WAVE2_RETRACE_ORIGIN"),
            ([100, 130, 110, 170, 130, 185], "WAVE4_OVERLAP"),
            ([100, 140, 120, 150, 145, 190], "WAVE3_SHORTEST"),
            ([100, 130, 110, 170, 140, 165], "TRUNCATED_WAVE5_UNSUPPORTED"),
        ]
        for prices, reason in cases:
            with self.subTest(reason=reason):
                self.assertIn(reason, validate_impulse(pivots(prices)))
        self.assertEqual(
            validate_impulse(pivots([100, 130, 110, 140, 135, 165])), []
        )  # equal lengths allowed

    def test_degenerate_nonfinite_and_unconfirmed(self):
        p = pivots([100, 130, 110, 170, 140, 185])
        self.assertIn("PIVOT_COUNT", validate_impulse(p[:5]))
        self.assertIn(
            "INVALID_PRICE", validate_impulse([replace(p[0], price=0)] + p[1:])
        )
        self.assertIn(
            "INVALID_PRICE", validate_impulse([replace(p[0], price=math.inf)] + p[1:])
        )
        self.assertIn(
            "UNCONFIRMED_PIVOT",
            validate_impulse([replace(p[0], available_at=0)] + p[1:]),
        )
        self.assertIn(
            "NOT_ALTERNATING", validate_impulse([replace(p[0], kind="high")] + p[1:])
        )


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.daily = daily_history()
        self.execution = execution_history()
        self.now = self.execution[-1].open_time / 1000 + H4

    def type1(self, daily=None, execution=None, now=None):
        return next(
            op
            for op in analyze(
                "TESTUSDT",
                self.daily if daily is None else daily,
                self.execution if execution is None else execution,
                self.now if now is None else now,
            )
            if op.setup == "1"
        )

    def test_actual_closed_bar_trigger_and_risk_contract(self):
        op = self.type1()
        self.assertEqual((op.state, op.reason), ("READY", "SWING_BREAK"))
        self.assertEqual(op.entry, 142)
        self.assertEqual(op.evidence["trigger_closed_at"], self.now)
        self.assertEqual(op.expires_at, self.now + 2 * H4)
        self.assertEqual(self.type1(now=self.now - 1).state, "WAIT")
        risk = assess(
            op,
            Instrument("TESTUSDT", 0.01, 0.01, 10000, 0.01, 5),
            10000,
            exposures=[],
            funding_rate_8h=0,
            spread_pct=0.01,
            now=self.now,
        )
        self.assertTrue(risk["allowed"], risk)

    def test_no_historical_trigger_or_id_rewrite_from_future(self):
        for n in range(len(self.daily) + 1):
            now = n * DAY
            self.assertEqual(
                analyze("TESTUSDT", self.daily[:n], self.execution, now),
                analyze("TESTUSDT", self.daily, self.execution, now),
                n,
            )
        for n in range(len(self.execution) + 1):
            now = 67 * DAY + n * H4
            self.assertEqual(
                analyze("TESTUSDT", self.daily, self.execution[:n], now),
                analyze("TESTUSDT", self.daily, self.execution, now),
                n,
            )
        waiting = self.type1(execution=self.execution[:-1], now=self.now - H4)
        ready = self.type1()
        self.assertEqual(waiting.id, ready.id)
        self.assertEqual(waiting.evidence["anchors"], ready.evidence["anchors"])
        self.assertEqual(
            waiting.evidence["daily_evidence_id"], ready.evidence["daily_evidence_id"]
        )

    def test_startup_type2_does_not_need_future_six_pivots(self):
        bars = self.daily[:43]
        self.assertLess(len(confirmed_zigzag(bars)), 6)
        result = analyze("TESTUSDT", bars, [], 43 * DAY)
        self.assertTrue(any(op.setup == "2" for op in result))
        self.assertEqual(result, analyze("TESTUSDT", self.daily, [], 43 * DAY))

    def test_long_short_trigger_mirroring(self):
        long = self.type1()
        short = next(
            op
            for op in analyze(
                "TESTUSDT", mirror(self.daily), mirror(self.execution), self.now
            )
            if op.setup == "1S"
        )
        self.assertEqual(short.state, "READY")
        self.assertEqual(short.side, "Sell")
        self.assertAlmostEqual(short.entry + long.entry, 400)
        self.assertAlmostEqual(short.target1 + long.target1, 400)
        self.assertAlmostEqual(short.target2 + long.target2, 400)
        self.assertEqual(
            short.evidence["trigger_closed_at"], long.evidence["trigger_closed_at"]
        )

    def test_touch_and_intrabar_cross_are_wait(self):
        self.assertEqual(
            self.type1(execution=self.execution[:-1], now=self.now - H4).state, "WAIT"
        )
        e = self.execution[:-1] + [
            replace(self.execution[-1], open=140, high=145, low=139.8, close=140)
        ]
        self.assertEqual(self.type1(execution=e).state, "WAIT")

    def test_expiry_does_not_rearm(self):
        ready = self.type1()
        expired = self.type1(now=ready.expires_at)
        later = self.type1(now=ready.expires_at + H4)
        self.assertEqual((expired.state, expired.reason), ("INVALID", "EXPIRED"))
        self.assertEqual(later.id, ready.id)
        self.assertEqual(
            later.evidence["trigger_evidence_id"], ready.evidence["trigger_evidence_id"]
        )
        self.assertEqual(later.reason, "EXPIRED")

    def test_outzone_chasing_stop_and_target_terminal(self):
        for price, expected in [
            (149.9, "ENTRY_OUTSIDE_ZONE"),
            (143, "CHASING"),
            (110, "HARD_STOP_BREACHED"),
            (181, "TARGET_ALREADY_PASSED"),
        ]:
            e = self.execution + candles([price], H4, self.now)
            op = self.type1(execution=e, now=self.now + H4)
            self.assertEqual((op.state, op.reason), ("INVALID", expected), price)

    def test_current_entry_outside_zone_cannot_be_a_trigger(self):
        e = self.execution[:-1] + [replace(self.execution[-1], high=160, close=160)]
        op = self.type1(execution=e)
        self.assertEqual(op.state, "WAIT")
        self.assertEqual(
            op.evidence["last_entry_attempt"]["reason"], "TRIGGER_OUTSIDE_ZONE"
        )
        self.assertIsNone(op.evidence["trigger_closed_at"])
        self.assertIsNone(op.evidence["terminal_status"])

    def test_missing_stale_and_invalid_data(self):
        self.assertEqual(self.type1(execution=[]).state, "WAIT")
        for value in (0, math.nan, math.inf):
            d = self.daily[:-1] + [replace(self.daily[-1], close=value)]
            self.assertEqual(analyze("TESTUSDT", d, [], self.now), [])
        self.assertEqual(analyze("TESTUSDT", self.daily, self.execution, math.nan), [])
        self.assertEqual(
            analyze("TESTUSDT", self.daily[:20] + self.daily[21:], [], self.now), []
        )

    def test_expired_untriggered_plan(self):
        op = self.type1(execution=[], now=110 * DAY)
        self.assertEqual((op.state, op.reason), ("INVALID", "EXPIRED"))

    def test_type2x_extended_rules_and_both_type5_sides(self):
        d = daily_history((95, 130, 110, 175, 150, 190, 120, 160, 140))
        allplans = _plans(d, confirmed_zigzag(d))
        extended = next(p for p in allplans if p.setup == "2X")
        self.assertGreaterEqual(extended.zone_low, extended.anchors[1].price * 1.01)
        self.assertEqual(extended.risk_multiplier, 0.5)
        p = next(p for p in allplans if p.setup == "5S")
        a = p.anchors[5].price - p.anchors[6].price
        self.assertAlmostEqual(p.target1, p.anchors[7].price - a)
        self.assertAlmostEqual(p.target2, p.anchors[7].price - 1.618 * a)
        reverse = _plans(mirror(d), confirmed_zigzag(mirror(d)))
        self.assertTrue(any(p.setup == "2XS" for p in reverse))
        short_correction = next(p for p in reverse if p.setup == "5L")
        self.assertAlmostEqual(p.target1 + short_correction.target1, 400)
        self.assertAlmostEqual(p.target2 + short_correction.target2, 400)

    def test_type2_closed_breakout_retest_and_short_mirror(self):
        d = self.daily[:43]
        e = candles([129, 132, 130.3], H4, 43 * DAY)
        now = 43 * DAY + 3 * H4
        op = analyze("TESTUSDT", d, e, now)[0]
        self.assertEqual(
            (op.setup, op.state, op.reason), ("2", "READY", "BREAKOUT_RETEST")
        )
        self.assertEqual(op.confirmation, 130.2)
        self.assertGreater(
            op.entry, op.evidence["zone_high"] * 1.01
        )  # explicit retest exception
        short = analyze("TESTUSDT", mirror(d), mirror(e), now)[0]
        self.assertEqual((short.setup, short.state), ("2S", "READY"))
        self.assertAlmostEqual(op.entry + short.entry, 400)
        self.assertEqual(analyze("TESTUSDT", d, e, now - 1)[0].state, "WAIT")

    def test_first_eligible_bar_uses_already_known_previous_close(self):
        d = self.daily[:43]
        start = 43 * DAY
        # The comparison candle ends at creation; only the later breakout and
        # still-later retest can trigger. No pre-creation signal is admitted.
        e = candles([129, 132, 130.3], H4, start - H4)
        for daily, execution, setup in ((d, e, "2"), (mirror(d), mirror(e), "2S")):
            before = analyze("TESTUSDT", daily, execution, start + H4)[0]
            self.assertEqual(before.state, "WAIT")
            self.assertIsNone(before.evidence["trigger_closed_at"])
            ready = analyze("TESTUSDT", daily, execution, start + 2 * H4)[0]
            self.assertEqual(
                (ready.setup, ready.state, ready.reason),
                (setup, "READY", "BREAKOUT_RETEST"),
            )
            self.assertEqual(ready.evidence["trigger_closed_at"], start + 2 * H4)
            self.assertEqual(
                ready,
                analyze(
                    "TESTUSDT",
                    daily,
                    execution
                    + [
                        replace(
                            execution[-1],
                            open_time=execution[-1].open_time + H4 * 1000,
                            close=math.nan,
                        )
                    ],
                    start + 2 * H4,
                )[0],
            )

    def test_retest_requires_later_bar_and_expires_after_three(self):
        d = self.daily[:43]
        # A breakout bar that also touches the level cannot retest itself.
        e = candles([129, 132], H4, 43 * DAY)
        e[-1] = replace(e[-1], low=130.1)
        self.assertEqual(analyze("TESTUSDT", d, e, 43 * DAY + 2 * H4)[0].state, "WAIT")
        late = candles([129, 132, 133, 134, 135, 130.3], H4, 43 * DAY)
        op = analyze("TESTUSDT", d, late, 44 * DAY)[0]
        self.assertEqual(op.state, "WAIT")
        self.assertEqual(op.evidence["retest_status"], "RETEST_EXPIRED")
        self.assertIsNone(op.evidence["trigger_closed_at"])
        failed = candles([129, 132, 130], H4, 43 * DAY)
        op = analyze("TESTUSDT", d, failed, 43 * DAY + 3 * H4)[0]
        self.assertEqual(op.state, "WAIT")
        self.assertEqual(op.evidence["retest_status"], "FAILED_FIRST_RETEST")
        # A later retest cannot reuse the skipped alternate's old clock.
        later = failed + candles([130.3], H4, 43 * DAY + 3 * H4)
        still_wait = analyze("TESTUSDT", d, later, 43 * DAY + 4 * H4)[0]
        self.assertEqual(still_wait.state, "WAIT")
        self.assertEqual(still_wait.evidence["retest_status"], "FAILED_FIRST_RETEST")

    def test_failed_retest_preserves_fresh_deep_zone_trigger_both_sides(self):
        d = self.daily[:43]
        start = 43 * DAY
        e = candles([129, 132, 130, 118, 120], H4, start)
        for daily, execution, side in ((d, e, "Buy"), (mirror(d), mirror(e), "Sell")):
            plan = _plans(daily, confirmed_zigzag(daily))[0]
            long = side == "Buy"
            swings = [
                Pivot(
                    "comparison",
                    "high" if long else "low",
                    128 if long else 272,
                    int((start - H4) * 1000),
                    start,
                    0,
                    1,
                ),
                Pivot(
                    "approach",
                    "high" if long else "low",
                    119 if long else 281,
                    int((start + H4) * 1000),
                    start + 3 * H4,
                    1,
                    1,
                ),
            ]
            now = start + 5 * H4
            op = _evaluate("TESTUSDT", plan, daily, execution, swings, now, 1, "crypto")
            self.assertEqual(
                (op.side, op.state, op.reason), (side, "READY", "SWING_BREAK")
            )
            self.assertEqual(op.evidence["retest_status"], "FAILED_FIRST_RETEST")
            self.assertEqual(op.evidence["trigger_closed_at"], now)
            self.assertEqual(op.entry, 120 if long else 280)
            self.assertEqual(op.evidence["entry_zone_low"], plan.zone_low)
            # A future pivot cannot make this trigger available early.
            future = swings[:-1] + [replace(swings[-1], available_at=now + H4)]
            self.assertEqual(
                _evaluate(
                    "TESTUSDT", plan, daily, execution, future, now, 1, "crypto"
                ).state,
                "WAIT",
            )

    def test_comparison_pivot_may_predate_anchor_but_latest_must_be_relevant(self):
        plan = next(
            p
            for p in _plans(self.daily, confirmed_zigzag(self.daily))
            if p.setup == "1"
        )
        swings = confirmed_zigzag(self.execution, H4, self.now, atr_multiple=1)
        highs = [p for p in swings if p.kind == "high"]
        before = plan.anchors[-1].open_time - DAY * 1000
        older = [
            replace(p, open_time=before) if p.id == highs[-2].id else p for p in swings
        ]
        op = _evaluate(
            "TESTUSDT", plan, self.daily, self.execution, older, self.now, 1, "crypto"
        )
        self.assertEqual(op.state, "READY")
        unrelated = [
            replace(p, open_time=before) if p.id == highs[-1].id else p for p in older
        ]
        self.assertEqual(
            _evaluate(
                "TESTUSDT",
                plan,
                self.daily,
                self.execution,
                unrelated,
                self.now,
                1,
                "crypto",
            ).state,
            "WAIT",
        )
        superseded = older + [replace(highs[-1], id="new-high", price=160)]
        self.assertEqual(
            _evaluate(
                "TESTUSDT",
                plan,
                self.daily,
                self.execution,
                superseded,
                self.now,
                1,
                "crypto",
            ).state,
            "WAIT",
        )

    def test_type2x_triggers_in_both_directions_and_tier2_waits(self):
        d = daily_history((95, 130, 110, 175, 150, 190, 120, 160, 140))
        e = [
            replace(
                b,
                open_time=b.open_time - 15 * DAY * 1000,
                open=b.open + 5,
                high=b.high + 5,
                low=b.low + 5,
                close=b.close + 5,
            )
            for b in self.execution
        ]
        now = e[-1].open_time / 1000 + H4
        op = next(o for o in analyze("TESTUSDT", d, e, now) if o.setup == "2X")
        self.assertEqual(op.state, "READY")
        self.assertEqual(op.evidence["risk_multiplier"], 0.5)
        short = next(
            o
            for o in analyze("TESTUSDT", mirror(d), mirror(e), now)
            if o.setup == "2XS"
        )
        self.assertEqual(short.state, "READY")
        watch = next(
            o for o in analyze("TESTUSDT", d, e, now, tier=2) if o.setup == "2X"
        )
        self.assertEqual((watch.state, watch.reason), ("WAIT", "TIER_NOT_ELIGIBLE"))

    def test_type5_requires_confirmed_a_b_then_a_new_execution_trigger(self):
        d = daily_history((95, 130, 110, 175, 150, 190, 120, 160, 140))
        d += candles([140, 140, 140], DAY, len(d) * DAY)
        e = [
            replace(b, open_time=b.open_time + 18 * DAY * 1000)
            for b in mirror(self.execution, 299)
        ]
        now = e[-1].open_time / 1000 + H4
        op = next(o for o in analyze("TESTUSDT", d, e, now) if o.setup == "5S")
        self.assertEqual(op.state, "READY")
        long = next(
            o for o in analyze("TESTUSDT", mirror(d), mirror(e), now) if o.setup == "5L"
        )
        self.assertEqual(long.state, "READY")
        self.assertEqual(
            op.evidence["trigger_closed_at"], long.evidence["trigger_closed_at"]
        )
        self.assertLess(op.created_at, op.evidence["trigger_closed_at"])
        self.assertFalse(
            any(o.setup == "5S" for o in analyze("TESTUSDT", d, e, 84 * DAY))
        )

    def test_target_passed_before_plan_available_never_signals_retroactively(self):
        d = daily_history((95, 130, 110, 165))
        # The first breakout daily bar jumps past the fixed Type 2 T1.
        d[42] = replace(d[42], high=150)
        result = analyze("TESTUSDT", d, [], 43 * DAY)
        op = next(o for o in result if o.setup == "2")
        self.assertEqual((op.state, op.reason), ("INVALID", "TARGET_ALREADY_PASSED"))
        self.assertIsNone(op.evidence["trigger_closed_at"])

    def test_2x_overlap_and_expanded_b_wicks_are_structural_failures(self):
        d = daily_history((95, 130, 110, 175, 150, 190, 120, 160, 140))
        for setup, created, price in [("2X", 52 * DAY, 130.2), ("5S", 85 * DAY, 190.2)]:
            e = candles([price], H4, created)
            op = next(
                o for o in analyze("TESTUSDT", d, e, created + H4) if o.setup == setup
            )
            self.assertEqual(
                (op.state, op.reason), ("INVALID", "STRUCTURE_INVALIDATED")
            )


if __name__ == "__main__":
    unittest.main()
