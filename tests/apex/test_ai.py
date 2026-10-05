"""Offline review/budget tests. Fake HTTP only; isolated transactional stores."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apex_bot import ai
from apex_bot.ai import AIReviewer
from apex_bot.models import Candle, Opportunity
from apex_bot.storage import Store, Transaction


NOW = 1_791_158_400.0  # UTC boundary; tests never depend on the machine clock.
KEY = "sk-test-secret-key-never-persist"


def fixtures():
    daily = [
        Candle(int((NOW - ai.DAY * n) * 1000), 100, 108, 95, 102, 10) for n in (3, 2, 1)
    ]
    execution = [
        Candle(int((NOW - ai.H4 * n) * 1000), 100, 106, 96, 101, 5) for n in (3, 2, 1)
    ]
    op = Opportunity(
        "op-1",
        "BTCUSDT",
        "Buy",
        "2",
        "READY",
        101,
        90,
        120,
        140,
        92,
        100,
        NOW - ai.DAY,
        NOW + ai.H4,
        "SWING_BREAK",
        {"structural_valid": True, "data_valid": True, "as_of": NOW, "anchor": 98},
    )
    return op, daily, execution


class MemoryStore:
    def __init__(self, state=None):
        self.state = copy.deepcopy(
            state or {"reviews": {}, "ai_budget": {}, "settings": {"risk_pct": 0.25}}
        )
        self.events = []
        self.lock = asyncio.Lock()
        self.leader = True

    async def read(self):
        return copy.deepcopy(self.state)

    async def update(self, callback):
        async with self.lock:
            if not self.leader:
                raise RuntimeError("not leader")
            tx = Transaction(copy.deepcopy(self.state))
            result = callback(tx)
            json.dumps(tx.state, allow_nan=False)  # Match production persistence.
            self.state = tx.state
            self.events.extend(copy.deepcopy(tx.events))
            return copy.deepcopy(result)


def envelope(value):
    return {
        "id": "resp-test",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(value),
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 120, "output_tokens": 80, "total_tokens": 200},
    }


def review_response(body, verdict="APPROVE", **changes):
    request = json.loads(body["input"][0]["content"])
    packet = request["packet"]
    refs = packet["required_refs"]
    value = {
        "opportunity_id": packet["opportunity"]["id"],
        "evidence_hash": request["evidence_hash"],
        "verdict": verdict,
        "reason": "Supplied numeric evidence supports this review",
        "refs": refs,
        "confidence": 0.6,
        "confidence_meaning": "review_quality_not_win_probability",
        "candidate_unchanged": True,
        "evidence_sufficient": True,
        "missing_fields": [],
        "concerns": [],
        "checks": [{"ref": r, "value": packet["numeric_refs"][r]} for r in refs],
    }
    value.update(changes)
    return envelope(value)


SOURCE = "https://www.federalreserve.gov/newsevents/pressreleases/example.htm"


def research_response(body, **changes):
    packet = json.loads(body["input"][0]["content"])["packet"]
    value = {
        "asof": packet["asof"],
        "uncertainty": "Source coverage may be incomplete",
        "advisory_only": True,
        "safe_to_trade": None,
        "facts": [
            {
                "symbol": "MACRO",
                "fact": "An official announcement was published",
                "source_url": SOURCE,
                "published": packet["asof"],
                "asof": packet["asof"],
                "uncertainty": "Impact on the listed symbols is unknown",
            }
        ],
    }
    value.update(changes)
    raw = envelope(value)
    raw["output"].insert(
        0,
        {
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "sources": [{"type": "url", "url": SOURCE}]},
        },
    )
    raw["output"][1]["content"][0]["annotations"] = [
        {"type": "url_citation", "url": SOURCE, "title": "Official release"}
    ]
    return raw


class FakeResponse:
    def __init__(self, session, body, action):
        self.session, self.body, self.action = session, body, action
        self.status = 200

    async def __aenter__(self):
        if self.session.before_send:
            await self.session.before_send()
        self.session.entered.set()
        if self.session.gate:
            await self.session.gate.wait()
        if isinstance(self.action, BaseException):
            raise self.action
        value = self.action(self.body) if callable(self.action) else self.action
        if isinstance(value, tuple):
            self.status, value = value
        self.value = value
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return self.value if isinstance(self.value, str) else json.dumps(self.value)


class FakeSession:
    def __init__(self, actions=None, before_send=None, gate=None):
        self.actions = list(actions or [])
        self.requests = []
        self.before_send = before_send
        self.gate = gate
        self.entered = asyncio.Event()

    def post(self, url, **kwargs):
        # No private exchange calls, GETs, SDK calls or real sockets exist here.
        if url != ai.RESPONSES_URL:
            raise AssertionError("unexpected network destination")
        self.requests.append((url, copy.deepcopy(kwargs)))
        action = self.actions.pop(0) if self.actions else review_response
        return FakeResponse(self, kwargs["json"], action)


class AIReviewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = patch.object(ai.time, "time", return_value=NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.store, self.session = MemoryStore(), FakeSession()
        self.op, self.daily, self.execution = fixtures()
        self.reviewer = AIReviewer(self.session, self.store, KEY, "test-model")

    async def review(self, reviewer=None, **kwargs):
        return await (reviewer or self.reviewer).review(
            self.op, self.daily, self.execution, **kwargs
        )

    def record(self):
        return next(iter(self.store.state["reviews"].values()))

    def archives(self):
        return [e for e in self.store.events if e[1] == "ai_archive"]

    def long_history(self, count=500):
        self.daily = [
            replace(self.daily[-1], open_time=int((NOW - ai.DAY * n) * 1000))
            for n in range(count, 0, -1)
        ]
        self.execution = [
            replace(self.execution[-1], open_time=int((NOW - ai.H4 * n) * 1000))
            for n in range(count, 0, -1)
        ]

    async def test_two_blind_reviews_same_packet_strict_schema_and_no_tools(self):
        result = await self.review()
        self.assertEqual(result["verdict"], "APPROVE")
        self.assertEqual(
            set(result),
            {
                "verdict",
                "reason",
                "reviewers",
                "model",
                "prompt_version",
                "created_at",
                "evidence_hash",
            },
        )
        self.assertEqual(len(result["reviewers"]), 2)
        self.assertEqual(len(self.session.requests), 2)
        first, second = [r[1] for r in self.session.requests]
        self.assertEqual(first, second)
        body = first["json"]
        self.assertFalse(first["allow_redirects"])
        self.assertEqual(first["headers"]["Authorization"], "Bearer " + KEY)
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse(body["text"]["format"]["schema"]["additionalProperties"])
        self.assertFalse(body["store"])
        self.assertEqual(body["max_output_tokens"], 1800)
        for forbidden in (
            "tools",
            "previous_response_id",
            "conversation",
            "temperature",
            "response_format",
        ):
            self.assertNotIn(forbidden, body)
        self.assertEqual(self.store.state["settings"], {"risk_pct": 0.25})

    async def test_budget_committed_before_each_send_and_usage_audited(self):
        async def check_reservation():
            state = await self.store.read()
            budget = state["ai_budget"][ai._utc_day()]
            self.assertEqual(budget["calls"], 2)
            self.assertEqual(len(budget["reservations"]), 2)

        self.session.before_send = check_reservation
        await self.review()
        archived = self.archives()[0][2]["record"]
        for attempt, original in zip(self.record()["attempts"], archived["attempts"]):
            self.assertEqual(attempt["usage"]["total_tokens"], 200)
            self.assertNotIn("raw_response", attempt)
            self.assertEqual(
                json.loads(original["raw_response"])["status"], "completed"
            )
            self.assertEqual(
                attempt["raw_response_hash"],
                sha256(original["raw_response"].encode()).hexdigest(),
            )
        budget = self.store.state["ai_budget"][ai._utc_day()]
        self.assertTrue(
            all(
                r["usage"]["total_tokens"] == 200
                for r in budget["reservations"].values()
            )
        )

    async def test_disagreement_and_wait_never_approve(self):
        for verdict, expected in (("REJECT", "REJECT"), ("WAIT", "WAIT")):
            with self.subTest(verdict=verdict):
                session = FakeSession(
                    [review_response, lambda b: review_response(b, verdict=verdict)]
                )
                reviewer = AIReviewer(session, MemoryStore(), KEY, "test-model")
                self.assertEqual((await self.review(reviewer))["verdict"], expected)
                self.assertEqual(len(session.requests), 2)

    async def test_invalid_review_fields_fail_closed_and_are_frozen(self):
        invalid = [
            {"opportunity_id": "wrong-id"},
            {"evidence_hash": "wrong-hash"},
            {
                "refs": ["/outside/price"],
                "checks": [{"ref": "/outside/price", "value": 123}],
            },
            {"confidence": -0.1},
            {"confidence": 1.1},
            {"confidence": float("nan")},
            {"confidence": float("inf")},
            {"confidence": True},
            {"confidence": "0.9"},
            {"confidence_meaning": "win_probability"},
            {"reason": " "},
            {"verdict": "approve"},
            {"candidate_unchanged": False},
            {"evidence_sufficient": False},
            {"missing_fields": ["confirmation"]},
            {"concerns": ["Unresolved contradiction"]},
            {"refs": [], "checks": []},
            {"entry": 999},
            {"confidence": None},
            {"api_key": KEY},
        ]
        for fields in invalid:
            with self.subTest(fields=fields):
                session = FakeSession(
                    [lambda b: review_response(b, **fields), review_response]
                )
                reviewer = AIReviewer(session, MemoryStore(), KEY, "test-model")
                result = await self.review(reviewer)
                self.assertEqual(result["verdict"], "WAIT")
                self.assertFalse(result["reviewers"][0]["valid"])
                self.assertEqual(await self.review(reviewer), result)
                self.assertEqual(len(session.requests), 2)

    async def test_altered_numeric_checks_duplicate_refs_missing_fields(self):
        def alter(body, mutation):
            raw = review_response(body)
            value = json.loads(raw["output"][0]["content"][0]["text"])
            mutation(value)
            return envelope(value)

        mutations = [
            lambda v: v.pop("reason"),
            lambda v: v["checks"][0].update(value=999),
            lambda v: v["refs"].append(v["refs"][0]),
            lambda v: v["checks"].pop(),
            lambda v: v["checks"][0].update(value=True),
            lambda v: v["checks"][0].update(ref="/opportunity/made_up"),
        ]
        for mutation in mutations:
            session = FakeSession([lambda b: alter(b, mutation), review_response])
            reviewer = AIReviewer(session, MemoryStore(), KEY, "test-model")
            self.assertEqual((await self.review(reviewer))["verdict"], "WAIT")

    async def test_refusal_malformed_incomplete_and_ambiguous_output(self):
        def mutated(body, mode):
            raw = review_response(body)
            msg = raw["output"][0]
            if mode == "refusal":
                msg["content"].append({"type": "refusal", "refusal": "No"})
            elif mode == "incomplete":
                raw.update(
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                )
            elif mode == "details":
                raw["incomplete_details"] = {"reason": "content_filter"}
            elif mode == "message":
                msg["status"] = "in_progress"
            elif mode == "missing_status":
                raw.pop("status")
            elif mode == "duplicate_text":
                msg["content"] *= 2
            elif mode == "duplicate_key":
                msg["content"][0]["text"] = '{"verdict":"REJECT","verdict":"APPROVE"}'
            elif mode == "tool":
                raw["output"].append({"type": "function_call", "name": "place_order"})
            elif mode == "no_output":
                raw["output"] = []
            elif mode == "json":
                msg["content"][0]["text"] = "```json\n{}\n```"
            return raw

        for mode in (
            "refusal",
            "incomplete",
            "details",
            "message",
            "missing_status",
            "duplicate_text",
            "duplicate_key",
            "tool",
            "no_output",
            "json",
        ):
            with self.subTest(mode=mode):
                session = FakeSession([lambda b: mutated(b, mode), review_response])
                reviewer = AIReviewer(session, MemoryStore(), KEY, "test-model")
                self.assertEqual((await self.review(reviewer))["verdict"], "WAIT")
        self.session.actions = ["not JSON", review_response]
        self.assertEqual((await self.review())["verdict"], "WAIT")
        self.assertNotIn("raw_response", self.record()["attempts"][0])
        self.assertEqual(
            self.archives()[0][2]["record"]["attempts"][0]["raw_response"], "not JSON"
        )

    async def test_missing_key_or_model_explicitly_unavailable_no_paid_calls(self):
        for key, model in (("", "test-model"), (KEY, ""), (" ", "test-model")):
            reviewer = AIReviewer(self.session, self.store, key, model)
            self.assertFalse(reviewer.available)
            result = await self.review(reviewer)
            self.assertEqual(result["verdict"], "WAIT")
            self.assertIn("unavailable", result["reason"])
            self.assertEqual(
                (await reviewer.research(["BTCUSDT"]))["status"], "UNAVAILABLE"
            )
        self.assertEqual(self.session.requests, [])
        self.assertEqual(self.store.state["ai_budget"], {})

    async def test_budget_requires_two_slots_and_restart_does_not_reset(self):
        too_small = AIReviewer(
            self.session, self.store, KEY, "test-model", daily_calls=1
        )
        self.assertEqual((await self.review(too_small))["verdict"], "WAIT")
        self.assertEqual(self.session.requests, [])
        limited = AIReviewer(self.session, self.store, KEY, "test-model", daily_calls=2)
        approved = await self.review(limited)
        restarted = AIReviewer(
            self.session,
            MemoryStore(await self.store.read()),
            KEY,
            "test-model",
            daily_calls=2,
        )
        self.assertEqual(await self.review(restarted), approved)
        self.op = replace(self.op, id="new-candidate")
        self.assertEqual((await self.review(restarted))["verdict"], "WAIT")
        self.assertEqual(len(self.session.requests), 2)

    async def test_cache_hash_contains_closed_history_and_context_provenance(self):
        context = {
            "macro": {
                "provenance": {
                    "source_url": "https://bls.gov/news.release/",
                    "asof": "2026-10-05",
                }
            }
        }
        initial = await self.review(context=context)
        self.op.evidence["as_of"] += 60  # Mere polling must not create paid requests.
        self.assertEqual(await self.review(context=copy.deepcopy(context)), initial)
        self.assertEqual(len(self.session.requests), 2)
        self.daily[0] = replace(self.daily[0], volume=20)
        changed_bar = await self.review(context=context)
        self.assertNotEqual(changed_bar["evidence_hash"], initial["evidence_hash"])
        context["macro"]["provenance"]["asof"] = "2026-10-04"
        changed_context = await self.review(context=context)
        self.assertNotEqual(
            changed_context["evidence_hash"], changed_bar["evidence_hash"]
        )
        self.assertEqual(len(self.session.requests), 6)

    async def test_packet_caps_both_histories_and_preserves_full_numeric_anchors(self):
        self.long_history()
        anchors = [
            {
                "id": "old-engine-pivot",
                "price": 97.25,
                "index": 4,
                "open_time": self.daily[4].open_time,
                "available_at": NOW - 490 * ai.DAY,
                "atr": 2.125,
                "nested": {"ratio": 0.618},
            }
        ]
        self.op.evidence["anchors"] = copy.deepcopy(anchors)
        expected = copy.deepcopy((self.op, self.daily, self.execution))
        result = await self.review()
        self.assertEqual(result["verdict"], "APPROVE")
        body = self.session.requests[0][1]["json"]
        packet = json.loads(body["input"][0]["content"])["packet"]
        for kind, bars in (("daily", self.daily), ("execution", self.execution)):
            self.assertEqual(len(packet[kind]), 120)
            self.assertEqual(packet[kind], [asdict(c) for c in bars[-120:]])
            provenance = packet["history_provenance"][kind]
            self.assertEqual(provenance["closed_count"], 500)
            self.assertEqual(provenance["sha256"], ai._hash([asdict(c) for c in bars]))
            self.assertIn(f"/{kind}/119/close", packet["required_refs"])
            self.assertNotIn(f"/{kind}/120/close", packet["numeric_refs"])
            self.assertEqual(
                len([r for r in packet["numeric_refs"] if r.startswith(f"/{kind}/")]),
                720,
            )
        self.assertEqual(packet["opportunity"]["evidence"]["anchors"], anchors)
        self.assertEqual(
            packet["numeric_refs"]["/opportunity/evidence/anchors/0/price"], 97.25
        )
        self.assertEqual(
            packet["numeric_refs"]["/opportunity/evidence/anchors/0/nested/ratio"],
            0.618,
        )
        self.assertEqual((self.op, self.daily, self.execution), expected)
        self.assertLess(len(body["input"][0]["content"]), 100_000)
        size = len(ai._json(packet))
        self.long_history(5000)
        larger = ai._packet(self.op, self.daily, self.execution, None, KEY)
        self.assertLess(abs(len(ai._json(larger)) - size), 1000)
        self.assertEqual(larger["daily"], packet["daily"])
        self.assertEqual(larger["execution"], packet["execution"])

    async def test_entire_history_is_validated_before_truncation_or_cache_lookup(self):
        self.long_history()
        self.assertEqual((await self.review())["verdict"], "APPROVE")
        mutations = [
            lambda bars: bars.pop(2),
            lambda bars: bars.insert(3, bars[2]),
            lambda bars: bars.__setitem__(0, replace(bars[0], high=90)),
            lambda bars: bars.__setitem__(0, replace(bars[0], volume=-1)),
            lambda bars: bars.__setitem__(0, replace(bars[0], close=float("nan"))),
            lambda bars: bars.__setitem__(
                0, replace(bars[0], open_time=bars[0].open_time + 1)
            ),
        ]
        daily, execution = self.daily, self.execution
        for kind in ("daily", "execution"):
            for mutate in mutations:
                with self.subTest(kind=kind, mutation=mutate):
                    self.daily, self.execution = daily[:], execution[:]
                    mutate(getattr(self, kind))
                    self.assertEqual((await self.review())["verdict"], "WAIT")
        self.assertEqual(len(self.session.requests), 2)
        self.assertEqual(len(self.archives()), 1)

    async def test_changed_older_bar_or_anchor_cannot_reuse_compacted_approval(self):
        self.long_history()
        original = await self.review()
        self.assertNotIn("packet", self.record())
        self.assertEqual(await self.review(), original)
        self.daily[0] = replace(self.daily[0], volume=11)
        changed = await self.review()
        self.assertNotEqual(original["evidence_hash"], changed["evidence_hash"])
        first_packet = json.loads(
            self.session.requests[0][1]["json"]["input"][0]["content"]
        )["packet"]
        later_packet = json.loads(
            self.session.requests[2][1]["json"]["input"][0]["content"]
        )["packet"]
        self.assertEqual(first_packet["daily"], later_packet["daily"])
        self.assertNotEqual(
            first_packet["history_provenance"], later_packet["history_provenance"]
        )
        self.op.evidence["anchor"] = 97
        changed_anchor = await self.review()
        self.assertNotEqual(changed["evidence_hash"], changed_anchor["evidence_hash"])
        self.assertEqual(len(self.session.requests), 6)
        self.assertEqual(len(self.archives()), 3)

    async def test_completed_record_archived_once_before_hot_state_compaction(self):
        self.long_history()
        original_event = Transaction.event
        observed = []

        def observe(tx, key, kind, payload, text=None):
            if kind == "ai_archive":
                hot = tx.state["reviews"][payload["key"]]
                self.assertIsNotNone(hot["result"])
                self.assertIn("packet", hot)
                self.assertTrue(all("raw_response" in a for a in hot["attempts"]))
                observed.append(key)
            return original_event(tx, key, kind, payload, text)

        with patch.object(Transaction, "event", new=observe):
            result = await self.review()
        self.assertEqual(len(observed), 1)
        archive = self.archives()[0]
        full = archive[2]["record"]
        hot = self.record()
        self.assertEqual(hot["archive_key"], archive[0])
        self.assertEqual(hot["archive_hash"], ai._hash(full))
        self.assertEqual(hot["evidence_hash"], ai._hash(full["packet"]))
        self.assertEqual(full["result"], result)
        self.assertNotIn("packet", hot)
        for attempt, saved in zip(hot["attempts"], full["attempts"]):
            self.assertNotIn("raw_response", attempt)
            self.assertEqual(attempt["parsed"], saved["parsed"])
            self.assertEqual(attempt["usage"], saved["usage"])
        self.assertLess(len(ai._json(hot)), len(ai._json(full)) / 4)
        self.assertIsNone(archive[3])  # The audit archive creates no Telegram message.
        restarted = AIReviewer(self.session, self.store, KEY, "test-model")
        self.assertEqual(await self.review(restarted), result)
        self.assertEqual(len(self.archives()), 1)
        self.assertEqual(len(self.session.requests), 2)

    async def test_archive_failure_rolls_back_result_without_losing_raw_or_reposting(
        self,
    ):
        original_event = Transaction.event

        def fail_archive(tx, key, kind, payload, text=None):
            original_event(tx, key, kind, payload, text)
            if kind == "ai_archive":
                raise RuntimeError("injected archive persistence failure")

        with patch.object(Transaction, "event", new=fail_archive):
            self.assertEqual((await self.review())["verdict"], "WAIT")
        self.assertIsNone(self.record()["result"])
        self.assertIn("packet", self.record())
        self.assertTrue(
            all(
                a["status"] == "completed" and a["raw_response"]
                for a in self.record()["attempts"]
            )
        )
        self.assertEqual(self.archives(), [])
        self.assertFalse(any(e[1] == "ai_review" for e in self.store.events))
        restarted = AIReviewer(self.session, self.store, KEY, "test-model")
        self.assertEqual((await self.review(restarted))["verdict"], "APPROVE")
        self.assertEqual(len(self.archives()), 1)
        self.assertNotIn("packet", self.record())
        self.assertEqual(len(self.session.requests), 2)

    async def test_legacy_completed_records_compact_without_reusing_unrelated_result(
        self,
    ):
        result = await self.review()
        state = await self.store.read()
        key = next(iter(state["reviews"]))
        state["reviews"][key] = copy.deepcopy(self.archives()[0][2]["record"])
        legacy = MemoryStore(state)
        reviewer = AIReviewer(self.session, legacy, KEY, "test-model", daily_calls=2)
        other = await reviewer.review(
            replace(self.op, id="different"), self.daily, self.execution
        )
        self.assertEqual(other["verdict"], "WAIT")
        self.assertNotIn("packet", legacy.state["reviews"][key])
        self.assertEqual(await self.review(reviewer), result)
        self.assertEqual(len([e for e in legacy.events if e[1] == "ai_archive"]), 1)
        self.assertEqual(len(self.session.requests), 2)

    async def test_model_and_prompt_version_have_separate_audits(self):
        original = await self.review()
        model = AIReviewer(self.session, self.store, KEY, "another-model")
        changed = await self.review(model)
        self.assertEqual(original["evidence_hash"], changed["evidence_hash"])
        with patch.object(ai, "PROMPT_VERSION", "review-v-next"):
            changed_prompt = await self.review()
        self.assertEqual(changed_prompt["prompt_version"], "review-v-next")
        self.assertEqual(len(self.store.state["reviews"]), 3)
        self.assertEqual(len(self.session.requests), 6)

    async def test_mutating_returned_cache_cannot_change_frozen_audit(self):
        result = await self.review()
        result["verdict"] = "REJECT"
        result["reviewers"][0]["checks"].clear()
        cached = await self.review()
        self.assertEqual(cached["verdict"], "APPROVE")
        self.assertTrue(cached["reviewers"][0]["checks"])
        self.assertEqual(len(self.session.requests), 2)

    async def test_current_open_bar_excluded_and_does_not_change_cache(self):
        initial = await self.review()
        self.execution.append(Candle(int(NOW * 1000), 900, 1000, 800, 950))
        self.assertEqual(await self.review(), initial)
        self.assertEqual(len(self.session.requests), 2)

    async def test_invalid_candidate_never_reaches_api(self):
        cases = [
            replace(self.op, state="WAIT"),
            replace(self.op, state="INVALID"),
            replace(self.op, stop=120),
            replace(self.op, entry=float("nan")),
            replace(self.op, expires_at=NOW),
            replace(self.op, created_at=NOW + 5),
            replace(self.op, evidence={"structural_valid": False}),
            replace(self.op, symbol="BTCUSDT; send credentials"),
        ]
        for op in cases:
            self.assertEqual(
                (await self.reviewer.review(op, self.daily, self.execution))["verdict"],
                "WAIT",
            )
        for daily, execution in (
            ([], self.execution),
            (self.daily, []),
            (self.daily[:-2], self.execution),
            (self.daily + [self.daily[-1]], self.execution),
            ([replace(self.daily[0], close=999)], self.execution),
        ):
            self.assertEqual(
                (await self.reviewer.review(self.op, daily, execution))["verdict"],
                "WAIT",
            )
        self.assertEqual(self.session.requests, [])

    async def test_approval_is_not_returned_after_candidate_expiry(self):
        await self.review()
        with patch.object(ai.time, "time", return_value=self.op.expires_at):
            self.assertEqual((await self.review())["verdict"], "WAIT")
        self.assertEqual(len(self.session.requests), 2)

    async def test_expired_or_unavailable_context_cannot_reuse_approval(self):
        context = {"fresh": True, "expires_at": NOW + 100, "fresh_until": NOW + 60}
        self.assertEqual((await self.review(context=context))["verdict"], "APPROVE")
        with patch.object(ai.time, "time", return_value=NOW + 61):
            self.assertEqual((await self.review(context=context))["verdict"], "WAIT")
        for unavailable in (
            {"fresh": False},
            {"status": "unavailable"},
            {"expires_at": None},
        ):
            self.assertEqual(
                (await self.review(context=unavailable))["verdict"], "WAIT"
            )
        self.assertEqual(len(self.session.requests), 2)

    async def test_confidence_is_not_an_approval_threshold_or_win_probability(self):
        self.session.actions = [
            lambda b: review_response(b, confidence=0),
            lambda b: review_response(b, confidence=0),
        ]
        result = await self.review()
        self.assertEqual(result["verdict"], "APPROVE")
        self.assertNotIn("win_probability", result)
        self.assertIn(
            "NEVER win probability", self.session.requests[0][1]["json"]["instructions"]
        )

    async def test_concurrent_reviewers_only_pay_twice(self):
        gate = asyncio.Event()
        self.session.gate = gate
        first = asyncio.create_task(self.review())
        await self.session.entered.wait()
        second = AIReviewer(self.session, self.store, KEY, "test-model")
        waiting = await asyncio.gather(*(self.review(second) for _ in range(10)))
        self.assertTrue(all(r["verdict"] == "WAIT" for r in waiting))
        gate.set()
        approved = await first
        self.assertEqual(approved["verdict"], "APPROVE")
        self.assertEqual(await self.review(second), approved)
        self.assertEqual(len(self.session.requests), 2)
        self.assertEqual(self.store.state["ai_budget"][ai._utc_day()]["calls"], 2)

    async def test_timeout_consumes_budget_no_retry_even_after_restart(self):
        self.session.gate = asyncio.Event()
        with patch.object(ai, "REQUEST_TIMEOUT", 0.01):
            result = await self.review()
        self.assertEqual(result["verdict"], "WAIT")
        self.assertTrue(
            all(
                a["error"] == "TIMEOUT_OUTCOME_UNKNOWN"
                for a in self.record()["attempts"]
            )
        )
        restarted = AIReviewer(
            self.session, MemoryStore(await self.store.read()), KEY, "test-model"
        )
        self.assertEqual(await self.review(restarted), result)
        self.assertEqual(len(self.session.requests), 2)

    async def test_cancellation_or_process_death_never_replays_paid_post(self):
        self.session.gate = asyncio.Event()
        task = asyncio.create_task(self.review())
        await self.session.entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        restarted = AIReviewer(
            self.session, MemoryStore(await self.store.read()), KEY, "test-model"
        )
        before = len(self.session.requests)
        self.assertEqual((await self.review(restarted))["verdict"], "WAIT")
        self.assertEqual(len(self.session.requests), before)
        self.assertEqual(self.store.state["ai_budget"][ai._utc_day()]["calls"], 2)
        self.assertIn("packet", self.record())
        self.assertEqual(self.archives(), [])

    async def test_lost_leader_or_unavailable_store_prevents_posts(self):
        self.store.leader = False
        self.assertEqual((await self.review())["verdict"], "WAIT")
        self.assertEqual(self.session.requests, [])

    async def test_lost_leader_after_reservation_prevents_posts(self):
        original = self.store.update

        async def lose_after_claim(callback):
            result = await original(callback)
            if isinstance(result, dict) and "owner" in result:
                self.store.leader = False
            return result

        self.store.update = lose_after_claim
        self.assertEqual((await self.review())["verdict"], "WAIT")
        self.assertEqual(self.session.requests, [])
        self.assertEqual(self.store.state["ai_budget"][ai._utc_day()]["calls"], 2)

    async def test_http_errors_and_transport_failures_never_retry_or_leak_key(self):
        for failure in (
            RuntimeError("Authorization Bearer " + KEY),
            (401, {"error": KEY}),
            (429, {"error": "rate limit"}),
            (500, "failure"),
            (302, "redirect"),
        ):
            with self.subTest(failure=type(failure)):
                store = MemoryStore()
                session = FakeSession([failure, review_response])
                reviewer = AIReviewer(session, store, KEY, "test-model")
                self.assertEqual((await self.review(reviewer))["verdict"], "WAIT")
                await self.review(reviewer)
                self.assertEqual(len(session.requests), 2)
                self.assertNotIn(KEY, json.dumps(store.state))
                self.assertNotIn(KEY, json.dumps(store.events))

    async def test_secrets_removed_before_packet_and_audit(self):
        context = {
            "api_key": KEY,
            "provenance": {"password": "dont-share", "note": KEY},
        }
        await self.review(context=context)
        wire = self.session.requests[0][1]["json"]
        self.assertNotIn(KEY, json.dumps(wire))
        self.assertNotIn("dont-share", json.dumps(wire))
        self.assertNotIn(KEY, json.dumps(self.store.state))

    async def test_inflight_context_mutation_cannot_change_blind_packet(self):
        context = {"provenance": {"revision": 1}}
        self.session.gate = asyncio.Event()
        task = asyncio.create_task(self.review(context=context))
        await self.session.entered.wait()
        context["provenance"]["revision"] = 2
        self.session.gate.set()
        self.assertEqual((await task)["verdict"], "WAIT")
        bodies = [r[1]["json"] for r in self.session.requests]
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(
            json.loads(bodies[0]["input"][0]["content"])["packet"]["context"][
                "provenance"
            ]["revision"],
            1,
        )

    async def test_utc_budget_rollover_does_not_delete_previous_day(self):
        await self.review()
        prior_day = ai._utc_day()
        self.session.actions = [research_response]
        with patch.object(ai.time, "time", return_value=NOW + ai.DAY):
            self.assertEqual(
                (await self.reviewer.research(["BTCUSDT"]))["status"], "COMPLETE"
            )
            self.assertEqual(self.store.state["ai_budget"][ai._utc_day()]["calls"], 1)
        self.assertEqual(self.store.state["ai_budget"][prior_day]["calls"], 2)

    async def test_reservation_cannot_fund_post_after_utc_midnight(self):
        original = self.store.update
        current = [NOW]

        async def cross_midnight(callback):
            result = await original(callback)
            if isinstance(result, dict) and result.get("owner"):
                current[0] += ai.DAY
            return result

        self.store.update = cross_midnight
        with patch.object(ai.time, "time", side_effect=lambda: current[0]):
            result = await self.review()
        self.assertEqual(result["verdict"], "WAIT")
        self.assertEqual(self.session.requests, [])
        self.assertEqual(self.store.state["ai_budget"][ai._utc_day()]["calls"], 2)

    async def test_competing_candidates_cannot_overspend_shared_budget(self):
        reviewer = AIReviewer(
            self.session, self.store, KEY, "test-model", daily_calls=2
        )
        results = await asyncio.gather(
            *(
                reviewer.review(
                    replace(self.op, id=f"candidate-{i}"), self.daily, self.execution
                )
                for i in range(10)
            )
        )
        self.assertEqual(sum(r["verdict"] == "APPROVE" for r in results), 1)
        self.assertEqual(len(self.session.requests), 2)
        self.assertEqual(self.store.state["ai_budget"][ai._utc_day()]["calls"], 2)

    async def test_crash_after_responses_before_finalization_recovers_without_post(
        self,
    ):
        with patch.object(
            self.reviewer, "_finalize", side_effect=RuntimeError("process stopped")
        ):
            self.assertEqual((await self.review())["verdict"], "WAIT")
        self.assertIsNone(self.record()["result"])
        self.assertIn("packet", self.record())
        self.assertEqual(self.archives(), [])
        recovered = MemoryStore(await self.store.read())
        restarted = AIReviewer(self.session, recovered, KEY, "test-model")
        self.assertEqual((await self.review(restarted))["verdict"], "APPROVE")
        self.assertEqual(len(self.session.requests), 2)
        self.assertNotIn("packet", next(iter(recovered.state["reviews"].values())))
        self.assertEqual(len([e for e in recovered.events if e[1] == "ai_archive"]), 1)

    async def test_research_domains_citations_daily_cache_and_outbox(self):
        self.session.actions = [research_response]
        result = await self.reviewer.research(["ETHUSDT", "BTCUSDT", "ETHUSDT"])
        self.assertEqual(result["status"], "COMPLETE")
        self.assertTrue(result["advisory_only"])
        self.assertIsNone(result["safe_to_trade"])
        body = self.session.requests[0][1]["json"]
        self.assertEqual(
            body["tools"],
            [
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": list(ai.PRIMARY_DOMAINS)},
                }
            ],
        )
        self.assertEqual(body["include"], ["web_search_call.action.sources"])
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(body["max_tool_calls"], 1)
        self.assertEqual(await self.reviewer.research(["BTCUSDT", "ETHUSDT"]), result)
        self.assertEqual(len(self.session.requests), 1)
        attempt = self.record()["attempts"][0]
        self.assertEqual(attempt["citations"][0]["url"], SOURCE)
        self.assertEqual(attempt["sources"][0]["url"], SOURCE)
        self.assertEqual(attempt["usage"]["total_tokens"], 200)
        self.assertNotIn("raw_response", attempt)
        self.assertNotIn("packet", self.record())
        archive = self.archives()[0][2]["record"]
        self.assertEqual(archive["packet"]["symbols"], ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(archive["result"], result)
        self.assertIn(SOURCE, archive["attempts"][0]["raw_response"])
        self.assertEqual(self.record()["archive_hash"], ai._hash(archive))
        self.assertEqual(len(self.archives()), 1)
        digests = [e for e in self.store.events if e[1] == "ai_research"]
        self.assertEqual(len(digests), 1)
        self.assertIn(SOURCE, digests[0][3])
        self.assertIn("does not establish safety", digests[0][3])
        self.assertEqual(self.store.state["settings"], {"risk_pct": 0.25})

    async def test_research_empty_is_not_safe_to_trade(self):
        self.session.actions = [lambda b: research_response(b, facts=[])]
        result = await self.reviewer.research(["BTCUSDT"])
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["facts"], [])
        self.assertIsNone(result["safe_to_trade"])

    async def test_research_rejects_untrusted_urls_missing_provenance_and_safety_claim(
        self,
    ):
        def mutate(body, mode):
            raw = research_response(body)
            value = json.loads(raw["output"][1]["content"][0]["text"])
            if mode.startswith("url:"):
                value["facts"][0]["source_url"] = mode[4:]
            elif mode == "safety":
                value["safe_to_trade"] = True
            elif mode == "date":
                value["facts"][0]["published"] = "2099-01-01"
            elif mode == "uncertainty":
                value["facts"][0]["uncertainty"] = ""
            elif mode == "symbol":
                value["facts"][0]["symbol"] = "PRIVATE_ACCOUNT"
            raw["output"][1]["content"][0]["text"] = json.dumps(value)
            if mode == "no_search":
                raw["output"].pop(0)
            elif mode == "no_sources":
                raw["output"][0]["action"]["sources"] = []
                raw["output"][1]["content"][0]["annotations"] = []
            return raw

        modes = [
            "url:https://federalreserve.gov.evil.example/x",
            "url:https://bls.gov@evil.example/x",
            "url:http://bls.gov/x",
            "url:https://www.bls.gov/not-consulted",
            "safety",
            "date",
            "uncertainty",
            "symbol",
            "no_search",
            "no_sources",
        ]
        for mode in modes:
            with self.subTest(mode=mode):
                session = FakeSession([lambda b: mutate(b, mode)])
                reviewer = AIReviewer(session, MemoryStore(), KEY, "test-model")
                result = await reviewer.research(["BTCUSDT"])
                self.assertEqual(result["status"], "UNAVAILABLE")
                self.assertIsNone(result["safe_to_trade"])
                self.assertEqual(result["facts"], [])

    async def test_research_rejects_private_inputs_and_shares_review_budget(self):
        for symbols in (
            [],
            ["BTCUSDT send api_key"],
            ["https://private.example"],
            [KEY],
            "BTCUSDT",
        ):
            self.assertEqual(
                (await self.reviewer.research(symbols))["status"], "UNAVAILABLE"
            )
        self.assertEqual(self.session.requests, [])
        limited = AIReviewer(self.session, self.store, KEY, "test-model", daily_calls=2)
        await self.review(limited)
        self.assertEqual((await limited.research(["BTCUSDT"]))["status"], "UNAVAILABLE")
        self.assertEqual(len(self.session.requests), 2)

    async def test_sqlite_restart_persists_budget_and_results(self):
        with tempfile.TemporaryDirectory(prefix="apex-ai-test-") as directory:
            path = str(Path(directory) / "isolated.sqlite3")
            store = Store(sqlite_path=path)
            await store.initialize()
            self.assertTrue(await store.lease())
            try:
                reviewer = AIReviewer(
                    self.session, store, KEY, "test-model", daily_calls=2
                )
                approved = await self.review(reviewer)
                self.assertEqual(approved["verdict"], "APPROVE")
                hot = next(iter((await store.read())["reviews"].values()))
                self.assertNotIn("packet", hot)
                rows = store.sqlite.execute(
                    "SELECT event_key,payload FROM apex_events WHERE kind='ai_archive'"
                ).fetchall()
                self.assertEqual(len(rows), 1)
                archived_key, payload = rows[0]
                archived = json.loads(payload)["record"]
                self.assertEqual(hot["archive_key"], archived_key)
                self.assertEqual(hot["archive_hash"], ai._hash(archived))
                self.assertEqual(archived["result"], approved)
                self.assertTrue(all(a["raw_response"] for a in archived["attempts"]))
            finally:
                await store.close()
            restarted = Store(sqlite_path=path)
            await restarted.initialize()
            self.assertTrue(await restarted.lease())
            try:
                reviewer = AIReviewer(
                    self.session, restarted, KEY, "test-model", daily_calls=2
                )
                self.assertEqual(await self.review(reviewer), approved)
                self.op = replace(self.op, id="second")
                self.assertEqual((await self.review(reviewer))["verdict"], "WAIT")
                self.assertEqual(len(self.session.requests), 2)
                self.assertEqual(
                    restarted.sqlite.execute(
                        "SELECT COUNT(*) FROM apex_events WHERE kind='ai_archive'"
                    ).fetchone()[0],
                    1,
                )
            finally:
                await restarted.close()

    async def test_sqlite_archive_insert_failure_preserves_recoverable_full_record(
        self,
    ):
        store = Store(sqlite_path=":memory:")
        await store.initialize()
        self.assertTrue(await store.lease())
        try:
            await asyncio.to_thread(
                store._run,
                lambda cur: cur.execute(
                    "CREATE TRIGGER fail_archive BEFORE INSERT ON apex_events "
                    "WHEN NEW.kind = 'ai_archive' BEGIN SELECT RAISE(ABORT, 'fixture archive failure'); END"
                ),
            )
            reviewer = AIReviewer(self.session, store, KEY, "test-model")
            self.assertEqual((await self.review(reviewer))["verdict"], "WAIT")
            record = next(iter((await store.read())["reviews"].values()))
            self.assertIsNone(record["result"])
            self.assertIn("packet", record)
            self.assertTrue(all(a["raw_response"] for a in record["attempts"]))
            self.assertEqual(
                store.sqlite.execute(
                    "SELECT COUNT(*) FROM apex_events WHERE kind IN ('ai_review','ai_archive')"
                ).fetchone()[0],
                0,
            )
            await asyncio.to_thread(
                store._run, lambda cur: cur.execute("DROP TRIGGER fail_archive")
            )
            restarted = AIReviewer(self.session, store, KEY, "test-model")
            self.assertEqual((await self.review(restarted))["verdict"], "APPROVE")
            self.assertNotIn(
                "packet", next(iter((await store.read())["reviews"].values()))
            )
            self.assertEqual(
                store.sqlite.execute(
                    "SELECT COUNT(*) FROM apex_events WHERE kind='ai_archive'"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(len(self.session.requests), 2)
        finally:
            await store.close()


if __name__ == "__main__":
    unittest.main()
