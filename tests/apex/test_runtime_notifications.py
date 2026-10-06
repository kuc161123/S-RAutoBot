"""Offline notification wording, durable observations and unchanged event identity."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import AsyncMock, patch

import pytest

from apex_bot import ai, runtime as runtime_module
from apex_bot.ai import AIReviewer
from apex_bot.notifications import (
    describe_reason,
    opportunity_alert,
    research_alert,
    research_failure_details,
    research_failure,
    review_alert,
    sanitize_text,
    setup_label,
    shadow_alert,
)
from .test_ai import (
    FakeSession,
    MemoryStore,
    NOW as AI_NOW,
    SOURCE,
    fixtures,
    research_response,
)
from .test_runtime import NOW, events, harness, ready, reviews


@pytest.mark.parametrize("side,label", [("Buy", "LONG"), ("Sell", "SHORT")])
@pytest.mark.parametrize(
    "state,reason",
    [
        ("WAIT", "AWAIT_CLOSED_4H_TRIGGER"),
        ("READY", "SWING_BREAK"),
        ("INVALID", "DAILY_INVALIDATION"),
    ],
)
def test_opportunity_alert_explains_direction_levels_and_transition(
    side, label, state, reason
):
    op = replace(fixtures()[0], side=side, state=state, reason=reason)
    before = op.to_dict()
    text = opportunity_alert(op)
    assert f"BTCUSDT · {label}" in text and side not in text
    assert "planned entry: 101" in text.lower()
    assert "Planned stop: 90" in text
    assert "Targets: 1 → 120 · 2 → 140" in text
    assert "Structural invalidation: 92" in text
    assert "/why BTCUSDT" in text and "/waves BTCUSDT" in text
    assert reason not in text
    assert setup_label("2") in text and "Elliott 2" not in text
    if state == "WAIT":
        assert "No trade opened" in text and "closed 4-hour trigger" in text
        assert "Watch confirmation level 100" in text
    elif state == "READY":
        assert "not executed" in text and "Risk and AI checks still apply" in text
        assert "not a fill" in text
    else:
        assert "Setup invalidated" in text
        assert "does not confirm a trade entry or exit" in text
    assert len(text.encode("utf-16-le")) // 2 < 1500
    assert op.to_dict() == before


@pytest.mark.parametrize(
    "reason,condition",
    [
        ("DATA_STALE_OR_MISSING", "fresh daily and 4-hour candle data"),
        ("CONFLICTING_DIRECTIONS", "conflicting direction signals to resolve"),
        ("TIER_NOT_ELIGIBLE", "eligible setup under the existing symbol rules"),
    ],
)
def test_wait_explains_the_actual_next_condition(reason, condition):
    text = opportunity_alert(replace(fixtures()[0], state="WAIT", reason=reason))
    assert "No trade opened" in text and condition in text
    assert reason not in text


def test_helpers_tolerate_missing_display_data_and_preserve_readable_reasons():
    assert describe_reason(None) == "Reason unavailable."
    assert describe_reason("Risk checks passed") == "Risk checks passed"
    assert setup_label(None) == "Elliott setup"
    op = replace(fixtures()[0], entry=None, stop=float("nan"), side="unknown")
    text = opportunity_alert(op)
    assert "Planned entry: unavailable" in text and "Planned stop: unavailable" in text
    assert "SHORT" not in text
    for verdict in ("WAIT", "REJECT", "APPROVE", None):
        assert "not a fill" in review_alert(op, {"verdict": verdict})


def test_simulation_notifications_do_not_invent_fills_or_zero_results():
    trade = {"symbol": "BTCUSDT", "side": "Sell", "arm": "baseline_shadow"}
    pending = shadow_alert({**trade, "status": "PENDING", "limit": 101, "stop": 110})
    assert "SHORT" in pending and "Rules baseline" in pending
    assert "no trade opened yet" in pending and "subsequent eligible candle" in pending
    expired = shadow_alert({**trade, "status": "EXPIRED", "exit_reason": "UNFILLED"})
    assert "no fill" in expired and "without a fill" in expired and "net" not in expired
    opened = shadow_alert({**trade, "status": "OPEN"})
    assert (
        "Simulated entry filled" in opened and "before funding: unavailable" in opened
    )
    closed = shadow_alert(
        {**trade, "status": "CLOSED", "net_pnl_before_funding": -2, "data_gap": True}
    )
    assert "Simulated trade closed" in closed and "before funding: $-2" in closed
    assert "not final" in closed


def test_runtime_transition_text_preserves_event_keys_payloads_and_dedup(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            base = replace(fixtures()[0], created_at=NOW, expires_at=NOW + 7200)
            current = base
            monkeypatch.setattr(
                runtime_module, "analyze", lambda *args, **kwargs: [current]
            )
            monkeypatch.setattr(h.runtime, "consider", AsyncMock())
            for state, reason in [
                ("WAIT", "AWAIT_CLOSED_4H_TRIGGER"),
                ("READY", "SWING_BREAK"),
                ("INVALID", "DAILY_INVALIDATION"),
            ]:
                current = replace(base, state=state, reason=reason)
                await h.runtime.scan()
                await h.runtime.scan()
            saved = await events(h.store, "opportunity")
            assert {key for key, _, _ in saved} == {
                f"op:{base.id}:{state}" for state in ("WAIT", "READY", "INVALID")
            }
            assert {payload["reason"] for _, _, payload in saved} == {
                "AWAIT_CLOSED_4H_TRIGGER",
                "SWING_BREAK",
                "DAILY_INVALIDATION",
            }
            alerts = {
                n["key"]: n["text"] for n in await h.store.pending_notifications()
            }
            assert "No trade opened" in alerts[f"op:{base.id}:WAIT"]
            assert "not executed" in alerts[f"op:{base.id}:READY"]
            assert "Setup invalidated" in alerts[f"op:{base.id}:INVALID"]
            assert not (await h.store.read())["trades"] and not h.market.submissions

    asyncio.run(scenario())


def test_health_recovery_preserves_history_policy_fields_and_event_dedup(harness):
    async def scenario():
        async with harness() as h:
            await h.runtime._health("telegram")
            await h.tick(10)
            await h.runtime._health(
                "telegram", TimeoutError("do-not-persist-this-secret")
            )
            await h.tick(1)
            await h.runtime._health(
                "telegram", TimeoutError("do-not-persist-this-secret")
            )
            health = (await h.store.read())["health"]
            assert health["telegram_at"] == h.runtime.last_loop["telegram"] == NOW
            assert health["telegram_error"] == "TimeoutError"
            assert (
                health["last_issue"]
                == health["issues"]["telegram"]
                == {
                    "component": "telegram",
                    "error_type": "TimeoutError",
                    "occurred_at": NOW + 11,
                    "recovered_at": None,
                }
            )
            await h.runtime._health("scan")
            assert (await h.store.read())["health"]["last_issue"][
                "recovered_at"
            ] is None
            await h.tick(1)
            await h.runtime._health("telegram")
            health = (await h.store.read())["health"]
            assert health["error"] == "telegram: TimeoutError"
            assert not [key for key in health if key.endswith("_error")]
            assert health["issues"]["telegram"]["recovered_at"] == NOW + 12
            assert health["last_issue"] == health["issues"]["telegram"]
            await h.tick(1)
            await h.runtime._health("telegram")
            assert (await h.store.read())["health"]["last_issue"][
                "recovered_at"
            ] == NOW + 12
            records = await events(h.store, "health_issue")
            assert records == [
                (
                    f"issue:telegram:{int((NOW + 10) // 900)}",
                    "health_issue",
                    {"component": "telegram", "error_type": "TimeoutError"},
                )
            ]
            assert "do-not-persist" not in json.dumps(health)
            await h.restart()
            assert (await h.store.read())["health"]["issues"] == health["issues"]

    asyncio.run(scenario())


def test_recovering_one_component_cannot_clear_another_or_rewrite_latest_failure(
    harness,
):
    async def scenario():
        async with harness() as h:
            await h.runtime._health("telegram", TimeoutError())
            await h.tick(1)
            await h.runtime._health("market_BTCUSDT", ValueError())
            latest = (await h.store.read())["health"]["last_issue"]
            await h.tick(1)
            await h.runtime._health("telegram")
            health = (await h.store.read())["health"]
            assert health["last_issue"] == latest
            assert health["market_BTCUSDT_error"] == "ValueError"
            assert health["issues"]["telegram"]["recovered_at"] == NOW + 2
            await h.tick(901)
            await h.runtime._health("telegram", RuntimeError())
            health = (await h.store.read())["health"]
            assert health["last_issue"]["recovered_at"] is None
            assert health["last_issue"]["error_type"] == "RuntimeError"
            assert len(await events(h.store, "health_issue")) == 3

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"issues": None, "last_issue": []},
        {"issues": {"telegram": "old"}},
        {"issues": {"telegram": {}}, "last_issue": {}},
    ],
)
def test_legacy_cloud_error_is_retained_without_fabricating_failure_or_recovery_time(
    metadata,
):
    health = {"error": "telegram: TelegramError", "telegram_at": NOW - 100, **metadata}
    runtime_module._health_history(health, "scan", None, NOW)
    assert health["last_issue"]["occurred_at"] is None
    assert health["last_issue"]["recovered_at"] is None
    assert health["last_issue"]["legacy"] is True
    assert not [key for key in health if key.endswith("_error")]
    runtime_module._health_history(health, "telegram", None, NOW + 1)
    assert health["last_issue"]["recovered_at"] == NOW + 1
    assert health["error"] == "telegram: TelegramError"


@pytest.mark.parametrize(
    "stamp", [None, False, 0, "bad", [], NOW + 1, float("inf"), 10**1000, NOW - 300]
)
def test_legacy_active_error_only_uses_a_valid_known_failure_time(stamp):
    health = {
        "error": "telegram: TelegramError",
        "telegram_error": "TelegramError",
        "telegram_attempt_at": stamp,
    }
    runtime_module._health_history(health, "telegram", None, NOW)
    assert health["issues"]["telegram"]["occurred_at"] == (
        NOW - 300 if stamp == NOW - 300 else None
    )
    assert health["issues"]["telegram"]["recovered_at"] == NOW
    assert (
        health["telegram_error"] == "TelegramError"
    )  # Metadata itself never makes policy decisions.


@pytest.mark.parametrize("mode", ["complete", "empty", "invalid", "timeout"])
def test_research_alert_matches_validated_result_and_retains_durable_identity(mode):
    async def scenario():
        def response(body):
            raw = research_response(body, **({"facts": []} if mode == "empty" else {}))
            if mode == "invalid":
                raw["output"][0]["action"]["sources"] = []
                raw["output"][1]["content"][0]["annotations"] = []
            return raw

        session = FakeSession(
            [TimeoutError("sensitive-error") if mode == "timeout" else response]
        )
        store = MemoryStore()
        reviewer = AIReviewer(session, store, "offline-key", "offline-model")
        with patch.object(ai.time, "time", return_value=AI_NOW):
            result = await reviewer.research(["BTCUSDT"])
            assert await reviewer.research(["BTCUSDT"]) == result
        alerts = [event for event in store.events if event[1] == "ai_research"]
        assert len(alerts) == len(session.requests) == 1
        key, _, payload, text = alerts[0]
        assert key.endswith(":result") and payload == result
        assert result["advisory_only"] and result["safe_to_trade"] is None
        if mode in {"invalid", "timeout"}:
            assert result["status"] == "UNAVAILABLE"
            assert "UNAVAILABLE" in text and result["reason"] in text
            assert "No news found" not in text and "Validated facts:" not in text
            assert "sensitive-error" not in text
        else:
            assert result["status"] == "COMPLETE" and "COMPLETE" in text
            assert f"Validated facts: {len(result['facts'])}" in text
            if mode == "complete":
                assert SOURCE in text and result["facts"][0]["fact"] in text
        assert "does not establish safety" in text

    asyncio.run(scenario())


def test_research_text_redacts_before_shortening_or_transport_and_does_not_lose_sources():
    fake_secret = "sk-test-redaction-boundary-never-a-real-key"
    fake_token = "123456789:synthetic-not-a-real-telegram-token"
    fact = {
        "symbol": "MACRO",
        "fact": "x" * 217 + fake_secret + " " + "y" * 217 + fake_token,
        "source_url": "https://www.bls.gov/" + fake_secret,
        "published": None,
        "asof": "2026-10-06",
        "uncertainty": "x" * 97 + fake_token,
    }
    result = {
        "status": "COMPLETE",
        "asof": "2026-10-06",
        "facts": [fact] * 8,
        "uncertainty": "x" * 197 + fake_secret,
    }
    before = deepcopy(result)
    text = research_alert(result, secrets=(fake_secret, fake_token))
    assert "sk-" not in text and "123456789:" not in text
    assert text.count("https://www.bls.gov/[redacted]") == 8
    assert "[redacted]" in text and text.endswith("Details: /research")
    assert result == before
    assert "sk-" not in describe_reason(fake_secret)
    assert fake_token not in sanitize_text(fake_token)
    unavailable = research_alert(
        {**result, "status": "UNAVAILABLE", "reason": "Failure " + fake_secret}
    )
    assert "Validated facts" not in unavailable and "https://" not in unavailable
    assert "sk-" not in unavailable


@pytest.mark.parametrize(
    "status,diagnosis",
    [
        (429, "API rate or quota limit"),
        (401, "API credentials were rejected"),
        (403, "API access was denied"),
        (404, "API model or endpoint path is unavailable"),
        (500, "API provider failure"),
        (503, "API provider failure"),
        (400, "API request failed"),
    ],
)
def test_unavailable_research_persists_safe_http_diagnosis_without_inferring_billing(
    status, diagnosis
):
    async def scenario():
        # Provider payloads are retained by existing archival policy, never copied
        # into the safe diagnosis or notification, even if they mention billing.
        session = FakeSession(
            [
                (
                    status,
                    {
                        "error": {
                            "message": "sensitive-provider-message-billing",
                            "code": "untrusted-code",
                            "type": "untrusted-type",
                        }
                    },
                )
            ]
        )
        store = MemoryStore()
        reviewer = AIReviewer(session, store, "offline-key", "offline-model")
        with patch.object(ai.time, "time", return_value=AI_NOW):
            result = await reviewer.research(["BTCUSDT"])
            assert await reviewer.research(["BTCUSDT"]) == result
        assert result["status"] == "UNAVAILABLE" and result["facts"] == []
        assert result["safe_to_trade"] is None
        assert result["http_status"] == status and result["error_code"] == "HTTP_ERROR"
        assert result["diagnostic"] == research_failure(
            {"http_status": status, "error": "HTTP_ERROR"}
        )
        alerts = [e for e in store.events if e[1] == "ai_research"]
        assert len(alerts) == len(session.requests) == 1
        assert alerts[0][2] == result
        text = alerts[0][3]
        assert diagnosis in text and f"HTTP {status}" in text
        assert "No news found" not in text
        assert (
            "billing" not in text
            and "untrusted" not in text
            and "sensitive" not in text
        )

    asyncio.run(scenario())


def test_diagnosis_rejects_untrusted_error_values_and_wrong_status_types():
    assert research_failure_details(
        {"error": "token=secret", "http_status": "429"}
    ) == {
        "error_code": None,
        "http_status": None,
    }
    assert research_failure_details(
        {"error": {}, "http_status": True, "valid": False}
    ) == {
        "error_code": "VALIDATION_FAILED",
        "http_status": None,
    }
    for status in [None, True, "429", [], 1000]:
        text = research_alert(
            {"status": "UNAVAILABLE", "http_status": status, "error_code": {}}
        )
        assert "API rate or quota limit" not in text


@pytest.mark.parametrize(
    "code,diagnosis",
    [
        ("billing_not_active", "API billing is not active"),
        ("insufficient_quota", "API quota is insufficient"),
        ("rate_limit_exceeded", "API rate limit exceeded"),
        ("invalid_api_key", "API credentials were rejected"),
        ("model_not_found", "Requested API model is unavailable"),
    ],
)
def test_allowlisted_provider_code_is_persisted_and_takes_precedence_over_http(
    code, diagnosis
):
    async def scenario():
        store = MemoryStore()
        session = FakeSession(
            [
                (
                    429,
                    {
                        "error": {
                            "code": code,
                            "type": code,
                            "message": "sensitive-provider-message",
                        }
                    },
                )
            ]
        )
        reviewer = AIReviewer(session, store, "offline-key", "offline-model")
        with patch.object(ai.time, "time", return_value=AI_NOW):
            result = await reviewer.research(["BTCUSDT"])
            assert await reviewer.research(["BTCUSDT"]) == result
        attempt = next(iter(store.state["reviews"].values()))["attempts"][0]
        assert attempt["error"] == "HTTP_ERROR" and attempt["error_code"] == code
        assert result["error_code"] == code and result["http_status"] == 429
        assert result["diagnostic"] == f"{diagnosis} (HTTP 429)."
        assert result["status"] == "UNAVAILABLE" and result["facts"] == []
        assert result["safe_to_trade"] is None and len(session.requests) == 1
        assert "sensitive-provider-message" not in json.dumps(result)
        assert research_failure(attempt) == result["diagnostic"]
        assert (
            research_failure({"http_status": 429, "error": "HTTP_ERROR"})
            == "API rate or quota limit (HTTP 429)."
        )

    asyncio.run(scenario())


def test_context_and_queued_notification_redaction_happens_before_truncation(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            secret = h.config.telegram_token
            context = deepcopy((await h.store.read())["context"])
            context.update(risk_state="CAUTION", reason="x" * 1497 + secret + " tail")
            monkeypatch.setattr(
                h.runtime.context, "build", AsyncMock(return_value=context)
            )
            await h.tick(1)  # A new context event key, after the startup context.
            await h.runtime.refresh_context()
            pending = await h.store.pending_notifications()
            text = next(
                item["text"]
                for item in pending
                if item["key"].startswith("context:") and "xxx" in item["text"]
            )
            assert secret[:3] not in text and secret not in text
            await h.store.update(
                lambda tx: tx.event(
                    "synthetic-redaction",
                    "test",
                    {},
                    "Message " + secret,
                )
            )
            await h.runtime.deliver()
            assert secret not in "\n".join(h.runtime.telegram.sent)
            assert "Message [redacted]" in h.runtime.telegram.sent

    asyncio.run(scenario())


def test_arm_decisions_survive_other_arm_and_scan_without_changing_legacy_decision(
    harness,
):
    async def scenario():
        async with harness() as h:
            await h.runtime.scan()
            baseline = deepcopy(
                ready(await h.store.read())["per_arm_decisions"]["baseline_shadow"]
            )
            await reviews(h.runtime)
            rec = ready(await h.store.read())
            assert rec["per_arm_decisions"]["baseline_shadow"] == baseline
            assert rec["per_arm_decisions"]["ai_shadow"]["decision"] == rec["decision"]
            ai_review = deepcopy(rec["per_arm_decisions"]["ai_shadow"])
            old_baseline_review = deepcopy(baseline["last_risk_review"])
            await h.runtime.scan()
            rec = ready(await h.store.read())
            assert rec["per_arm_decisions"]["ai_shadow"] == ai_review
            assert (
                rec["per_arm_decisions"]["baseline_shadow"]["last_risk_review"]
                == old_baseline_review
            )
            assert len((await h.store.read())["trades"]) == 2
            assert not h.market.submissions
            await h.restart()
            assert (
                ready(await h.store.read())["per_arm_decisions"]
                == rec["per_arm_decisions"]
            )

    asyncio.run(scenario())


def test_testnet_observations_keep_ai_simulation_and_exchange_assessments_separate(
    harness,
):
    async def scenario():
        async with harness("testnet") as h:
            await h.runtime.scan()
            await reviews(h.runtime)
            rec = ready(await h.store.read())
            assert set(rec["per_arm_decisions"]) == {
                "baseline_shadow",
                "ai_shadow",
                "testnet",
            }
            for arm in rec["per_arm_decisions"]:
                assert rec["per_arm_decisions"][arm]["last_risk_review"]["arm"] == arm
            assert rec["decision"] == rec["per_arm_decisions"]["testnet"]["decision"]
            assert len(h.market.submissions) == 1  # Offline fake exchange only.

    asyncio.run(scenario())
