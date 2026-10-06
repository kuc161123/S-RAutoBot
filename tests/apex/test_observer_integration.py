"""Runtime/Store/risk/observer/Telegram integration without external services."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from apex_bot import runtime as runtime_module
from apex_bot.models import Candle, Opportunity
from apex_bot.shadow_observer import ShadowObserver
from apex_bot.storage import NotLeader
from apex_bot.telegram import TelegramController

from .test_runtime import NOW, harness, ready, reviews
from .test_shadow_observer import records
from .test_telegram import FakeSession, callback, message


async def reserve_capacity(h, arms=("baseline_shadow",), count=6):
    """Start with a real engine candidate and reserve independent portfolio slots."""
    await h.runtime.scan()
    state = await h.store.read()
    op = Opportunity(**ready(state)["opportunity"])
    template = deepcopy(next(iter(state["trades"].values())))
    reservations = {}
    for arm in arms:
        for i in range(count):
            identity = f"{arm}:existing-{i}"
            reservations[identity] = dict(
                deepcopy(template),
                id=identity,
                opportunity_id=f"existing-{i}",
                symbol=f"OTHER{i}USDT",
                bucket=f"independent-{i}",
                arm=arm,
                risk_cash=1,
                notional=100,
            )
    await h.store.update(lambda tx: tx.state.update(trades=reservations))
    return op


def test_runtime_initializes_shared_observer_and_empty_service_view(harness):
    async def scenario():
        async with harness() as h:
            observer = h.runtime.observer
            assert isinstance(observer, ShadowObserver)
            assert observer.store is h.store
            assert observer.client is h.runtime.client
            assert observer.cache is h.runtime.cache
            assert h.runtime.service.observer is observer
            before = await h.store.read()
            summary = await observer.summary()  # Table created by runtime.initialize.
            assert summary["total"] == summary["complete_closed"] == 0
            assert summary["wr"] is None and summary["mean_r"] is None
            view = await h.runtime.service.snapshot("observer")
            assert "CAPACITY OBSERVER" in view and "SHADOW ONLY" in view
            assert "Win rate: N/A" in view and "Mean net outcome: N/A" in view
            assert "Win rate: 0.0%" not in view
            assert "not portfolio returns" in view
            assert await h.store.read() == before

    asyncio.run(scenario())


def test_baseline_capacity_capture_no_ai_duplicate_and_restart_preserves_record(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            op = await reserve_capacity(h, ("baseline_shadow", "ai_shadow"))
            before = await h.store.read()
            spy = AsyncMock(wraps=h.runtime.observer.consider)
            monkeypatch.setattr(h.runtime.observer, "consider", spy)
            await h.runtime.consider(op, None)
            captured = await records(h.store)
            assert len(captured) == 1
            assert captured[0]["payload"]["reasons"] == ["POSITION_CAP"]
            assert captured[0]["payload"]["independent_assessment"]["allowed"] is True
            assert captured[0]["payload"]["opportunity"]["id"] == op.id
            assert captured[0]["trade"]["status"] == "PENDING"
            assert "POSITION_CAP" in ready(await h.store.read())["decision"]
            spy.assert_awaited_once()

            # Actual blind AI review must not create another observer record.
            await reviews(h.runtime)
            state = await h.store.read()
            assert ready(state)["ai"]["verdict"] == "APPROVE"
            assert "POSITION_CAP" in ready(state)["decision"]
            spy.assert_awaited_once()
            assert await records(h.store) == captured
            assert state["trades"] == before["trades"]
            assert state["orders"] == before["orders"]
            assert h.market.submissions == []
            assert len(h.session.requests) == 2  # Observer adds no model calls.

            await h.runtime.consider(op, None)
            assert await records(h.store) == captured
            await h.restart()
            assert await records(h.store) == captured
            assert h.runtime.service.observer is h.runtime.observer
            assert (await h.runtime.observer.summary())["total"] == 1

    asyncio.run(scenario())


def test_allowed_baseline_and_ai_only_capacity_rejection_are_not_observed(harness):
    async def scenario():
        async with harness() as h:
            op = await reserve_capacity(h, ("ai_shadow",))
            await h.runtime.consider(op, None)
            state = await h.store.read()
            assert f"baseline_shadow:{op.id}" in state["trades"]
            assert (await h.runtime.observer.summary())["total"] == 0
            await reviews(h.runtime)
            state = await h.store.read()
            assert ready(state)["ai"]["verdict"] == "APPROVE"
            assert ready(state)["last_risk_review"]["arm"] == "ai_shadow"
            assert "POSITION_CAP" in ready(state)["decision"]
            assert (await h.runtime.observer.summary())["total"] == 0
            assert f"baseline_shadow:{op.id}" in state["trades"]
            assert f"ai_shadow:{op.id}" not in state["trades"]
            assert state["orders"] == {} and h.market.submissions == []

    asyncio.run(scenario())


@pytest.mark.parametrize("gate", ["spread", "funding", "reward_risk", "loss_halt"])
def test_noncapacity_rejection_stays_excluded_even_with_full_portfolio(
    harness, monkeypatch, gate
):
    async def scenario():
        async with harness() as h:
            op = await reserve_capacity(h)
            before = await h.store.read()
            if gate == "spread":
                h.market.ticker.update(ask1Price="150", bid1Price="135")
            elif gate == "funding":
                h.market.ticker["fundingRate"] = ".01"
            elif gate == "reward_risk":
                op = replace(op, target1=op.entry + 0.01)
                await h.store.update(
                    lambda tx: tx.state["opportunities"][op.id].update(
                        opportunity=op.to_dict()
                    )
                )
            else:
                monkeypatch.setattr(
                    h.runtime,
                    "_risk_snapshot",
                    AsyncMock(
                        return_value=dict(
                            daily_loss_pct=2, weekly_loss_pct=0, drawdown_pct=0
                        )
                    ),
                )
            await h.runtime.consider(op, None)
            state = await h.store.read()
            assessment = ready(state)["last_risk_review"]["assessment"]
            assert not assessment["allowed"]
            reason = {
                "spread": "SPREAD_TOO_WIDE",
                "funding": "ADVERSE_FUNDING",
                "reward_risk": "RR_TARGET1",
                "loss_halt": "DAILY_LOSS_HALT",
            }[gate]
            assert reason in assessment["reasons"]
            assert (await h.runtime.observer.summary())["total"] == 0
            assert state["trades"] == before["trades"]
            assert state["orders"] == before["orders"]
            assert h.market.submissions == []

    asyncio.run(scenario())


@pytest.mark.parametrize("health_write_fails", [False, True])
def test_observer_exception_preserves_rejection_and_no_order(
    harness, monkeypatch, caplog, health_write_fails
):
    async def scenario():
        async with harness() as h:
            op = await reserve_capacity(h)
            before = await h.store.read()
            monkeypatch.setattr(
                h.runtime.observer,
                "consider",
                AsyncMock(side_effect=RuntimeError("private-observer-error")),
            )
            if health_write_fails:
                monkeypatch.setattr(
                    h.runtime, "_health", AsyncMock(side_effect=OSError())
                )
            await h.runtime.consider(op, None)
            state = await h.store.read()
            assert "POSITION_CAP" in ready(state)["decision"]
            assert ready(state)["last_risk_review"]["assessment"]["allowed"] is False
            if not health_write_fails:
                assert state["health"]["shadow_observer_error"] == "RuntimeError"
            assert state["trades"] == before["trades"]
            assert state["orders"] == before["orders"]
            assert (await h.runtime.observer.summary())["total"] == 0
            assert h.market.submissions == []
            assert "private-observer-error" not in caplog.text

    asyncio.run(scenario())


@pytest.mark.parametrize("exception", [NotLeader, asyncio.CancelledError])
def test_observer_lost_authority_and_cancellation_propagate(
    harness, monkeypatch, exception
):
    async def scenario():
        async with harness() as h:
            op = await reserve_capacity(h)
            before = await h.store.read()
            monkeypatch.setattr(
                h.runtime.observer, "consider", AsyncMock(side_effect=exception())
            )
            with pytest.raises(exception):
                await h.runtime.consider(op, None)
            after = await h.store.read()
            assert after["trades"] == before["trades"]
            assert after["orders"] == before["orders"]
            assert h.market.submissions == []

    asyncio.run(scenario())


def test_runtime_observer_resolves_loss_without_touching_standard_ledgers(harness):
    async def scenario():
        async with harness() as h:
            op = await reserve_capacity(h)
            await h.runtime.consider(op, None)
            row = (await records(h.store))[0]
            before = await h.store.read()
            price, stop = row["trade"]["limit"], row["trade"]["stop"]
            h.market.intraday = [
                Candle(int(NOW * 1000), price, price + 1, price - 1, price),
                Candle(int((NOW + 180) * 1000), price, price + 1, stop - 1, stop),
            ]
            await h.tick(360)
            await h.runtime.observe_shadows()
            summary = await h.runtime.observer.summary()
            assert (
                summary["total"] == summary["closed"] == summary["complete_closed"] == 1
            )
            assert summary["wins"] == 0 and summary["losses"] == 1
            assert summary["wr"] == 0 and summary["net_r"] < -1
            assert "Win rate: 0.0%" in await h.runtime.service.snapshot("observer")
            after = await h.store.read()
            assert after["trades"] == before["trades"]
            assert after["orders"] == before["orders"]
            assert h.market.submissions == []
            persisted = await records(h.store)
            await h.runtime.observe_shadows()
            assert await records(h.store) == persisted

    asyncio.run(scenario())


def test_real_telegram_observer_menu_command_button_help_and_authorization(harness):
    async def scenario():
        async with harness() as h:
            session = FakeSession()
            chat, user = int(h.config.telegram_chat_id), next(
                iter(h.config.telegram_user_ids)
            )
            control = TelegramController(
                session, "123456:OFFLINE", str(chat), {user}, h.runtime.service
            )
            control.WRITE_INTERVAL = 0
            before = await h.store.read()
            await control.initialize()
            commands = session.payloads("setMyCommands")[0]["commands"]
            assert sum(c["command"] == "observer" for c in commands) == 1
            await control._handle_update(
                {"message": message("/help", chat=chat, user=user)}
            )
            assert "/observer" in session.text_payloads()[-1]["text"]
            await control._handle_update(
                {"message": message("/dashboard", chat=chat, user=user)}
            )
            home_buttons = [
                b
                for row in session.text_payloads()[-1]["reply_markup"][
                    "inline_keyboard"
                ]
                for b in row
            ]
            assert any(b["callback_data"] == "pg:performance:0" for b in home_buttons)
            await control._handle_update(
                {"callback_query": callback("pg:performance:0", chat=chat, user=user)}
            )
            buttons = [
                b
                for row in session.text_payloads()[-1]["reply_markup"][
                    "inline_keyboard"
                ]
                for b in row
            ]
            assert any(
                b["callback_data"] == "pg:observer:0" and "Observer" in b["text"]
                for b in buttons
            )
            await control._handle_update(
                {"message": message("/observer", chat=chat, user=user)}
            )
            expected = await h.runtime.service.snapshot("observer")
            assert session.text_payloads()[-1]["text"] == expected
            assert "Win rate: N/A" in expected
            await control._handle_update(
                {"callback_query": callback("v:observer", chat=chat, user=user)}
            )
            assert session.payloads("answerCallbackQuery")
            assert session.payloads("editMessageText")[-1]["text"] == expected
            count = len(session.text_payloads())
            await control._handle_update(
                {"message": message("/observer", chat=chat, user=user + 1)}
            )
            await control._handle_update(
                {"callback_query": callback("v:observer", chat=chat, user=user + 1)}
            )
            assert len(session.text_payloads()) == count
            assert await h.store.read() == before
            assert (await h.runtime.observer.summary())["total"] == 0

    asyncio.run(scenario())


def test_observer_loop_is_supervised_retries_and_does_not_stop_other_workers(
    harness, monkeypatch
):
    async def scenario():
        async with harness() as h:
            recovered = asyncio.Event()
            calls, loops = [], []
            original_loop, original_health = h.runtime._loop, h.runtime._health

            async def resolve(now):
                calls.append(now)
                if len(calls) == 1:
                    raise TimeoutError("offline observer retry")
                return {"processed": 0, "symbols": 0}

            async def health(name, error=None):
                await original_health(name, error)
                if name == "shadow_observer" and error is None:
                    recovered.set()

            async def loop(name, callback_fn, seconds):
                loops.append((name, seconds))
                await original_loop(
                    name, callback_fn, 0 if name == "shadow_observer" else seconds
                )

            async def idle():
                await asyncio.Event().wait()

            with monkeypatch.context() as patch:
                for name in (
                    "renew",
                    "refresh_context",
                    "refresh_risk",
                    "refresh_instruments",
                    "refresh_universe",
                    "scan",
                    "simulate",
                    "funding",
                    "reconcile",
                    "poll",
                    "deliver",
                    "research",
                    "review_worker",
                ):
                    patch.setattr(h.runtime, name, idle)
                patch.setattr(h.runtime.references, "refresh", idle)
                patch.setattr(h.runtime.observer, "resolve", resolve)
                patch.setattr(h.runtime, "_health", health)
                patch.setattr(h.runtime, "_loop", loop)
                patch.setattr(h.runtime.cache, "close", AsyncMock())
                patch.setattr(h.store, "close", AsyncMock())
                task = asyncio.create_task(h.runtime.run())
                try:
                    await asyncio.wait_for(recovered.wait(), 4)
                    assert loops.count(("shadow_observer", 30)) == 1
                    assert len(calls) >= 2 and all(t == h.clock.now for t in calls)
                    assert (
                        "shadow_observer_error" not in (await h.store.read())["health"]
                    )
                    assert not task.done() and not h.runtime.stopping.is_set()
                finally:
                    h.runtime.stopping.set()
                    await asyncio.wait_for(task, 2)
                assert all(t.done() for t in h.runtime.worker_tasks)

    asyncio.run(scenario())
