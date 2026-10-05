import asyncio
import time

import pytest

from apex_bot.models import Candle
from apex_bot.storage import NotLeader, Store


def test_restart_settings_outbox_and_single_worker(tmp_path):
    async def run():
        path = str(tmp_path / "state.sqlite")
        first = Store(sqlite_path=path)
        second = Store(sqlite_path=path)
        await first.initialize()
        await second.initialize()
        assert await first.lease()
        assert not await second.lease()

        def save(tx):
            tx.state["settings"].update(profile="ultra_cautious", paused=True)
            tx.event("once", "test", {"x": 1}, "A persistent notification")

        await first.update(save)
        await first.update(save)
        assert len(await first.pending_notifications()) == 1
        with pytest.raises(NotLeader):
            await second.update(save)
        await first.close()
        assert await second.lease()
        assert (await second.read())["settings"]["paused"] is True
        notification = (await second.pending_notifications())[0]
        await second.notification_result(notification["key"], retry_after=60)
        assert await second.pending_notifications() == []
        assert (await second.outbox_health())["pending"] == 1
        await second.notification_result("once", message_id="123")
        assert (await second.outbox_health())["pending"] == 0
        await second.close()

    asyncio.run(run())


def test_transaction_rollback_does_not_emit_or_modify():
    async def run():
        store = Store()
        await store.initialize()
        await store.lease()

        def fail(tx):
            tx.state["settings"]["paused"] = True
            tx.event("never", "test", {}, "must not send")
            raise RuntimeError("rollback")

        with pytest.raises(RuntimeError):
            await store.update(fail)
        assert (await store.read())["settings"]["paused"] is False
        assert await store.pending_notifications() == []
        await store.close()

    asyncio.run(run())


def test_concurrent_reservations_are_serialized():
    async def run():
        store = Store()
        await store.initialize()
        await store.lease()

        def increment(tx):
            tx.state["counter"] = tx.state.get("counter", 0) + 1

        await asyncio.gather(*(store.update(increment) for _ in range(40)))
        assert (await store.read())["counter"] == 40
        await store.close()

    asyncio.run(run())


def test_fixed_candle_origin_and_changed_history_rejection(tmp_path):
    async def run():
        store = Store(sqlite_path=str(tmp_path / "candles.sqlite"))
        await store.initialize()
        await store.lease()
        bars = [Candle(i * 86400000, 100, 102, 99, 101, 10) for i in range(5)]
        assert await store.candle_history("XUSDT", "D", bars[:4]) == bars[:4]
        assert await store.candle_history("XUSDT", "D", bars[2:]) == bars
        changed = Candle(2 * 86400000, 100, 103, 99, 102, 10)
        with pytest.raises(ValueError):
            await store.candle_history("XUSDT", "D", [changed])
        assert await store.candle_history("XUSDT", "D", []) == bars
        await store.close()

    asyncio.run(run())


def test_expired_lease_cannot_write():
    async def run():
        store = Store()
        await store.initialize()
        await store.lease(ttl=-1)
        with pytest.raises(NotLeader):
            await store.update(lambda tx: tx.state.update(x=1))
        await store.close()

    asyncio.run(run())
