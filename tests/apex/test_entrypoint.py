"""Deployment liveness ordering without sockets, credentials or external services."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from apex_bot import __main__ as entry
from apex_bot import runtime as runtime_module
from apex_bot.config import Config
from apex_bot.storage import Store


def install(monkeypatch):
    state = SimpleNamespace(
        app=None, runtime=None, reading=asyncio.Event(), allow=asyncio.Event()
    )

    class Runner:
        def __init__(self, app, **kwargs):
            state.app = app

        async def setup(self):
            pass

        async def cleanup(self):
            pass

    class Site:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            pass

    class FakeRuntime:
        def __init__(self, config, store, session):
            self.stopping = asyncio.Event()
            self.store = store
            self.telegram = None
            self.worker_tasks = []
            self.started = False
            self.last_loop = {}
            self.cache = SimpleNamespace(close=self.close)
            state.runtime = self

        async def close(self):
            pass

        async def refresh_instruments(self):
            state.reading.set()
            await state.allow.wait()

        async def initialize(self):
            self.started = True
            self.last_loop["scan"] = entry.time.time()

        async def run(self):
            await self.stopping.wait()

    monkeypatch.setattr(entry.web, "AppRunner", Runner)
    monkeypatch.setattr(entry.web, "TCPSite", Site)
    monkeypatch.setattr(runtime_module, "Runtime", FakeRuntime)
    asyncio.get_running_loop().add_signal_handler = lambda *args: None
    return state


async def response(state, path):
    handler = next(
        r.handler for r in state.app.router.routes() if r.resource.canonical == path
    )
    return await handler(None)


def test_liveness_waits_for_public_preflight_and_readiness_waits_for_leadership(
    tmp_path, monkeypatch
):
    async def scenario():
        state = install(monkeypatch)
        task = asyncio.create_task(
            entry.serve(Config(sqlite_path=str(tmp_path / "entry.sqlite")))
        )
        try:
            await asyncio.wait_for(state.reading.wait(), 2)
            assert (await response(state, "/healthz")).status == 503
            assert (await response(state, "/readyz")).status == 503
            state.allow.set()
            for _ in range(100):
                if state.runtime.started:
                    break
                await asyncio.sleep(0.01)
            assert state.runtime.started
            assert (await response(state, "/healthz")).status == 200
            assert json.loads((await response(state, "/readyz")).body) == {
                "ready": True
            }
        finally:
            if state.runtime:
                state.runtime.stopping.set()
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


def test_incompatible_execution_ledger_never_passes_deployment_health(
    tmp_path, monkeypatch
):
    async def scenario():
        path = str(tmp_path / "existing.sqlite")
        store = Store(sqlite_path=path)
        await store.initialize()
        assert await store.lease()
        await store.update(
            lambda tx: tx.state.update(
                execution_venue="live", orders={"old": {"status": "CLOSED"}}
            )
        )
        await store.close()
        state = install(monkeypatch)
        with pytest.raises(RuntimeError, match="another mode"):
            await entry.serve(Config(sqlite_path=path, mode="shadow"))
        assert not state.reading.is_set()
        assert (await response(state, "/healthz")).status == 503

    asyncio.run(scenario())
