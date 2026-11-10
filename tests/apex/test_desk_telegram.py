"""The desk panel becomes the Telegram home screen when attached."""

import asyncio

from apex_bot.desk import Desk
from apex_bot.desk_panel import DESK_COMMANDS, DeskPanel
from apex_bot.storage import Store
from apex_bot.telegram import TelegramController

from .test_telegram import CHAT, TOKEN, USER, FakeService, FakeSession, callback, message


async def controller(tmp_path):
    store = Store(sqlite_path=str(tmp_path / "t.sqlite"))
    await store.initialize()
    await store.lease()
    session = FakeSession()
    service = FakeService()
    ctl = TelegramController(session, TOKEN, CHAT, {USER}, service)
    ctl.WRITE_INTERVAL = 0
    ctl.attach_desk(DeskPanel(store, Desk(store, None)))
    await ctl.initialize()
    return store, session, service, ctl


def test_start_dashboard_and_commands_route_to_desk(tmp_path):
    async def run():
        store, session, service, ctl = await controller(tmp_path)
        published = session.payloads("setMyCommands")[0]["commands"]
        assert {c["command"] for c in published} == set(DESK_COMMANDS)
        await ctl._handle_update({"message": message("/start")})
        text = session.text_payloads()[-1]["text"]
        assert text.startswith("🏦 APEX DESK") and "not connected" in text
        await ctl._handle_update({"message": message("/risk 80")})
        assert "$80.00/trade" in session.text_payloads()[-1]["text"]
        # Old engine views still work and their Home button returns to the desk.
        await ctl._handle_update({"message": message("/status")})
        payload = session.text_payloads()[-1]
        assert "status" in payload["text"]
        homes = [b["callback_data"] for row in payload["reply_markup"]["inline_keyboard"] for b in row if "Home" in b["text"]]
        assert homes == ["d:home"]
        await store.close()

    asyncio.run(run())


def test_desk_buttons_render_and_stale_commands_are_refused(tmp_path):
    async def run():
        store, session, service, ctl = await controller(tmp_path)
        await ctl._handle_update({"callback_query": callback("d:risk")})
        assert "Risk settings" in session.text_payloads()[-1]["text"]
        await ctl._handle_update({"message": message("/pause", age=600)})
        assert "stale" in session.text_payloads()[-1]["text"]
        state = await store.read()
        assert not state.get("desk", {}).get("settings", {}).get("paused")
        await ctl._handle_update({"message": message("/pause")})
        state = await store.read()
        assert state["desk"]["settings"]["paused"] is True
        await ctl._handle_update({"message": message("/pause", user=999)})  # not authorized
        assert service.proposals == []
        await store.close()

    asyncio.run(run())
