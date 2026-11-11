"""Desk-only deployment: Telegram is the Hyperliquid desk and nothing else."""

import asyncio
import types
from types import SimpleNamespace

from apex_bot.config import Config
from apex_bot.desk import AUTO_LEVERAGE, Desk, desk_state, normalize_settings, short_id
from apex_bot.desk_panel import DeskPanel, plain
from apex_bot.runtime import Runtime
from apex_bot.storage import Store
from apex_bot.telegram import TelegramController

from .test_desk import NOW, make_desk, outbox, signal
from .test_telegram import CHAT, TOKEN, USER, FakeResponse, FakeService, FakeSession, callback, message


async def exclusive_controller(tmp_path, desk=None):
    store = Store(sqlite_path=str(tmp_path / "t.sqlite"))
    await store.initialize()
    await store.lease()
    session = FakeSession()
    ctl = TelegramController(session, TOKEN, CHAT, {USER}, FakeService())
    ctl.WRITE_INTERVAL = 0
    ctl.attach_desk(DeskPanel(store, desk or Desk(store, None, engine_books=False)), exclusive=True)
    await ctl.initialize()
    return store, session, ctl


def test_exclusive_bot_refreshes_its_identity_and_menus(tmp_path):
    async def run():
        store, session, ctl = await exclusive_controller(tmp_path)
        scopes = [p["scope"]["type"] for p in session.payloads("setMyCommands")]
        assert scopes == ["chat", "default", "all_private_chats"]
        assert session.payloads("setMyName")[0] == {"name": "Apex Desk"}
        assert "Shadow Book" in session.payloads("setMyDescription")[0]["description"]
        assert session.payloads("setMyShortDescription")
        await store.close()

    asyncio.run(run())


def test_profile_failures_never_block_startup(tmp_path):
    async def run():
        store = Store(sqlite_path=str(tmp_path / "t.sqlite"))
        await store.initialize()
        await store.lease()
        session = FakeSession()
        session.responses["setMyName"].append(
            FakeResponse({"ok": False, "error_code": 429, "description": "Too Many Requests", "parameters": {"retry_after": 1}}, status=429)
        )
        ctl = TelegramController(session, TOKEN, CHAT, {USER}, FakeService())
        ctl.WRITE_INTERVAL = 0
        ctl.attach_desk(DeskPanel(store, Desk(store, None, engine_books=False)), exclusive=True)
        await ctl.initialize()  # does not raise
        await store.close()

    asyncio.run(run())


def test_old_engine_commands_and_buttons_open_the_desk(tmp_path):
    async def run():
        store, session, ctl = await exclusive_controller(tmp_path)
        await ctl._handle_update({"message": message("/opportunities")})
        payload = session.text_payloads()[-1]
        text = plain(payload["text"])
        assert "not a desk command" in text and "APEX DESK" in text
        assert payload["parse_mode"] == "HTML"
        await ctl._handle_update({"callback_query": callback("pg:dashboard:0")})
        text = plain(session.text_payloads()[-1]["text"])
        assert "old bot" in text and "APEX DESK" in text
        # No engine button anywhere on the desk-only home screen.
        keyboard = str(session.text_payloads()[-1]["reply_markup"])
        assert "pg:" not in keyboard and "Signal engine" not in keyboard
        await ctl._handle_update({"message": message("/status")})
        assert "System status" in plain(session.text_payloads()[-1]["text"])
        await store.close()

    asyncio.run(run())


def test_unparseable_html_falls_back_to_plain_text(tmp_path):
    async def run():
        store, session, ctl = await exclusive_controller(tmp_path)
        session.responses["sendMessage"].append(
            FakeResponse(
                {"ok": False, "error_code": 400, "description": "Bad Request: can't parse entities: unclosed tag"},
                status=400,
            )
        )
        await ctl.send("<b>bold &amp; broken", html=True)
        sent = session.payloads("sendMessage")
        assert sent[-2]["parse_mode"] == "HTML"
        assert "parse_mode" not in sent[-1] and sent[-1]["text"] == "bold & broken"
        await store.close()

    asyncio.run(run())


def test_desk_only_ignores_engine_books_and_forces_the_shadow_book(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)  # stored book = baseline_shadow
        desk.engine_books = False
        await desk.tick()
        assert desk_state(await store.read())["trades"] == {}  # engine signals are not traded
        assert (await desk.settings())["book"] == "shadow_book"
        await store.close()

    asyncio.run(run())


def test_old_10x_default_migrates_to_auto_once(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path, settings={"max_leverage": 10, "risk_usd": 60})
        await desk.tick()
        stored = desk_state(await store.read())["settings"]
        assert stored["max_leverage"] == AUTO_LEVERAGE and stored["version"] == 2 and stored["risk_usd"] == 60
        # A later explicit 10x choice sticks.
        panel = DeskPanel(store, desk, clock=clock)
        await panel.command("lev", ["10"], "telegram:1:2")
        await desk.tick()
        assert normalize_settings(desk_state(await store.read())["settings"])["max_leverage"] == 10
        text, _ = await panel.command("lev", ["auto"], "telegram:1:2")
        assert "Auto (safe max)" in plain(text)
        await store.close()

    asyncio.run(run())


def test_skip_messages_can_be_switched_off(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(
            tmp_path, shadow_trades=[signal(symbol="NOPEUSDT")], settings={"notify_skips": False}
        )
        await desk.tick()
        assert desk_state(await store.read())["trades"]["baseline_shadow:op1"]["status"] == "SKIPPED"
        assert not any("not taken" in x for x in await outbox(store))
        await store.close()

    asyncio.run(run())


def test_breakeven_button_only_for_winners_and_tightens_the_stop(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        hl.fill_entry("SOL")
        await desk.tick()
        panel = DeskPanel(store, desk, clock=clock)
        actor, fresh = "telegram:1:2", {"date": int(NOW)}
        sid = short_id("baseline_shadow:op1")
        text, keyboard = await panel.callback("d:pos", actor, fresh)
        assert f"d:be:{sid}" not in str(keyboard)  # price at entry: no break-even button
        hl.mid["SOL"] = 104.0
        await desk.tick()
        text, keyboard = await panel.callback("d:pos", actor, fresh)
        assert f"d:be:{sid}" in str(keyboard)
        text, keyboard = await panel.callback(f"d:be:{sid}", actor, fresh)
        assert "stop moved to entry" in plain(text)
        t = desk_state(await store.read())["trades"]["baseline_shadow:op1"]
        assert t["stop"] == 100.0
        stops = [o for o in hl.orders.values() if o["kind"] == "sl"]
        assert len(stops) == 1 and stops[0]["trigger_px"] == 100.0
        # A losing trade is refused instead of being closed by a stop above price.
        hl.mid["SOL"] = 99.0
        assert "not in profit" in await desk.breakeven("baseline_shadow:op1") or "already" in await desk.breakeven(
            "baseline_shadow:op1"
        )
        await store.close()

    asyncio.run(run())


def test_status_view_and_colon_coins_in_buttons(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path, settings={"symbol_leverage": {"xyz:GOLD": 3}, "blocked": ["xyz:GOLD"]})
        await desk.tick()
        panel = DeskPanel(store, desk, clock=clock)
        actor, fresh = "telegram:1:2", {"date": int(NOW)}
        text, _ = await panel.callback("d:st", actor, fresh)
        text = plain(text)
        assert "System status" in text and "Hyperliquid ✅ live" in text
        await panel.callback("d:lc:xyz:GOLD", actor, fresh)
        await panel.callback("d:ub:xyz:GOLD", actor, fresh)
        s = normalize_settings(desk_state(await store.read())["settings"])
        assert s["symbol_leverage"] == {} and s["blocked"] == []
        text, _ = await panel.callback("d:ns:0", actor, fresh)
        assert "Signals not taken (e.g. price already past target): off" in plain(text)
        await store.close()

    asyncio.run(run())


def test_every_view_is_valid_telegram_html(tmp_path):
    """Escaping: views must contain only the tags Telegram accepts, balanced."""
    import re

    async def run():
        store, desk, hl, clock = await make_desk(tmp_path, settings={"blocked": ["A<B&C"]})
        await desk.tick()
        hl.fill_entry("SOL")
        await desk.tick()
        panel = DeskPanel(store, desk, clock=clock)
        for view in ("home", "pos", "risk", "lev", "flt", "alr", "perf", "help", "sig", "st"):
            text, keyboard = await panel.render(view)
            tags = re.findall(r"</?([a-z]+)>", text)
            assert set(tags) <= {"b", "i"}, (view, tags)
            for name in ("b", "i"):
                assert text.count(f"<{name}>") == text.count(f"</{name}>"), view
            stripped = re.sub(r"</?(b|i)>", "", text)
            assert "<" not in stripped, (view, stripped)
            for row in keyboard["inline_keyboard"]:
                for button in row:
                    assert len(button["callback_data"].encode()) <= 64
        await store.close()

    asyncio.run(run())


# ----------------------------------------------------------- runtime pieces


class PinTelegram:
    def __init__(self):
        self.sent, self.edits, self.pins = [], [], []
        self.edit_ok = True

    async def send(self, text, keyboard=None, *, html=False):
        self.sent.append((text, keyboard, html))
        return str(500 + len(self.sent))

    async def edit(self, message_id, text, keyboard, *, html=True):
        self.edits.append((message_id, text))
        return self.edit_ok

    async def pin(self, message_id, *, previous=None):
        self.pins.append((message_id, previous))


def fake_runtime(store, desk, telegram):
    rt = SimpleNamespace(
        store=store,
        desk=desk,
        panel=DeskPanel(store, desk),
        telegram=telegram,
        desk_only=True,
        service=SimpleNamespace(_safe=lambda text: text),
    )
    rt._save_pin = types.MethodType(Runtime._save_pin, rt)
    return rt


def test_pinned_dashboard_posts_once_per_boot_then_edits_in_place(tmp_path, monkeypatch):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        tg = PinTelegram()
        rt = fake_runtime(store, desk, tg)
        refresh = types.MethodType(Runtime.refresh_dashboard, rt)
        await refresh()
        assert len(tg.sent) == 1 and tg.sent[0][2] is True and tg.pins == [(501, None)]
        assert "APEX DESK" in plain(tg.sent[0][0])
        await refresh()
        assert len(tg.sent) == 1 and tg.edits == [(501, tg.edits[0][1])]
        # The owner opened Positions on the pinned message: no refresh fights it.
        panel = DeskPanel(store, desk)
        await panel.callback("d:pos", "telegram:1:2", {"message_id": 501, "date": int(NOW)})
        await refresh()
        assert len(tg.edits) == 1
        # Message deleted by the user -> a new one is posted and pinned.
        pin = desk_state(await store.read())["pin"]
        pin["touched"] = 0
        await rt._save_pin(pin)
        tg.edit_ok = False
        await refresh()
        assert len(tg.sent) == 2 and tg.pins[-1] == (502, 501)
        await store.close()

    asyncio.run(run())


def test_desk_only_outbox_drops_engine_messages_and_sends_desk_html(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)

        def queue(tx):
            tx.event("boot:old", "startup", {}, "🌊 Apex started · signal engine SHADOW")
            tx.event("opp:1", "opportunity", {}, "WAIT BTCUSDT")
            tx.event("desk:x:skip", "desk", {}, "⏭ <b>Signal not taken</b>")
            tx.event("issue:desk:1", "health_issue", {}, "⚠️ desk unavailable (HLError).")

        await store.update(queue)
        tg = PinTelegram()
        rt = fake_runtime(store, desk, tg)
        await types.MethodType(Runtime.deliver, rt)()
        sent = [(text, html) for text, keyboard, html in tg.sent]
        assert sent == [("⏭ <b>Signal not taken</b>", True), ("⚠️ desk unavailable (HLError).", False)]
        assert await store.pending_notifications() == []
        await store.close()

    asyncio.run(run())


def test_config_desk_only_switch():
    assert Config.from_env({}).desk_only == "auto"
    assert Config.from_env({"APEX_DESK_ONLY": "OFF"}).desk_only == "off"
    try:
        Config.from_env({"APEX_DESK_ONLY": "maybe"})
    except Exception as exc:  # ConfigError
        assert "APEX_DESK_ONLY" in str(exc)
    else:
        raise AssertionError("invalid value accepted")


# ------------------------------------------------- Hyperliquid blips and alerts


def test_reads_retry_server_errors_but_not_bad_requests(monkeypatch):
    from hyperliquid.utils.error import ClientError, ServerError

    from apex_bot import hl_client
    from apex_bot.hl_client import HLError, HyperliquidClient

    monkeypatch.setattr(hl_client, "_backoff", lambda exc, attempt: 0)
    client = HyperliquidClient("0x" + "ab" * 20, "0x" + "cd" * 32)
    client._exchange = object()  # skip the network build

    async def run():
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ServerError(502, "bad gateway")
            return {"ok": True}

        assert await client._read(flaky) == {"ok": True} and len(calls) == 3

        def down():
            raise ServerError(502, "bad gateway")

        try:
            await client._read(down)
        except HLError as exc:
            assert str(exc) == "Hyperliquid request failed (ServerError 502)"
        else:
            raise AssertionError("expected HLError")

        bad = []

        def rejected():
            bad.append(1)
            raise ClientError(400, None, "bad request", {})

        try:
            await client._read(rejected)
        except HLError:
            pass
        assert len(bad) == 1  # a 400 is not retried

        writes = []

        def order():
            writes.append(1)
            raise ClientError(429, None, "rate limited", {})

        try:
            await client._call(order)  # exchange actions are never retried
        except HLError:
            pass
        assert len(writes) == 1

    asyncio.run(run())


def test_desk_only_pages_after_two_minutes_of_failure_and_reports_recovery(tmp_path, monkeypatch):
    from apex_bot import runtime as runtime_module
    from apex_bot.hl_client import HLError

    from .test_runtime import events

    async def run():
        store = Store(sqlite_path=str(tmp_path / "h.sqlite"))
        await store.initialize()
        await store.lease()
        import time as real_time

        now = [real_time.time()]
        monkeypatch.setattr(runtime_module.time, "time", lambda: now[0])
        await store.lease()  # renew under the patched clock
        rt = SimpleNamespace(
            store=store,
            desk_only=True,
            last_loop={},
            _failing_since={},
            _alerted=set(),
            cache=SimpleNamespace(status="ok"),
        )
        health = types.MethodType(Runtime._health, rt)
        await health("desk", HLError("Hyperliquid request failed (ClientError 429)"))
        now[0] += 4
        await health("desk")  # a single blip: recovered without paging
        assert await store.pending_notifications() == []
        for _ in range(35):  # ~140 s of continuous failure
            now[0] += 4
            await health("desk", HLError("Hyperliquid request failed (ServerError 502)"))
            await store.lease()
        texts = [n["text"] for n in await store.pending_notifications()]
        assert len(texts) == 1 and "failing for 2 min (Hyperliquid request failed (ServerError 502))" in texts[0]
        now[0] += 4
        await health("desk")
        texts = [n["text"] for n in await store.pending_notifications()]
        assert len(texts) == 2 and texts[1].startswith("✅ desk is back to normal")
        # Every failure is still recorded for diagnostics, just not sent.
        assert len(await events(store, "health_issue")) >= 2
        await store.close()

    asyncio.run(run())


def test_a_warning_sent_before_a_restart_still_gets_its_all_clear(tmp_path, monkeypatch):
    from apex_bot import runtime as runtime_module
    from apex_bot.hl_client import HLError
    from apex_bot.runtime import Runtime

    async def run():
        store = Store(sqlite_path=str(tmp_path / "h2.sqlite"))
        await store.initialize()
        import time as real_time

        now = [real_time.time()]
        monkeypatch.setattr(runtime_module.time, "time", lambda: now[0])
        await store.lease()

        def process():
            rt = SimpleNamespace(store=store, desk_only=True, last_loop={}, _failing_since={}, cache=SimpleNamespace(status="ok"))
            return types.MethodType(Runtime._health, rt)

        old = process()
        for _ in range(35):
            now[0] += 4
            await old("desk", HLError("Hyperliquid request failed (ClientError 429)"))
            await store.lease()
        assert len(await store.pending_notifications()) == 1  # paged once
        new = process()  # restart mid-outage: fresh memory, same saved state
        for _ in range(35):  # still failing after the restart: no second page
            now[0] += 4
            await new("desk", HLError("Hyperliquid request failed (ClientError 429)"))
            await store.lease()
        assert len(await store.pending_notifications()) == 1
        now[0] += 4
        await new("desk")
        texts = [n["text"] for n in await store.pending_notifications()]
        assert len(texts) == 2 and texts[1].startswith("✅ desk is back to normal (it was failing for about 5 min)")
        await store.close()

    asyncio.run(run())


def test_status_shows_the_last_hyperliquid_hiccup(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()

        async def broken():
            from apex_bot.hl_client import HLError

            raise HLError("Hyperliquid request failed (ClientError 429)")

        hl.open_orders = broken
        try:
            await desk.tick()
        except Exception:
            pass
        panel = DeskPanel(store, desk, clock=clock)
        text, _ = await panel.render("st")
        assert "Last Hyperliquid hiccup: Hyperliquid request failed (ClientError 429)" in plain(text)
        assert "1 since restart" in plain(text)
        await store.close()

    asyncio.run(run())
