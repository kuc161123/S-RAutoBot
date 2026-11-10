import asyncio
import itertools

import pytest

from apex_bot.desk import (
    Desk,
    Skip,
    desk_state,
    liquidation_safe_leverage,
    normalize_settings,
    plan_entry,
    short_id,
)
from apex_bot.desk_panel import DeskPanel, plain, signal_card
from apex_bot.hl_client import (
    HLError,
    cloid_for,
    coin_candidates,
    map_coin,
    parse_statuses,
    round_price,
    round_size,
)
from apex_bot.storage import Store

NOW = 1_800_000_000.0
ASSETS = {
    "SOL": {"sz_decimals": 2, "max_leverage": 20, "only_isolated": False, "delisted": False},
    "kPEPE": {"sz_decimals": 0, "max_leverage": 10, "only_isolated": False, "delisted": False},
    "BTC": {"sz_decimals": 5, "max_leverage": 40, "only_isolated": False, "delisted": False},
    "OLD": {"sz_decimals": 1, "max_leverage": 3, "only_isolated": False, "delisted": True},
}


# ----------------------------------------------------------------- precision


def test_price_and_size_rounding_follow_hyperliquid_rules():
    assert round_price(123.456789, 2) == 123.46  # 5 significant figures
    assert round_price(0.0123456789, 0) == 0.012346  # <= 6 decimals for szDecimals 0
    assert round_price(0.0123456789, 2) == 0.0123  # <= 4 decimals for szDecimals 2
    assert round_price(65432.7, 5) == 65433.0  # integers always allowed
    assert round_price(1.23456, 1, "down") == 1.2345
    assert round_size(1.239, 2) == 1.23  # always rounds down
    assert round_size(-1, 2) == 0.0


def test_symbol_mapping_and_client_ids():
    assert coin_candidates("SOLUSDT") == ["SOL"]
    assert coin_candidates("1000PEPEUSDT") == ["kPEPE"]
    assert coin_candidates("SHIB1000USDT") == ["kSHIB"]
    assert coin_candidates("10000SATSUSDT") == []
    assert map_coin("1000PEPEUSDT", ASSETS) == "kPEPE"
    assert map_coin("OLDUSDT", ASSETS) is None  # delisted
    assert map_coin("XYZUSDT", ASSETS) is None
    cid = cloid_for("baseline_shadow:abc", "entry")
    assert cid.startswith("0x") and len(cid) == 34
    assert cid == cloid_for("baseline_shadow:abc", "entry")
    assert cid != cloid_for("baseline_shadow:abc", "sl1")


def test_parse_statuses():
    response = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {
                "statuses": [
                    {"resting": {"oid": 7}},
                    {"filled": {"totalSz": "1.5", "avgPx": "100.1", "oid": 8}},
                    {"error": "Order must have minimum value of $10."},
                    "waitingForFill",
                ]
            },
        },
    }
    out = parse_statuses(response)
    assert [o["state"] for o in out] == ["resting", "filled", "error", "waiting"]
    assert out[0]["oid"] == 7 and out[1]["filled"] == 1.5 and not out[2]["ok"]
    with pytest.raises(HLError):
        parse_statuses({"status": "err", "response": "bad"})


# ------------------------------------------------------------------- sizing


def signal(**overrides):
    base = {
        "id": "baseline_shadow:op1",
        "arm": "baseline_shadow",
        "symbol": "SOLUSDT",
        "side": "Buy",
        "setup": "2X",
        "status": "PENDING",
        "entry_style": "confirmed",
        "limit": 100.0,
        "stop": 95.0,
        "target1": 110.0,
        "target2": 125.0,
        "created_at": NOW - 30,
        "expires_at": NOW + 3600,
        "tp1_done": False,
    }
    base.update(overrides)
    return base


def test_plan_sizes_to_fixed_dollar_risk_including_fees():
    s = normalize_settings({})
    plan = plan_entry(signal(), s, ASSETS["SOL"], "SOL", 5000, 5000, 100.2)
    # $50 / (5 + 0.00045 * 195) = 9.82..., rounded down to 0.01
    assert plan["size"] == 9.82
    assert 49 < plan["risk_usd"] <= 50
    # Stop 5% away on a 20x market: liquidation-safe cap is 10x = the default max.
    assert plan["leverage"] == 10
    assert plan["tp1_size"] == 4.91  # half of 982 lots, odd rounds up
    assert plan["margin"] == pytest.approx(plan["notional"] / 10)


def test_percent_risk_and_symbol_leverage_and_liquidation_cap():
    s = normalize_settings({"risk_mode": "pct", "risk_pct": 2, "symbol_leverage": {"SOL": 20}})
    plan = plan_entry(signal(), s, ASSETS["SOL"], "SOL", 10000, 10000, 100.0)
    assert 199 < plan["risk_usd"] <= 200
    # Stop 5% away on a 20x market: 1 / (1.5 * 0.05 + 0.025) = 10x max
    assert liquidation_safe_leverage(0.05, 20) == 10
    assert plan["leverage"] == 10 and plan["leverage_capped_by"] == "liquidation safety"
    s = normalize_settings({"max_leverage": 50})
    plan = plan_entry(signal(stop=99.0, target1=101.0, target2=103.0), s, ASSETS["kPEPE"] | {"sz_decimals": 2}, "kPEPE", 5000, 5000, 100.0)
    assert plan["leverage"] == 10 and plan["leverage_capped_by"] == "exchange maximum"


@pytest.mark.parametrize(
    "mid, kwargs, reason",
    [
        (110.0, {}, "more than 5%"),
        (94.0, {"stop": 95.0}, "more than 5%"),
        (100.0, {"limit": 100.0, "stop": 50.0}, "minimum"),
        (None, {}, "No Hyperliquid price"),
        (100.0, {"target1": 99.0}, "valid order"),
    ],
)
def test_plan_skips_with_reasons(mid, kwargs, reason):
    s = normalize_settings({"risk_usd": 1})
    with pytest.raises(Skip) as info:
        plan_entry(signal(**kwargs), s, ASSETS["SOL"], "SOL", 5000, 5000, mid)
    assert reason in info.value.reason


def test_margin_skip_is_retryable():
    s = normalize_settings({"max_leverage": 1})
    with pytest.raises(Skip) as info:
        plan_entry(signal(), s, ASSETS["SOL"], "SOL", 100, 100, 100.0)
    assert info.value.retry and "free margin" in info.value.reason


def test_settings_are_clamped_and_defaulted():
    s = normalize_settings({"risk_usd": -5, "max_leverage": 999, "direction": "sideways", "watchlist_alerts": True})
    assert s["risk_usd"] == 1.0 and s["max_leverage"] == 50 and s["direction"] == "both"
    assert s["watchlist_alerts"] == "on"


# ------------------------------------------------------- simulated exchange


class FakeHL:
    address = "0x" + "ab" * 20

    def __init__(self):
        self.mid = {"SOL": 100.0, "BTC": 60000.0}
        self.positions = {}
        self.orders = {}
        self.fills = []
        self.leverage = {}
        self.placed = []
        self.cancelled = []
        self.oids = itertools.count(100)
        self.reject_next = False
        self.equity = 5000.0

    async def assets(self):
        return dict(ASSETS)

    async def account(self):
        return {
            "equity": self.equity,
            "available": self.equity * 0.9,
            "margin_used": 0.0,
            "positions": {c: dict(p) for c, p in self.positions.items()},
        }

    async def open_orders(self):
        return [dict(o) for o in self.orders.values()]

    async def mids(self):
        return dict(self.mid)

    async def fills_since(self, start_ms):
        return list(self.fills)

    async def set_leverage(self, coin, leverage, isolated=True):
        self.leverage[coin] = (leverage, isolated)

    async def place(self, orders):
        out = []
        for o in orders:
            self.placed.append(o)
            if self.reject_next:
                self.reject_next = False
                out.append({"ok": False, "state": "error", "oid": None, "filled": 0.0, "avg_px": None, "error": "rejected"})
                continue
            if o["kind"] == "limit" and o.get("tif") == "Ioc":
                pos = self.positions.get(o["coin"])
                if pos:
                    pos["size"] += o["size"] if o["is_buy"] else -o["size"]
                    if abs(pos["size"]) < 1e-9:
                        del self.positions[o["coin"]]
                out.append({"ok": True, "state": "filled", "oid": next(self.oids), "filled": o["size"], "avg_px": self.mid[o["coin"]], "error": None})
                continue
            oid = next(self.oids)
            self.orders[oid] = {
                "coin": o["coin"],
                "oid": oid,
                "cloid": o.get("cloid"),
                "side": "B" if o["is_buy"] else "A",
                "size": o["size"],
                "orig_size": o["size"],
                "limit_px": o["limit_px"],
                "trigger_px": o.get("trigger_px"),
                "is_trigger": o["kind"] in ("sl", "tp"),
                "reduce_only": o.get("reduce_only", False),
                "order_type": o["kind"],
                "kind": o["kind"],
            }
            out.append({"ok": True, "state": "resting", "oid": oid, "filled": 0.0, "avg_px": None, "error": None})
        return out

    async def cancel(self, coin, oid):
        self.cancelled.append(oid)
        return self.orders.pop(oid, None) is not None

    async def order_status(self, cloid):
        for o in self.orders.values():
            if o.get("cloid") == cloid:
                return {"status": "order", "order": {"order": {"oid": o["oid"]}, "status": "open"}}
        return {"status": "unknownOid"}

    # -- helpers to simulate the market
    def fill_entry(self, coin, size=None, px=None):
        entry = next(o for o in self.orders.values() if o["coin"] == coin and o["kind"] == "limit")
        size = entry["size"] if size is None else size
        sign = 1 if entry["side"] == "B" else -1
        self.positions[coin] = {"size": sign * size, "entry_px": px or entry["limit_px"], "liq_px": 50.0, "upnl": 0.0}
        entry["size"] -= size
        if entry["size"] <= 1e-12:
            del self.orders[entry["oid"]]

    def trigger(self, oid, pnl=0.0):
        order = self.orders.pop(oid)
        pos = self.positions[order["coin"]]
        reduce = min(order["size"], abs(pos["size"]))
        pos["size"] -= reduce if pos["size"] > 0 else -reduce
        self.fills.append({"coin": order["coin"], "oid": oid, "px": order["trigger_px"], "size": reduce, "side": order["side"], "time": 0, "closed_pnl": pnl, "fee": 0.1, "dir": "Close Long"})
        if abs(pos["size"]) < 1e-9:
            del self.positions[order["coin"]]

    def kinds(self, coin):
        return sorted(o["kind"] for o in self.orders.values() if o["coin"] == coin)


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


async def make_desk(tmp_path, shadow_trades=None, settings=None):
    store = Store(sqlite_path=str(tmp_path / "desk.sqlite"))
    await store.initialize()
    assert await store.lease()
    clock = Clock()
    hl = FakeHL()

    def seed(tx):
        for t in shadow_trades or [signal()]:
            tx.state["trades"][t["id"]] = t
        desk_state(tx.state)["settings"] = {"book": "baseline_shadow", **(settings or {})}

    await store.update(seed)
    desk = Desk(store, hl, clock=clock)
    return store, desk, hl, clock


async def outbox(store):
    return [plain(n["text"]) for n in await store.pending_notifications(limit=100)]


async def trade(store, tid="baseline_shadow:op1"):
    return desk_state(await store.read())["trades"][tid]


def test_full_lifecycle_entry_fill_tp1_breakeven_trail_and_stop(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        t = await trade(store)
        assert t["status"] == "ORDERED" and hl.leverage["SOL"] == (10, True)
        assert hl.kinds("SOL") == ["limit"]
        texts = await outbox(store)
        assert any("Desk online" in x for x in texts)
        assert any("LONG SOL · Apex signal" in x and "Risk $" in x for x in texts)

        await desk.tick()  # idempotent: no second entry
        assert sum(1 for o in hl.placed if o["kind"] == "limit") == 1

        hl.fill_entry("SOL")
        await desk.tick()
        t = await trade(store)
        assert t["status"] == "OPEN"
        assert hl.kinds("SOL") == ["sl", "tp", "tp"]
        sl = hl.orders[t["sl_oid"]]
        assert sl["reduce_only"] and sl["trigger_px"] == 95.0 and sl["size"] == t["size"]
        assert hl.orders[t["tp1_oid"]]["size"] == t["tp1_size"]
        assert any("FILLED" in x for x in await outbox(store))

        hl.trigger(t["tp1_oid"], pnl=49.0)
        hl.mid["SOL"] = 110.5
        await desk.tick()
        t = await trade(store)
        assert t["tp1_done"] and t["stop"] == 100.0
        assert hl.orders[t["sl_oid"]]["trigger_px"] == 100.0
        assert any("STOP MOVED" in x for x in await outbox(store))

        def ratchet(tx):
            tx.state["trades"]["baseline_shadow:op1"]["stop"] = 104.0

        await store.update(ratchet)
        hl.mid["SOL"] = 112.0
        await desk.tick()
        t = await trade(store)
        assert t["stop"] == 104.0 and hl.orders[t["sl_oid"]]["trigger_px"] == 104.0
        assert len([o for o in hl.orders.values() if o["kind"] == "sl"]) == 1

        hl.trigger(t["sl_oid"], pnl=19.0)
        await desk.tick()
        t = await trade(store)
        assert t["status"] == "CLOSED" and t["pnl"] == pytest.approx(49 + 19 - 0.2)
        assert t["exit"] == "Trailing stop hit" or "Stop" in t["exit"]
        assert hl.orders == {}  # leftover target cancelled
        assert any("CLOSED" in x and "Result: +$" in x for x in await outbox(store))
        await store.close()

    asyncio.run(run())


def test_partial_fill_is_protected_and_expiry_cancels_remainder(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        hl.fill_entry("SOL", size=3.0)
        await desk.tick()
        t = await trade(store)
        assert t["status"] == "ORDERED"
        assert hl.orders[t["sl_oid"]]["size"] == 3.0  # stop covers the partial fill
        clock.now = t["expires_at"] + 1
        await desk.tick()
        t = await trade(store)
        assert t["status"] == "OPEN" and t["filled"] == 3.0
        assert "limit" not in hl.kinds("SOL")
        await store.close()

    asyncio.run(run())


def test_unfilled_entry_cancelled_when_signal_expires(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        clock.now = NOW + 4000
        await desk.tick()
        t = await trade(store)
        assert t["status"] == "CANCELLED" and hl.orders == {}
        assert any("ENTRY CANCELLED" in x for x in await outbox(store))
        await store.close()

    asyncio.run(run())


def test_missing_stop_is_replaced(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        hl.fill_entry("SOL")
        await desk.tick()
        t = await trade(store)
        hl.orders.pop(t["sl_oid"])  # someone cancelled it on the website
        await desk.tick()
        t2 = await trade(store)
        assert t2["sl_oid"] != t["sl_oid"] and hl.orders[t2["sl_oid"]]["kind"] == "sl"
        await store.close()

    asyncio.run(run())


def test_structural_engine_exit_closes_position(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        hl.fill_entry("SOL")
        await desk.tick()

        def close(tx):
            tx.state["trades"]["baseline_shadow:op1"].update(status="CLOSED", exit_reason="DAILY_INVALIDATION")

        await store.update(close)
        await desk.tick()  # market close sent
        await desk.tick()  # position gone -> finalized
        t = await trade(store)
        assert t["status"] == "CLOSED" and t["exit"].startswith("Engine exit")
        await store.close()

    asyncio.run(run())


def test_skips_are_recorded_once_and_filters_apply(tmp_path):
    async def run():
        trades = [
            signal(id="baseline_shadow:a", symbol="XYZUSDT"),
            signal(id="baseline_shadow:b", side="Sell", limit=100.0, stop=105.0, target1=90.0, target2=80.0),
            signal(id="ai_shadow:c", arm="ai_shadow"),
        ]
        store, desk, hl, clock = await make_desk(tmp_path, trades, {"direction": "long"})
        await desk.tick()
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        assert trades["baseline_shadow:a"]["status"] == "SKIPPED"
        assert "not listed" in trades["baseline_shadow:a"]["reason"]
        assert trades["baseline_shadow:b"]["reason"] == "Shorts are switched off"
        assert "ai_shadow:c" not in trades  # other book is not followed
        texts = await outbox(store)
        assert sum("not taken" in x for x in texts) == 2
        await store.close()

    asyncio.run(run())


def test_max_positions_defers_and_pause_blocks_new_entries(tmp_path):
    async def run():
        trades = [signal(id="baseline_shadow:a"), signal(id="baseline_shadow:b", symbol="BTCUSDT", limit=60000.0, stop=59000.0, target1=62000.0, target2=65000.0)]
        store, desk, hl, clock = await make_desk(tmp_path, trades, {"max_positions": 1})
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        assert trades["baseline_shadow:a"]["status"] == "ORDERED"
        assert "baseline_shadow:b" not in trades  # deferred, may still be taken later
        texts = await outbox(store)
        assert any("Max open positions" in x for x in texts)
        await desk.tick()
        assert sum("Max open positions" in x for x in await outbox(store)) == 1

        def pause(tx):
            desk_state(tx.state)["settings"] = {"paused": True, "max_positions": 5, "book": "baseline_shadow"}

        await store.update(pause)
        await desk.tick()
        assert "baseline_shadow:b" not in desk_state(await store.read())["trades"]
        await store.close()

    asyncio.run(run())


def test_rejected_entry_is_recorded_as_error(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        hl.reject_next = True
        await desk.tick()
        t = await trade(store)
        assert t["status"] == "ERROR" and hl.orders == {}
        await store.close()

    asyncio.run(run())


def test_no_keys_reports_not_connected(tmp_path):
    async def run():
        store = Store(sqlite_path=str(tmp_path / "x.sqlite"))
        await store.initialize()
        await store.lease()
        desk = Desk(store, None)
        await desk.tick()
        assert desk.snapshot["status"] == "no_keys"
        panel = DeskPanel(store, desk)
        text, keyboard = await panel.home()
        text = plain(text)
        assert "not connected" in text and "HL_ACCOUNT_ADDRESS" in text
        await store.close()

    asyncio.run(run())


# --------------------------------------------------------------------- panel


def test_panel_commands_change_settings(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        panel = DeskPanel(store, desk, clock=clock)
        actor = "telegram:1:2"
        text, _ = await panel.command("risk", ["75"], actor)
        text = plain(text)
        assert "$75.00/trade" in text
        text, _ = await panel.command("risk", ["1.5%"], actor)
        text = plain(text)
        assert "1.5%" in text
        text, _ = await panel.command("risk", ["500%"], actor)
        text = plain(text)
        assert text.startswith("⚠️")
        text, _ = await panel.command("lev", ["10"], actor)
        text = plain(text)
        assert "Max for all symbols: 10x" in text
        text, _ = await panel.command("lev", ["sol", "8"], actor)
        text = plain(text)
        assert "SOL 8x" in text
        text, _ = await panel.command("lev", ["1000pepe", "3"], actor)
        text = plain(text)
        assert "kPEPE 3x" in text
        text, _ = await panel.command("lev", ["SOL", "auto"], actor)
        text = plain(text)
        assert "SOL 8x" not in text
        await panel.command("maxpos", ["8"], actor)
        await panel.command("dailystop", ["off"], actor)
        await panel.command("direction", ["short"], actor)
        await panel.command("block", ["sol"], actor)
        s = normalize_settings(desk_state(await store.read())["settings"])
        assert s["risk_mode"] == "pct" and s["risk_pct"] == 1.5
        assert s["max_leverage"] == 10 and s["symbol_leverage"] == {"kPEPE": 3}
        assert s["max_positions"] == 8 and s["daily_loss_usd"] == 0 and s["direction"] == "short"
        assert s["blocked"] == ["SOL"]
        assert await panel.command("opportunities", [], actor) is None  # engine view falls through
        await store.close()

    asyncio.run(run())


def test_panel_buttons_and_close_confirmation(tmp_path):
    async def run():
        store, desk, hl, clock = await make_desk(tmp_path)
        await desk.tick()
        hl.fill_entry("SOL")
        await desk.tick()
        panel = DeskPanel(store, desk, clock=clock)
        actor = "telegram:1:2"
        fresh = {"date": int(NOW)}
        text, keyboard = await panel.callback("d:home", actor, fresh)
        text = plain(text)
        assert "APEX DESK" in text and "1 open" in text
        text, _ = await panel.callback("d:ru:100", actor, fresh)
        text = plain(text)
        assert "$100.00/trade" in text
        text, _ = await panel.callback("d:ml:20", actor, {"date": int(NOW) - 3600})
        text = plain(text)
        assert "old" in text  # stale settings buttons do nothing
        text, keyboard = await panel.callback("d:pos", actor, fresh)
        text = plain(text)
        sid = short_id("baseline_shadow:op1")
        assert f"d:cl:{sid}" in str(keyboard)
        text, keyboard = await panel.callback(f"d:cl:{sid}", actor, fresh)
        text = plain(text)
        confirm = keyboard["inline_keyboard"][0][0]["callback_data"]
        assert confirm.startswith("d:clok:")
        text, _ = await panel.callback(confirm, "telegram:1:999", fresh)
        text = plain(text)
        assert "expired" in text  # another user cannot confirm
        text, keyboard = await panel.callback(f"d:cl:{sid}", actor, fresh)
        text = plain(text)
        confirm = keyboard["inline_keyboard"][0][0]["callback_data"]
        text, _ = await panel.callback(confirm, actor, fresh)
        text = plain(text)
        assert "Closing SOL" in text
        assert "SOL" not in hl.positions
        await desk.tick()
        assert (await trade(store))["status"] == "CLOSED"
        await store.close()

    asyncio.run(run())


def test_signal_card_reads_professionally():
    t = {
        "id": "baseline_shadow:x",
        "coin": "SOL",
        "side": "Buy",
        "is_buy": True,
        "entry": 100.0,
        "stop": 95.0,
        "target1": 110.0,
        "target2": 125.0,
        "size": 9.82,
        "tp1_size": 4.91,
        "notional": 982.0,
        "leverage": 5,
        "leverage_note": None,
        "margin": 196.4,
        "risk_usd": 49.98,
        "expires_at": NOW,
        "setup": "2X",
    }
    html = signal_card(t)
    assert html.splitlines()[0].startswith("🟢 <b>LONG SOL</b> · Apex signal #")
    card = plain(html)
    for needle in ("Entry 100 ·", "Stop 95 (−5.00%)", "Target 1 110 (+10.00%)", "Target 2 125", "2.0R · 5.0R", "Risk $49.98"):
        assert needle in card


# ------------------------------------------------- real SDK wire compatibility


def test_every_desk_order_encodes_and_signs_with_the_real_sdk(tmp_path):
    """Prices/sizes the desk sends must pass the SDK's exact float_to_wire checks."""
    import eth_account
    from hyperliquid.utils.signing import order_request_to_order_wire, order_wires_to_order_action, sign_l1_action

    from apex_bot.hl_client import HyperliquidClient

    async def run():
        trades = [
            signal(),
            signal(id="baseline_shadow:pepe", symbol="1000PEPEUSDT", limit=0.0123457, stop=0.0117771, target1=0.0136123, target2=0.0151234),
            signal(id="baseline_shadow:btc", symbol="BTCUSDT", side="Sell", limit=60123.4, stop=61987.7, target1=57111.1, target2=54000.9),
        ]
        store, desk, hl, clock = await make_desk(tmp_path, trades, {"max_leverage": 50, "risk_usd": 37.5})
        hl.mid.update({"kPEPE": 0.0123, "BTC": 60100.0})
        await desk.tick()
        for coin in ("SOL", "kPEPE", "BTC"):
            hl.fill_entry(coin)
        await desk.tick()
        hl.trigger(next(o["oid"] for o in hl.orders.values() if o["coin"] == "SOL" and o["kind"] == "tp" and o["size"] < 9), pnl=1)
        hl.mid["SOL"] = 110.4
        await desk.tick()
        await store.close()
        return hl.placed

    placed = asyncio.run(run())
    assert {o["kind"] for o in placed} == {"limit", "sl", "tp"}
    client = HyperliquidClient("0x" + "ab" * 20, "0x" + "11" * 32)
    wallet = eth_account.Account.from_key("0x" + "11" * 32)
    for order in placed:
        wire = order_request_to_order_wire(client._wire(order), 3)
        action = order_wires_to_order_action([wire])
        signature = sign_l1_action(wallet, action, None, 1_700_000_000_000, None, True)
        assert set(signature) == {"r", "s", "v"}


def test_config_requires_valid_hyperliquid_pair():
    from apex_bot.config import Config, ConfigError

    base = {"APEX_MODE": "shadow"}
    assert Config.from_env(base).hl_account_address == ""
    good = dict(base, HL_ACCOUNT_ADDRESS="0x" + "ab" * 20, HL_API_PRIVATE_KEY="11" * 32)
    cfg = Config.from_env(good)
    assert cfg.hl_account_address and "11" * 32 not in repr(cfg)
    for broken in (
        dict(base, HL_ACCOUNT_ADDRESS="0x" + "ab" * 20),
        dict(good, HL_ACCOUNT_ADDRESS="0x123"),
        dict(good, HL_API_PRIVATE_KEY="nothex"),
        dict(good, HL_API_URL="https://evil.example"),
    ):
        with pytest.raises(ConfigError):
            Config.from_env(broken)


def test_unified_account_counts_spot_usdc_as_free_margin():
    from apex_bot.hl_client import HyperliquidClient

    class Info:
        def user_state(self, address, dex=""):
            if dex:
                return {"marginSummary": {"accountValue": "0", "totalMarginUsed": "0"}, "withdrawable": "0", "assetPositions": []}
            return {
                "marginSummary": {"accountValue": "712.2", "totalMarginUsed": "566.1"},
                "withdrawable": "31.15",
                "assetPositions": [{"position": {"coin": "XRP", "szi": "6444", "entryPx": "1.0", "unrealizedPnl": "12.5", "leverage": {"type": "cross", "value": 12}}}],
            }

        def spot_user_state(self, address):
            return {"balances": [{"coin": "USDC", "total": "1170.83", "hold": "679.33"}]}

        def query_user_abstraction_state(self, address):
            return "unifiedAccount"

    client = HyperliquidClient("0x" + "ab" * 20, "0x" + "11" * 32)
    client._info, client._exchange, client.dexs = Info(), object(), ("", "xyz")
    account = asyncio.run(client.account())
    assert account["mode"] == "unifiedAccount"
    assert account["available"] == pytest.approx(491.5)
    assert account["equity"] == pytest.approx(1183.33)
    assert "XRP" in account["positions"]
