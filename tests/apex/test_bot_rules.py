"""BotRules: the Shadow Book tab that says which Side + Setup letters the bot may trade."""

import asyncio
import csv
import io
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from apex_bot.desk import RULES_DOWN_NOTICE, desk_state
from apex_bot.desk_panel import DeskPanel, plain
from apex_bot.shadow_book import (
    RULES_GID,
    ShadowBookSource,
    SourceError,
    parse_rows,
    parse_rules,
    rule_for,
    rules_links,
    setup_letter,
)

from .test_desk import ASSETS, outbox
from .test_shadow_book import HEADER, START, make, row, rule

for _coin in ("xyz:SP500", "xyz:XYZ100", "xyz:SILVER"):
    ASSETS[_coin] = {"sz_decimals": 3, "max_leverage": 20, "only_isolated": True, "delisted": False, "dex": "xyz"}

RULES_HEADER = [
    "Rule ID", "Active", "Side", "Setup", "Setup name", "Closed trades (n)", "Win rate %",
    "Expectancy R", "Since UTC", "Reason / evidence", "Rules version",
]


def rules_row(rid, active, side, setup, name, version="2026-11-17-1"):
    return [rid, active, side, setup, name, "10", "60", "0.3", "2026-11-17 14:15", "evidence", version]


# The tab as Google's export CSV returns it on 17 Nov 2026 (blank rows kept).
REAL_RULES = [
    RULES_HEADER,
    rules_row("R-SHORT-C", "TRUE", "Short", "C", "Breakdown-retest sell"),
    rules_row("R-SHORT-B", "TRUE", "Short", "B", "Wave-(iv)/4 bounce sell"),
    rules_row("R-SHORT-A", "FALSE", "Short", "A", "Wave-(ii)/B bounce sell"),
    rules_row("R-SHORT-D", "FALSE", "Short", "D", "Corrective-completion sell"),
    rules_row("R-LONG-A", "FALSE", "Long", "A", "Wave-(ii) pullback buy"),
    rules_row("R-LONG-B", "FALSE", "Long", "B", "Wave-(iv)/2 pullback buy"),
    rules_row("R-LONG-C", "FALSE", "Long", "C", "Breakout-retest buy"),
    rules_row("R-LONG-D", "FALSE", "Long", "D", "Corrective-completion buy"),
    [""] * 11,
    [""] * 11,
    ["HOW THIS TAB WORKS (rows 1-10 are the rule table)"] + [""] * 10,
    ["1. The Shadow Desk paper-trades every setup it believes in."] + [""] * 10,
    [""] * 11,
    ["CHANGE LOG", "Rules version", "Change", "Reason"] + [""] * 7,
    ["2026-11-17 14:15 UTC", "2026-11-17-1", "Created. Active: Short C, Short B.", "Lual"] + [""] * 7,
]


def test_parse_rules_reads_the_rule_table_only():
    out = parse_rules(REAL_RULES)
    assert out["version"] == "2026-11-17-1"
    assert len(out["rules"]) == 8  # stops at the first blank row; notes and change log never read
    active = [(r["side"], r["setup"]) for r in out["rules"] if r["active"]]
    assert active == [("Short", "C"), ("Short", "B")]
    assert out["rules"][1] == {
        "id": "R-SHORT-B", "active": True, "side": "Short", "setup": "B",
        "name": "Wave-(iv)/4 bounce sell", "version": "2026-11-17-1",
    }
    # Google's by-name CSV drops blank rows and blanks off-type cells: the notes row
    # (no Active/Side/Setup) still ends the table and "Rules version" in the change
    # log can't leak into the rules.
    gviz = [r for r in REAL_RULES if any(c for c in r)]
    gviz[-2] = ["CHANGE LOG", "", "Change", "Reason"] + [""] * 7
    assert parse_rules(gviz) == out
    # Lower-case and spaced cells, extra columns, any order.
    shuffled = [["Side", "Notes", "Setup", "Active"], [" short ", "x", "b — wave iv", "true"], ["", "", "", ""]]
    assert parse_rules(shuffled)["rules"][0]["side"] == "Short"
    assert parse_rules(shuffled)["rules"][0]["setup"] == "B" and parse_rules(shuffled)["rules"][0]["active"]
    assert parse_rules([RULES_HEADER, [""] * 11]) == {"rules": [], "version": ""}
    with pytest.raises(SourceError, match="Active"):
        parse_rules([["Rule ID", "Side", "Setup"], ["R-1", "Short", "C"]])
    with pytest.raises(SourceError):
        parse_rules([])


def test_rule_matching_uses_side_and_setup_letter():
    rules = parse_rules(REAL_RULES)["rules"]
    assert rule_for(rules, "Sell", "B")["id"] == "R-SHORT-B"
    assert rule_for(rules, "Short", "c")["id"] == "R-SHORT-C"
    assert rule_for(rules, "Sell", "A") is None  # listed but not active
    assert rule_for(rules, "Buy", "C") is None  # Long C is inactive
    assert rule_for(rules, "Sell", "") is None and rule_for([], "Sell", "B") is None
    assert setup_letter("B — wave-(iv) bounce sell") == "B"
    assert setup_letter("  c — breakdown retest") == "C" and setup_letter("") == ""


def test_rules_links_follow_the_trades_links():
    live = "https://docs.google.com/spreadsheets/d/abc/export?format=csv&gid=1"
    pub = "https://docs.google.com/spreadsheets/d/e/2PACX-abc/pub?gid=1&single=true&output=csv"
    links = rules_links([live, pub])
    assert links == [
        f"https://docs.google.com/spreadsheets/d/abc/export?format=csv&gid={RULES_GID}",
        f"https://docs.google.com/spreadsheets/d/e/2PACX-abc/pub?single=true&output=csv&gid={RULES_GID}",
        "https://docs.google.com/spreadsheets/d/abc/gviz/tq?tqx=out%3Acsv&headers=1&sheet=BotRules",
    ]
    assert all(link.startswith("https://docs.google.com/") for link in links)
    assert rules_links([pub], gid="77")[0].endswith("gid=77")


def _csv(rows) -> bytes:
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    return out.getvalue().encode()


class Response:
    def __init__(self, status, body):
        self.url = SimpleNamespace(host="docs.google.com")
        self.status = status

        async def chunks(size):
            yield body

        self.content = SimpleNamespace(iter_chunked=chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_source_reads_bot_rules_with_every_trades_read():
    trades = _csv([HEADER, row("S-1", "SOL", "Short", 100, 105, 95, 90, layer="SIGNAL")])
    tabs = {"trades": (200, trades), "rules": (200, _csv(REAL_RULES))}
    calls = []

    class Session:
        def get(self, url, **kwargs):
            tab = "rules" if f"gid={RULES_GID}" in url or "/gviz/" in url else "trades"
            calls.append(tab)
            return Response(*tabs[tab])

    live = "https://docs.google.com/spreadsheets/d/abc/export?format=csv&gid=1"
    source = ShadowBookSource(Session(), csv_url=live, clock=lambda: START)
    assert asyncio.run(source.refresh(force=True))
    assert calls == ["trades", "rules"]
    assert source.rules_ok and source.rules_version == "2026-11-17-1" and source.rules_link_used == "live sheet"
    assert [(r["side"], r["setup"]) for r in source.rules if r["active"]] == [("Short", "C"), ("Short", "B")]

    # BotRules unreadable: Trades still read (open trades stay managed), rules dropped.
    tabs["rules"] = (500, b"")
    assert asyncio.run(source.refresh(force=True))
    assert source.ok and not source.rules_ok and source.rules == []
    assert "HTTP 500" in source.rules_error

    # The by-name fallback is used when the tab id stops working (tab recreated).
    class ByName(Session):
        def get(self, url, **kwargs):
            if f"gid={RULES_GID}" in url:
                return Response(400, b"")
            return Response(*tabs["rules" if "/gviz/" in url else "trades"])

    tabs["rules"] = (200, _csv(REAL_RULES))
    source = ShadowBookSource(ByName(), csv_url=live, clock=lambda: START)
    assert asyncio.run(source.refresh(force=True)) and source.rules_ok and source.rules_link_used == "by tab name"

    # A tab with headers but no rule rows counts as unreadable: nothing new is placed.
    source = ShadowBookSource(Session(), csv_url=live, clock=lambda: START)
    tabs["rules"] = (200, _csv([RULES_HEADER, [""] * 11]))
    assert asyncio.run(source.refresh(force=True))
    assert not source.rules_ok and source.rules_error == "BotRules has no rule rows"

    # The wrong tab (e.g. Config) behind the id is rejected, not read as "no rules".
    tabs["rules"] = (200, _csv([["Key", "Value"], ["start_equity", "10000"]]))
    assert asyncio.run(source.refresh(force=True))
    assert not source.rules_ok and "header missing" in source.rules_error


# ------------------------------------------------------------------ desk

def ts(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()


def real_row(trade_id, market, setup, entry, stop, tp1, tp2, status, layer, opened, valid):
    values = row(trade_id, market, "Short", entry, stop, tp1, tp2, status=status, layer=layer, opened=opened, valid=valid)
    values[HEADER.index("Setup (EW count)")] = setup
    return values


# The Trades rows on 17 Nov 2026 (S-261116-01 is OPEN and -02 CLOSED in the Sheet;
# both are also tried as PENDING to prove the BOOK layer is what keeps them out).
REAL_TRADES = [
    HEADER,
    real_row("S-261116-01", "xyz:SILVER", "A — sell the wave-(ii) bounce of a new five-wave decline",
             47.88, 49.16, 46.6, 45.32, "PENDING", "BOOK", "2026-11-16 05:25", "2026-11-18 05:25"),
    real_row("S-261116-02", "xyz:XYZ100", "B — wave-(iv) bounce sell inside the decline from 28,176",
             27350, 27605, 27095, 26840, "PENDING", "BOOK", "2026-11-16 22:05", "2026-11-18 22:05"),
    real_row("S-261116-03", "xyz:SP500", "B — wave-(iv) bounce sell inside the decline from 7,591.6",
             7452, 7522, 7382, 7312, "PENDING", "SIGNAL", "2026-11-16 22:05", "2026-11-18 22:05"),
]


def test_real_rows_sp500_passes_and_book_rows_stay_ignored(tmp_path):
    async def run():
        rules = parse_rules(REAL_RULES)["rules"]
        store, desk, hl, clock, source = await make(tmp_path, REAL_TRADES, rules=rules)
        source.rules_version = "2026-11-17-1"
        clock.now = ts("2026-11-17 14:00")
        hl.mid.update({"xyz:SP500": 7418.4, "xyz:XYZ100": 27199.0, "xyz:SILVER": 47.445})
        assert parse_rows(REAL_TRADES)["shadowbook:S-261116-03"]["setup_letter"] == "B"
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        assert set(trades) == {"shadowbook:S-261116-03"}  # SP500 Short B placed; BOOK rows never seen
        assert trades["shadowbook:S-261116-03"]["status"] == "ORDERED"
        assert {o["coin"] for o in hl.placed} == {"xyz:SP500"}
        texts = await outbox(store)
        assert any("Bot rules loaded" in t and "version 2026-11-17-1" in t and "Short · B" in t for t in texts)
        await store.close()

    asyncio.run(run())


def test_only_active_setups_are_placed_and_others_are_handled_once(tmp_path):
    async def run():
        rows = [
            HEADER,
            row("S-1", "SOL", "Short", 100, 105, 95, 90, layer="SIGNAL"),  # Short C: allowed
            row("S-2", "BTC", "Short", 70000, 71000, 69000, 68000, layer="SIGNAL"),  # Short A below
            row("S-3", "xyz:GOLD", "Long", 4060, 4000, 4120, 4200, layer="SIGNAL"),  # Long C: inactive
            row("T-1", "SOL", "Short", 100, 105, 95, 90, layer="TEST"),
        ]
        rows[2][HEADER.index("Setup (EW count)")] = "A — sell the wave-(ii) bounce"
        rules = [rule("Short", "C", version="v1"), rule("Short", "B", version="v1"), rule("Long", "C", active=False, version="v1")]
        store, desk, hl, clock, source = await make(tmp_path, rows, rules=rules)
        source.rules_version = "v1"
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        assert trades["shadowbook:S-1"]["status"] == "ORDERED"
        for tid, why in (("shadowbook:S-2", "Setup A Short"), ("shadowbook:S-3", "Setup C Long")):
            assert trades[tid]["status"] == "SKIPPED" and why in trades[tid]["reason"]
            assert "not active in BotRules (rules v1)" in trades[tid]["reason"]
        assert "shadowbook:T-1" not in trades
        assert {o["coin"] for o in hl.placed} == {"SOL"}
        texts = await outbox(store)
        assert sum("Bot rules loaded" in t for t in texts) == 1
        assert any("Signal not taken" in t and "Setup A Short" in t for t in texts)

        # New version: Short A turned on, Short C turned off.
        source.set_rules([rule("Short", "A"), rule("Short", "C", active=False)], "v2")
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        # Handled signals stay handled; the placed SOL order is never cancelled by a rule change.
        assert trades["shadowbook:S-2"]["status"] == "SKIPPED"
        assert trades["shadowbook:S-1"]["status"] == "ORDERED" and not hl.cancelled
        texts = await outbox(store)
        updated = [t for t in texts if "Bot rules updated" in t]
        assert len(updated) == 1 and "version v2" in updated[0] and "Short · A" in updated[0]
        assert "Short · C" not in updated[0]
        await desk.tick()  # same version again: no second message
        assert len([t for t in await outbox(store) if "Bot rules updated" in t]) == 1

        # A new Short A signal is placed under the new rules.
        rows.append(row("S-4", "BTC", "Short", 70000, 71000, 69000, 68000, layer="SIGNAL"))
        rows[-1][HEADER.index("Setup (EW count)")] = "A — sell the wave-(ii) bounce"
        await desk.tick()
        assert desk_state(await store.read())["trades"]["shadowbook:S-4"]["status"] == "ORDERED"

        panel = DeskPanel(store, desk, clock=clock)
        text = plain((await panel.command("rules", [], "telegram:1:2"))[0])
        assert "Version v2" in text and "The bot places" in text and "Short · A" in text
        assert "Paper only" in text and "Short · C" in text
        status = plain((await panel.status())[0])
        assert "Rules v2: Short A · /rules" in status
        await store.close()

    asyncio.run(run())


def test_unreadable_rules_place_nothing_but_keep_existing_orders(tmp_path):
    async def run():
        rows = [HEADER, row("S-1", "SOL", "Short", 100, 105, 95, 90, layer="SIGNAL")]
        store, desk, hl, clock, source = await make(tmp_path, rows)
        await desk.tick()
        assert desk_state(await store.read())["trades"]["shadowbook:S-1"]["status"] == "ORDERED"

        rows.append(row("S-2", "BTC", "Short", 70000, 71000, 69000, 68000, layer="SIGNAL"))
        source.rules, source.rules_ok, source.rules_error = [], False, "BotRules CSV error HTTP 500"
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        assert "shadowbook:S-2" not in trades  # not placed, not marked handled
        assert trades["shadowbook:S-1"]["status"] == "ORDERED" and not hl.cancelled
        assert not any("can't be read" in t for t in await outbox(store))
        clock.now += RULES_DOWN_NOTICE + 1
        await desk.tick()
        await desk.tick()
        warnings = [t for t in await outbox(store) if "BotRules can't be read" in t]
        assert len(warnings) == 1 and "HTTP 500" in warnings[0]
        panel = DeskPanel(store, desk, clock=clock)
        assert "BotRules unavailable" in plain((await panel.status())[0])
        assert "can't be read" in plain((await panel.command("rules", [], "telegram:1:2"))[0])

        # Back: the waiting signal is placed.
        source.set_rules([rule("Short", "C")], "t-1")
        await desk.tick()
        assert desk_state(await store.read())["trades"]["shadowbook:S-2"]["status"] == "ORDERED"
        await store.close()

    asyncio.run(run())


def test_rules_gid_override_from_config():
    from apex_bot.config import Config, ConfigError

    assert Config.from_env({"APEX_SHADOW_BOOK_RULES_GID": "123"}).shadow_book_rules_gid == "123"
    with pytest.raises(ConfigError):
        Config.from_env({"APEX_SHADOW_BOOK_RULES_GID": "BotRules"})
