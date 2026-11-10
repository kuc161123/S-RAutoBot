import asyncio

import pytest

from apex_bot.desk import Desk, desk_state
from apex_bot.desk_panel import DeskPanel, signal_card
from apex_bot.shadow_book import (
    SourceError,
    ShadowBookSource,
    is_price_exit,
    parse_rows,
    setup_title,
    valid_csv_url,
)
from apex_bot.storage import Store

from .test_desk import ASSETS, NOW, Clock, FakeHL, outbox

HEADER = [
    "ID", "Opened UTC", "Shift", "Market (HL)", "Class", "Side", "Setup (EW count)",
    "Thesis / invalidation", "Entry type", "Entry", "Stop", "TP1", "TP2", "TP1 share %",
    "Valid until UTC", "Risk $", "Size (units)", "Notional $", "Status", "Filled UTC",
    "Fill price", "TP1 hit UTC", "Exit UTC", "Exit price", "Exit reason", "Gross P&L $",
    "Fees $", "Funding $", "Net P&L $", "R multiple", "Price source", "Underlying check",
    "Desk", "Notes", "Layer",
]


def row(trade_id, market, side, entry, stop, tp1, tp2, status="PENDING", layer="BOOK", opened="2027-01-15 08:00", valid="2027-01-17 08:00", reason="", tp1_hit=""):
    values = dict(zip(HEADER, [""] * len(HEADER)))
    values.update({
        "ID": trade_id, "Opened UTC": opened, "Market (HL)": market, "Side": side,
        "Setup (EW count)": "C — sell the retest of a broken floor. 1D lower highs since Oct; more detail.",
        "Entry type": "LIMIT", "Entry": str(entry), "Stop": str(stop), "TP1": str(tp1), "TP2": str(tp2),
        "TP1 share %": "50", "Valid until UTC": valid, "Status": status, "Exit reason": reason,
        "TP1 hit UTC": tp1_hit, "Layer": layer,
    })
    return [values[h] for h in HEADER]


def test_parse_rows_reads_the_real_header_layout():
    rows = [
        HEADER,
        row("S-1", "SOL", "Short", 100, 105, 95, 90),
        row("S-2", "xyz:GOLD", "Long", "4,078", "4,050", "4,120", "4,200", status="TP1", layer="SIGNAL", tp1_hit="2027-01-16 01:00"),
        row("S-3", "BTC", "Short", 70000, 71000, 69000, 68000, status="CLOSED", reason="STOP 71000 hit on the 15m bar"),
        ["", "", "", ""],  # blank row
    ]
    out = parse_rows(rows)
    assert set(out) == {"shadowbook:S-1", "shadowbook:S-2", "shadowbook:S-3"}
    s1 = out["shadowbook:S-1"]
    assert s1["side"] == "Sell" and s1["limit"] == 100 and s1["coin"] == "SOL" and s1["layer"] == "BOOK"
    assert s1["expires_at"] - s1["created_at"] == 48 * 3600
    assert s1["setup_text"] == "C — sell the retest of a broken floor"
    assert out["shadowbook:S-2"]["limit"] == 4078 and out["shadowbook:S-2"]["tp1_done"]
    assert out["shadowbook:S-3"]["price_exit"]
    with pytest.raises(SourceError):
        parse_rows([["ID", "Side"], ["x", "Long"]])


def test_exit_reason_classification_and_links():
    assert is_price_exit("TP2") and is_price_exit("Stop: filled 07:00 UTC") and is_price_exit("EXPIRED unfilled")
    assert not is_price_exit("Closed by desk: count invalidated") and not is_price_exit("")
    assert setup_title("") == "Elliott setup"
    assert valid_csv_url("https://docs.google.com/spreadsheets/d/e/2PACX-abc/pub?gid=1&single=true&output=csv")
    assert valid_csv_url("https://docs.google.com/spreadsheets/d/abc/export?format=csv&gid=123")
    assert not valid_csv_url("https://evil.example/pub?output=csv")
    assert not valid_csv_url("https://docs.google.com/forms/x?output=csv")
    with pytest.raises(ValueError):
        ShadowBookSource(None, sheet_id="1aCcEcbodqcBzf3eM8THxxTyTww3aoOCE6PXCLtirXe4")  # no Google key


class FakeSource:
    configured = True
    mode = "csv"

    def __init__(self, rows):
        self.rows = rows
        self.ok = True
        self.error = None
        self.signals = parse_rows(rows)

    async def refresh(self, force=False):
        self.signals = parse_rows(self.rows)
        return True


START = 1_800_000_000.0  # 2027-01-15 08:00 UTC


async def make(tmp_path, rows, settings=None):
    store = Store(sqlite_path=str(tmp_path / "sb.sqlite"))
    await store.initialize()
    assert await store.lease()
    clock = Clock()
    clock.now = START + 600
    hl = FakeHL()
    hl.mid.update({"SOL": 99.0, "xyz:GOLD": 4070.0, "BTC": 69800.0})
    source = FakeSource(rows)

    def seed(tx):
        desk_state(tx.state)["settings"] = settings or {}

    await store.update(seed)
    return store, Desk(store, hl, clock=clock, source=source), hl, clock, source


ASSETS["xyz:GOLD"] = {"sz_decimals": 4, "max_leverage": 25, "only_isolated": True, "delisted": False, "dex": "xyz"}


def test_shadow_book_signal_layer_goes_live_with_xyz_markets(tmp_path):
    async def run():
        rows = [
            HEADER,
            row("S-1", "SOL", "Short", 100, 105, 95, 90, layer="SIGNAL"),
            row("S-2", "xyz:GOLD", "Long", 4060, 4000, 4120, 4200, layer="SIGNAL"),
            row("S-3", "BTC", "Short", 70000, 71000, 69000, 68000, layer="BOOK"),
            row("S-4", "SOL", "Short", 100, 105, 95, 90, status="CLOSED", reason="TP2", layer="SIGNAL"),
        ]
        store, desk, hl, clock, source = await make(tmp_path, rows)
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        assert set(trades) == {"shadowbook:S-1", "shadowbook:S-2"}  # SIGNAL layer, PENDING only
        assert {o["coin"] for o in hl.placed} == {"SOL", "xyz:GOLD"}
        assert hl.leverage["xyz:GOLD"][1] is True  # isolated
        cards = [t for t in await outbox(store) if "Apex signal" in t]
        assert any("Apex Kapital Shadow Book · S-1 · SIGNAL" in c for c in cards)
        assert any("C — sell the retest of a broken floor" in c for c in cards)

        # Switch to SIGNAL + BOOK: the BOOK-layer BTC idea is taken next tick.
        def both(tx):
            desk_state(tx.state)["settings"]["layer"] = "all"

        await store.update(both)
        await desk.tick()
        assert "shadowbook:S-3" in desk_state(await store.read())["trades"]
        await store.close()

    asyncio.run(run())


def test_sheet_status_changes_drive_live_management(tmp_path):
    async def run():
        rows = [HEADER, row("S-1", "SOL", "Short", 100, 105, 95, 90), row("S-2", "xyz:GOLD", "Long", 4060, 4000, 4120, 4200)]
        store, desk, hl, clock, source = await make(tmp_path, rows, {"layer": "book"})
        await desk.tick()
        # GOLD expires in the Sheet before filling -> live entry cancelled.
        source.rows[2] = row("S-2", "xyz:GOLD", "Long", 4060, 4000, 4120, 4200, status="EXPIRED", reason="EXPIRED unfilled")
        # SOL fills live, then the Sheet reports TP1 -> stop to entry.
        hl.fill_entry("SOL")
        await desk.tick()
        trades = desk_state(await store.read())["trades"]
        assert trades["shadowbook:S-2"]["status"] == "CANCELLED"
        assert trades["shadowbook:S-1"]["status"] == "OPEN"
        source.rows[1] = row("S-1", "SOL", "Short", 100, 105, 95, 90, status="TP1", tp1_hit="2027-01-15 12:00")
        hl.mid["SOL"] = 96.0
        await desk.tick()
        t = desk_state(await store.read())["trades"]["shadowbook:S-1"]
        assert t["tp1_done"] and t["stop"] == 100.0
        # The desk closes it for its own reason -> live position closed at market.
        source.rows[1] = row("S-1", "SOL", "Short", 100, 105, 95, 90, status="CLOSED", reason="Closed by desk: count invalidated")
        await desk.tick()
        await desk.tick()
        t = desk_state(await store.read())["trades"]["shadowbook:S-1"]
        assert t["status"] == "CLOSED" and t["exit"].startswith("Shadow Desk exit")
        await store.close()

    asyncio.run(run())


def test_unavailable_sheet_pauses_intake_and_is_visible(tmp_path):
    async def run():
        rows = [HEADER, row("S-1", "SOL", "Short", 100, 105, 95, 90)]
        store, desk, hl, clock, source = await make(tmp_path, rows)
        source.ok, source.error = False, "Google denied access"

        async def failing(force=False):
            return False

        source.refresh = failing
        await desk.tick()
        assert desk_state(await store.read())["trades"] == {}
        panel = DeskPanel(store, desk, clock=clock)
        text, _ = await panel.home()
        assert "Shadow Book unavailable: Google denied access" in text
        await store.close()

    asyncio.run(run())


def test_panel_coin_lookup_accepts_xyz_names(tmp_path):
    async def run():
        store, desk, hl, clock, source = await make(tmp_path, [HEADER])
        await desk.tick()
        panel = DeskPanel(store, desk, clock=clock)
        text, _ = await panel.command("lev", ["gold", "3"], "telegram:1:2")
        assert "xyz:GOLD 3x" in text
        await store.close()

    asyncio.run(run())
