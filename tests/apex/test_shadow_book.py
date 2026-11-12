import asyncio

import pytest

from apex_bot.desk import Desk, desk_state
from apex_bot.desk_panel import DeskPanel, plain, signal_card
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
    # Real Shadow Desk wording: the exchange stop already covers these.
    assert is_price_exit("FILL and STOP on the same 15m bar 14:30 UTC 4 Nov")
    assert is_price_exit("BE stop after TP1: 10:00 UTC 15m bar") and is_price_exit("TP1 966 then BE STOP 946")
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
        self.fetched_at = 0.0
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
        text = plain(text)
        assert "Shadow Book unavailable: Google denied access" in text
        await store.close()

    asyncio.run(run())


def test_panel_coin_lookup_accepts_xyz_names(tmp_path):
    async def run():
        store, desk, hl, clock, source = await make(tmp_path, [HEADER])
        await desk.tick()
        panel = DeskPanel(store, desk, clock=clock)
        text, _ = await panel.command("lev", ["gold", "3"], "telegram:1:2")
        text = plain(text)
        assert "xyz:GOLD 3x" in text
        await store.close()

    asyncio.run(run())


def test_signals_view_and_setup_checklist(tmp_path):
    async def run():
        rows = [
            HEADER,
            row("S-1", "SOL", "Short", 100, 105, 95, 90, layer="SIGNAL"),
            row("S-9", "XRP", "Short", 1.2, 1.25, 1.15, 1.1, layer="SIGNAL"),  # not listed -> skipped
            row("S-3", "BTC", "Short", 70000, 71000, 69000, 68000, layer="BOOK"),
        ]
        store, desk, hl, clock, source = await make(tmp_path, rows)
        panel = DeskPanel(store, desk, clock=clock)
        text, keyboard = await panel.home()
        text = plain(text)
        assert "🧩 Setup" in text and "Hyperliquid connecting" in text  # keys present, no tick yet
        await desk.tick()
        text, keyboard = await panel.home()
        text = plain(text)
        assert "🧩 Setup" not in text  # everything connected
        assert "d:sig" in str(keyboard)
        text, _ = await panel.callback("d:sig", "telegram:1:2", {"date": int(clock.now)})
        text = plain(text)
        assert "S-1" in text and "live limit order" in text
        assert "S-9" in text and "skipped" in text
        assert "S-3" not in text  # BOOK layer not followed
        await store.close()

    asyncio.run(run())


# --------------------------------------- entry history (Shadow Book fill rules)

from apex_bot.desk import LEGACY_TP1_SKIP, entry_history  # noqa: E402
from apex_bot.hl_client import HLError  # noqa: E402


def test_entry_history_matches_shadow_book_fills():
    short = dict(is_buy=False, entry=74250, stop=75020, target1=73450)
    # Never traded the entry: still a valid resting limit, wherever price is now.
    assert entry_history(**short, bars=[(1, 74033, 73082), (2, 73700, 73300)]) == (False, None)
    # Entry traded, no stop/target yet: live, may still be entered.
    assert entry_history(**short, bars=[(1, 74300, 74000)]) == (True, None)
    # Entry traded, then target 1: already played out.
    assert entry_history(**short, bars=[(1, 74300, 74000), (2, 73900, 73400)]) == (True, "target")
    # Same bar touches entry and stop: stop first, like the Shadow Book.
    assert entry_history(**short, bars=[(1, 75100, 73000)]) == (True, "stop")
    long = dict(is_buy=True, entry=100, stop=95, target1=110)
    assert entry_history(**long, bars=[(1, 112, 101)]) == (False, None)  # ran up without the dip to 100
    assert entry_history(**long, bars=[(1, 104, 99.5), (2, 111, 103)]) == (True, "target")


def _ms(ts):
    return int(ts * 1000)


def test_untouched_short_below_target1_is_still_placed(tmp_path):
    """The 10 Nov BTC case: limit above market, price dipped below TP1, entry never traded."""

    async def run():
        rows = [HEADER, row("S-BTC", "BTC", "Short", 74250, 75020, 73450, 72535, layer="SIGNAL")]
        store, desk, hl, clock, source = await make(tmp_path, rows)
        hl.mid["BTC"] = 73300.0  # below target 1
        hl.bars["BTC"] = [(_ms(START + 300), 74033.0, 73082.0), (_ms(START + 600), 73600.0, 73250.0)]
        await desk.tick()
        trade = desk_state(await store.read())["trades"]["shadowbook:S-BTC"]
        assert trade["status"] == "ORDERED" and trade["entry"] == 74250
        assert [o["limit_px"] for o in hl.placed if o["coin"] == "BTC" and o["kind"] == "limit"] == [74250]
        # its stop rides on the entry order, waiting for the fill
        assert [o["trigger_px"] for o in hl.placed if o["coin"] == "BTC" and o["kind"] == "sl"] == [75020]
        await store.close()

    asyncio.run(run())


def test_entry_traded_then_target_is_skipped_as_played_out(tmp_path):
    """The 10 Nov SOL case: filled and ran to target before the sheet's next update."""

    async def run():
        rows = [HEADER, row("S-SOL", "SOL", "Short", 100, 101, 98, 96, layer="SIGNAL")]
        store, desk, hl, clock, source = await make(tmp_path, rows)
        hl.mid["SOL"] = 99.0
        hl.bars["SOL"] = [(_ms(START + 300), 100.2, 99.5), (_ms(START + 600), 99.6, 97.8)]
        await desk.tick()
        trade = desk_state(await store.read())["trades"]["shadowbook:S-SOL"]
        assert trade["status"] == "SKIPPED" and trade["rule"] == 2
        assert trade["reason"].startswith("Already played out on Hyperliquid: entry 100 traded, then target 1")
        assert not [o for o in hl.placed if o["coin"] == "SOL"]
        await store.close()

    asyncio.run(run())


def test_legacy_target1_skips_get_one_recheck_and_history_errors_retry(tmp_path):
    async def run():
        rows = [HEADER, row("S-BTC", "BTC", "Short", 74250, 75020, 73450, 72535, layer="SIGNAL")]
        store, desk, hl, clock, source = await make(tmp_path, rows)

        def legacy(tx):
            desk_state(tx.state)["trades"]["shadowbook:S-BTC"] = {
                "id": "shadowbook:S-BTC", "status": "SKIPPED", "reason": LEGACY_TP1_SKIP, "coin": "BTC",
            }

        await store.update(legacy)
        hl.mid["BTC"] = 73300.0

        async def down(*args):
            raise HLError("Hyperliquid request failed (ServerError 502)")

        real = hl.candles
        hl.candles = down
        await desk.tick()  # no history: nothing decided, nothing recorded
        assert desk_state(await store.read())["trades"]["shadowbook:S-BTC"]["reason"] == LEGACY_TP1_SKIP
        hl.candles = real
        hl.bars["BTC"] = [(_ms(START + 300), 74033.0, 73082.0)]
        await desk.tick()
        assert desk_state(await store.read())["trades"]["shadowbook:S-BTC"]["status"] == "ORDERED"
        await store.close()

    asyncio.run(run())


def test_status_shows_how_fresh_the_shadow_book_copy_is(tmp_path):
    async def run():
        store, desk, hl, clock, source = await make(tmp_path, [HEADER, row("S-1", "SOL", "Short", 100, 105, 95, 90)])
        source.ok_at, source.newest_at = clock.now - 30, START
        await desk.tick()
        text, _ = await DeskPanel(store, desk, clock=clock).render("st")
        assert "Last good read 30s ago · newest row in it opened 15 Jan 08:00 UTC" in plain(text)
        await store.close()

    asyncio.run(run())


def test_csv_reads_ask_google_for_a_fresh_copy():
    import csv as _csv
    import io as _io
    from types import SimpleNamespace

    out = _io.StringIO()
    _csv.writer(out).writerows([HEADER, row("S-1", "SOL", "Short", 100, 105, 95, 90)])
    body = out.getvalue().encode()
    seen = {}

    class Response:
        url = SimpleNamespace(host="docs.google.com")
        status = 200
        content = SimpleNamespace(read=None)

        async def __aenter__(self):
            async def chunks(size):
                for i in range(0, len(body), 7):  # many small chunks, like a real socket
                    yield body[i : i + 7]

            self.content = SimpleNamespace(iter_chunked=chunks)
            return self

        async def __aexit__(self, *exc):
            return False

    class Session:
        def get(self, url, **kwargs):
            seen["url"], seen["headers"] = url, kwargs.get("headers")
            return Response()

    link = "https://docs.google.com/spreadsheets/d/e/2PACX-abc/pub?gid=1&single=true&output=csv"
    source = ShadowBookSource(Session(), csv_url=link, clock=lambda: START)
    assert asyncio.run(source.refresh(force=True))
    assert seen["url"] == link + f"&_ts={int(START)}"
    assert seen["headers"]["Cache-Control"] == "no-cache"
    assert source.ok_at == START and source.newest_at == START


def test_live_link_first_published_copy_as_fallback():
    import csv as _csv
    import io as _io
    from types import SimpleNamespace

    from apex_bot.config import Config
    from apex_bot.shadow_book import valid_csv_links

    out = _io.StringIO()
    _csv.writer(out).writerows([HEADER, row("S-1", "SOL", "Short", 100, 105, 95, 90)])
    body = out.getvalue().encode()
    live = "https://docs.google.com/spreadsheets/d/abc/export?format=csv&gid=1"
    pub = "https://docs.google.com/spreadsheets/d/e/2PACX-abc/pub?gid=1&single=true&output=csv"
    status = {"export": 429, "pub": 200}
    calls = []

    class Response:
        def __init__(self, url):
            self.url = SimpleNamespace(host="docs.google.com")
            self.status = status["export" if "/export" in url else "pub"]

            async def chunks(size):
                for i in range(0, len(body), 11):
                    yield body[i : i + 11]

            self.content = SimpleNamespace(iter_chunked=chunks)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class Session:
        def get(self, url, **kwargs):
            calls.append("export" if "/export" in url else "pub")
            return Response(url)

    assert valid_csv_links(f"{live} {pub}") and valid_csv_links(f"{live},{pub}")
    assert not valid_csv_links(f"{live} https://evil.example/x.csv")
    assert Config.from_env({"APEX_SHADOW_BOOK_CSV_URL": f"{live} {pub}"}).shadow_book_csv_url
    source = ShadowBookSource(Session(), csv_url=f"{live} {pub}", clock=lambda: START)
    assert asyncio.run(source.refresh(force=True))
    assert calls == ["export", "pub"] and source.link_used == "published copy"
    assert source.link_errors == {"live sheet": "Shadow Book CSV error HTTP 429"}
    status["export"] = 200
    assert asyncio.run(source.refresh(force=True))
    assert calls[-1] == "export" and source.link_used == "live sheet" and source.link_errors == {}


def test_whole_csv_is_read_even_when_it_arrives_in_pieces():
    """10-11 Nov: content.read(n) returned only the first chunk, so new rows vanished."""
    import csv as _csv
    import io as _io
    from types import SimpleNamespace

    out = _io.StringIO()
    rows = [HEADER] + [row(f"S-{i}", "SOL", "Short", 100, 105, 95, 90, opened=f"2027-01-{10 + i % 5:02d} 08:00") for i in range(60)]
    rows.append(row("S-NEW", "BTC", "Short", 74250, 75020, 73450, 72535, layer="SIGNAL", opened="2027-01-15 08:00"))
    _csv.writer(out).writerows(rows)
    body = out.getvalue().encode()

    class Response:
        url = SimpleNamespace(host="docs.google.com")
        status = 200

        async def __aenter__(self):
            async def chunks(size):
                for i in range(0, len(body), 4096):
                    yield body[i : i + 4096]

            self.content = SimpleNamespace(iter_chunked=chunks)
            return self

        async def __aexit__(self, *exc):
            return False

    class Session:
        def get(self, url, **kwargs):
            return Response()

    link = "https://docs.google.com/spreadsheets/d/abc/export?format=csv&gid=1"
    source = ShadowBookSource(Session(), csv_url=link, clock=lambda: START)
    assert asyncio.run(source.refresh(force=True))
    assert len(body) > 2 * 4096 and len(source.signals) == 61
    assert source.signals["shadowbook:S-NEW"]["status"] == "PENDING" and source.newest_at == START


def test_resting_shadow_limit_is_not_cancelled_when_price_passes_target1(tmp_path):
    """11 Nov oil short: entry 88.1 untouched, price below TP1 86.7 -> keep waiting."""

    async def run():
        rows = [HEADER, row("S-CL", "SOL", "Short", 100, 101, 98, 96, layer="SIGNAL")]
        store, desk, hl, clock, source = await make(tmp_path, rows)
        hl.mid["SOL"] = 99.0
        await desk.tick()
        assert desk_state(await store.read())["trades"]["shadowbook:S-CL"]["status"] == "ORDERED"
        hl.mid["SOL"] = 97.5  # beyond target 1, entry never traded
        await desk.tick()
        await desk.tick()
        trade = desk_state(await store.read())["trades"]["shadowbook:S-CL"]
        assert trade["status"] == "ORDERED" and not hl.cancelled
        # The Shadow Book expiring it still cancels it.
        source.rows[1] = row("S-CL", "SOL", "Short", 100, 101, 98, 96, status="EXPIRED", reason="EXPIRED unfilled", layer="SIGNAL")
        await desk.tick()
        trade = desk_state(await store.read())["trades"]["shadowbook:S-CL"]
        assert trade["status"] == "CANCELLED" and "Shadow Book closed" in trade["reason"]
        await store.close()

    asyncio.run(run())


def test_wrongly_cancelled_limit_is_replaced_with_fresh_order_ids(tmp_path):
    from apex_bot.desk import LEGACY_TP1_CANCEL
    from apex_bot.hl_client import cloid_for

    async def run():
        rows = [HEADER, row("S-CL", "SOL", "Short", 100, 101, 98, 96, layer="SIGNAL")]
        store, desk, hl, clock, source = await make(tmp_path, rows)

        def legacy(tx):
            desk_state(tx.state)["trades"]["shadowbook:S-CL"] = {
                "id": "shadowbook:S-CL", "status": "CANCELLED", "reason": LEGACY_TP1_CANCEL, "coin": "SOL",
                "entry_cloid": cloid_for("shadowbook:S-CL", "entry"),
            }

        await store.update(legacy)
        hl.mid["SOL"] = 97.5
        await desk.tick()
        trade = desk_state(await store.read())["trades"]["shadowbook:S-CL"]
        assert trade["status"] == "ORDERED" and trade["attempt"] == 2 and trade["rule"] == 2
        assert trade["entry_cloid"] == cloid_for("shadowbook:S-CL", "entry#2")
        cards = [t for t in await outbox(store) if "Apex signal" in t]
        assert len(cards) == 1
        await store.close()

    asyncio.run(run())
