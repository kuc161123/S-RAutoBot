"""Telegram control panel and trade messages for the Hyperliquid desk.

Views and trade messages are Telegram HTML (sent with parse_mode=HTML): every
dynamic value goes through h(). Every view returns (text, inline_keyboard).
Settings buttons apply immediately and show the result; closing positions
always needs a second, fresh confirmation tap by the same user.
"""

from __future__ import annotations

import math
import re
import secrets
import time
from datetime import datetime, timezone
from decimal import Decimal
from html import escape, unescape

from .desk import (
    ACTIVE,
    AUTO_LEVERAGE,
    BOOKS,
    DEFAULT_SETTINGS,
    LIMITS,
    desk_state,
    leverage_label,
    liquidation_safe_leverage,
    normalize_settings,
    risk_dollars,
    short_id,
)
from .hl_client import coin_candidates
from .notifications import setup_label

CONFIRM_SECONDS = 120
LAYERS = {"signal": "SIGNAL layer", "book": "BOOK layer", "all": "SIGNAL + BOOK"}
SETTINGS_BUTTON_SECONDS = 900
PIN_IDLE_SECONDS = 300  # the pinned dashboard returns home after 5 idle minutes
VIEWS = ("home", "pos", "risk", "lev", "flt", "alr", "perf", "help", "sig", "st")

# ---------------------------------------------------------------- formatting


def h(value) -> str:
    """Escape one dynamic value for Telegram HTML."""
    return escape(str(value), quote=False)


def plain(text: str) -> str:
    """HTML message -> plain text (fallback transport and tests)."""
    return unescape(re.sub(r"</?(b|i|u|s|code|pre)>", "", str(text)))


def px(value) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    text = format(Decimal(repr(float(value))).normalize(), "f")
    return text


def money(value, signed=False) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    if signed:
        return ("+" if value >= 0 else "−") + f"${abs(value):,.2f}"
    return f"${value:,.2f}"


def pct(a, b) -> str:
    if not a or not b:
        return "—"
    change = (b / a - 1) * 100
    return ("+" if change >= 0 else "−") + f"{abs(change):.2f}%"


def when(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%d %b %H:%M UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return "—"


def clock_time(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%H:%M:%S UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return "—"


def ago(ts, now) -> str:
    try:
        seconds = max(0, int(float(now) - float(ts)))
    except (TypeError, ValueError):
        return "—"
    if seconds < 90:
        return f"{seconds}s ago"
    if seconds < 5400:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"


def held_for(start, end) -> str:
    try:
        seconds = max(0, int(float(end) - float(start)))
    except (TypeError, ValueError):
        return "—"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h {rem // 60}m"


def side_word(trade) -> str:
    return "LONG" if trade.get("side") == "Buy" or trade.get("is_buy") else "SHORT"


def side_dot(trade) -> str:
    return "🟢" if side_word(trade) == "LONG" else "🔴"


def tag(trade) -> str:
    return h(f"{side_word(trade)} {trade.get('coin') or trade.get('symbol')} #{short_id(trade['id'])}")


def _sign_r(value) -> str:
    return f"{'+' if value >= 0 else '−'}{abs(value):.2f}R"


# ----------------------------------------------------------- trade messages


def signal_card(trade: dict, signal: dict | None = None) -> str:
    entry, stop = trade["entry"], trade["stop"]
    t1, t2 = trade["target1"], trade["target2"]
    risk_dist = abs(entry - stop) or 1
    r1 = abs(t1 - entry) / risk_dist
    r2 = abs(t2 - entry) / risk_dist
    source = (signal or {}).get("source") or ("shadow_book" if str(trade["id"]).startswith("shadowbook:") else None)
    setup = trade.get("setup_text") or (signal or {}).get("setup_text") or setup_label((signal or {}).get("setup", trade.get("setup")))
    origin = (
        f"Apex Kapital Shadow Book · {trade.get('source_id') or (signal or {}).get('source_id')} · {trade.get('layer') or (signal or {}).get('layer', 'BOOK')}"
        if source == "shadow_book"
        else "Apex signal engine"
    )
    lev_note = f" (capped by {h(trade['leverage_note'])})" if trade.get("leverage_note") else ""
    share = round(100 * trade["tp1_size"] / trade["size"]) if trade.get("tp1_size") and trade.get("size") else 0
    t1_note = f" → close {share}%, stop to entry" if share else " → stop to entry"
    lines = [
        f"{side_dot(trade)} <b>{side_word(trade)} {h(trade['coin'])}</b> · Apex signal #{short_id(trade['id'])}",
        f"<i>{h(setup)}</i>",
        h(origin),
        "",
        f"📍 Entry <b>{px(entry)}</b> · limit until {when(trade.get('expires_at'))}",
        f"🛑 Stop {px(stop)} ({pct(entry, stop)})",
        f"🎯 Target 1 {px(t1)} ({pct(entry, t1)}){t1_note}",
        f"🎯 Target 2 {px(t2)} ({pct(entry, t2)})",
        f"⚖️ Reward {r1:.1f}R · {r2:.1f}R",
        "",
        f"💼 {px(trade['size'])} {h(trade['coin'])} ({money(trade['notional'])}) · "
        f"{trade['leverage']}x isolated{lev_note} · margin {money(trade['margin'])}",
        f"💵 Risk <b>{money(trade['risk_usd'])}</b> if the stop is hit",
        "⏳ Limit order live on Hyperliquid",
    ]
    return "\n".join(lines)


def skip_text(signal: dict, coin, reason: str) -> str:
    name = coin or signal.get("symbol")
    word = "LONG" if signal.get("side") == "Buy" else "SHORT"
    return (
        f"⏭ <b>Signal not taken</b> · {h(word)} {h(name)} #{short_id(signal['id'])}\n"
        f"{h(reason)}\n"
        f"Entry {px(signal.get('limit'))} · stop {px(signal.get('stop'))} · "
        f"targets {px(signal.get('target1'))} / {px(signal.get('target2'))}"
    )


def fill_text(trade: dict, held: float, pos: dict | None) -> str:
    entry = (pos or {}).get("entry_px") or trade["entry"]
    liq = (pos or {}).get("liq_px")
    return (
        f"✅ <b>FILLED</b> · {tag(trade)}\n"
        f"Entry <b>{px(entry)}</b> · size {px(held)} {h(trade['coin'])} · {trade['leverage']}x"
        + (f" · liq {px(liq)}" if liq else "")
        + f"\n🛑 Stop {px(trade['stop'])} is on the exchange\n"
        f"🎯 Targets {px(trade['target1'])} / {px(trade['target2'])}\n"
        f"💵 Risk {money(trade['risk_usd'])}"
    )


def stop_text(trade: dict, old: float, new: float, why: str) -> str:
    entry = trade.get("avg_entry") or trade["entry"]
    held = trade.get("filled_after_tp1") or trade.get("filled") or trade["size"]
    sign = 1 if trade.get("is_buy") else -1
    locked = sign * (new - entry) * held
    risk_line = (
        f"Worst case now <b>{money(locked, signed=True)}</b>"
        if locked >= -0.005
        else f"Risk now {money(-locked)}"
    )
    return f"🛡 <b>STOP MOVED</b> · {tag(trade)}\n{px(old)} → <b>{px(new)}</b>\n{h(why)}\n{risk_line}"


def cancel_text(trade: dict, reason: str) -> str:
    return f"✖️ <b>ENTRY CANCELLED</b> · {tag(trade)}\n{h(reason)}\nNo position was opened."


def close_text(trade: dict) -> str:
    pnl = trade.get("pnl", 0.0)
    icon = "🏆" if pnl > 0 else "🔻" if pnl < 0 else "➖"
    r = trade.get("r")
    r_text = f" ({_sign_r(r)})" if isinstance(r, (int, float)) else ""
    note = "" if trade.get("pnl_complete", True) else "\nFill history unavailable; result may be incomplete."
    return (
        f"{icon} <b>CLOSED</b> · {tag(trade)}\n"
        f"{h(trade.get('exit', 'Closed'))}\n"
        f"Result: <b>{money(pnl, signed=True)}</b>{r_text}\n"
        f"Entry {px(trade.get('avg_entry') or trade.get('entry'))} · held "
        f"{held_for(trade.get('opened_at') or trade.get('created_at'), trade.get('closed_at'))}"
        + note
    )


# ---------------------------------------------------------------- the panel

HELP = (
    "🏦 <b>Apex Desk · how it works</b>\n\n"
    "The Shadow Desk writes trade ideas to the Apex Kapital Shadow Book. "
    "Every new signal in the layer you follow becomes a live Hyperliquid limit order with "
    "its stop on the exchange. Target 1 closes half and moves the stop to entry; "
    "target 2 closes the rest. Everything runs on its own — use this panel only to change "
    "the rules.\n\n"
    "<b>Screens</b>\n"
    "/dashboard — home (also pinned at the top of this chat)\n"
    "/signals — Shadow Book signals and what the desk did\n"
    "/positions — open trades with close and break-even buttons\n"
    "/performance — live results\n"
    "/status — connections and health\n\n"
    "<b>Risk</b>\n"
    "/risk 50 — risk $50 per trade · /risk 1% — 1% of equity\n"
    "/maxpos 6 — max open positions\n"
    "/dailystop 200 — no new trades after $200 lost today (off to disable)\n\n"
    "<b>Leverage</b>\n"
    "/lev auto — safest maximum per trade (default)\n"
    "/lev 10 — cap all symbols · /lev SOL 5 — one symbol · /lev SOL auto — clear it\n\n"
    "<b>Filters</b>\n"
    "/direction both|long|short · /block DOGE · /unblock DOGE\n\n"
    "<b>Control</b>\n"
    "/pause · /resume — stop or restart new entries (open trades stay managed)\n"
    "/be SOL — move a winning trade's stop to entry\n"
    "/close SOL · /closeall — close at market (asks to confirm)\n\n"
    "Leverage changes margin, not dollars at risk: risk comes from the stop distance. "
    "Liquidation is always kept at least 1.5× the stop distance away."
)

DESK_COMMANDS = {
    "dashboard": "Home: account, P&L, positions and controls",
    "signals": "Shadow Book signals and what the desk did with each",
    "positions": "Open positions and pending entries",
    "performance": "Live trading results",
    "status": "Connections and health",
    "risk": "Risk per trade: /risk 50 or /risk 1%",
    "lev": "Leverage: /lev auto, /lev 10 or /lev SOL 5",
    "maxpos": "Max open positions: /maxpos 6",
    "dailystop": "Daily loss stop: /dailystop 200 or off",
    "direction": "Trade both, long or short",
    "block": "Block a symbol: /block DOGE",
    "unblock": "Unblock a symbol",
    "pause": "Stop taking new signals",
    "resume": "Start taking signals again",
    "be": "Move a winning trade's stop to entry: /be SOL",
    "close": "Close one position: /close SOL",
    "closeall": "Close all positions and entries",
    "alerts": "Notification settings",
    "help": "How the desk works and all commands",
}


def _button(label, data):
    return {"text": label, "callback_data": data}


def _rows(buttons, width=2):
    return [buttons[i : i + width] for i in range(0, len(buttons), width)]


def _nav(refresh="d:home"):
    return [_button("🔄 Refresh", refresh), _button("🏠 Home", "d:home")]


def _check(on: bool, label: str) -> str:
    return ("✅ " if on else "") + label


class DeskPanel:
    def __init__(self, store, desk, *, clock=time.time):
        self.store = store
        self.desk = desk
        self.clock = clock
        self._confirm: dict = {}

    @property
    def engine(self) -> bool:
        """False on a desk-only deployment (the Railway engine is switched off)."""
        return bool(getattr(self.desk, "engine_books", True)) if self.desk else True

    # -------------------------------------------------------------- state
    async def _state(self):
        state = await self.store.read()
        desk = desk_state(state)
        raw = desk.get("settings")
        settings = self.desk.effective(raw) if self.desk and hasattr(self.desk, "effective") else normalize_settings(raw)
        return state, desk, settings

    async def _change(self, mutate, note: str):
        def write(tx):
            desk = desk_state(tx.state)
            settings = normalize_settings(desk.get("settings"))
            mutate(settings)
            desk["settings"] = normalize_settings(settings)
            tx.event(
                f"desk:settings:{time.time_ns()}",
                "desk_settings",
                {"change": note, "settings": desk["settings"]},
                None,
            )

        await self.store.update(write)

    def _coin(self, text: str) -> str | None:
        """User input (SOL, sol, SOLUSDT, 1000PEPE, kPEPE, gold) -> Hyperliquid coin."""
        assets = (self.desk.assets if self.desk else None) or {}
        raw = text.strip()
        names = {name.lower(): name for name in assets}
        if raw.lower() in names:
            return names[raw.lower()]
        for prefix in ("xyz:",):
            if (prefix + raw.lower()) in names:
                return names[prefix + raw.lower()]
        upper = raw.upper()
        symbol = upper if upper.endswith("USDT") else upper + "USDT"
        for candidate in coin_candidates(symbol):
            if candidate.lower() in names:
                return names[candidate.lower()]
        return None if assets else (upper if upper.isalnum() and len(upper) <= 20 else None)

    def _snap(self) -> dict:
        return (self.desk.snapshot if self.desk else None) or {}

    def _live(self) -> bool:
        return bool(self.desk and self.desk.configured and self._snap().get("status") == "live")

    # -------------------------------------------------------------- views
    async def render(self, view: str):
        if view == "pos":
            return await self.positions()
        if view == "risk":
            return await self.risk()
        if view == "lev":
            return await self.leverage()
        if view == "flt":
            return await self.filters()
        if view == "alr":
            return await self.alerts()
        if view == "perf":
            return await self.performance()
        if view == "help":
            return HELP, {"inline_keyboard": [[_button("📟 Status", "d:st"), _button("🏠 Home", "d:home")]]}
        if view == "sig":
            return await self.signals()
        if view == "st":
            return await self.status()
        return await self.home()

    def _connection_lines(self):
        snap = self._snap()
        if not self.desk or not self.desk.configured or snap.get("status") == "no_keys":
            return None, [
                "🔌 <b>Hyperliquid not connected</b>",
                "Add HL_ACCOUNT_ADDRESS and HL_API_PRIVATE_KEY in Railway Variables, then redeploy.",
            ]
        if snap.get("status") != "live":
            return snap, ["🔌 Connecting to Hyperliquid…"]
        age = self.clock() - snap.get("at", 0)
        stale = " ⚠️ data stale" if age > 60 else ""
        return snap, [f"<i>Updated {clock_time(snap.get('at'))}{stale}</i>"]

    def _source_label(self, s) -> str:
        if s["book"] == "shadow_book":
            return f"Shadow Book · {LAYERS[s['layer']]}"
        return BOOKS[s["book"]]

    def _source_line(self, s) -> str:
        if not self.desk:
            return ""
        if s["book"] != "shadow_book":
            return "✅ engine connected"
        source = self.desk.source
        if source is None or not source.configured:
            return "❌ Shadow Book NOT connected (add APEX_SHADOW_BOOK_CSV_URL)"
        if source.ok:
            active = sum(
                1
                for sig in source.signals.values()
                if sig.get("status") in ("PENDING", "OPEN", "TP1") and self._wanted(s)(sig)
            )
            read = getattr(source, "fetched_at", 0)
            return f"✅ read {clock_time(read) if read else 'OK'} · {active} active in your layer"
        return f"⚠️ Shadow Book unavailable: {h(source.error or 'waiting for first read')}"

    @staticmethod
    def _wanted(s):
        if s["layer"] == "all":
            return lambda sig: True
        return lambda sig: sig.get("layer") == s["layer"].upper()

    def _lev_text(self, s) -> str:
        return "Auto (safe max)" if s["max_leverage"] >= AUTO_LEVERAGE else f"{s['max_leverage']}x"

    @staticmethod
    def _direction_text(s) -> str:
        return {"both": "longs + shorts", "long": "longs only", "short": "shorts only"}[s["direction"]]

    async def home(self):
        state, desk, s = await self._state()
        snap, conn = self._connection_lines()
        trades = desk["trades"].values()
        open_trades = sorted((t for t in trades if t.get("status") == "OPEN"), key=lambda t: t.get("opened_at") or 0)
        pending = sorted(
            (t for t in trades if t.get("status") in ("ORDERED", "PLACING")), key=lambda t: t.get("created_at") or 0
        )
        day = datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")
        closed_today = [
            t
            for t in trades
            if t.get("status") == "CLOSED"
            and datetime.fromtimestamp(t.get("closed_at", 0), timezone.utc).strftime("%Y-%m-%d") == day
        ]
        realized = sum(t.get("pnl", 0.0) for t in closed_today)
        wins = sum(1 for t in closed_today if t.get("pnl", 0) > 0)
        losses = sum(1 for t in closed_today if t.get("pnl", 0) < 0)
        if not (self.desk and self.desk.configured):
            status = "🔌 <b>NOT CONNECTED</b> · no trading until Hyperliquid keys are added"
        elif self._snap().get("status") != "live":
            status = "⏳ <b>CONNECTING</b> to Hyperliquid…"
        elif s["paused"]:
            status = "⏸ <b>PAUSED</b> · no new entries (open trades still managed)"
        else:
            status = "🟢 <b>LIVE</b> · taking signals automatically"
        lines = ["🏦 <b>APEX DESK</b> · Hyperliquid", status]
        if snap and snap.get("status") == "live":
            positions = snap.get("positions", {}) or {}
            upnl = sum(p.get("upnl", 0.0) for p in positions.values())
            open_risk = sum(self._live_risk(t, snap) for t in open_trades)
            lines += [
                "",
                f"💰 <b>{money(snap.get('equity'))}</b> equity · {money(snap.get('available'))} free margin",
                f"📅 Today <b>{money(realized, signed=True)}</b> ({wins}W {losses}L) · open P&amp;L "
                f"<b>{money(upnl, signed=True)}</b>",
                "",
                f"📂 <b>Book</b> · {len(open_trades)} open · {len(pending)} pending",
            ]
            for t in open_trades[:6]:
                pos = positions.get(t["coin"], {})
                pnl = pos.get("upnl")
                risk = t.get("risk_usd") or 0
                r_now = f" ({_sign_r(pnl / risk)})" if isinstance(pnl, (int, float)) and risk else ""
                guard = " · 🛡 stop at entry" if t.get("tp1_done") else ""
                lines.append(
                    f"{side_dot(t)} {h(t['coin'])} {side_word(t).lower()} · {money(pnl, signed=True)}{r_now}{guard}"
                )
            for t in pending[:4]:
                lines.append(
                    f"⏳ {h(t['coin'])} {side_word(t).lower()} · limit {px(t['entry'])} · until {when(t.get('expires_at'))}"
                )
            if not open_trades and not pending:
                lines.append("Flat — waiting for the next signal.")
            if open_trades:
                lines.append(f"At risk if every stop hits: {money(open_risk)}")
            mine = {t["coin"] for t in (*open_trades, *pending)}
            foreign = sorted(c for c in positions if c not in mine)
            if foreign:
                lines.append(f"<i>Also on the account, not managed: {h(', '.join(foreign))}</i>")
        lines += [
            "",
            "🛡 <b>Rules</b>",
            f"{self._risk_text(s, snap)} · max {s['max_positions']} positions · daily stop "
            f"{money(s['daily_loss_usd']) if s['daily_loss_usd'] else 'off'}",
            f"Leverage {self._lev_text(s)} · {self._direction_text(s)}"
            + (f" · {len(s['symbol_leverage'])} symbol settings" if s["symbol_leverage"] else "")
            + (f" · {len(s['blocked'])} blocked" if s["blocked"] else ""),
            "",
            f"📡 <b>Signals</b> · {h(self._source_label(s))}",
            self._source_line(s),
            "",
            *conn,
        ]
        setup = self._setup_lines(s)
        if setup:
            lines += ["", "🧩 <b>Setup</b>", *setup]
        buttons = [
            _button("📡 Signals", "d:sig"),
            _button("📂 Positions", "d:pos"),
            _button("📈 Performance", "d:perf"),
            _button("📟 Status", "d:st"),
            _button("🛡 Risk", "d:risk"),
            _button("⚙️ Leverage", "d:lev"),
            _button("🎯 Filters", "d:flt"),
            _button("🔔 Alerts", "d:alr"),
            _button("▶️ Resume trading" if s["paused"] else "⏸ Pause trading", "d:resume" if s["paused"] else "d:pause"),
            _button("🚨 Close all", "d:ca"),
        ]
        keyboard = _rows(buttons)
        if self.engine:
            keyboard.append([_button("🔬 Signal engine", "pg:dashboard:0")])
        keyboard.append([_button("🔄 Refresh", "d:home"), _button("❓ Help", "d:help")])
        return "\n".join(lines), {"inline_keyboard": keyboard}

    def _setup_lines(self, s):
        """A checklist only while something still needs doing."""
        desk = self.desk
        hl_ok = bool(desk and desk.configured)
        live = hl_ok and self._snap().get("status") == "live"
        if s["book"] == "shadow_book":
            source = desk.source if desk else None
            src_ok = bool(source and source.configured and source.ok)
            src_line = (
                "✅ Shadow Book connected"
                if src_ok
                else "❌ Shadow Book link — add APEX_SHADOW_BOOK_CSV_URL in Railway"
                if not (source and source.configured)
                else f"⚠️ Shadow Book: {h(source.error or 'connecting…')}"
            )
        else:
            src_ok, src_line = True, "✅ Railway engine signals"
        if live and src_ok:
            return []
        return [
            "✅ Telegram connected",
            (
                "✅ Hyperliquid connected"
                if live
                else "⏳ Hyperliquid connecting…"
                if hl_ok
                else "❌ Hyperliquid keys — add HL_ACCOUNT_ADDRESS and HL_API_PRIVATE_KEY in Railway"
            ),
            src_line,
            "Trading starts automatically when every line is ✅.",
        ]

    async def status(self):
        state, desk, s = await self._state()
        snap = self._snap()
        now = self.clock()
        lines = ["📟 <b>System status</b>", ""]
        if self._live():
            client = self.desk.client
            address = getattr(client, "address", "") or ""
            dexs = [d for d in getattr(client, "dexs", ()) if d]
            lines += [
                f"<b>Hyperliquid</b> ✅ live · updated {ago(snap.get('at'), now)}",
                f"Account {h(address[:6])}…{h(address[-4:])}"
                + (f" · {h(snap.get('mode'))} margin" if snap.get("mode") else ""),
                "Markets: crypto" + (f" + {h(', '.join(dexs))}" if dexs else ""),
                f"Equity {money(snap.get('equity'))} · free {money(snap.get('available'))} · "
                f"in use {money(snap.get('margin_used'))}",
            ]
        else:
            lines += self._connection_lines()[1]
        lines += ["", f"<b>Signals</b> · {h(self._source_label(s))}", self._source_line(s)]
        if s["book"] == "shadow_book":
            lines.append("New signals arrive with each Shadow Desk shift (3× a day).")
        trades = desk["trades"].values()
        counts = {}
        for t in trades:
            counts[t.get("status")] = counts.get(t.get("status"), 0) + 1
        lines += [
            "",
            "<b>Desk</b> " + ("⏸ paused" if s["paused"] else "🟢 taking signals") + " · checks every 4 s",
            "Stops and targets live on the exchange, so open trades stay protected even if this bot restarts.",
            f"Trades: {counts.get('OPEN', 0)} open · {counts.get('ORDERED', 0) + counts.get('PLACING', 0)} pending · "
            f"{counts.get('CLOSED', 0)} closed · {counts.get('SKIPPED', 0)} skipped · {counts.get('CANCELLED', 0)} cancelled",
            f"Running since {when(desk.get('since'))}",
        ]
        errors = sorted(
            (t for t in trades if t.get("status") == "ERROR"), key=lambda t: t.get("closed_at") or 0, reverse=True
        )[:3]
        if errors:
            lines += ["", "<b>Recent order errors</b>"]
            lines += [f"⚠️ {h(t.get('coin'))}: {h(str(t.get('reason', ''))[:120])}" for t in errors]
        return "\n".join(lines), {"inline_keyboard": [_nav("d:st")]}

    async def signals(self):
        state, desk, s = await self._state()
        trades = desk["trades"]
        lines = [f"📡 <b>Signals</b> · {h(self._source_label(s))}"]
        layer_row = [
            _button(_check(s["layer"] == v, {"signal": "SIGNAL", "book": "BOOK", "all": "BOTH"}[v]), f"d:lay:{v}:sig")
            for v in ("signal", "book", "all")
        ]
        keyboard = [layer_row] if s["book"] == "shadow_book" else []
        if self.engine:
            keyboard.append([_button("🎯 Change source", "d:flt")])
        keyboard.append(_nav("d:sig"))
        if s["book"] != "shadow_book":
            engine = [
                t for t in state.get("trades", {}).values()
                if t.get("arm") == s["book"] and t.get("status") in ("PENDING", "OPEN")
            ]
            lines += [f"{len(engine)} active", ""]
            rows = sorted(engine, key=lambda t: t.get("created_at", 0), reverse=True)[:10]
        else:
            source = self.desk.source if self.desk else None
            if not source or not source.configured:
                lines += ["", "Shadow Book not connected yet.", "Add APEX_SHADOW_BOOK_CSV_URL in Railway Variables."]
                return "\n".join(lines), {"inline_keyboard": keyboard}
            lines += [self._source_line(s), ""]
            rows = sorted(
                (x for x in source.signals.values() if x.get("status") in ("PENDING", "OPEN", "TP1") and self._wanted(s)(x)),
                key=lambda x: x.get("created_at", 0),
                reverse=True,
            )[:10]
        if not rows:
            lines.append("No active signals right now. New ones arrive with each Shadow Desk shift.")
        for sig in rows:
            mine = trades.get(sig["id"])
            if mine is None:
                state_text = (
                    "🆕 new — taking it on the next check"
                    if sig.get("status") == "PENDING" and not s["paused"]
                    else "⏸ paused — not taken"
                    if s["paused"] and sig.get("status") == "PENDING"
                    else "— already running before the desk saw it (not taken)"
                )
            else:
                state_text = {
                    "PLACING": "⏳ placing order",
                    "ORDERED": "📍 live limit order",
                    "OPEN": "✅ in position" + (" · TP1 done" if mine.get("tp1_done") else ""),
                    "CLOSED": f"🏁 closed {money(mine.get('pnl'), signed=True)}",
                    "CANCELLED": "✖️ entry cancelled",
                    "SKIPPED": f"⏭ skipped: {h(mine.get('reason', ''))}",
                    "ERROR": f"⚠️ {h(mine.get('reason', 'error'))}",
                }.get(mine.get("status"), h(mine.get("status", "")))
            word = "LONG" if sig.get("side") == "Buy" else "SHORT"
            label = sig.get("source_id") or short_id(sig["id"])
            lines += [
                f"{'🟢' if word == 'LONG' else '🔴'} <b>{word} {h(sig.get('coin') or sig.get('symbol'))}</b> · "
                f"{h(label)} · sheet {h(str(sig.get('status', '')).lower())}",
                f"Entry {px(sig.get('limit'))} · stop {px(sig.get('stop'))} · TP {px(sig.get('target1'))} / {px(sig.get('target2'))}",
                f"Desk: {state_text}",
                "",
            ]
        return "\n".join(lines).rstrip(), {"inline_keyboard": keyboard}

    def _risk_text(self, s, snap):
        if s["risk_mode"] == "pct":
            equity = (snap or {}).get("equity") or 0
            return f"{s['risk_pct']:g}% (≈{money(risk_dollars(s, equity))})/trade"
        return f"{money(s['risk_usd'])}/trade"

    @staticmethod
    def _live_risk(trade, snap):
        pos = (snap.get("positions") or {}).get(trade["coin"])
        if not pos:
            return 0.0
        entry = pos.get("entry_px") or trade.get("avg_entry") or trade["entry"]
        sign = 1 if trade.get("is_buy") else -1
        return max(0.0, sign * (entry - trade["stop"]) * abs(pos.get("size", 0)))

    async def positions(self):
        state, desk, s = await self._state()
        snap = self._snap()
        mids = snap.get("mids") or {}
        positions = snap.get("positions") or {}
        active = sorted(
            (t for t in desk["trades"].values() if t.get("status") in ACTIVE),
            key=lambda t: t.get("created_at", 0),
        )
        open_trades = [t for t in active if t["status"] == "OPEN"]
        pending = [t for t in active if t["status"] != "OPEN"]
        lines = [f"📂 <b>Positions</b> · {len(open_trades)} open · {len(pending)} pending", ""]
        keyboard = []
        for t in open_trades:
            pos = positions.get(t["coin"], {})
            entry = pos.get("entry_px") or t.get("avg_entry") or t["entry"]
            mid = mids.get(t["coin"])
            upnl = pos.get("upnl")
            risk = t.get("risk_usd") or 0
            r_now = f" ({_sign_r(upnl / risk)})" if isinstance(upnl, (int, float)) and risk else ""
            sign = 1 if t.get("is_buy") else -1
            at_entry = sign * (t["stop"] - entry) >= 0
            lines += [
                f"{side_dot(t)} <b>{tag(t)}</b> · {t['leverage']}x",
                f"Entry {px(entry)} → now {px(mid)} ({pct(entry, mid) if mid else '—'})",
                f"P&amp;L <b>{money(upnl, signed=True)}</b>{r_now} · size {px(abs(pos.get('size', t.get('filled', 0))))}",
                f"🛑 {px(t['stop'])}" + (" (at entry)" if at_entry else "") + " · 🎯 "
                + ("✅ " if t.get("tp1_done") else "")
                + f"{px(t['target1'])} / {px(t['target2'])}"
                + (f" · liq {px(pos.get('liq_px'))}" if pos.get("liq_px") else ""),
                "",
            ]
            row = []
            in_profit = isinstance(mid, (int, float)) and sign * (mid - entry) > 0
            if not at_entry and in_profit:
                row.append(_button(f"🛡 Stop→entry {t['coin']}", f"d:be:{short_id(t['id'])}"))
            row.append(_button(f"✖️ Close {t['coin']}", f"d:cl:{short_id(t['id'])}"))
            keyboard.append(row)
        for t in pending:
            lines += [
                f"⏳ <b>{tag(t)}</b> · limit {px(t['entry'])} · stop {px(t['stop'])}",
                f"Until {when(t.get('expires_at'))} · risk {money(t.get('risk_usd'))}",
                "",
            ]
            keyboard.append([_button(f"✖️ Cancel {t['coin']}", f"d:cl:{short_id(t['id'])}")])
        foreign = [c for c in positions if c not in {t["coin"] for t in active}]
        if foreign:
            lines.append(f"<i>Not managed by the desk: {h(', '.join(sorted(foreign)))}</i>")
        if not active:
            lines.append("No open positions or pending entries.")
        keyboard = keyboard[:10]
        if active:
            keyboard.append([_button("🚨 Close all", "d:ca")])
        keyboard.append(_nav("d:pos"))
        return "\n".join(lines).rstrip(), {"inline_keyboard": keyboard}

    def _typical_margin(self, s, equity):
        """Margin a typical trade ties up: 2% stop on a 20x market at current settings."""
        risk = risk_dollars(s, equity)
        notional = risk / 0.02
        leverage = max(1, min(s["max_leverage"], 20, liquidation_safe_leverage(0.02, 20)))
        return notional / leverage, leverage

    async def risk(self):
        state, desk, s = await self._state()
        snap = self._snap()
        equity = snap.get("equity") or 0
        mode = "fixed dollars" if s["risk_mode"] == "usd" else "% of equity"
        lines = [
            "🛡 <b>Risk settings</b>",
            "",
            f"Per trade: <b>{self._risk_text(s, snap)}</b> ({mode})",
            f"Max open positions: <b>{s['max_positions']}</b> → worst case "
            f"{money(risk_dollars(s, equity) * s['max_positions'])} if every stop hits",
            f"Daily loss stop: <b>{money(s['daily_loss_usd']) if s['daily_loss_usd'] else 'off'}</b> "
            "— no new entries for the rest of the UTC day",
        ]
        if self._live():
            margin, lev = self._typical_margin(s, equity)
            free = snap.get("available") or 0
            active = sum(1 for t in desk["trades"].values() if t.get("status") in ACTIVE)
            room = min(int(free * 0.95 // margin) if margin > 0 else 0, max(0, s["max_positions"] - active))
            lines.append(
                f"Free margin {money(free)} · a typical trade (2% stop, {lev}x) ties up ≈ {money(margin)} "
                f"→ room for about <b>{room}</b> more now"
            )
        lines += [
            "",
            "Tap a preset or type /risk 75 · /risk 1.5% · /maxpos 8 · /dailystop 300",
            "<i>Stops are stop-market orders on the exchange: a gap can cost a little more than the set risk.</i>",
        ]
        usd = s["risk_mode"] == "usd"
        ru = [_button(_check(usd and s["risk_usd"] == v, f"${v}"), f"d:ru:{v}") for v in (25, 50, 75, 100, 200)]
        rp = [_button(_check(not usd and s["risk_pct"] == v, f"{v:g}%"), f"d:rp:{v:g}") for v in (0.5, 1, 2, 3)]
        mp = [_button(_check(s["max_positions"] == v, f"{v} pos"), f"d:mp:{v}") for v in (3, 4, 6, 8, 10)]
        dl = [
            _button(_check(s["daily_loss_usd"] == v, f"Stop ${v}" if v else "No stop"), f"d:dl:{v}")
            for v in (100, 200, 300, 500, 0)
        ]
        keyboard = [ru, rp, mp, dl[:3], dl[3:], _nav("d:risk")]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def leverage(self):
        state, desk, s = await self._state()
        lines = [
            "⚙️ <b>Leverage</b>",
            "",
            f"Max for all symbols: <b>{self._lev_text(s)}</b> (isolated margin)",
        ]
        if s["symbol_leverage"]:
            lines.append("Symbol settings: " + h(", ".join(f"{c} {v}x" for c, v in sorted(s["symbol_leverage"].items()))))
        else:
            lines.append("Symbol settings: none")
        lines += [
            "",
            "Each trade uses the symbol setting if there is one, otherwise the max above — never more "
            "than the exchange maximum or the liquidation-safe level (liquidation at least 1.5× the stop "
            "distance away).",
            "Leverage only changes the margin a trade ties up; the dollars at risk come from the stop.",
            "",
            "On a 20x market: stop 1% away → up to "
            f"{min(20, liquidation_safe_leverage(0.01, 20))}x · 2% → {liquidation_safe_leverage(0.02, 20)}x · "
            f"5% → {liquidation_safe_leverage(0.05, 20)}x",
            "",
            "Type /lev SOL 5 for one symbol, /lev SOL auto to clear it.",
        ]
        presets = [_button(_check(s["max_leverage"] == v, f"{v}x"), f"d:ml:{v}") for v in (3, 5, 10, 15, 20, 25)]
        keyboard = _rows(presets, 3) + [
            [_button(_check(s["max_leverage"] >= AUTO_LEVERAGE, "Auto (safe max)"), f"d:ml:{AUTO_LEVERAGE}")]
        ]
        clears = [_button(f"✖️ {c}", f"d:lc:{c}") for c in sorted(s["symbol_leverage"])[:6]]
        keyboard += _rows(clears, 3) + [_nav("d:lev")]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def filters(self):
        state, desk, s = await self._state()
        lines = [
            "🎯 <b>Filters</b>",
            "",
            f"Direction: <b>{self._direction_text(s)}</b>",
            f"Signal source: <b>{h(BOOKS[s['book']])}</b>"
            + (f" · {LAYERS[s['layer']]}" if s["book"] == "shadow_book" else ""),
            "Blocked: " + (h(", ".join(s["blocked"])) if s["blocked"] else "none"),
            "",
            "SIGNAL layer = every setup the Shadow Desk believes in (the profitable record). "
            "BOOK layer = its capped, realistic subset. BOTH = everything.",
        ]
        if self.engine:
            lines.append("Engine = the Railway bot's own Elliott scanner on Bybit data.")
        lines += ["", "Type /block DOGE or /unblock DOGE."]
        dirs = [
            _button(_check(s["direction"] == d, label), f"d:dir:{d}")
            for d, label in (("both", "Both"), ("long", "Longs only"), ("short", "Shorts only"))
        ]
        layers = [
            _button(_check(s["layer"] == v, {"signal": "SIGNAL", "book": "BOOK", "all": "BOTH"}[v]), f"d:lay:{v}")
            for v in ("signal", "book", "all")
        ]
        keyboard = [dirs, layers]
        if self.engine:
            keyboard.append(
                [
                    _button(_check(s["book"] == "shadow_book", "Shadow Book"), "d:book:s"),
                    _button(_check(s["book"] == "baseline_shadow", "Engine rules"), "d:book:b"),
                    _button(_check(s["book"] == "ai_shadow", "Engine AI"), "d:book:a"),
                ]
            )
        unblock = [_button(f"✖️ {c}", f"d:ub:{c}") for c in s["blocked"][:6]]
        keyboard += _rows(unblock, 3) + [_nav("d:flt")]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def alerts(self):
        state, desk, s = await self._state()
        lines = [
            "🔔 <b>Alerts</b>",
            "",
            "Always sent: new orders, fills, stop moves, closes, cancelled entries and faults.",
            f"Signals not taken (e.g. price already past target): <b>{'on' if s['notify_skips'] else 'off'}</b>",
        ]
        keyboard = [
            [
                _button(_check(s["notify_skips"], "Skipped: on"), "d:ns:1"),
                _button(_check(not s["notify_skips"], "Skipped: off"), "d:ns:0"),
            ]
        ]
        if self.engine:
            lines += [
                f"Watchlist alerts (engine setups, AI reviews, universe changes): <b>{s['watchlist_alerts']}</b>",
                f"Shadow simulation alerts: <b>{s['shadow_alerts']}</b>",
                "",
                "auto = on only while Hyperliquid is not connected, so live trading stays uncluttered.",
            ]
            keyboard += [
                [_button(_check(s["watchlist_alerts"] == v, f"Watchlist {v}"), f"d:wl:{v}") for v in ("auto", "on", "off")],
                [_button(_check(s["shadow_alerts"] == v, f"Shadow {v}"), f"d:sa:{v}") for v in ("auto", "on", "off")],
            ]
        keyboard.append(_nav("d:alr"))
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def performance(self):
        state, desk, s = await self._state()
        closed = [t for t in desk["trades"].values() if t.get("status") == "CLOSED"]
        now = self.clock()

        def block(title, rows):
            if not rows:
                return [f"<b>{title}</b> · no closed trades"]
            wins = [t for t in rows if t.get("pnl", 0) > 0]
            losses = [t for t in rows if t.get("pnl", 0) < 0]
            net = sum(t.get("pnl", 0) for t in rows)
            rs = [t["r"] for t in rows if isinstance(t.get("r"), (int, float))]
            gross_win = sum(t["pnl"] for t in wins)
            gross_loss = -sum(t["pnl"] for t in losses)
            pf = f"{gross_win / gross_loss:.2f}" if gross_loss > 0 else "∞" if gross_win > 0 else "—"
            out = [
                f"<b>{title}</b> · <b>{money(net, signed=True)}</b> · {len(rows)} trades · "
                f"{len(wins)}W {len(losses)}L ({len(wins) / len(rows):.0%})"
            ]
            if rs:
                out.append(f"   avg {sum(rs) / len(rs):+.2f}R · profit factor {pf}")
            return out

        day = [t for t in closed if now - t.get("closed_at", 0) <= 86400]
        week = [t for t in closed if now - t.get("closed_at", 0) <= 7 * 86400]
        lines = ["📈 <b>Live performance</b> · Hyperliquid", ""]
        lines += block("24h", day)
        lines += block("7 days", week)
        lines += block("All time", closed)
        if closed:
            best = max(closed, key=lambda t: t.get("pnl", 0))
            worst = min(closed, key=lambda t: t.get("pnl", 0))
            lines += [
                "",
                f"Best {tag(best)} {money(best.get('pnl'), signed=True)}",
                f"Worst {tag(worst)} {money(worst.get('pnl'), signed=True)}",
                "",
                "<b>Recent</b>",
            ]
            for t in sorted(closed, key=lambda t: t.get("closed_at", 0), reverse=True)[:8]:
                lines.append(
                    f"{'🏆' if t.get('pnl', 0) > 0 else '🔻'} {tag(t)} {money(t.get('pnl'), signed=True)} · {h(t.get('exit', ''))}"
                )
        skipped = sum(1 for t in desk["trades"].values() if t.get("status") == "SKIPPED")
        lines += ["", f"Signals not taken: {skipped}"]
        keyboard = []
        if self.engine:
            keyboard.append([_button("🔬 Shadow performance", "pg:performance:0")])
        keyboard.append(_nav("d:perf"))
        return "\n".join(lines), {"inline_keyboard": keyboard}

    # ----------------------------------------------------------- actions
    def _fresh(self, message, seconds):
        if message is None:
            return True
        stamp = message.get("edit_date") or message.get("date")
        return isinstance(stamp, int) and -5 <= self.clock() - stamp <= seconds

    async def _touch(self, message, view: str):
        """Remember what the pinned dashboard shows, so auto-refresh never fights a tap."""
        mid = (message or {}).get("message_id")
        if not mid:
            return

        def write(tx):
            pin = desk_state(tx.state).get("pin")
            if isinstance(pin, dict) and pin.get("message_id") == mid:
                pin["view"] = view
                pin["touched"] = self.clock()

        await self.store.update(write)

    async def callback(self, data: str, actor: str, message: dict | None = None):
        parts = data.split(":")
        view = parts[1] if len(parts) == 2 and parts[1] in VIEWS else "other"
        await self._touch(message, view)
        return await self._callback(parts, actor, message)

    async def _callback(self, parts, actor, message):
        if len(parts) == 2 and parts[1] in VIEWS:
            return await self.render(parts[1])
        action = parts[1] if len(parts) > 1 else ""
        if action in ("clok", "caok"):
            return await self._confirmed(action, parts[2] if len(parts) > 2 else "", actor, message)
        if action == "x":
            return await self.home()
        if not self._fresh(message, SETTINGS_BUTTON_SECONDS):
            text, keyboard = await self.home()
            return "⌛ Those buttons were old, so nothing changed. Here is a fresh panel.\n\n" + text, keyboard
        arg = parts[2] if len(parts) > 2 else ""
        back = parts[3] if len(parts) > 3 else ""
        coin_arg = ":".join(parts[2:])  # builder markets contain a colon, e.g. xyz:GOLD
        if action in ("pause", "resume"):
            paused = action == "pause"
            await self._change(lambda s: s.update(paused=paused), f"paused={paused}")
            return await self.home()
        if action == "ru" and arg.isdigit():
            await self._change(lambda s: s.update(risk_mode="usd", risk_usd=float(arg)), f"risk_usd={arg}")
            return await self.risk()
        if action == "rp":
            try:
                value = float(arg)
            except ValueError:
                value = None
            if value is not None:
                await self._change(lambda s: s.update(risk_mode="pct", risk_pct=value), f"risk_pct={value}")
            return await self.risk()
        if action == "mp" and arg.isdigit():
            await self._change(lambda s: s.update(max_positions=int(arg)), f"max_positions={arg}")
            return await self.risk()
        if action == "dl" and arg.isdigit():
            await self._change(lambda s: s.update(daily_loss_usd=float(arg)), f"daily_loss={arg}")
            return await self.risk()
        if action == "ml" and arg.isdigit():
            await self._change(lambda s: s.update(max_leverage=int(arg)), f"max_leverage={arg}")
            return await self.leverage()
        if action == "lc" and coin_arg:
            await self._change(lambda s: s["symbol_leverage"].pop(coin_arg, None), f"clear_leverage={coin_arg}")
            return await self.leverage()
        if action == "dir" and arg in ("both", "long", "short"):
            await self._change(lambda s: s.update(direction=arg), f"direction={arg}")
            return await self.filters()
        if action == "book" and arg in ("a", "b", "s"):
            book = {"a": "ai_shadow", "b": "baseline_shadow", "s": "shadow_book"}[arg]
            await self._change(lambda s: s.update(book=book), f"book={book}")
            return await self.filters()
        if action == "lay" and arg in ("signal", "book", "all"):
            await self._change(lambda s: s.update(layer=arg), f"layer={arg}")
            return await (self.signals() if back == "sig" else self.filters())
        if action == "ub" and coin_arg:
            await self._change(
                lambda s: s.update(blocked=[c for c in s["blocked"] if c != coin_arg]), f"unblock={coin_arg}"
            )
            return await self.filters()
        if action == "ns" and arg in ("0", "1"):
            await self._change(lambda s: s.update(notify_skips=arg == "1"), f"notify_skips={arg}")
            return await self.alerts()
        if action == "wl" and arg in ("auto", "on", "off"):
            await self._change(lambda s: s.update(watchlist_alerts=arg), f"watchlist={arg}")
            return await self.alerts()
        if action == "sa" and arg in ("auto", "on", "off"):
            await self._change(lambda s: s.update(shadow_alerts=arg), f"shadow_alerts={arg}")
            return await self.alerts()
        if action == "be" and arg:
            return await self._breakeven(arg)
        if action == "cl" and arg:
            return await self._ask_close(arg, actor)
        if action == "ca":
            return self._ask("caok", "", actor, "🚨 <b>Close ALL</b> desk positions at market and cancel all pending entries?")
        return await self.home()

    def _ask(self, action, arg, actor, question):
        nonce = secrets.token_hex(4)
        self._confirm[nonce] = (action, arg, actor, self.clock() + CONFIRM_SECONDS)
        keyboard = [[_button("✅ Yes, do it", f"d:{action}:{nonce}"), _button("✖️ No", "d:x")]]
        return question + f"\n\nConfirm within {CONFIRM_SECONDS // 60} minutes.", {"inline_keyboard": keyboard}

    async def _find(self, sid, statuses=ACTIVE):
        state, desk, s = await self._state()
        return next(
            (t for t in desk["trades"].values() if short_id(t["id"]) == sid and t.get("status") in statuses), None
        )

    async def _breakeven(self, sid):
        trade = await self._find(sid, ("OPEN",))
        result = await self.desk.breakeven(trade["id"]) if trade and self.desk else "That position is no longer open."
        text, keyboard = await self.positions()
        return "🛡 " + h(result) + "\n\n" + text, keyboard

    async def _ask_close(self, sid, actor):
        trade = await self._find(sid)
        if not trade:
            text, keyboard = await self.positions()
            return "That position is no longer active.\n\n" + text, keyboard
        verb = "Close" if trade["status"] == "OPEN" else "Cancel the pending entry for"
        return self._ask("clok", f"{sid}", actor, f"{verb} <b>{tag(trade)}</b> at market?")

    async def _confirmed(self, action, nonce, actor, message):
        pending = self._confirm.pop(nonce, None)
        if not pending or pending[0] != action or pending[2] != actor or pending[3] < self.clock():
            text, keyboard = await self.home()
            return "⌛ That confirmation expired. Nothing was closed.\n\n" + text, keyboard
        if action == "caok":
            result = await self.desk.close_all()
        else:
            state, desk, s = await self._state()
            trade = next((t for t in desk["trades"].values() if short_id(t["id"]) == pending[1]), None)
            result = await self.desk.close_trade(trade["id"]) if trade else "That position is no longer active."
        text, keyboard = await self.positions()
        return h(result) + "\n\n" + text, keyboard

    async def command(self, command: str, args: list[str], actor: str):
        """Return (text, keyboard) for desk commands, or None to fall through."""
        if command in ("start", "dashboard", "desk", "settings", "home"):
            return await self.home()
        if command == "positions":
            return await self.positions()
        if command == "signals":
            return await self.signals()
        if command == "performance":
            return await self.performance()
        if command == "status":
            return await self.status()
        if command == "alerts":
            return await self.alerts()
        if command == "help":
            return await self.render("help")
        if command in ("pause", "resume"):
            paused = command == "pause"
            await self._change(lambda s: s.update(paused=paused), f"paused={paused}")
            return await self.home()
        if command == "risk":
            if not args:
                return await self.risk()
            value = args[0].strip()
            try:
                if value.endswith("%"):
                    number = float(value[:-1])
                    low, high = LIMITS["risk_pct"]
                    if not low <= number <= high:
                        return self._error(f"Percent risk must be {low:g}–{high:g}%.", "d:risk")
                    await self._change(lambda s: s.update(risk_mode="pct", risk_pct=number), f"risk_pct={number}")
                else:
                    number = float(value.lstrip("$"))
                    low, high = LIMITS["risk_usd"]
                    if not low <= number <= high:
                        return self._error(f"Dollar risk must be ${low:g}–${high:,.0f}.", "d:risk")
                    await self._change(lambda s: s.update(risk_mode="usd", risk_usd=number), f"risk_usd={number}")
            except ValueError:
                return self._error("Use /risk 50 for dollars or /risk 1% for percent of equity.", "d:risk")
            return await self.risk()
        if command == "lev":
            if not args:
                return await self.leverage()
            if len(args) == 1:
                if args[0].lower() in ("auto", "max"):
                    await self._change(lambda s: s.update(max_leverage=AUTO_LEVERAGE), "max_leverage=auto")
                    return await self.leverage()
                try:
                    number = int(args[0].lower().rstrip("x"))
                except ValueError:
                    return self._error("Use /lev auto, /lev 5 or /lev SOL 10.", "d:lev")
                if not 1 <= number <= 50:
                    return self._error("Leverage must be 1–50 (or auto).", "d:lev")
                await self._change(lambda s: s.update(max_leverage=number), f"max_leverage={number}")
                return await self.leverage()
            coin = self._coin(args[0])
            if not coin:
                return self._error(f"{args[0]} is not a Hyperliquid market.", "d:lev")
            if args[1].lower() in ("auto", "off", "clear", "default"):
                await self._change(lambda s: s["symbol_leverage"].pop(coin, None), f"clear_leverage={coin}")
                return await self.leverage()
            try:
                number = int(args[1].lower().rstrip("x"))
            except ValueError:
                return self._error("Use /lev SOL 10 or /lev SOL auto.", "d:lev")
            if not 1 <= number <= 50:
                return self._error("Leverage must be 1–50.", "d:lev")
            await self._change(lambda s: s["symbol_leverage"].__setitem__(coin, number), f"leverage {coin}={number}")
            return await self.leverage()
        if command == "maxpos":
            if len(args) != 1 or not args[0].isdigit() or not 1 <= int(args[0]) <= 30:
                return self._error("Use /maxpos 6 (1–30).", "d:risk")
            await self._change(lambda s: s.update(max_positions=int(args[0])), f"max_positions={args[0]}")
            return await self.risk()
        if command == "dailystop":
            if len(args) != 1:
                return self._error("Use /dailystop 200 or /dailystop off.", "d:risk")
            value = 0.0 if args[0].lower() == "off" else None
            if value is None:
                try:
                    value = float(args[0].lstrip("$"))
                except ValueError:
                    return self._error("Use /dailystop 200 or /dailystop off.", "d:risk")
            if value < 0:
                return self._error("Use a positive amount or off.", "d:risk")
            await self._change(lambda s: s.update(daily_loss_usd=value), f"daily_loss={value}")
            return await self.risk()
        if command == "direction":
            if len(args) != 1 or args[0].lower() not in ("both", "long", "short"):
                return self._error("Use /direction both, /direction long or /direction short.", "d:flt")
            direction = args[0].lower()
            await self._change(lambda s: s.update(direction=direction), f"direction={direction}")
            return await self.filters()
        if command in ("block", "unblock"):
            if len(args) != 1:
                return self._error(f"Use /{command} DOGE.", "d:flt")
            coin = self._coin(args[0])
            if not coin:
                return self._error(f"{args[0]} is not a Hyperliquid market.", "d:flt")
            if command == "block":
                await self._change(lambda s: s.update(blocked=sorted(set(s["blocked"]) | {coin})), f"block={coin}")
            else:
                await self._change(lambda s: s.update(blocked=[c for c in s["blocked"] if c != coin]), f"unblock={coin}")
            return await self.filters()
        if command in ("close", "be"):
            if len(args) != 1:
                return self._error(f"Use /{command} SOL.", "d:pos")
            coin = self._coin(args[0])
            state, desk, s = await self._state()
            statuses = ("OPEN",) if command == "be" else ACTIVE
            trade = next((t for t in desk["trades"].values() if t.get("coin") == coin and t.get("status") in statuses), None)
            if not trade:
                return self._error(f"No {'open' if command == 'be' else 'active'} desk position on {args[0].upper()}.", "d:pos")
            if command == "be":
                return await self._breakeven(short_id(trade["id"]))
            return await self._ask_close(short_id(trade["id"]), actor)
        if command == "closeall":
            return self._ask("caok", "", actor, "🚨 <b>Close ALL</b> desk positions at market and cancel all pending entries?")
        return None

    @staticmethod
    def _error(text, back):
        return "⚠️ " + h(text), {"inline_keyboard": [[_button("← Back", back), _button("🏠 Home", "d:home")]]}


def _alert_mode(state: dict, key: str, desk_live: bool) -> bool:
    mode = normalize_settings(state.get("desk", {}).get("settings")).get(key)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return not desk_live


def shadow_alerts_enabled(state: dict, desk_live: bool) -> bool:
    return _alert_mode(state, "shadow_alerts", desk_live)


def watchlist_alerts_enabled(state: dict, desk_live: bool) -> bool:
    return _alert_mode(state, "watchlist_alerts", desk_live)


__all__ = [
    "DeskPanel",
    "DESK_COMMANDS",
    "HELP",
    "plain",
    "signal_card",
    "skip_text",
    "fill_text",
    "stop_text",
    "cancel_text",
    "close_text",
    "shadow_alerts_enabled",
    "watchlist_alerts_enabled",
    "DEFAULT_SETTINGS",
    "leverage_label",
]
