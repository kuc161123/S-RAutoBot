"""Telegram control panel and trade messages for the Hyperliquid desk.

Plain text only (the transport sends no parse_mode). Every view returns
(text, inline_keyboard). Settings buttons apply immediately and show the result;
closing positions always needs a second, fresh confirmation tap.
"""

from __future__ import annotations

import math
import secrets
import time
from datetime import datetime, timezone
from decimal import Decimal

from .desk import (
    ACTIVE,
    BOOKS,
    DEFAULT_SETTINGS,
    LIMITS,
    desk_state,
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

# ---------------------------------------------------------------- formatting


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
    return f"{side_word(trade)} {trade.get('coin') or trade.get('symbol')} #{short_id(trade['id'])}"


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
    lev_note = f" (capped by {trade['leverage_note']})" if trade.get("leverage_note") else ""
    share = round(100 * trade["tp1_size"] / trade["size"]) if trade.get("tp1_size") and trade.get("size") else 0
    t1_note = f" → close {share}%, stop to entry" if share else " → stop to entry"
    lines = [
        f"{side_dot(trade)} {side_word(trade)} {trade['coin']} · Apex signal #{short_id(trade['id'])}",
        setup,
        origin,
        "",
        f"📍 Entry: {px(entry)} (limit, valid until {when(trade.get('expires_at'))})",
        f"🛑 Stop: {px(stop)} ({pct(entry, stop)})",
        f"🎯 Target 1: {px(t1)} ({pct(entry, t1)}){t1_note}",
        f"🎯 Target 2: {px(t2)} ({pct(entry, t2)})",
        f"⚖️ Reward/risk: {r1:.1f}R · {r2:.1f}R",
        "",
        f"💼 Size {px(trade['size'])} {trade['coin']} ({money(trade['notional'])}) · "
        f"{trade['leverage']}x isolated{lev_note} · margin {money(trade['margin'])}",
        f"💵 Risk {money(trade['risk_usd'])} if the stop is hit",
        "⏳ Limit order live on Hyperliquid",
    ]
    return "\n".join(lines)


def skip_text(signal: dict, coin, reason: str) -> str:
    name = coin or signal.get("symbol")
    word = "LONG" if signal.get("side") == "Buy" else "SHORT"
    return (
        f"⏭ Signal not taken · {word} {name} #{short_id(signal['id'])}\n"
        f"Reason: {reason}\n"
        f"Levels: entry {px(signal.get('limit'))} · stop {px(signal.get('stop'))} · "
        f"targets {px(signal.get('target1'))} / {px(signal.get('target2'))}"
    )


def fill_text(trade: dict, held: float, pos: dict | None) -> str:
    entry = (pos or {}).get("entry_px") or trade["entry"]
    liq = (pos or {}).get("liq_px")
    return (
        f"✅ FILLED · {tag(trade)}\n"
        f"Entry {px(entry)} · size {px(held)} {trade['coin']} · {trade['leverage']}x"
        + (f" · liq {px(liq)}" if liq else "")
        + f"\n🛑 Stop {px(trade['stop'])} is on the exchange\n"
        f"🎯 Targets {px(trade['target1'])} / {px(trade['target2'])}\n"
        f"Risk {money(trade['risk_usd'])}"
    )


def stop_text(trade: dict, old: float, new: float, why: str) -> str:
    entry = trade.get("avg_entry") or trade["entry"]
    held = trade.get("filled_after_tp1") or trade.get("filled") or trade["size"]
    sign = 1 if trade.get("is_buy") else -1
    locked = sign * (new - entry) * held
    risk_line = (
        f"Worst case now {money(locked, signed=True)}"
        if locked >= -0.005
        else f"Risk now {money(-locked)}"
    )
    return f"🛡 STOP MOVED · {tag(trade)}\n{px(old)} → {px(new)}\n{why}\n{risk_line}"


def cancel_text(trade: dict, reason: str) -> str:
    return f"✖️ ENTRY CANCELLED · {tag(trade)}\nReason: {reason}\nNo position was opened."


def close_text(trade: dict) -> str:
    pnl = trade.get("pnl", 0.0)
    icon = "🏆" if pnl > 0 else "🔻" if pnl < 0 else "➖"
    r = trade.get("r")
    r_text = f" ({'+' if r >= 0 else '−'}{abs(r):.2f}R)" if isinstance(r, (int, float)) else ""
    note = "" if trade.get("pnl_complete", True) else "\nFill history unavailable; result may be incomplete."
    return (
        f"{icon} CLOSED · {tag(trade)}\n"
        f"{trade.get('exit', 'Closed')}\n"
        f"Result: {money(pnl, signed=True)}{r_text}\n"
        f"Entry {px(trade.get('avg_entry') or trade.get('entry'))} · held "
        f"{held_for(trade.get('opened_at') or trade.get('created_at'), trade.get('closed_at'))}"
        + note
    )


# ---------------------------------------------------------------- the panel

HELP = (
    "🏦 Apex Desk · commands\n\n"
    "/dashboard — home: account, P&L, positions\n"
    "/signals — Shadow Book signals and what the desk did with each\n"
    "/positions — open positions and pending entries\n"
    "/performance — live results\n\n"
    "Risk\n"
    "/risk 50 — risk $50 per trade\n"
    "/risk 1% — risk 1% of equity per trade\n"
    "/maxpos 6 — max open positions\n"
    "/dailystop 200 — stop new trades after $200 lost today (/dailystop off)\n\n"
    "Leverage\n"
    "/lev 5 — max leverage for all symbols\n"
    "/lev SOL 10 — fixed leverage for one symbol\n"
    "/lev SOL auto — remove a symbol setting\n\n"
    "Filters\n"
    "/direction both|long|short\n"
    "/block DOGE · /unblock DOGE\n\n"
    "Control\n"
    "/pause · /resume — stop or restart new entries\n"
    "/close SOL — close one position (asks to confirm)\n"
    "/closeall — close everything (asks to confirm)\n\n"
    "Signal engine views: /opportunities /waves /why /status /system\n\n"
    "Leverage changes margin, not dollars at risk. Risk is set by the stop distance "
    "and your risk amount. The desk always keeps liquidation well beyond the stop."
)

DESK_COMMANDS = {
    "dashboard": "Home: account, P&L, positions and controls",
    "signals": "Shadow Book signals and what the desk did with each",
    "positions": "Open positions and pending entries",
    "performance": "Live trading results",
    "risk": "Risk per trade: /risk 50 or /risk 1%",
    "lev": "Leverage: /lev 5 or /lev SOL 10",
    "maxpos": "Max open positions: /maxpos 6",
    "dailystop": "Daily loss stop: /dailystop 200 or off",
    "direction": "Trade both, long or short",
    "block": "Block a symbol: /block DOGE",
    "unblock": "Unblock a symbol",
    "pause": "Stop taking new signals",
    "resume": "Start taking signals again",
    "close": "Close one position: /close SOL",
    "closeall": "Close all positions and entries",
    "alerts": "Notification settings",
    "opportunities": "Signal engine: current setups",
    "status": "Signal engine status",
    "help": "All commands",
}


def _button(label, data):
    return {"text": label, "callback_data": data}


def _rows(buttons, width=2):
    return [buttons[i : i + width] for i in range(0, len(buttons), width)]


def _nav(*extra):
    row = [_button("🔄 Refresh", extra[0] if extra else "d:home"), _button("🏠 Home", "d:home")]
    return row


class DeskPanel:
    def __init__(self, store, desk, *, clock=time.time):
        self.store = store
        self.desk = desk
        self.clock = clock
        self._confirm: dict = {}

    # -------------------------------------------------------------- state
    async def _state(self):
        state = await self.store.read()
        desk = desk_state(state)
        return state, desk, normalize_settings(desk.get("settings"))

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
        """User input (SOL, sol, SOLUSDT, 1000PEPE, kPEPE) -> Hyperliquid coin."""
        assets = self.desk.assets or {}
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
            return HELP, {"inline_keyboard": [_nav("d:help")]}
        if view == "sig":
            return await self.signals()
        return await self.home()

    def _connection_lines(self):
        snap = self.desk.snapshot if self.desk else {"status": "no_keys"}
        if not self.desk or not self.desk.configured or snap.get("status") == "no_keys":
            return None, [
                "🔌 Hyperliquid not connected",
                "Add HL_ACCOUNT_ADDRESS and HL_API_PRIVATE_KEY in Railway Variables, then redeploy.",
            ]
        if snap.get("status") != "live":
            return snap, ["🔌 Connecting to Hyperliquid…"]
        age = self.clock() - snap.get("at", 0)
        stale = " (data stale)" if age > 60 else ""
        return snap, [f"Updated {when(snap.get('at'))}{stale}"]

    async def home(self):
        state, desk, s = await self._state()
        snap, conn = self._connection_lines()
        trades = desk["trades"].values()
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        pending = [t for t in trades if t.get("status") in ("ORDERED", "PLACING")]
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
        status = "⏸ PAUSED — no new entries" if s["paused"] else "🟢 LIVE — taking signals"
        lines = ["🏦 APEX DESK · Hyperliquid", status, ""]
        if snap and snap.get("status") == "live":
            upnl = sum(p.get("upnl", 0.0) for p in snap.get("positions", {}).values())
            open_risk = sum(self._live_risk(t, snap) for t in open_trades)
            lines += [
                f"💰 Equity {money(snap.get('equity'))} · free {money(snap.get('available'))}",
                f"📅 Today {money(realized, signed=True)} realized ({wins}W {losses}L) · "
                f"open P&L {money(upnl, signed=True)}",
                f"📂 {len(open_trades)} open · {len(pending)} pending · open risk {money(open_risk)}",
            ]
        lines += [
            "",
            f"🛡 Risk {self._risk_text(s, snap)} · max {s['max_positions']} positions",
            f"⚙️ Leverage max {s['max_leverage']}x"
            + (f" · {len(s['symbol_leverage'])} symbol settings" if s["symbol_leverage"] else ""),
            f"🧯 Daily stop {money(s['daily_loss_usd']) if s['daily_loss_usd'] else 'off'} · {s['direction']}",
            f"📡 Signals: {BOOKS[s['book']]}"
            + (f" · {LAYERS[s['layer']]}" if s["book"] == "shadow_book" else "")
            + (f"\n   {self.desk.source_status(s)}" if self.desk else ""),
            "",
            *conn,
        ]
        setup = self._setup_lines(s)
        if setup:
            lines += ["", "🧩 Setup", *setup]
        buttons = [
            _button("📡 Signals", "d:sig"),
            _button("📂 Positions", "d:pos"),
            _button("📈 Performance", "d:perf"),
            _button("🛡 Risk", "d:risk"),
            _button("⚙️ Leverage", "d:lev"),
            _button("🎯 Filters", "d:flt"),
            _button("🔔 Alerts", "d:alr"),
            _button("▶️ Resume" if s["paused"] else "⏸ Pause", "d:resume" if s["paused"] else "d:pause"),
            _button("🚨 Close all", "d:ca"),
            _button("❓ Help", "d:help"),
            _button("🔬 Signal engine", "pg:dashboard:0"),
        ]
        keyboard = _rows(buttons) + [_nav("d:home")]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    def _setup_lines(self, s):
        """A checklist only while something still needs doing."""
        desk = self.desk
        hl_ok = bool(desk and desk.configured)
        live = hl_ok and desk.snapshot.get("status") == "live"
        if s["book"] == "shadow_book":
            source = desk.source if desk else None
            src_ok = bool(source and source.configured and source.ok)
            src_line = (
                "✅ Shadow Book connected"
                if src_ok
                else "❌ Shadow Book link — add APEX_SHADOW_BOOK_CSV_URL in Railway"
                if not (source and source.configured)
                else f"⚠️ Shadow Book: {source.error or 'connecting…'}"
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

    async def signals(self):
        state, desk, s = await self._state()
        trades = desk["trades"]
        lines = ["📡 Signals"]
        if s["book"] != "shadow_book":
            engine = [
                t for t in state.get("trades", {}).values()
                if t.get("arm") == s["book"] and t.get("status") in ("PENDING", "OPEN")
            ]
            lines += [f"Source: {BOOKS[s['book']]} · {len(engine)} active", ""]
            rows = sorted(engine, key=lambda t: t.get("created_at", 0), reverse=True)[:10]
        else:
            source = self.desk.source if self.desk else None
            if not source or not source.configured:
                lines += ["", "Shadow Book not connected yet.", "Add APEX_SHADOW_BOOK_CSV_URL in Railway Variables."]
                return "\n".join(lines), {"inline_keyboard": [_nav("d:sig")]}
            status = f"read {when(getattr(source, 'fetched_at', 0))}" if source.ok else f"unavailable: {source.error}"
            lines += [f"Source: Apex Kapital Shadow Book · {LAYERS[s['layer']]} · {status}", ""]
            wanted = (lambda sig: True) if s["layer"] == "all" else (lambda sig: sig.get("layer") == s["layer"].upper())
            rows = sorted(
                (x for x in source.signals.values() if x.get("status") in ("PENDING", "OPEN", "TP1") and wanted(x)),
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
                    else "— started before the desk saw it (not taken)"
                )
            else:
                state_text = {
                    "PLACING": "⏳ placing order",
                    "ORDERED": "📍 live limit order",
                    "OPEN": "✅ in position" + (" · TP1 done" if mine.get("tp1_done") else ""),
                    "CLOSED": f"🏁 closed {money(mine.get('pnl'), signed=True)}",
                    "CANCELLED": "✖️ entry cancelled",
                    "SKIPPED": f"⏭ skipped: {mine.get('reason', '')}",
                    "ERROR": f"⚠️ {mine.get('reason', 'error')}",
                }.get(mine.get("status"), mine.get("status", ""))
            word = "LONG" if sig.get("side") == "Buy" else "SHORT"
            label = sig.get("source_id") or short_id(sig["id"])
            lines += [
                f"{'🟢' if word == 'LONG' else '🔴'} {word} {sig.get('coin') or sig.get('symbol')} · {label} · sheet {sig.get('status', '').lower()}",
                f"Entry {px(sig.get('limit'))} · stop {px(sig.get('stop'))} · TP {px(sig.get('target1'))} / {px(sig.get('target2'))}",
                f"Desk: {state_text}",
                "",
            ]
        return "\n".join(lines).rstrip(), {"inline_keyboard": [[_button("🎯 Change source", "d:flt")], _nav("d:sig")]}

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
        snap = self.desk.snapshot if self.desk else {}
        mids = snap.get("mids") or {}
        positions = snap.get("positions") or {}
        active = sorted(
            (t for t in desk["trades"].values() if t.get("status") in ACTIVE),
            key=lambda t: t.get("created_at", 0),
        )
        open_trades = [t for t in active if t["status"] == "OPEN"]
        pending = [t for t in active if t["status"] != "OPEN"]
        lines = [f"📂 Positions · {len(open_trades)} open · {len(pending)} pending", ""]
        buttons = []
        for t in open_trades:
            pos = positions.get(t["coin"], {})
            entry = pos.get("entry_px") or t.get("avg_entry") or t["entry"]
            mid = mids.get(t["coin"])
            upnl = pos.get("upnl")
            risk = t.get("risk_usd") or 0
            r_now = f" ({upnl / risk:+.2f}R)" if isinstance(upnl, (int, float)) and risk else ""
            lines += [
                f"{side_dot(t)} {tag(t)} · {t['leverage']}x",
                f"Entry {px(entry)} → now {px(mid)} ({pct(entry, mid) if mid else '—'})",
                f"P&L {money(upnl, signed=True)}{r_now} · size {px(abs(pos.get('size', t.get('filled', 0))))}",
                f"🛑 {px(t['stop'])} · 🎯 "
                + ("✅ " if t.get("tp1_done") else "")
                + f"{px(t['target1'])} / {px(t['target2'])}"
                + (f" · liq {px(pos.get('liq_px'))}" if pos.get("liq_px") else ""),
                "",
            ]
            buttons.append(_button(f"✖️ Close {t['coin']}", f"d:cl:{short_id(t['id'])}"))
        for t in pending:
            lines += [
                f"⏳ {tag(t)} · limit {px(t['entry'])} · stop {px(t['stop'])}",
                f"Until {when(t.get('expires_at'))} · risk {money(t.get('risk_usd'))}",
                "",
            ]
            buttons.append(_button(f"✖️ Cancel {t['coin']}", f"d:cl:{short_id(t['id'])}"))
        foreign = [c for c in positions if c not in {t["coin"] for t in active}]
        if foreign:
            lines.append("Not managed by the desk: " + ", ".join(sorted(foreign)))
        if not active:
            lines.append("No open positions or pending entries.")
        keyboard = _rows(buttons[:12]) + [_nav("d:pos")]
        return "\n".join(lines).rstrip(), {"inline_keyboard": keyboard}

    async def risk(self):
        state, desk, s = await self._state()
        snap = self.desk.snapshot if self.desk else {}
        equity = snap.get("equity") or 0
        mode = "fixed dollars" if s["risk_mode"] == "usd" else "% of equity"
        lines = [
            "🛡 Risk settings",
            "",
            f"Per trade: {self._risk_text(s, snap)} ({mode})",
            f"Max open positions: {s['max_positions']} → worst case "
            f"{money(risk_dollars(s, equity) * s['max_positions'])} if every stop hits",
            f"Daily loss stop: {money(s['daily_loss_usd']) if s['daily_loss_usd'] else 'off'} "
            "(no new entries for the rest of the UTC day)",
            "",
            "Tap to change, or type /risk 75 · /risk 1.5% · /maxpos 8 · /dailystop 300",
            "Stops are exchange stop-market orders: a fast move or gap can lose a little more than the set risk.",
        ]
        ru = [_button(("✅ " if s["risk_mode"] == "usd" and s["risk_usd"] == v else "") + f"${v}", f"d:ru:{v}") for v in (25, 50, 100, 200)]
        rp = [_button(("✅ " if s["risk_mode"] == "pct" and s["risk_pct"] == v else "") + f"{v:g}%", f"d:rp:{v:g}") for v in (0.5, 1, 2)]
        mp = [_button(("✅ " if s["max_positions"] == v else "") + f"{v} pos", f"d:mp:{v}") for v in (3, 6, 10)]
        dl = [_button(("✅ " if s["daily_loss_usd"] == v else "") + (f"Stop ${v}" if v else "Stop off"), f"d:dl:{v}") for v in (100, 200, 500, 0)]
        keyboard = [ru, rp, mp, dl[:2], dl[2:], _nav("d:risk")]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def leverage(self):
        state, desk, s = await self._state()
        lines = [
            "⚙️ Leverage",
            "",
            f"Max for all symbols: {s['max_leverage']}x (isolated margin)",
        ]
        if s["symbol_leverage"]:
            lines.append("Symbol settings: " + ", ".join(f"{c} {v}x" for c, v in sorted(s["symbol_leverage"].items())))
        else:
            lines.append("Symbol settings: none")
        lines += [
            "",
            "Each trade uses the symbol setting if you set one, otherwise the max above, "
            "then the lower of the exchange maximum and the liquidation-safe level "
            "(liquidation kept at least 1.5× the stop distance away).",
            "Leverage changes the margin a trade uses, not the dollars at risk.",
            "",
            "Examples: stop 2% away allows up to "
            f"{liquidation_safe_leverage(0.02, 20)}x · 5% away up to {liquidation_safe_leverage(0.05, 20)}x · "
            f"10% away up to {liquidation_safe_leverage(0.10, 20)}x (on a 20x market).",
            "",
            "Type /lev SOL 10 for one symbol, /lev SOL auto to clear it.",
        ]
        presets = [_button(("✅ " if s["max_leverage"] == v else "") + f"{v}x", f"d:ml:{v}") for v in (1, 2, 3, 5, 10, 20)]
        keyboard = _rows(presets, 3) + [[_button(("✅ " if s["max_leverage"] == 50 else "") + "Exchange max", "d:ml:50")]]
        clears = [_button(f"✖️ {c}", f"d:lc:{c}") for c in sorted(s["symbol_leverage"])[:6]]
        keyboard += _rows(clears, 3) + [_nav("d:lev")]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def filters(self):
        state, desk, s = await self._state()
        lines = [
            "🎯 Signal filters",
            "",
            f"Direction: {s['direction']}",
            f"Signal source: {BOOKS[s['book']]}"
            + (f" · {LAYERS[s['layer']]}" if s["book"] == "shadow_book" else ""),
            "Blocked: " + (", ".join(s["blocked"]) if s["blocked"] else "none"),
            "",
            "Shadow Book = the Claude Shadow Desk's trades in your Google Sheet. SIGNAL layer = the desk's signal trades; BOOK layer = the capped book it tracks for its equity curve.",
            "Railway engine = the bot's own automated Elliott scanner on Bybit data.",
            "",
            "Type /block DOGE or /unblock DOGE.",
        ]
        dirs = [_button(("✅ " if s["direction"] == d else "") + d.capitalize(), f"d:dir:{d}") for d in ("both", "long", "short")]
        books = [
            _button(("✅ " if s["book"] == "shadow_book" else "") + "Shadow Book", "d:book:s"),
            _button(("✅ " if s["book"] == "baseline_shadow" else "") + "Engine rules", "d:book:b"),
            _button(("✅ " if s["book"] == "ai_shadow" else "") + "Engine AI", "d:book:a"),
        ]
        layers = [
            _button(("✅ " if s["layer"] == v else "") + LAYERS[v], f"d:lay:{v}")
            for v in ("signal", "book", "all")
        ]
        unblock = [_button(f"✖️ {c}", f"d:ub:{c}") for c in s["blocked"][:6]]
        keyboard = [dirs, books, layers] + _rows(unblock, 3) + [_nav("d:flt")]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def alerts(self):
        state, desk, s = await self._state()
        lines = [
            "🔔 Alerts",
            "",
            "Always sent: live orders, fills, stop moves, closes, skipped signals and faults.",
            f"Watchlist alerts (engine setups WAIT/READY/invalidated, AI reviews, universe changes): {s['watchlist_alerts']}",
            f"Shadow simulation alerts: {s['shadow_alerts']}",
            "",
            "auto = on only while Hyperliquid is not connected, so live trading stays uncluttered.",
        ]
        keyboard = [
            [_button(("✅ " if s["watchlist_alerts"] == v else "") + f"Watchlist {v}", f"d:wl:{v}") for v in ("auto", "on", "off")],
            [_button(("✅ " if s["shadow_alerts"] == v else "") + f"Shadow {v}", f"d:sa:{v}") for v in ("auto", "on", "off")],
            _nav("d:alr"),
        ]
        return "\n".join(lines), {"inline_keyboard": keyboard}

    async def performance(self):
        state, desk, s = await self._state()
        closed = [t for t in desk["trades"].values() if t.get("status") == "CLOSED"]
        now = self.clock()

        def block(title, rows):
            if not rows:
                return [f"{title}: no closed trades"]
            wins = [t for t in rows if t.get("pnl", 0) > 0]
            losses = [t for t in rows if t.get("pnl", 0) < 0]
            net = sum(t.get("pnl", 0) for t in rows)
            rs = [t["r"] for t in rows if isinstance(t.get("r"), (int, float))]
            gross_win = sum(t["pnl"] for t in wins)
            gross_loss = -sum(t["pnl"] for t in losses)
            pf = f"{gross_win / gross_loss:.2f}" if gross_loss > 0 else "∞" if gross_win > 0 else "—"
            return [
                f"{title}: {money(net, signed=True)} · {len(rows)} trades · "
                f"{len(wins)}W {len(losses)}L ({len(wins) / len(rows):.0%})",
                f"   avg {sum(rs) / len(rs):+.2f}R · profit factor {pf}" if rs else "",
            ]

        day = [t for t in closed if now - t.get("closed_at", 0) <= 86400]
        week = [t for t in closed if now - t.get("closed_at", 0) <= 7 * 86400]
        lines = ["📈 Live performance · Hyperliquid", ""]
        lines += block("24h", day)
        lines += block("7 days", week)
        lines += block("All time", closed)
        if closed:
            best = max(closed, key=lambda t: t.get("pnl", 0))
            worst = min(closed, key=lambda t: t.get("pnl", 0))
            lines += ["", f"Best {tag(best)} {money(best.get('pnl'), signed=True)}", f"Worst {tag(worst)} {money(worst.get('pnl'), signed=True)}"]
            lines += ["", "Recent:"]
            for t in sorted(closed, key=lambda t: t.get("closed_at", 0), reverse=True)[:8]:
                lines.append(f"{'🏆' if t.get('pnl', 0) > 0 else '🔻'} {tag(t)} {money(t.get('pnl'), signed=True)} · {t.get('exit', '')}")
        skipped = sum(1 for t in desk["trades"].values() if t.get("status") == "SKIPPED")
        lines += ["", f"Signals not taken: {skipped} · shadow results: Signal engine → Performance"]
        keyboard = [[_button("🔬 Shadow performance", "pg:performance:0")], _nav("d:perf")]
        return "\n".join(x for x in lines if x is not None), {"inline_keyboard": keyboard}

    # ----------------------------------------------------------- actions
    def _fresh(self, message, seconds):
        if message is None:
            return True
        stamp = message.get("edit_date") or message.get("date")
        return isinstance(stamp, int) and -5 <= self.clock() - stamp <= seconds

    async def callback(self, data: str, actor: str, message: dict | None = None):
        parts = data.split(":")
        if len(parts) == 2 and parts[1] in ("home", "pos", "risk", "lev", "flt", "alr", "perf", "help", "sig"):
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
        if action == "lc" and arg:
            await self._change(lambda s: s["symbol_leverage"].pop(arg, None), f"clear_leverage={arg}")
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
            return await self.filters()
        if action == "ub" and arg:
            await self._change(lambda s: s.update(blocked=[c for c in s["blocked"] if c != arg]), f"unblock={arg}")
            return await self.filters()
        if action == "wl" and arg in ("auto", "on", "off"):
            await self._change(lambda s: s.update(watchlist_alerts=arg), f"watchlist={arg}")
            return await self.alerts()
        if action == "sa" and arg in ("auto", "on", "off"):
            await self._change(lambda s: s.update(shadow_alerts=arg), f"shadow_alerts={arg}")
            return await self.alerts()
        if action == "cl" and arg:
            return await self._ask_close(arg, actor)
        if action == "ca":
            return self._ask("caok", "", actor, "🚨 Close ALL desk positions at market and cancel all pending entries?")
        return await self.home()

    def _ask(self, action, arg, actor, question):
        nonce = secrets.token_hex(4)
        self._confirm[nonce] = (action, arg, actor, self.clock() + CONFIRM_SECONDS)
        keyboard = [[_button("✅ Yes, do it", f"d:{action}:{nonce}"), _button("✖️ No", "d:x")]]
        return question + f"\n\nConfirm within {CONFIRM_SECONDS // 60} minutes.", {"inline_keyboard": keyboard}

    async def _ask_close(self, sid, actor):
        state, desk, s = await self._state()
        trade = next((t for t in desk["trades"].values() if short_id(t["id"]) == sid and t.get("status") in ACTIVE), None)
        if not trade:
            text, keyboard = await self.positions()
            return "That position is no longer active.\n\n" + text, keyboard
        verb = "Close" if trade["status"] == "OPEN" else "Cancel the pending entry for"
        return self._ask("clok", f"{sid}", actor, f"{verb} {tag(trade)} at market?")

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
        return result + "\n\n" + text, keyboard

    async def command(self, command: str, args: list[str], actor: str):
        """Return (text, keyboard) for desk commands, or None to fall through."""
        if command in ("start", "dashboard", "desk", "settings"):
            return await self.home()
        if command == "positions":
            return await self.positions()
        if command == "signals":
            return await self.signals()
        if command == "performance":
            return await self.performance()
        if command == "alerts":
            return await self.alerts()
        if command == "help":
            return HELP, {"inline_keyboard": [_nav("d:help")]}
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
                try:
                    number = int(args[0].lower().rstrip("x"))
                except ValueError:
                    return self._error("Use /lev 5 or /lev SOL 10.", "d:lev")
                if not 1 <= number <= 50:
                    return self._error("Leverage must be 1–50.", "d:lev")
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
        if command == "close":
            if len(args) != 1:
                return self._error("Use /close SOL.", "d:pos")
            coin = self._coin(args[0])
            state, desk, s = await self._state()
            trade = next((t for t in desk["trades"].values() if t.get("coin") == coin and t.get("status") in ACTIVE), None)
            if not trade:
                return self._error(f"No active desk position on {args[0].upper()}.", "d:pos")
            return await self._ask_close(short_id(trade["id"]), actor)
        if command == "closeall":
            return self._ask("caok", "", actor, "🚨 Close ALL desk positions at market and cancel all pending entries?")
        return None

    @staticmethod
    def _error(text, back):
        return "⚠️ " + text, {"inline_keyboard": [[_button("← Back", back), _button("🏠 Home", "d:home")]]}


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
    "signal_card",
    "skip_text",
    "fill_text",
    "stop_text",
    "cancel_text",
    "close_text",
    "shadow_alerts_enabled",
    "watchlist_alerts_enabled",
    "DEFAULT_SETTINGS",
]
