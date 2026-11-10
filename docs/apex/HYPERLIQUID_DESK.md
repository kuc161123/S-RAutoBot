# Apex Desk — live Hyperliquid trading

Added 2026-11-10. The desk turns the Apex shadow signals into live Hyperliquid
perpetual trades and gives Telegram a full control panel.

## How it fits together

```
Claude Shadow Desk (3 shifts/day) ─► Apex Kapital Shadow Book (Google Sheet, Trades tab)
                                                     │  shadow_book.py (every 60 s)
                                                     ▼
                                   desk.py  ── Hyperliquid (hl_client.py; crypto + xyz:)
                                     │
                                     ▼
                       Telegram (desk_panel.py via telegram.py)
```

Default signal source: the **Shadow Book**, BOOK layer (the capped book whose
equity curve is reported). Filters can add the SIGNAL layer, or switch to the
Railway engine's own books (`baseline_shadow` / `ai_shadow`).

* `apex_bot/hl_client.py` — Hyperliquid SDK wrapper: price/size precision,
  client order IDs, Bybit→Hyperliquid symbol mapping, sanitized errors.
* `apex_bot/shadow_book.py` — reads the Sheet by header name (Sheets API with a
  service account, or a published CSV link); PENDING rows become signals; status
  changes (EXPIRED, TP1, CLOSED + exit reason) drive management.
* `apex_bot/desk.py` — the live book. Every new `PENDING` signal in the followed
  source becomes a live order.
* `apex_bot/desk_panel.py` — Telegram home screen, settings, trade messages.
* `runtime.py` runs `desk.tick` every 4 seconds; `telegram.py` routes `/start`,
  `/dashboard` and every `d:*` button to the panel when it is attached.

The engine's levels are used unchanged: limit entry, stop, target 1, target 2.

## Order lifecycle

1. **Intake.** Followed-book PENDING trade, not expiring within 60 s. Filters:
   listed on Hyperliquid, not blocked, direction allowed, no existing position
   or order on that coin, max positions, daily loss stop, price within 5% of
   the signal entry, price not already through the stop or target 1.
2. **Sizing.** `size = risk$ / (|entry − stop| + fee_rate × (entry + stop))`,
   rounded down. Risk is fixed dollars or % of equity.
3. **Leverage.** `min(symbol setting or max leverage, exchange max,
   liquidation-safe)`; liquidation-safe keeps the isolated liquidation distance
   ≥ 1.5× the stop distance. Isolated margin, set before every entry.
4. **Entry.** GTC limit at the signal entry. Cancelled at the signal's expiry,
   when the engine closes the signal, or if price crosses the stop or target 1
   first.
5. **Protection.** As soon as any size is filled, a reduce-only stop-market
   trigger covers the whole position. Every tick re-checks it and re-places it
   if missing; two failures in a row close the position at market.
6. **Targets.** Target 1 closes half (one lot stays intact, odd lots round up —
   same as the shadow book; skipped if too small for the $10 minimum). Target 2
   closes the rest.
7. **Management.** Target 1 hit (or the Sheet/engine reporting it) → stop to
   entry. Engine trailing stop moves → live stop follows, tighten only. Sheet row
   CLOSED with a non-price reason (the desk closed it), or an engine structural
   exit → market close. Sheet EXPIRED/CLOSED before a fill → entry cancelled.
   If the Sheet cannot be read, no new entries are taken and exchange orders keep
   protecting open positions.
8. **Close.** Realized P&L = Σ closedPnl − Σ fees from Hyperliquid fills since
   the order was placed; R = P&L ÷ planned risk.

State lives in `apex_state` under `desk` and is written right after every
exchange action. A trade saved as `PLACING` before a crash is resolved by its
client order ID on restart.

## Telegram

Home: equity, free margin, today's realized P&L, open P&L, positions, pending
entries, open risk, settings summary. Buttons: Positions, Performance, Risk,
Leverage, Filters, Alerts, Pause/Resume, Close all, Signal engine, Help.

Commands: `/risk 50`, `/risk 1%`, `/lev 5`, `/lev SOL 10`, `/lev SOL auto`,
`/maxpos 6`, `/dailystop 200|off`, `/direction both|long|short`,
`/block X`, `/unblock X`, `/pause`, `/resume`, `/close X`, `/closeall`.
Typed changes must be fresh (≤120 s); settings buttons older than 15 minutes
are refused; closing needs a second tap within 2 minutes by the same user.

Alerts: live trade messages always. Watchlist (WAIT/READY setups, AI reviews)
and shadow-simulation messages are `auto` (on only while Hyperliquid is not
connected), `on` or `off`.

## Defaults

risk $50 per trade · max leverage 5x · max 6 positions · daily loss stop $200 ·
both directions · Shadow Book BOOK layer · fee estimate 0.045% per side.

## Railway variables

```
HL_ACCOUNT_ADDRESS        main Hyperliquid wallet address (0x + 40 hex)
HL_API_PRIVATE_KEY        API wallet private key (app.hyperliquid.xyz → More → API)
HL_PERP_DEXS              builder markets to trade besides crypto (default xyz)
APEX_SHADOW_BOOK_CSV_URL  "Publish to the web" CSV link of the Trades tab, OR
APEX_SHADOW_BOOK_SHEET_ID + APEX_GOOGLE_SERVICE_ACCOUNT_JSON (Sheet shared as Viewer)
```

For xyz markets the desk asks Hyperliquid once to let them use the main perp
USDC balance (dex abstraction); the "Desk online" message reports the result.

Both present = live trading. Both absent = the desk shows "not connected" and
the engine keeps running in shadow. The Bybit path and `release.py` are
unchanged; `APEX_MODE` stays `shadow`.

## Known limits

* Stops are stop-market triggers: gaps and fast moves can lose more than the
  set risk. Several positions can stop out together.
* Prices come from Bybit data; Hyperliquid prices differ slightly.
* The desk does not manage positions it did not open, and will not trade a
  coin where such a position exists.
