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

Default signal source: the **Shadow Book, SIGNAL layer** (+$533.51 over 17
closed trades to 10 Nov 2026; the BOOK layer made +$203.36 over 23). Filters can
switch to BOOK or both, or to the Railway engine's own books.

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
   ≥ 1.5× the stop distance. Isolated margin, set before every entry. Leverage
   only sets margin: dollars at risk come from the stop. With isolated margin the
   higher, still liquidation-safe level also bounds a gap-through-the-stop loss
   at about 1.5R, where a low leverage would put the whole larger margin at risk.
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

## Desk-only mode (default once Hyperliquid keys are set)

`APEX_DESK_ONLY=auto` (default) turns the bot into the desk and nothing else as
soon as `HL_ACCOUNT_ADDRESS`/`HL_API_PRIVATE_KEY` are present and `APEX_MODE` is
`shadow`. `on` forces it, `off` keeps the old engine running beside the desk.

In desk-only mode:

* only the lease, Telegram, desk, dashboard and outbox loops run — no Bybit
  scans, AI reviews, shadow books, universe rotation or engine alerts;
* the only signal source is the Shadow Book (engine books are hidden);
* engine messages still queued from before are dropped, never sent;
* old engine commands and buttons open the desk home instead;
* the bot renames itself **Apex Desk**, rewrites its description and replaces
  its command menus (default, private-chat and owner-chat scopes);
* each boot posts a fresh dashboard and pins it; it is edited in place every
  30 s while showing home, and returns home after 5 idle minutes on another
  screen;
* `/readyz` follows the desk loop instead of the engine scan.

## Telegram

Messages and screens are Telegram HTML (anything Telegram cannot parse is
resent as plain text).

Home: status, equity, free margin, today's realized P&L, open P&L, every open
and pending trade, open risk, rules summary, signal-source health. Buttons:
Signals, Positions, Performance, Status, Risk, Leverage, Filters, Alerts,
Pause/Resume, Close all, Refresh, Help.

Screens: Signals (with SIGNAL/BOOK/BOTH switch), Positions (per trade:
stop→entry when in profit, close/cancel), Performance, Status (connections,
counts, recent order errors), Risk (presets + how many more trades fit in free
margin), Leverage, Filters, Alerts (skipped-signal messages on/off).

Commands: `/risk 50`, `/risk 1%`, `/lev auto`, `/lev 10`, `/lev SOL 5`,
`/lev SOL auto`, `/maxpos 6`, `/dailystop 200|off`,
`/direction both|long|short`, `/block X`, `/unblock X`, `/pause`, `/resume`,
`/be X`, `/close X`, `/closeall`, `/status`.
Typed changes must be fresh (≤120 s); settings buttons older than 15 minutes
are refused; closing needs a second tap within 2 minutes by the same user.

Alerts: live trade messages always; skipped-signal messages on/off. With the
engine running, watchlist and shadow-simulation messages are `auto` (on only
while Hyperliquid is not connected), `on` or `off`.

## Defaults

risk $50 per trade · leverage Auto (the liquidation-safe level, capped by the exchange
maximum; settings saved with the old 10x default migrate to Auto once) · max 6 positions · daily loss stop $200 ·
both directions · Shadow Book SIGNAL layer · fee estimate 0.045% per side.

## Railway variables

```
HL_ACCOUNT_ADDRESS        main Hyperliquid wallet address (0x + 40 hex)
HL_API_PRIVATE_KEY        API wallet private key (app.hyperliquid.xyz → More → API)
HL_PERP_DEXS              builder markets to trade besides crypto (default xyz)
APEX_SHADOW_BOOK_CSV_URL  1–3 CSV links of the Trades tab, space separated, tried in order:
                          the live export (…/d/<id>/export?format=csv&gid=<Trades gid>,
                          needs "Anyone with the link: Viewer") first — no delay — then
                          the "Publish to the web" link as fallback (Google refreshes it
                          on its own schedule; on 10 Nov that took hours), OR
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
