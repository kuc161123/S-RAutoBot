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
switch to BOOK or both, or to the Railway engine's own books. `/layer
signal|book|both` sets it by command. BOTH means SIGNAL + BOOK only: a `TEST`
row (paper only) is never traded whatever the setting, and BotRules applies to
every layer (17 Nov 2026: BOOK Short B/C had closed 19 trades, 68% won, +0.51R
each; the other BOOK setups 15 trades, -0.27R each). Since playbook v1.6 every
BOOK pick is also written as a SIGNAL row with its own ID and the same numbers;
the desk places the first of such a pair and records the copy as handled
without a Telegram message ("Same trade as S-…"). A copy = same market, side,
entry, stop, TP1 and TP2, opened within an hour of the other row.

**BotRules** (since 17 Nov 2026, Lual: the bot trades only the best setups). The
Shadow Book's `BotRules` tab lists which Side + Setup letters the bot may trade:
row 1 = headers (found by name: `Active`, `Side`, `Setup`, `Rules version`;
`Rule ID` and `Setup name` are shown when present), rule rows from row 2 to the
first blank row. It is read with Trades on every poll (the Trades links with the
BotRules tab id `gid=423805298`, override `APEX_SHADOW_BOOK_RULES_GID`; Google's
by-name CSV as fallback). A Shadow Book row is placed only if an `Active = TRUE`
rule has the same Side and the same Setup letter (the first character of
`Setup (EW count)`); otherwise it is skipped and recorded as handled. If
BotRules can't be read or has no rule rows, nothing new is placed that cycle
(no cached rules); after 10 minutes Telegram says so once. Rules apply to new
signals only — a rule change never closes or cancels an order or position.
When `Rules version` changes, Telegram lists the active setups; `/rules` shows
the current list.

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
   an active BotRules row for its Side + Setup letter (Shadow Book only),
   listed on Hyperliquid, not blocked, direction allowed, no existing position
   or order on that coin, max positions, daily loss stop, price within 5% of
   the signal entry, price not already through the stop or target 1.
2. **Sizing.** `size = risk$ / (|entry − stop| + fee_rate × (entry + stop))`,
   rounded down. Risk is fixed dollars or % of equity. On a unified account
   equity is the spot USDC total, which already carries open P&L (it equals
   Hyperliquid's own account value).
3. **Leverage.** `min(symbol setting or max leverage, exchange max,
   liquidation-safe)`; liquidation-safe keeps the isolated liquidation distance
   ≥ 1.5× the stop distance. Isolated margin, set before every entry. Leverage
   only sets margin: dollars at risk come from the stop. With isolated margin the
   higher, still liquidation-safe level also bounds a gap-through-the-stop loss
   at about 1.5R, where a low leverage would put the whole larger margin at risk.
4. **Entry.** GTC limit at the signal entry, sent together with its stop
   (Hyperliquid order grouping `normalTpsl`): the exchange itself places the
   stop the moment the entry is completely filled, so a fill during an outage of
   this process or of the API is protected from the first second. Cancelled at
   the signal's expiry, when the engine closes the signal, or if price crosses
   the stop first (engine signals also on target 1 first; Shadow Book limits
   rest until their entry trades). If the exchange refuses the pair as a whole,
   the entry goes out alone and the stop follows the fill as below.
5. **Protection.** As soon as any size is filled, a reduce-only stop-market
   trigger covers the whole position. On a full fill this is the attached stop,
   adopted by its client order ID (or, failing that, by its exact level, size and
   side). Hyperliquid holds the attached stop back while the entry is only
   partly filled, so the desk places its own stop for the filled part and hands
   over to the attached one when the rest fills (cancelling the unfilled rest
   cancels the attached stop; the desk's stop stays). Every tick re-checks the
   stop and re-places it if missing; two failures in a row close the position at
   market. Exactly one stop is kept on the book.
6. **Targets.** Target 1 closes half (one lot stays intact, odd lots round up —
   same as the shadow book; skipped if too small for the $10 minimum). Target 2
   closes the rest.
7. **Management.** Target 1 hit (or the Sheet/engine reporting it) → stop to
   entry, retried every 30 s until it is there. Engine trailing stop moves → live stop follows, tighten only.
   Sheet row CLOSED for any reason (stop, target 2, break-even or the desk's own
   call) or EXPIRED while a position is still open here → the rest is closed at
   market, so the live book never holds a trade the Shadow Book has closed.
   Hyperliquid triggers stops and targets on the mark price, the Shadow Book on
   traded prices; on a thin book (stock perps at the weekend) they part — 14 Nov
   TSLA traded under its stop at 16:45 UTC, the mark only at 20:54 UTC. Engine
   structural exit → market close. Sheet EXPIRED/CLOSED before a fill → entry
   cancelled. Every Sheet-driven close or cancel waits until the row has read
   CLOSED/EXPIRED for 3 minutes, so a mis-aimed Sheet write fixed within that is
   ignored.
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
`/be X`, `/close X`, `/closeall`, `/status`, `/rules` (BotRules: what the bot may trade),
`/layer signal|book|both`.
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
APEX_SHADOW_BOOK_RULES_GID  optional: the BotRules tab id if the tab is ever recreated
                          (default 423805298; the tab is also found by name as a fallback)
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
