#!/usr/bin/env python3
"""
LIQ ANALYSIS — at what risk-per-trade does margin/liquidation become the
binding constraint, rather than the stop-loss?

Read-only analysis against halt_universe_live.parquet (38,753 signals,
2023-06 -> present, 277-symbol live universe). All output written under
risk_study/agent_out/.
"""
import pandas as pd
import numpy as np
import heapq
import os

OUT = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out"
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet("/Users/lualakol/AutoTrading Bot/halt_universe_live.parquet")
df = df.sort_values("entry_time").reset_index(drop=True)
print(f"Loaded {len(df)} rows, {df.entry_time.min()} -> {df.exit_time.max()}")

# ─────────────────────────────────────────────────────────────────────────
# 1. stop_frac distribution
# ─────────────────────────────────────────────────────────────────────────
df["stop_frac"] = (df["entry_price"] - df["sl_price"]).abs() / df["entry_price"]
df["notional_mult"] = 1.0 / df["stop_frac"]  # notional = risk_usd * notional_mult

pcts = [1, 5, 25, 50, 75, 95, 99]
stop_frac_pcts = df["stop_frac"].describe(percentiles=[p/100 for p in pcts])
notional_mult_pcts = df["notional_mult"].describe(percentiles=[p/100 for p in pcts])

tbl1 = pd.DataFrame({
    "percentile": pcts,
    "stop_frac": [df["stop_frac"].quantile(p/100) for p in pcts],
    "notional_mult (1/stop_frac)": [df["notional_mult"].quantile(p/100) for p in pcts],
})
tbl1.to_csv(f"{OUT}/1_stop_frac_distribution.csv", index=False)
print("\n=== 1. stop_frac & notional multiplier distribution ===")
print(tbl1.to_string(index=False))
print(f"mean stop_frac={df['stop_frac'].mean():.5f}  mean notional_mult={df['notional_mult'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────
# Leverage table — REUSED VERBATIM from backtest_production_correct.py
# (get_leverage / TOP_ALTS_50X), since that's the one place in-repo that
# encodes an explicit per-symbol leverage assumption tied to Bybit risk
# limits. This is NOT fabricated for this study.
# ─────────────────────────────────────────────────────────────────────────
TOP_ALTS_50X = {
    'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT',
    'DOTUSDT', 'MATICUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'APTUSDT',
    'NEARUSDT', 'FILUSDT', 'ARBUSDT', 'OPUSDT', 'MKRUSDT', 'AAVEUSDT',
    'ATOMUSDT', 'XLMUSDT', 'TRXUSDT', 'ICPUSDT', 'SUIUSDT', 'SEIUSDT',
    'TIAUSDT', 'STXUSDT', 'INJUSDT', 'IMXUSDT', 'RUNEUSDT', 'FETUSDT',
}

def get_leverage(symbol: str) -> int:
    if symbol in ('BTCUSDT', 'ETHUSDT'):
        return 100
    if symbol.startswith('1000') or symbol.startswith('10000'):
        return 25
    if symbol in TOP_ALTS_50X:
        return 50
    return 20

df["leverage"] = df["symbol"].apply(get_leverage)

# ─────────────────────────────────────────────────────────────────────────
# 2. Concurrency + notional-multiple timeline
# ─────────────────────────────────────────────────────────────────────────
RISK_LEVELS = [0.003, 0.01, 0.02, 0.03]
GROSS_CAP = 0.30  # config.yaml risk.gross_open_risk_cap

def build_hourly_timeline(trades, risk_col_f=None, cap=None):
    """
    Event-driven: at every entry, optionally gate on gross_open_risk_cap
    (greedy, equity held constant = 1.0). Returns an hourly-resampled
    DataFrame of concurrency, notional_mult_sum (= total_notional/equity
    per unit f), and open_risk_sum/equity.
    Also returns the (possibly filtered) trades actually taken.
    """
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    equity = 1.0
    open_risk_sum = 0.0
    exit_heap = []  # (exit_time, risk_usd)
    taken_idx = []
    events = []  # (time, +-1 concurrency, +- notional_mult*f, +- risk contribution, +- margin contribution)

    for i, row in trades.iterrows():
        et, xt = row.entry_time, row.exit_time
        # release everything that exited by et
        while exit_heap and exit_heap[0][0] <= et:
            _, r = heapq.heappop(exit_heap)
            open_risk_sum -= r
        risk_usd = (risk_col_f if risk_col_f is not None else 1.0) * equity
        if cap is not None:
            budget = cap * equity
            if open_risk_sum + risk_usd > budget:
                continue  # blocked by gross_open_risk_cap
        heapq.heappush(exit_heap, (xt, risk_usd))
        open_risk_sum += risk_usd
        taken_idx.append(i)
        f_ = risk_col_f if risk_col_f else 1.0
        events.append((et, 1, row.notional_mult * f_, risk_usd, (row.notional_mult / row.leverage) * f_))
        events.append((xt, -1, -row.notional_mult * f_, -risk_usd, -(row.notional_mult / row.leverage) * f_))

    taken = trades.loc[taken_idx]
    ev = pd.DataFrame(events, columns=["time", "dconc", "dnotional_mult", "drisk", "dmargin"]).sort_values("time")
    ev = ev.groupby("time", as_index=False).sum()
    ev["concurrency"] = ev["dconc"].cumsum()
    ev["notional_over_equity"] = ev["dnotional_mult"].cumsum()  # = f * (notional/equity), unitless per f=1
    ev["open_risk_over_equity"] = ev["drisk"].cumsum()
    ev["margin_over_equity"] = ev["dmargin"].cumsum()  # = f * (initial-margin/equity), leverage-weighted

    # resample to hourly grid over the full span, forward-fill (state between events)
    full_range = pd.date_range(trades.entry_time.min().floor("h"),
                                trades.exit_time.max().ceil("h"), freq="1h")
    ev = ev.set_index("time")[["concurrency", "notional_over_equity", "open_risk_over_equity", "margin_over_equity"]]
    hourly = ev.reindex(ev.index.union(full_range)).sort_index().ffill().fillna(0.0)
    hourly = hourly.reindex(full_range).ffill().fillna(0.0)
    return hourly, taken

summary_rows = []
notional_series = {}
for f in RISK_LEVELS:
    # uncapped
    hourly_u, taken_u = build_hourly_timeline(df, risk_col_f=f, cap=None)
    # gross_open_risk_cap applied greedily
    hourly_c, taken_c = build_hourly_timeline(df, risk_col_f=f, cap=GROSS_CAP)

    for label, hourly, taken in [("uncapped", hourly_u, taken_u), ("gross_cap_applied", hourly_c, taken_c)]:
        conc = hourly["concurrency"]
        notmult = hourly["notional_over_equity"]  # notional / equity, directly (already scaled by f inside)
        row = {
            "f": f, "scenario": label,
            "n_trades_taken": len(taken), "n_trades_total": len(df),
            "conc_median": conc.median(), "conc_p90": conc.quantile(0.90),
            "conc_p99": conc.quantile(0.99), "conc_max": conc.max(),
            "notional/equity_median": notmult.median(),
            "notional/equity_p90": notmult.quantile(0.90),
            "notional/equity_p99": notmult.quantile(0.99),
            "notional/equity_max": notmult.max(),
        }
        summary_rows.append(row)
    notional_series[f] = {"uncapped": hourly_u, "capped": hourly_c,
                           "taken_u": taken_u, "taken_c": taken_c}

tbl2 = pd.DataFrame(summary_rows)
tbl2.to_csv(f"{OUT}/2_concurrency_notional_summary.csv", index=False)
print("\n=== 2. Concurrency & notional/equity by risk level (uncapped vs gross_open_risk_cap-gated) ===")
print(tbl2.to_string(index=False))

# ─── 2b. Initial margin / equity (leverage-weighted, uses exchange-max leverage
#          table copied from backtest_production_correct.py: BTC/ETH=100x,
#          1000x-prefixed=25x, top-50 alts=50x, other alts=20x) ───
margin_rows = []
for f in RISK_LEVELS:
    for label in ["uncapped", "capped"]:
        hourly = notional_series[f][label]["margin_over_equity"]
        margin_rows.append({
            "f": f, "scenario": label,
            "margin/equity_median": hourly.median(),
            "margin/equity_p90": hourly.quantile(0.90),
            "margin/equity_p99": hourly.quantile(0.99),
            "margin/equity_max": hourly.max(),
        })
tbl2b = pd.DataFrame(margin_rows)
tbl2b.to_csv(f"{OUT}/2b_initial_margin_over_equity.csv", index=False)
print("\n=== 2b. Initial margin/equity (exchange-max leverage per symbol) ===")
print(tbl2b.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────
# 3. Margin stress estimate
# ─────────────────────────────────────────────────────────────────────────
# ASSUMPTION (explicit, conservative, stated): flat maintenance-margin rate
# (MMR) of 1.0% of notional across the book. Bybit's real base-tier MMR is
# roughly 0.4-0.5% for BTC/ETH and typically 1-2% for smaller-cap alts at
# their base tier (higher tiers step up further); this repo's universe is
# ~90% alts, so 1.0% flat is a round, defensible, *conservative-in-the-
# sense-of-detecting-stress-early* single number. We could NOT find any
# Bybit tier table checked into this repo, so this is NOT sourced from the
# codebase -- it is an external, clearly-labelled assumption.
MMR_ASSUMED = 0.01

# We define two stress thresholds on (total_maintenance_margin / equity):
#   YELLOW  > 0.50  -- maintenance margin alone already consumes half of
#                      equity with ZERO adverse price movement; almost no
#                      buffer left for the book to move against you.
#   RED     > 1.00  -- maintenance margin alone exceeds total equity before
#                      any loss has even happened. This is not a "close
#                      call" -- cross-margin liquidation logic means the
#                      account is already under water on margin math alone.
stress_rows = []
for f in RISK_LEVELS:
    hourly_c = notional_series[f]["capped"]["notional_over_equity"]
    hourly_u = notional_series[f]["uncapped"]["notional_over_equity"]
    for label, series in [("uncapped", hourly_u), ("gross_cap_applied", hourly_c)]:
        mm_over_eq = MMR_ASSUMED * series
        frac_yellow = (mm_over_eq > 0.50).mean()
        frac_red = (mm_over_eq > 1.00).mean()
        stress_rows.append({
            "f": f, "scenario": label,
            "MMR_assumed": MMR_ASSUMED,
            "max_MM/equity": mm_over_eq.max(),
            "frac_hours_MM/equity>0.50": frac_yellow,
            "frac_hours_MM/equity>1.00": frac_red,
        })
tbl3 = pd.DataFrame(stress_rows)
tbl3.to_csv(f"{OUT}/3_margin_stress.csv", index=False)
print(f"\n=== 3. Margin stress (MMR assumed = {MMR_ASSUMED:.1%} flat of notional) ===")
print(tbl3.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────
# 4. Correlated-loss scenario: worst single hour / day by simultaneous
#    stop-outs, and resulting equity loss at each f.
# ─────────────────────────────────────────────────────────────────────────
losers = df[df["r_net"] < 0].copy()
losers["exit_hour"] = losers["exit_time"].dt.floor("h")
losers["exit_day"] = losers["exit_time"].dt.floor("D")

by_hour = losers.groupby("exit_hour").agg(
    n_stopouts=("r_net", "size"),
    sum_r_net=("r_net", "sum"),   # negative number, sum of R lost (fees included)
).sort_values("sum_r_net")  # most negative first

by_day = losers.groupby("exit_day").agg(
    n_stopouts=("r_net", "size"),
    sum_r_net=("r_net", "sum"),
).sort_values("sum_r_net")

worst_hour = by_hour.iloc[0]
worst_day = by_day.iloc[0]

by_hour.reset_index().to_csv(f"{OUT}/4_worst_hours.csv", index=False)
by_day.reset_index().to_csv(f"{OUT}/4_worst_days.csv", index=False)

print("\n=== 4. Worst correlated-loss clusters ===")
print(f"Worst single HOUR: {by_hour.index[0]}  n_stopouts={worst_hour.n_stopouts}  sum_r_net={worst_hour.sum_r_net:.2f}")
print(by_hour.head(10).to_string())
print(f"\nWorst single DAY: {by_day.index[0]}  n_stopouts={worst_day.n_stopouts}  sum_r_net={worst_day.sum_r_net:.2f}")
print(by_day.head(10).to_string())

# equity loss fraction = f * |sum_r_net|  (since each trade's risk_usd = f*equity,
# and loss_usd = r_net * risk_usd; assumes equity ~constant across the cluster window)
ruin_rows = []
for f in RISK_LEVELS:
    hour_loss_frac = f * abs(worst_hour.sum_r_net)
    day_loss_frac = f * abs(worst_day.sum_r_net)
    ruin_rows.append({
        "f": f,
        "worst_hour_equity_loss_frac": hour_loss_frac,
        "worst_day_equity_loss_frac": day_loss_frac,
    })
tbl4 = pd.DataFrame(ruin_rows)
tbl4.to_csv(f"{OUT}/4_ruin_fraction_by_f.csv", index=False)
print("\n=== equity loss fraction at each f (worst historical cluster) ===")
print(tbl4.to_string(index=False))

# solve f at which loss frac crosses 0.5 and 1.0
f_50_hour = 0.50 / abs(worst_hour.sum_r_net)
f_100_hour = 1.00 / abs(worst_hour.sum_r_net)
f_50_day = 0.50 / abs(worst_day.sum_r_net)
f_100_day = 1.00 / abs(worst_day.sum_r_net)
print(f"\nf at which worst HOUR cluster = 50% equity loss: {f_50_hour:.4f}")
print(f"f at which worst HOUR cluster = 100% equity loss (ruin): {f_100_hour:.4f}")
print(f"f at which worst DAY cluster = 50% equity loss: {f_50_day:.4f}")
print(f"f at which worst DAY cluster = 100% equity loss (ruin): {f_100_day:.4f}")

with open(f"{OUT}/4_breakeven_f.txt", "w") as fh:
    fh.write(f"worst_hour: {by_hour.index[0]}  n={worst_hour.n_stopouts}  sum_r_net={worst_hour.sum_r_net:.3f}\n")
    fh.write(f"worst_day:  {by_day.index[0]}  n={worst_day.n_stopouts}  sum_r_net={worst_day.sum_r_net:.3f}\n")
    fh.write(f"f @ 50% equity loss (hour cluster): {f_50_hour:.4f}\n")
    fh.write(f"f @ 100% equity loss / ruin (hour cluster): {f_100_hour:.4f}\n")
    fh.write(f"f @ 50% equity loss (day cluster): {f_50_day:.4f}\n")
    fh.write(f"f @ 100% equity loss / ruin (day cluster): {f_100_day:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────
# 4b. HONEST version: worst-day/hour loss when gross_open_risk_cap actually
#     gates entries (fewer trades taken at higher f -> different, smaller
#     set of realized R). This is the figure that should be trusted over
#     the uncapped one above, since gross_open_risk_cap=0.30 is live in
#     config.yaml today.
# ─────────────────────────────────────────────────────────────────────────
cap_cluster_rows = []
for f in RISK_LEVELS:
    taken_c = notional_series[f]["taken_c"].copy()
    losers_c = taken_c[taken_c["r_net"] < 0].copy()
    losers_c["exit_hour"] = losers_c["exit_time"].dt.floor("h")
    losers_c["exit_day"] = losers_c["exit_time"].dt.floor("D")
    bh = losers_c.groupby("exit_hour")["r_net"].agg(["size", "sum"]).sort_values("sum")
    bd = losers_c.groupby("exit_day")["r_net"].agg(["size", "sum"]).sort_values("sum")
    wh = bh.iloc[0] if len(bh) else pd.Series({"size": 0, "sum": 0.0})
    wd = bd.iloc[0] if len(bd) else pd.Series({"size": 0, "sum": 0.0})
    cap_cluster_rows.append({
        "f": f,
        "worst_hour_date": bh.index[0] if len(bh) else None,
        "worst_hour_n_stopouts": wh["size"],
        "worst_hour_sum_r_net": wh["sum"],
        "worst_hour_equity_loss_frac": f * abs(wh["sum"]),
        "worst_day_date": bd.index[0] if len(bd) else None,
        "worst_day_n_stopouts": wd["size"],
        "worst_day_sum_r_net": wd["sum"],
        "worst_day_equity_loss_frac": f * abs(wd["sum"]),
    })
tbl4b = pd.DataFrame(cap_cluster_rows)
tbl4b.to_csv(f"{OUT}/4b_worst_cluster_gross_cap_applied.csv", index=False)
print("\n=== 4b. HONEST (gross_open_risk_cap=0.30 applied) worst-cluster equity loss ===")
print(tbl4b.to_string(index=False))

print("\nDone. Outputs written to", OUT)
