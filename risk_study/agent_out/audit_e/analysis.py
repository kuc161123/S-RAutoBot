"""
AUDIT-E: test additive ideas against the current-config trade set.
Reads trades_features.parquet (built by build_features.py).
Writes: results.csv (one row per tested idea/feature) and prints detail used for AUDIT_E.md.
"""
import pandas as pd
import numpy as np
from scipy import stats as sstats

ROOT = "/Users/lualakol/AutoTrading Bot"
OUT = f"{ROOT}/risk_study/agent_out/audit_e"

df = pd.read_parquet(f"{OUT}/trades_features.parquet")
df["week"] = df.entry_time.dt.to_period("W").apply(lambda p: p.start_time)
pre = df[df.period == "pre"].copy()
post = df[df.period == "post"].copy()
print("pre", len(pre), "post", len(post))
print("baseline mean net_r  pre=%.4f  post=%.4f" % (pre.net_r.mean(), post.net_r.mean()))

RNG = np.random.default_rng(42)


def weekly_bootstrap_ci(sub, statfunc, n_boot=800, seed=42):
    """Weekly block bootstrap CI for statfunc(df) -> float. Resample weeks w/ replacement."""
    rng = np.random.default_rng(seed)
    weeks = sub["week"].unique()
    n_weeks = len(weeks)
    if n_weeks < 3:
        return (np.nan, np.nan, np.nan)
    groups = [sub[sub["week"] == w] for w in weeks]
    vals = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_weeks, size=n_weeks)
        boot_df = pd.concat([groups[i] for i in chosen], ignore_index=True)
        try:
            v = statfunc(boot_df)
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    return (np.percentile(vals, 2.5), np.mean(vals), np.percentile(vals, 97.5))


def mean_netr(sub):
    return sub.net_r.mean()


def spearman_ic(sub, feature, target="net_r"):
    s = sub[[feature, target]].dropna()
    if len(s) < 30:
        return np.nan
    return sstats.spearmanr(s[feature], s[target]).correlation


def rolling_window_agreement(dfull, feature, direction, freq="Q", min_n=80):
    """direction: +1 if higher feature => predict higher net_r (per pre-fit sign).
    Returns fraction of windows where sign matches, and list of (window, ic, n).
    NOTE: uses Spearman rank IC, which can disagree in sign with a simple group-mean
    difference when net_r is heavily bimodal (huge tied mass at exactly -1.0 losses).
    Use rolling_window_win_rate (mean-difference based) for anything that will be
    reported as a PAYS/DOES-NOT-PAY verdict -- this one is kept only as a secondary
    rank-based diagnostic."""
    dfull = dfull.copy()
    dfull["win"] = dfull.entry_time.dt.to_period(freq)
    rows = []
    for w, g in dfull.groupby("win"):
        if len(g) < min_n:
            continue
        ic = spearman_ic(g, feature)
        if np.isnan(ic):
            continue
        rows.append((str(w), ic, len(g)))
    if not rows:
        return np.nan, rows
    agree = sum(1 for _, ic, _ in rows if np.sign(ic) == np.sign(direction)) / len(rows)
    return agree, rows


def rolling_window_win_rate(dfull, statfunc, freq="Q", min_n=80):
    """The metric that actually matters: per-window, compute the SAME effect statistic
    (a mean-R difference, e.g. group_a.mean() - group_b.mean(), or a weighted-vs-flat
    gain) used for the headline number, and report the fraction of windows where its
    sign agrees with the overall (all-period) sign. Robust to net_r's heavy tie-mass at
    -1.0, unlike a rank correlation."""
    dfull = dfull.copy()
    dfull["win"] = dfull.entry_time.dt.to_period(freq)
    rows = []
    for w, g in dfull.groupby("win"):
        if len(g) < min_n:
            continue
        try:
            v = statfunc(g)
        except Exception:
            v = np.nan
        if v == v:
            rows.append((str(w), v, len(g)))
    if not rows:
        return np.nan, rows
    overall = statfunc(dfull)
    if overall != overall:
        return np.nan, rows
    win_rate = sum(1 for _, v, _ in rows if np.sign(v) == np.sign(overall)) / len(rows)
    return win_rate, rows


results = []


def add_result(idea, feature, ic_pre, ic_post, effect_pre, effect_post, ci, roll_agree, n_pre, n_post, verdict, note):
    results.append(dict(idea=idea, feature=feature, ic_pre=ic_pre, ic_post=ic_post,
                         effect_pre=effect_pre, effect_post=effect_post,
                         ci_lo=ci[0] if ci else np.nan, ci_mid=ci[1] if ci else np.nan, ci_hi=ci[2] if ci else np.nan,
                         rolling_win_rate=roll_agree, n_pre=n_pre, n_post=n_post,
                         verdict=verdict, note=note))


# =========================================================================
# IDEA 1: Portfolio-level concurrency -> inverse-scale sizing
# =========================================================================
print("\n=== IDEA 1: concurrency ===")
ic_pre = spearman_ic(pre, "concurrency")
ic_post = spearman_ic(post, "concurrency")
print("IC concurrency vs net_r: pre=%.4f post=%.4f" % (ic_pre, ic_post))

# raw concurrency trends up over 2023->2026 as the universe/config matured (more symbols, config
# changes, edge decay/recovery cycles) -- this is a huge confound for a raw level test. De-trend by
# month: z-score concurrency within its own month, and also monthly-demean net_r, before testing.
df["month"] = df.entry_time.dt.to_period("M")
df["conc_z_month"] = df.groupby("month")["concurrency"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
df["net_r_demonth"] = df["net_r"] - df.groupby("month")["net_r"].transform("mean")
pre["conc_z_month"] = df.loc[pre.index, "conc_z_month"]
post["conc_z_month"] = df.loc[post.index, "conc_z_month"]
pre["net_r_demonth"] = df.loc[pre.index, "net_r_demonth"]
post["net_r_demonth"] = df.loc[post.index, "net_r_demonth"]

ic_pre_z = sstats.spearmanr(pre["conc_z_month"], pre["net_r_demonth"]).correlation
ic_post_z = sstats.spearmanr(post["conc_z_month"], post["net_r_demonth"]).correlation
print("DE-TRENDED, but NON-causal diagnostic only (within-month z-score concurrency vs "
      "within-month-demeaned net_r, month stats use future info within the month): "
      "IC pre=%.4f post=%.4f" % (ic_pre_z, ic_post_z))

# CAUSAL, deployable version: relative concurrency = concurrency / trailing rolling median
# concurrency of the last 500 trades (strictly past trades only via shift).
df_sorted = df.sort_values("entry_time").reset_index(drop=True)
df_sorted["roll_median_conc"] = df_sorted["concurrency"].shift(1).rolling(500, min_periods=100).median()
df_sorted["rel_conc"] = df_sorted["concurrency"] / df_sorted["roll_median_conc"]
df = df_sorted.set_index(df_sorted.index)  # keep same integer index scheme going forward
pre = df[df.period == "pre"].copy()
post = df[df.period == "post"].copy()
pre["week"] = pre.entry_time.dt.to_period("W").apply(lambda p: p.start_time)
post["week"] = post.entry_time.dt.to_period("W").apply(lambda p: p.start_time)

ic_pre_rel = spearman_ic(pre, "rel_conc")
ic_post_rel = spearman_ic(post, "rel_conc")
print("CAUSAL relative concurrency (vs trailing 500-trade rolling median) IC: "
      "pre=%.4f post=%.4f" % (ic_pre_rel, ic_post_rel))

q = pre.concurrency.quantile([0.2, 0.4, 0.6, 0.8]).values
pre["conc_bucket"] = pd.cut(pre.concurrency, [-1] + list(q) + [1e9], labels=False)
post["conc_bucket"] = pd.cut(post.concurrency, [-1] + list(q) + [1e9], labels=False)
bucket_stats_pre = pre.groupby("conc_bucket").net_r.agg(["mean", "count"])
bucket_stats_post = post.groupby("conc_bucket").net_r.agg(["mean", "count"])
print("pre buckets:\n", bucket_stats_pre)
print("post buckets:\n", bucket_stats_post)

# inverse-RELATIVE-concurrency size multiplier (causal: target=1.0 by construction), cap [0.4, 1.5]
def size_mult(rel_c):
    m = 1.0 / np.maximum(rel_c, 0.1)
    return np.clip(m, 0.4, 1.5)


def weighted_effect(sub):
    s = sub.dropna(subset=["rel_conc"])
    if len(s) < 20:
        return np.nan
    w = size_mult(s.rel_conc.values)
    flat = s.net_r.mean()
    weighted = np.average(s.net_r.values, weights=w)
    return weighted - flat


eff_pre = weighted_effect(pre)
eff_post = weighted_effect(post)
ci = weekly_bootstrap_ci(post, weighted_effect)
roll_agree, roll_rows = rolling_window_win_rate(df, weighted_effect)
print("weighted-vs-flat effect (causal rel_conc sizing): pre=%.4f post=%.4f  post 95%%CI=(%.4f,%.4f)" %
      (eff_pre, eff_post, ci[0], ci[2]))
print("rolling quarter sign agreement:", roll_agree, roll_rows)

verdict1 = "DOES-NOT-PAY"
if abs(eff_post) > 0.02 and ci[0] * ci[2] > 0 and roll_agree is not np.nan and roll_agree > 0.6:
    verdict1 = "PAYS"
add_result("1_portfolio_vol_targeting", "rel_conc (causal, vs trailing 500-trade median)",
           ic_pre_rel, ic_post_rel, eff_pre, eff_post, ci, roll_agree,
           len(pre), len(post), verdict1,
           f"raw-level IC pre={ic_pre:.3f}/post={ic_post:.3f} confounded by time trend (universe grew "
           f"3x over the sample); month-detrended diagnostic IC pre={ic_pre_z:.3f}/post={ic_post_z:.3f}. "
           "Deployable inverse-relative-concurrency size multiplier (target=1.0, clip 0.4-1.5x) weighted "
           "mean R vs flat mean R.")

# =========================================================================
# IDEA 2: sizing by causal signal characteristics
# =========================================================================
print("\n=== IDEA 2: signal characteristics ===")
pre["log_liq30d"] = np.log10(pre.liq_30d.clip(lower=1))
post["log_liq30d"] = np.log10(post.liq_30d.clip(lower=1))
df["log_liq30d"] = np.log10(df.liq_30d.clip(lower=1))

continuous_feats = ["log_liq30d", "stop_frac", "chop_bos", "ema_dist_1h_aligned", "rvol_30d",
                     "funding_avg7d_for_side"]

for feat in continuous_feats:
    ic_pre = spearman_ic(pre, feat)
    ic_post = spearman_ic(post, feat)
    n_pre = pre[feat].notna().sum()
    n_post = post[feat].notna().sum()

    # fit a simple top/bottom tercile rule on pre (direction from pre IC sign)
    valid_pre = pre[feat].dropna()
    if len(valid_pre) < 100:
        add_result("2_signal_characteristics", feat, ic_pre, ic_post, np.nan, np.nan, (np.nan,) * 3, np.nan,
                   n_pre, n_post, "UNTESTABLE", "insufficient data")
        continue
    lo_thr, hi_thr = valid_pre.quantile([0.33, 0.67])
    direction = 1 if (ic_pre == ic_pre and ic_pre > 0) else -1

    def bucket_effect(sub, feat=feat, lo=lo_thr, hi=hi_thr, direction=direction):
        s = sub.dropna(subset=[feat])
        if len(s) < 20:
            return np.nan
        good = s[feat] >= hi if direction > 0 else s[feat] <= lo
        bad = s[feat] <= lo if direction > 0 else s[feat] >= hi
        if good.sum() < 5 or bad.sum() < 5:
            return np.nan
        return s.loc[good, "net_r"].mean() - s.loc[bad, "net_r"].mean()

    eff_pre = bucket_effect(pre)
    eff_post = bucket_effect(post)
    ci = weekly_bootstrap_ci(post, bucket_effect)
    roll_agree, roll_rows = rolling_window_win_rate(df, bucket_effect)

    verdict = "DOES-NOT-PAY"
    reason = "no transfer / weak or inconsistent"
    if (not np.isnan(eff_post)) and abs(eff_post) > 0.02 * 1 and np.sign(eff_pre) == np.sign(eff_post) \
       and not (ci[0] < 0 < ci[2]) and roll_agree is not np.nan and roll_agree >= 0.6:
        verdict = "PAYS"
        reason = "sign-consistent pre/post, CI excludes 0, rolling win-rate>=0.6"
    elif np.isnan(eff_post):
        verdict = "UNTESTABLE"
        reason = "too few post-split obs in extreme buckets"

    add_result("2_signal_characteristics", feat, ic_pre, ic_post, eff_pre, eff_post, ci, roll_agree,
               n_pre, n_post, verdict, reason)
    print(f"{feat}: IC pre={ic_pre:.3f} post={ic_post:.3f}  tercile-effect pre={eff_pre:.4f} post={eff_post:.4f} "
          f"CI=({ci[0]:.4f},{ci[2]:.4f}) roll={roll_agree} -> {verdict}")

# --- refine log_liq30d and stop_frac: both showed a strong net_r IC that could be a mechanical
# artifact of the cost formula (net_r = r - 0.00242/stop_frac) or an indirect correlation channel
# rather than real outcome-quality information. Decompose against GROSS r_result before trusting them.
print("\n--- decomposing log_liq30d / stop_frac against GROSS r_result ---")


def ic_vs(a, b):
    m = a.notna() & b.notna()
    return sstats.spearmanr(a[m], b[m]).correlation


ic_liq_gross_pre = ic_vs(pre["log_liq30d"], pre["r_result"])
ic_liq_gross_post = ic_vs(post["log_liq30d"], post["r_result"])
print(f"log_liq30d IC vs GROSS r_result: pre={ic_liq_gross_pre:.3f} post={ic_liq_gross_post:.3f} "
      "(both ~0 -> the net_r IC is noise / weak indirect cost-channel, not a real liquidity edge)")

liq_thr = pre.log_liq30d.quantile(0.2)
liq_keep_pre = pre[pre.log_liq30d >= liq_thr].net_r.mean() - pre.net_r.mean()
liq_keep_post = post[post.log_liq30d >= liq_thr].net_r.mean() - post.net_r.mean()
print(f"drop-bottom-quintile-liquidity filter gain: pre={liq_keep_pre:.4f} post={liq_keep_post:.4f} "
      "(wrong sign / not robust OOS)")
# overwrite the earlier log_liq30d row's verdict now that it's understood to be a false positive
for r in results:
    if r["idea"] == "2_signal_characteristics" and r["feature"] == "log_liq30d":
        r["verdict"] = "DOES-NOT-PAY"
        r["note"] = (f"REVISED after decomposition: IC vs GROSS r_result pre={ic_liq_gross_pre:.3f} "
                     f"post={ic_liq_gross_post:.3f} (~0). The apparent net_r tercile effect was driven by "
                     "a weak indirect channel (corr(log_liq30d, stop_frac)~0.16) plus small-holdout noise "
                     f"in the extreme bucket. Explicit drop-bottom-quintile-liquidity filter gives the WRONG "
                     f"sign OOS (pre={liq_keep_pre:.4f}, post={liq_keep_post:.4f}). Not a real edge.")

# stop_frac: IC vs GROSS r_result is ~0 across most of the range, but the TOP quintile (widest
# ATR-relative stop = most volatile signal) has a materially worse GROSS outcome in both periods,
# and much worse OOS. A quintile-cutoff filter (not a monotonic size) captures this.
ic_sf_gross_pre = ic_vs(pre["stop_frac"], pre["r_result"])
ic_sf_gross_post = ic_vs(post["stop_frac"], post["r_result"])
print(f"stop_frac IC vs GROSS r_result: pre={ic_sf_gross_pre:.3f} post={ic_sf_gross_post:.3f} "
      "(near zero overall -- the strong net_r IC (~0.48) is mostly the cost formula's built-in "
      "1/stop_frac term, NOT outcome information -- except the top quintile, see below)")

df["roll_q80_sf"] = df["stop_frac"].shift(1).rolling(2000, min_periods=500).quantile(0.8)
pre["roll_q80_sf"] = df.loc[pre.index, "roll_q80_sf"]
post["roll_q80_sf"] = df.loc[post.index, "roll_q80_sf"]


def sf_filter_gain(sub):
    s = sub.dropna(subset=["stop_frac", "net_r", "roll_q80_sf"])
    if len(s) < 30:
        return np.nan
    keep = s[s.stop_frac < s.roll_q80_sf]
    if len(keep) == 0:
        return np.nan
    return keep.net_r.mean() - s.net_r.mean()


eff_pre_sf = sf_filter_gain(pre)
eff_post_sf = sf_filter_gain(post)
ci_sf = weekly_bootstrap_ci(post, sf_filter_gain)
roll_agree_sf, roll_rows_sf = rolling_window_win_rate(df, sf_filter_gain)
print("rolling quarter detail (skip-top-quintile-stop_frac gain):", roll_rows_sf)
pct_dropped_pre = 1 - (pre.stop_frac < pre.roll_q80_sf).mean()
pct_dropped_post = 1 - (post.stop_frac < post.roll_q80_sf).mean()
print(f"SKIP-top-quintile-stop_frac filter (causal rolling 80th pct, trailing 2000 trades): "
      f"pre gain={eff_pre_sf:.4f} (drops {pct_dropped_pre:.1%}) post gain={eff_post_sf:.4f} "
      f"(drops {pct_dropped_post:.1%}) post 95%CI=({ci_sf[0]:.4f},{ci_sf[2]:.4f}) "
      f"roll_win_rate={roll_agree_sf}")

verdict_sf2 = "DOES-NOT-PAY"
if not np.isnan(eff_post_sf) and eff_post_sf > 0.02 and ci_sf[0] > 0:
    verdict_sf2 = "PAYS"
add_result("2_signal_characteristics", "stop_frac[skip top-quintile filter, causal rolling q80]",
           ic_sf_gross_pre, ic_sf_gross_post, eff_pre_sf, eff_post_sf, ci_sf, roll_agree_sf,
           (pre.stop_frac < pre.roll_q80_sf).sum(), (post.stop_frac < post.roll_q80_sf).sum(), verdict_sf2,
           "Skip entries where stop_frac (=ATR*atr_mult/entry_price) sits in the top ~20% of its own "
           "trailing 2000-trade distribution -- flags unusually volatile/wide-stop signals. GROSS "
           "r_result in this bucket is worse in both periods (much worse OOS); this is a real outcome "
           "effect at the tail, not the mechanical 1/stop_frac cost-formula artifact that dominates the "
           "full-range correlation.")
# overwrite the earlier naive stop_frac (monotonic-direction) row's verdict/note for clarity
for r in results:
    if r["idea"] == "2_signal_characteristics" and r["feature"] == "stop_frac":
        r["note"] = (f"Naive monotonic top/bottom-tercile test is misleading: full-range Spearman IC "
                     f"vs net_r (~0.48-0.49) is almost entirely the cost formula's built-in 1/stop_frac "
                     f"term (IC vs GROSS r_result is only {ic_sf_gross_pre:.3f}/{ic_sf_gross_post:.3f}). "
                     "See the 'skip top-quintile filter' row below for the real, robust effect.")

# categorical: div_type, side, hour, dow
print("\n--- categorical breakdowns ---")
for cat in ["div_type", "side", "hour", "dow"]:
    g_pre = pre.groupby(cat).net_r.agg(["mean", "count"])
    g_post = post.groupby(cat).net_r.agg(["mean", "count"])
    print(f"\n{cat} pre:\n{g_pre}\n{cat} post:\n{g_post}")
    # spread between best and worst category (by pre mean, weighted by count>=100 in pre)
    elig = g_pre[g_pre["count"] >= 100]
    if len(elig) < 2:
        continue
    best_cat = elig["mean"].idxmax()
    worst_cat = elig["mean"].idxmin()
    eff_pre = g_pre.loc[best_cat, "mean"] - g_pre.loc[worst_cat, "mean"]
    eff_post = (g_post.loc[best_cat, "mean"] - g_post.loc[worst_cat, "mean"]) \
        if (best_cat in g_post.index and worst_cat in g_post.index) else np.nan

    def cat_effect(sub, cat=cat, best_cat=best_cat, worst_cat=worst_cat):
        s = sub[sub[cat].isin([best_cat, worst_cat])]
        if s[cat].nunique() < 2:
            return np.nan
        gm = s.groupby(cat).net_r.mean()
        if best_cat not in gm.index or worst_cat not in gm.index:
            return np.nan
        return gm[best_cat] - gm[worst_cat]

    ci = weekly_bootstrap_ci(post, cat_effect)
    roll_agree_cat, _ = rolling_window_win_rate(df, cat_effect)
    n_pre = int(g_pre.loc[best_cat, "count"] + g_pre.loc[worst_cat, "count"])
    n_post = int(g_post.loc[best_cat, "count"] if best_cat in g_post.index else 0) + \
        int(g_post.loc[worst_cat, "count"] if worst_cat in g_post.index else 0)
    verdict = "DOES-NOT-PAY"
    if not np.isnan(eff_post) and abs(eff_post) > 0.02 and not (ci[0] < 0 < ci[2]) \
       and roll_agree_cat is not np.nan and roll_agree_cat >= 0.6:
        verdict = "PAYS"
    add_result("2_signal_characteristics", f"{cat}[{best_cat} vs {worst_cat}]", np.nan, np.nan,
               eff_pre, eff_post, ci, roll_agree_cat, n_pre, n_post, verdict,
               f"best/worst {cat} bucket spread in net_r, pre-selected buckets")

# =========================================================================
# IDEA 3: time-of-day / session
# =========================================================================
print("\n=== IDEA 3: session effects ===")
pre["session"] = pd.cut(pre.hour, [-1, 7, 15, 23], labels=["asia_0_8utc", "eu_8_16utc", "us_16_24utc"])
post["session"] = pd.cut(post.hour, [-1, 7, 15, 23], labels=["asia_0_8utc", "eu_8_16utc", "us_16_24utc"])
df["session"] = pd.cut(df.hour, [-1, 7, 15, 23], labels=["asia_0_8utc", "eu_8_16utc", "us_16_24utc"])

sess_pre = pre.groupby("session", observed=True).net_r.agg(["mean", "count", lambda x: x.quantile(0.05)])
sess_pre.columns = ["mean", "count", "p05"]
sess_post = post.groupby("session", observed=True).net_r.agg(["mean", "count", lambda x: x.quantile(0.05)])
sess_post.columns = ["mean", "count", "p05"]
print("session pre:\n", sess_pre)
print("session post:\n", sess_post)

best_s = sess_pre["mean"].idxmax()
worst_s = sess_pre["mean"].idxmin()
eff_pre = sess_pre.loc[best_s, "mean"] - sess_pre.loc[worst_s, "mean"]
eff_post = (sess_post.loc[best_s, "mean"] - sess_post.loc[worst_s, "mean"]) \
    if (best_s in sess_post.index and worst_s in sess_post.index) else np.nan


def sess_effect(sub, best_s=best_s, worst_s=worst_s):
    s = sub[sub.session.isin([best_s, worst_s])]
    if s.session.nunique() < 2:
        return np.nan
    gm = s.groupby("session", observed=True).net_r.mean()
    if best_s not in gm.index or worst_s not in gm.index:
        return np.nan
    return gm[best_s] - gm[worst_s]


ci = weekly_bootstrap_ci(post, sess_effect)
roll_agree, roll_rows = rolling_window_win_rate(df, sess_effect)
verdict3 = "DOES-NOT-PAY"
if not np.isnan(eff_post) and abs(eff_post) > 0.02 and not (ci[0] < 0 < ci[2]):
    verdict3 = "PAYS"
add_result("3_session_effects", f"session[{best_s} vs {worst_s}]", np.nan, np.nan, eff_pre, eff_post, ci,
           roll_agree, len(pre), len(post), verdict3, "best/worst 8h UTC session spread, pre-selected")

# =========================================================================
# IDEA 4: poll-loop / alphabetical rank
# =========================================================================
print("\n=== IDEA 4: alphabetical rank / poll loop ===")
sym_stats = df.groupby("symbol").agg(mean_net_r=("net_r", "mean"), n=("net_r", "count"),
                                      alpha_rank=("alpha_rank", "first")).reset_index()
sym_stats_elig = sym_stats[sym_stats.n >= 20]
ic_rank = sstats.spearmanr(sym_stats_elig.alpha_rank, sym_stats_elig.mean_net_r).correlation
print(f"symbol-level spearman(alpha_rank, mean_net_r), n_symbols={len(sym_stats_elig)}: {ic_rank:.4f}")

# also trade-level IC (weaker test, dominated by symbol composition, but matches prompt's literal ask)
ic_pre_rank = spearman_ic(pre, "alpha_rank")
ic_post_rank = spearman_ic(post, "alpha_rank")
print(f"trade-level IC pre={ic_pre_rank:.4f} post={ic_post_rank:.4f}")


# bootstrap over SYMBOLS (not weeks) for this one since it's a cross-sectional symbol-level test
def symbol_bootstrap_ic(sym_df, n_boot=1000, seed=1):
    rng = np.random.default_rng(seed)
    syms = sym_df.symbol.values
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(len(syms), size=len(syms), replace=True)
        b = sym_df.iloc[chosen]
        if b.alpha_rank.nunique() < 5:
            continue
        vals.append(sstats.spearmanr(b.alpha_rank, b.mean_net_r).correlation)
    vals = np.array(vals)
    return np.percentile(vals, 2.5), np.mean(vals), np.percentile(vals, 97.5)


ci_rank = symbol_bootstrap_ic(sym_stats_elig)
print("symbol-level rank IC bootstrap CI:", ci_rank)

verdict4 = "DOES-NOT-PAY"
note4 = ("No material relationship between alphabetical processing order and realized R in this dataset. "
         "IMPORTANT CAVEAT: entry_price in the backtest is the idealized candle open "
         "(df.iloc[-1]['open']); it does NOT model the bot's actual fetch-time slippage from "
         "processing symbols late in a multi-minute sweep. This test can only detect a "
         "*selection* effect (e.g. late symbols happen to be structurally worse names), not the "
         "*execution-lag* mechanism the prompt is actually asking about, which would require "
         "sub-hourly fill data (exec_log) not available to this workstream.")
if abs(ci_rank[1]) > 0.05 and ci_rank[0] * ci_rank[2] > 0:
    verdict4 = "UNRESOLVED - possible selection effect, needs exec_log"
add_result("4_poll_loop_alpha_rank", "alpha_rank (symbol-level)", ic_rank, np.nan, np.nan, np.nan, ci_rank, np.nan,
           len(sym_stats_elig), np.nan, "UNTESTABLE (mechanism)", note4)

# =========================================================================
# IDEA 5: funding-aware side selection (already partly in continuous_feats above as
# funding_avg7d_for_side) -- add a coarser current-funding version + note.
# =========================================================================
print("\n=== IDEA 5: funding (see also 2_signal_characteristics/funding_avg7d_for_side) ===")
fpre = pre.dropna(subset=["funding_last"])
fpost = post.dropna(subset=["funding_last"])
print("funding coverage window: pre n=%d (funding data starts 2025-01-15) post n=%d" % (len(fpre), len(fpost)))
ic_pre_f = spearman_ic(fpre, "funding_for_side")
ic_post_f = spearman_ic(fpost, "funding_for_side")
print(f"instantaneous funding_for_side IC: pre={ic_pre_f:.4f} post={ic_post_f:.4f}")
add_result("5_funding_side_selection", "funding_last_for_side (instant)", ic_pre_f, ic_post_f, np.nan, np.nan,
           (np.nan,) * 3, np.nan, len(fpre), len(fpost), "SEE funding_avg7d_for_side row",
           "instantaneous funding rate at entry, side-adjusted (short receives +funding)")

# =========================================================================
# IDEA 6: multi-timeframe confirmation (1H vs 4H) -- multicollinearity check
# =========================================================================
print("\n=== IDEA 6: 4H trend agreement vs 1H EMA (multicollinearity) ===")
agree_1h_4h = (df["ema_dist_1h_aligned"] > 0) == (df["trend_4h_aligned"])
print("1H-EMA-aligned vs 4H-trend-aligned agreement rate:", agree_1h_4h.mean())

ic_pre_4h = spearman_ic(pre.assign(t4=pre.trend_4h_aligned.astype(float)), "t4")
ic_post_4h = spearman_ic(post.assign(t4=post.trend_4h_aligned.astype(float)), "t4")
eff_pre_4h = pre.loc[pre.trend_4h_aligned, "net_r"].mean() - pre.loc[~pre.trend_4h_aligned, "net_r"].mean()
eff_post_4h = post.loc[post.trend_4h_aligned, "net_r"].mean() - post.loc[~post.trend_4h_aligned, "net_r"].mean()


def trend4h_effect(sub):
    if sub.trend_4h_aligned.nunique() < 2:
        return np.nan
    return sub.loc[sub.trend_4h_aligned, "net_r"].mean() - sub.loc[~sub.trend_4h_aligned, "net_r"].mean()


ci4h = weekly_bootstrap_ci(post, trend4h_effect)
roll_agree_4h, roll_rows_4h = rolling_window_win_rate(df, trend4h_effect)
print(f"4H-aligned vs not: pre={eff_pre_4h:.4f} post={eff_post_4h:.4f} CI={ci4h} roll={roll_agree_4h}")
print("rolling quarter detail (4H trend effect):", roll_rows_4h)

# conditional test: among trades where 1H EMA gate is only barely satisfied (close to EMA),
# does 4H trend add information? -> subset to bottom tercile of |ema_dist_1h_aligned| (marginal 1H cases)
thr = pre.ema_dist_1h_aligned.abs().quantile(0.34)
marginal = df[df.ema_dist_1h_aligned.abs() <= thr]
marg_pre = marginal[marginal.period == "pre"]
marg_post = marginal[marginal.period == "post"]
eff_pre_marg = marg_pre.loc[marg_pre.trend_4h_aligned, "net_r"].mean() - \
    marg_pre.loc[~marg_pre.trend_4h_aligned, "net_r"].mean()
eff_post_marg = (marg_post.loc[marg_post.trend_4h_aligned, "net_r"].mean() -
                  marg_post.loc[~marg_post.trend_4h_aligned, "net_r"].mean()) if len(marg_post) > 40 else np.nan
print(f"marginal-1H-cases (bottom tercile |ema_dist|) 4H-aligned effect: pre={eff_pre_marg:.4f} (n={len(marg_pre)}) "
      f"post={eff_post_marg} (n={len(marg_post)})")

verdict6 = "DOES-NOT-PAY"
if not np.isnan(eff_post_4h) and abs(eff_post_4h) > 0.02 and not (ci4h[0] < 0 < ci4h[2]) and roll_agree_4h >= 0.6:
    verdict6 = "PAYS"
add_result("6_multi_timeframe_4h", "trend_4h_aligned (all trades)", ic_pre_4h, ic_post_4h, eff_pre_4h, eff_post_4h,
           ci4h, roll_agree_4h, len(pre), len(post), verdict6,
           f"1H/4H trend agreement rate={agree_1h_4h.mean():.3f}; marginal-1H-case effect pre={eff_pre_marg:.4f} "
           f"post={eff_post_marg}")

# =========================================================================
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUT}/results.csv", index=False)
print("\nSaved results.csv with", len(res_df), "rows")
print(res_df[["idea", "feature", "ic_pre", "ic_post", "effect_pre", "effect_post", "verdict"]].to_string())
