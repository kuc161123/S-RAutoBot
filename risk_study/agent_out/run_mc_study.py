"""
Monte Carlo drawdown / risk-of-ruin study, AGENT-MC.

Reads risk_study/universe_chopBOS.parquet, splits into DEV / VAL / HOLDOUT by
entry_time, computes net R per trade at two cost levels, then for every
(period, f) runs:
  1. IID trade-sequence bootstrap (5000 iters)
  2. Block bootstrap, 1-week blocks (5000 iters)
  3. Block bootstrap, 1-month blocks (5000 iters)
using the event-driven engine in mc_engine.py (entry-time sizing / exit-time
booking, not a sequential collapse).

Outputs:
  agent_out/mc_results.csv   - one row per period x cost x method x f
  agent_out/MC_RESULTS.md    - full write-up
"""
import numpy as np
import pandas as pd
import time
from mc_engine import simulate

OUT_DIR = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out"
SRC = "/Users/lualakol/AutoTrading Bot/risk_study/universe_chopBOS.parquet"

SEED = 42
N_ITER = 5000
START_EQUITY = 10000.0
F_VALUES = np.array([0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03])
COST_CENTRAL = 0.00341
COST_OPTIMISTIC = 0.0018

df = pd.read_parquet(SRC)
df = df.sort_values('entry_time').reset_index(drop=True)

# integer hour index, anchored at the dataset's minimum entry_time, for exact
# integer event-time arithmetic in the numba engine.
t0 = df['entry_time'].min()
df['entry_h'] = ((df['entry_time'] - t0) / pd.Timedelta(hours=1)).round().astype(np.int64)
df['exit_h'] = ((df['exit_time'] - t0) / pd.Timedelta(hours=1)).round().astype(np.int64)

PERIODS = {
    'DEV':     df[df['entry_time'] < '2025-07-01'].copy(),
    'VAL':     df[(df['entry_time'] >= '2025-07-01') & (df['entry_time'] < '2026-05-25')].copy(),
    'HOLDOUT': df[df['entry_time'] >= '2026-05-25'].copy(),
}

for name, d in PERIODS.items():
    print(name, len(d), d['entry_time'].min(), '->', d['entry_time'].max(),
          'winrate', (d['r_result'] > 0).mean())


def block_length(d, unit_hours):
    span_h = (d['entry_time'].max() - d['entry_time'].min()).total_seconds() / 3600.0
    if span_h <= 0:
        return 1
    rate_per_hour = len(d) / span_h
    return max(1, int(round(rate_per_hour * unit_hours)))


def analytic_ruin_prob(net_R, f, barrier=np.log(10.0)):
    """Diffusion-approximation risk-of-ruin cross-check.
    Models log(equity) as a random walk with per-trade increment log(1+f*R).
    mu>0: P(ever hit lower barrier at distance `barrier`) = exp(-2*mu*barrier/sigma^2)
    mu<=0: ruin probability -> 1 over an infinite horizon.
    This assumes IID increments (matches the IID bootstrap best) and an
    infinite time horizon, so it is an upper bound on any finite-horizon
    simulated P(ruin).
    """
    x = 1.0 + f * net_R
    x = np.clip(x, 1e-12, None)
    logx = np.log(x)
    mu = logx.mean()
    sigma2 = logx.var()
    if sigma2 <= 0:
        return 1.0 if mu <= 0 else 0.0
    if mu <= 0:
        return 1.0
    p = np.exp(-2.0 * mu * barrier / sigma2)
    return float(min(1.0, p))


rows = []
t_start = time.time()

for cost_name, cost_bps in [('central', COST_CENTRAL), ('optimistic', COST_OPTIMISTIC)]:
    for period_name, d in PERIODS.items():
        net_R_full = (d['r_result'] - cost_bps / d['stop_frac']).values.astype(np.float64)
        entry_h = d['entry_h'].values
        exit_h = d['exit_h'].values
        n_pool = len(d)

        # pool sorted by exit_time for block bootstrap, per spec
        order_exit = np.argsort(exit_h, kind='stable')
        entry_h_bx = entry_h[order_exit]
        exit_h_bx = exit_h[order_exit]
        net_R_bx = net_R_full[order_exit]

        L_week = block_length(d, 7 * 24)
        L_month = block_length(d, 30.44 * 24)

        wr = (d['r_result'] > 0).mean()
        avg_win = d.loc[d['r_result'] > 0, 'r_result'].mean() if wr > 0 else np.nan

        # one simulate() call per method covers ALL f values at once (the event
        # order per iteration is independent of f, so the sort is shared).
        for method_name, args, block_L in [
            ('iid',         dict(entry_t=entry_h, exit_t=exit_h, net_R=net_R_full, method='iid', block_L=0), np.nan),
            ('block_week',  dict(entry_t=entry_h_bx, exit_t=exit_h_bx, net_R=net_R_bx, method='block', block_L=L_week), L_week),
            ('block_month', dict(entry_t=entry_h_bx, exit_t=exit_h_bx, net_R=net_R_bx, method='block', block_L=L_month), L_month),
        ]:
            res = simulate(n_iter=N_ITER, f_values=F_VALUES,
                            start_equity=START_EQUITY, seed=SEED, **args)
            for fi, f in enumerate(F_VALUES):
                final_eq = res['final_eq'][:, fi]
                max_dd = res['max_dd'][:, fi]
                breach = res['breach'][:, fi, :]  # [iter, 5] cols: 10,20,30,50,ruin
                analytic_p_ruin = analytic_ruin_prob(net_R_full, f)

                row = dict(
                    cost_scenario=cost_name, cost_bps=cost_bps, period=period_name,
                    n_trades=n_pool, method=method_name, block_L=block_L,
                    f=f, win_rate=wr, avg_win_R=avg_win,
                    median_final_eq=np.median(final_eq),
                    p10_final_eq=np.percentile(final_eq, 10),
                    p90_final_eq=np.percentile(final_eq, 90),
                    mean_final_eq=np.mean(final_eq),
                    median_maxdd=np.median(max_dd),
                    p90_maxdd=np.percentile(max_dd, 90),
                    p95_maxdd=np.percentile(max_dd, 95),
                    p99_maxdd=np.percentile(max_dd, 99),
                    p_drop10=breach[:, 0].mean(),
                    p_drop20=breach[:, 1].mean(),
                    p_drop30=breach[:, 2].mean(),
                    p_drop50=breach[:, 3].mean(),
                    p_ruin_sim=breach[:, 4].mean(),
                    p_ruin_analytic=analytic_p_ruin,
                )
                rows.append(row)
            print(f"  done cost={cost_name} period={period_name} method={method_name}  "
                  f"t={time.time()-t_start:.1f}s")

res_df = pd.DataFrame(rows)
res_df.to_csv(f"{OUT_DIR}/mc_results.csv", index=False)
print("TOTAL TIME", time.time() - t_start)
print(res_df.shape)
