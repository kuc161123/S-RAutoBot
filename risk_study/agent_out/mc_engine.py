"""
Numba-accelerated event-driven equity simulator for the risk-of-ruin / drawdown
Monte Carlo study.

Core idea: trades have entry_time and exit_time. Up to ~45 positions can be open
concurrently in the live bot. To size correctly we must NOT collapse to a
sequential one-trade-at-a-time model: risk_usd for a trade is fixed at its ENTRY
using the equity AT THAT MOMENT (which reflects only exits that have already
happened), and P&L is booked into equity at the trade's EXIT. This requires a
chronological event-driven pass over interleaved entry/exit events, which is why
this is written as a tight numba loop rather than a vectorized pandas op.

Tie-break rule: when an entry and an exit fall on the exact same timestamp
(common at hourly granularity), ENTRIES are processed before EXITS, matching the
live bot's own hourly cycle order (bot.py: queued entries are executed in step 5
of process_symbol, before monitor_active_trades' exit detection in step 9).

Two resampling schemes share this same event engine:
  method=0  IID bootstrap      - draw n_pool trades with replacement, independently.
  method=1  BLOCK bootstrap    - draw circular contiguous blocks of length block_L
                                  from the pool (pool pre-sorted by exit_time, per
                                  the study spec), concatenating blocks (with
                                  wraparound) until n_pool trades are filled. This
                                  preserves within-block correlation and clustering
                                  (e.g. a basket of correlated alts stopping out in
                                  the same week) while destroying cross-block
                                  ordering - exactly what an IID bootstrap erases.

Both methods draw trades by ORIGINAL historical (entry_time, exit_time) pairs -
i.e. we resample which historical trade-outcome fills a slot, but we keep that
slot's own real timestamps. This is what lets the concurrency (up to ~45 open at
once) in the resampled path resemble the concurrency actually observed in
history, rather than inventing a synthetic arrival process.

Equity is floored at a tiny epsilon (1e-8) rather than allowed to go negative -
a real losing streak asymptotically shrinks position size toward zero rather than
producing negative equity (risk_usd = f * equity, and equity cannot cross zero
under this compounding rule for reasonable f - it is floored purely for numerical
safety since concurrent clusters of worst-case trades could in principle drive it
extremely close to zero).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _simulate_core(entry_t, exit_t, net_R, n_pool, method, block_L,
                    n_iter, f_values, start_equity, seed,
                    out_final_eq, out_max_dd, out_breach):
    np.random.seed(seed)
    n_f = f_values.shape[0]
    n_period = n_pool  # resample same number of trades as the historical period

    risk_at_entry = np.zeros((n_period, n_f))
    draw_idx = np.zeros(n_period, dtype=np.int64)
    times = np.zeros(2 * n_period, dtype=np.int64)
    keys = np.zeros(2 * n_period, dtype=np.int64)
    local_idx = np.zeros(2 * n_period, dtype=np.int64)
    event_type = np.zeros(2 * n_period, dtype=np.int64)

    # threshold fractions BELOW start: 10%,20%,30%,50%,ruin(90% i.e. equity<=10% start)
    thresholds = np.array([0.10, 0.20, 0.30, 0.50, 0.90])
    n_thresh = thresholds.shape[0]

    for it in range(n_iter):
        # ---- build draw_idx for this iteration ----
        if method == 0:
            for k in range(n_period):
                draw_idx[k] = np.random.randint(0, n_pool)
        else:
            k = 0
            while k < n_period:
                s = np.random.randint(0, n_pool)
                take = block_L
                if k + take > n_period:
                    take = n_period - k
                for j in range(take):
                    draw_idx[k + j] = (s + j) % n_pool
                k += take

        # ---- build interleaved entry/exit event arrays ----
        for k in range(n_period):
            di = draw_idx[k]
            times[k] = entry_t[di]
            local_idx[k] = k
            event_type[k] = 0
            times[n_period + k] = exit_t[di]
            local_idx[n_period + k] = k
            event_type[n_period + k] = 1

        for k in range(2 * n_period):
            keys[k] = times[k] * 2 + event_type[k]  # entries(0) sort before exits(1) at equal time
        order = np.argsort(keys)

        # ---- run the equity path once per f, reusing the same event order ----
        for fi in range(n_f):
            f = f_values[fi]
            equity = start_equity
            peak = start_equity
            min_eq = start_equity
            max_dd = 0.0

            for oi in range(2 * n_period):
                ev = order[oi]
                li = local_idx[ev]
                et = event_type[ev]
                if et == 0:
                    risk_at_entry[li, fi] = f * equity
                else:
                    r = net_R[draw_idx[li]]
                    pnl = r * risk_at_entry[li, fi]
                    equity += pnl
                    if equity < 1e-8:
                        equity = 1e-8
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak
                    if dd > max_dd:
                        max_dd = dd
                    if equity < min_eq:
                        min_eq = equity

            out_final_eq[it, fi] = equity
            out_max_dd[it, fi] = max_dd
            for ti in range(n_thresh):
                thresh_eq = start_equity * (1.0 - thresholds[ti])
                out_breach[it, fi, ti] = 1 if min_eq <= thresh_eq else 0


def simulate(entry_t, exit_t, net_R, method, block_L, n_iter, f_values,
             start_equity, seed):
    """
    method: 'iid' or 'block'
    entry_t, exit_t: int64 arrays (hours since epoch), same length as net_R
                      -- for method='block' these MUST already be sorted by exit_t.
    Returns dict with final_eq[n_iter,n_f], max_dd[n_iter,n_f], breach[n_iter,n_f,5]
    (breach columns: drop>=10%, >=20%, >=30%, >=50%, ruin(>=90%/equity<=10%start))
    """
    n_pool = len(net_R)
    n_f = len(f_values)
    out_final_eq = np.zeros((n_iter, n_f))
    out_max_dd = np.zeros((n_iter, n_f))
    out_breach = np.zeros((n_iter, n_f, 5), dtype=np.uint8)
    m = 0 if method == 'iid' else 1
    bl = 0 if method == 'iid' else int(block_L)
    _simulate_core(entry_t.astype(np.int64), exit_t.astype(np.int64),
                   net_R.astype(np.float64), n_pool, m, bl,
                   n_iter, np.asarray(f_values, dtype=np.float64),
                   float(start_equity), int(seed),
                   out_final_eq, out_max_dd, out_breach)
    return {'final_eq': out_final_eq, 'max_dd': out_max_dd, 'breach': out_breach}
