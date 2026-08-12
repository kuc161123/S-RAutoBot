"""Shared s3_a1 trail resolver, factored out so the live-config and global-parameter
universe builders provably run the SAME exit rule rather than two copies of it."""
import numpy as np

TRIGGER_R = 3.0
TRAIL_ATR = 1.0


def resolve_trail(e, n, long_, entry, sl0, tp, risk_dist, h, l, atr):
    """Walk bars from e; return (r, exit_idx) under the s3_a1 trail. None if unresolved."""
    stop = sl0
    for k in range(e, n):
        # --- exit check against the stop in force (set at close of bar k-1) ---
        if long_:
            hit_stop = l[k] <= stop
            hit_tp = h[k] >= tp
        else:
            hit_stop = h[k] >= stop
            hit_tp = l[k] <= tp
        if hit_stop:                              # stop wins ties
            r = (stop - entry) / risk_dist if long_ else (entry - stop) / risk_dist
            return r, k
        if hit_tp:
            r = (tp - entry) / risk_dist if long_ else (entry - tp) / risk_dist
            return r, k

        # --- bar k has closed: update the trail from ITS OWN excursion ---
        a = atr[k]
        if not (np.isfinite(a) and a > 0):
            continue
        if long_:
            mfe = (h[k] - entry) / risk_dist
            if mfe >= TRIGGER_R:
                cand = h[k] - TRAIL_ATR * a
                if cand > stop:
                    stop = cand
        else:
            mfe = (entry - l[k]) / risk_dist
            if mfe >= TRIGGER_R:
                cand = l[k] + TRAIL_ATR * a
                if cand < stop:
                    stop = cand
    return None
