import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

def summarize_scaled(sub, scale_mask, factor, label=""):
    s = sub.copy()
    s.loc[scale_mask, "net_r"] = s.loc[scale_mask, "net_r"] * factor
    d = summarize(s, label)
    w10, _ = worst_n_days(s, 10)
    d["worst10"] = w10
    return d

print("=== Family 3: side-asymmetric sizing, ALWAYS (scale ALL shorts) ===")
for pname, sub in [("train", train), ("hold", hold)]:
    base = summarize(sub, "base")
    w10b, _ = worst_n_days(sub, 10)
    print(f"{pname} baseline: n={base['n']} mean_r={base['mean_r']:.4f} total_r={base['total_r']:.1f} worst10={w10b:.1f}")
    for f in [0.5, 0.65, 0.8]:
        mask = sub.side == "short"
        d = summarize_scaled(sub, mask, f, f"scale_all_shorts={f}")
        print(f"  factor={f}: total_r={d['total_r']:.1f} (delta {d['total_r']-base['total_r']:+.1f})  "
              f"worst10={d['worst10']:.1f} (delta {d['worst10']-w10b:+.1f})")

print("\n=== Family 3b: scale shorts ONLY when density_all_72h>160 (conditional sizing, not hard skip) ===")
for pname, sub in [("train", train), ("hold", hold)]:
    base = summarize(sub, "base")
    w10b, _ = worst_n_days(sub, 10)
    for f in [0.0, 0.5, 0.65, 0.8]:
        mask = (sub.side == "short") & (sub["density_all_72h"] > 160)
        d = summarize_scaled(sub, mask, f, f"cond_scale={f}")
        print(f"{pname} factor={f} (n_scaled={mask.sum()}): total_r={d['total_r']:.1f} (delta {d['total_r']-base['total_r']:+.1f})  "
              f"worst10={d['worst10']:.1f} (delta {d['worst10']-w10b:+.1f})")

print("\n=== combo: density_all_72h>160 OR rvol_30d_causal>0.0298 (stack best state gate on best density gate) ===")
for pname, sub in [("train", train), ("hold", hold)]:
    base = summarize(sub, "base")
    w10b, _ = worst_n_days(sub, 10)
    is_short = sub.side == "short"
    mask_skip = is_short & ((sub["density_all_72h"] > 160) | (sub["rvol_30d_causal"] > 0.0298))
    kept = sub[~mask_skip]
    d = summarize(kept, "combo")
    w10, _ = worst_n_days(kept, 10)
    print(f"{pname}: n_skipped={mask_skip.sum()} total_r={d['total_r']:.1f} (delta {d['total_r']-base['total_r']:+.1f})  "
          f"worst10={w10:.1f} (delta {w10-w10b:+.1f})")
