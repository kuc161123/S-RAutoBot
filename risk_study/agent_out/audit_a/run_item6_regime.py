#!/usr/bin/env python3
"""AUDIT-A item 6: regime-tier window length sweep, 20 (live) vs {10, 40, 60}.
Tier thresholds themselves (wr>=0.18/avg_r>=0.15 etc) held fixed -- only the trailing
trade-count window used to compute wr/avg_r changes. Also includes a REGIME-OFF run
(flat tapered risk, no multiplier) as the zero point, for reference (already covered in
risk_study/results/variations.csv but reproduced here on the halted baseline for a
consistent IS/holdout table).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

WINDOWS = [10, 20, 40, 60]


def main():
    chop = H.load_chop_cache()
    d = H.load_gated_halted()
    print(f"gated+halted population: {len(d)} rows")

    rows = []
    for w in WINDOWS:
        H.reset_globals()
        tag = "live" if w == 20 else f"window={w}"
        H.P.get_regime = H.make_get_regime(window=w)
        row = H.run_variant(f"regime_window={w} ({tag})", d, chop,
                             cfg_diff=f"regime window: 20 -> {w}", note=tag)
        rows.append(row)
        H.reset_globals()
        pd.DataFrame(rows).to_csv(HERE / "item6_regime_results.csv", index=False)

    # REGIME OFF (scenario='chop_only' keeps CHOP + taper, drops the multiplier)
    H.reset_globals()
    row = H.run_variant("regime OFF (flat tapered risk)", d, chop,
                         cfg_diff="scenario: production -> chop_only", note="no regime multiplier",
                         scenario="chop_only")
    rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "item6_regime_results.csv", index=False)

    print("\nwrote item6_regime_results.csv")


if __name__ == "__main__":
    main()
