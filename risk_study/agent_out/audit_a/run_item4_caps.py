#!/usr/bin/env python3
"""AUDIT-A item 4: net_directional_cap and gross_open_risk_cap sweep, on the CURRENT
config (short_gate=False -- risk_study/search_roidd.py and variations.py both still
carry short_gate=True via LIVE_SIM defaults and were NOT re-checked against the current
config; this script fixes that so results here are the honest current-config numbers).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

NET_DIR = [0.05, 0.10, 0.15, 0.20, None]
GROSS = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, None]


def main():
    chop = H.load_chop_cache()
    d = H.load_gated_halted()
    print(f"gated+halted population: {len(d)} rows")

    rows = []
    print("\n-- net_directional_cap sweep (gross fixed at live 0.10) --")
    for nd in NET_DIR:
        tag = "live" if nd == 0.10 else ("none" if nd is None else f"{nd}")
        row = H.run_variant(f"net_dir_cap={nd} ({tag})", d, chop,
                             cfg_diff=f"net_directional_cap: 0.10 -> {nd}", note=tag,
                             net_dir_cap=nd, open_risk_cap=0.10)
        row["axis"] = "net_dir"
        rows.append(row)
        pd.DataFrame(rows).to_csv(HERE / "item4_caps_results.csv", index=False)

    print("\n-- gross_open_risk_cap sweep (net_dir fixed at live 0.10) --")
    for gr in GROSS:
        tag = "live" if gr == 0.10 else ("none" if gr is None else f"{gr}")
        row = H.run_variant(f"gross_cap={gr} ({tag})", d, chop,
                             cfg_diff=f"gross_open_risk_cap: 0.10 -> {gr}", note=tag,
                             net_dir_cap=0.10, open_risk_cap=gr)
        row["axis"] = "gross"
        rows.append(row)
        pd.DataFrame(rows).to_csv(HERE / "item4_caps_results.csv", index=False)

    print("\nwrote item4_caps_results.csv")


if __name__ == "__main__":
    main()
