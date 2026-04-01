#!/usr/bin/env python3
"""
Merge Three Config Sources into Unified config.yaml
=====================================================
Combines:
  1. 3yr walk-forward (37 all-weather configs) — tier 0, highest confidence
  2. 1yr bear-tested (378 configs)             — tier 1
  3. Bull-period validated (241 configs)        — tier 1

Priority: 3yr walk-forward always wins. Between bear/bull, higher avg_r_per_trade wins.
Per-symbol: at most ONE long config + ONE short config.
"""

import argparse
import pandas as pd
import yaml
import shutil
from datetime import datetime
from collections import defaultdict

# === File paths ===
BEAR_CSV = "1yr_improved_validated.csv"
ALLWEATHER_CSV = "3yr_walkforward_validated.csv"
BULL_CSV = "bull_validated_configs.csv"
CONFIG_FILE = "config.yaml"
AUDIT_CSV = "merged_configs_audit.csv"
CONFLICTS_CSV = "merge_conflicts.csv"

LONG_TYPES = {"REG_BULL", "HID_BULL"}
SHORT_TYPES = {"REG_BEAR", "HID_BEAR"}


def load_sources():
    """Load and normalize all three CSV sources."""
    # Source 1: bear-tested (1yr improved)
    bear = pd.read_csv(BEAR_CSV)
    bear["source"] = "bear_tested"
    bear["tier"] = 1
    bear["avg_r_per_trade"] = bear["test_r"] / bear["test_trades"]
    bear["test_trades_norm"] = bear["test_trades"]
    bear = bear[["symbol", "div_type", "atr", "rr", "avg_r_per_trade", "source", "tier", "test_trades_norm"]]

    # Source 2: 3yr walk-forward (different column names)
    aw = pd.read_csv(ALLWEATHER_CSV)
    aw["source"] = "all_weather"
    aw["tier"] = 0
    aw["avg_r_per_trade"] = aw["total_test_r"] / aw["total_test_trades"]
    aw["test_trades_norm"] = aw["total_test_trades"]
    aw = aw[["symbol", "div_type", "atr", "rr", "avg_r_per_trade", "source", "tier", "test_trades_norm"]]

    # Source 3: bull-period validated
    bull = pd.read_csv(BULL_CSV)
    bull["source"] = "bull_tested"
    bull["tier"] = 1
    bull["avg_r_per_trade"] = bull["test_r"] / bull["test_trades"]
    bull["test_trades_norm"] = bull["test_trades"]
    bull = bull[["symbol", "div_type", "atr", "rr", "avg_r_per_trade", "source", "tier", "test_trades_norm"]]

    print(f"Loaded sources:")
    print(f"  Bear-tested:  {len(bear)} configs")
    print(f"  All-weather:  {len(aw)} configs")
    print(f"  Bull-tested:  {len(bull)} configs")

    combined = pd.concat([bear, aw, bull], ignore_index=True)
    print(f"  Combined:     {len(combined)} total rows")
    return combined


def resolve_conflicts(df):
    """For each (symbol, div_type), pick the winner by priority."""
    conflicts = []
    winners = []

    for (sym, dt), group in df.groupby(["symbol", "div_type"]):
        if len(group) == 1:
            winners.append(group.iloc[0])
            continue

        # Multiple sources for same (symbol, div_type)
        sorted_g = group.sort_values(["tier", "avg_r_per_trade"], ascending=[True, False])
        winner = sorted_g.iloc[0]
        losers = sorted_g.iloc[1:]

        winners.append(winner)

        for _, loser in losers.iterrows():
            if winner["tier"] < loser["tier"]:
                reason = f"tier {winner['tier']} ({winner['source']}) beats tier {loser['tier']} ({loser['source']})"
            else:
                reason = (
                    f"higher avg_r ({winner['avg_r_per_trade']:.3f} vs {loser['avg_r_per_trade']:.3f}), "
                    f"{winner['source']} over {loser['source']}"
                )
            conflicts.append({
                "symbol": sym,
                "div_type": dt,
                "winner_source": winner["source"],
                "winner_avg_r": round(winner["avg_r_per_trade"], 4),
                "loser_source": loser["source"],
                "loser_avg_r": round(loser["avg_r_per_trade"], 4),
                "reason": reason,
            })

    resolved = pd.DataFrame(winners)
    conflicts_df = pd.DataFrame(conflicts) if conflicts else pd.DataFrame()

    print(f"\nConflict resolution:")
    print(f"  Unique (symbol, div_type) pairs: {len(resolved)}")
    print(f"  Conflicts resolved: {len(conflicts_df)}")
    if len(conflicts_df):
        aw_wins = (conflicts_df["winner_source"] == "all_weather").sum()
        bear_wins = (conflicts_df["winner_source"] == "bear_tested").sum()
        bull_wins = (conflicts_df["winner_source"] == "bull_tested").sum()
        print(f"    All-weather won: {aw_wins}")
        print(f"    Bear-tested won: {bear_wins}")
        print(f"    Bull-tested won: {bull_wins}")

    return resolved, conflicts_df


def select_best_per_side(df):
    """Keep at most one long and one short config per symbol."""
    longs = df[df["div_type"].isin(LONG_TYPES)].copy()
    shorts = df[df["div_type"].isin(SHORT_TYPES)].copy()

    best_long = (
        longs.sort_values("avg_r_per_trade", ascending=False)
        .drop_duplicates("symbol", keep="first")
    )
    best_short = (
        shorts.sort_values("avg_r_per_trade", ascending=False)
        .drop_duplicates("symbol", keep="first")
    )

    final = pd.concat([best_long, best_short], ignore_index=True)

    print(f"\nPer-side selection:")
    print(f"  Long candidates: {len(longs)} → best per symbol: {len(best_long)}")
    print(f"  Short candidates: {len(shorts)} → best per symbol: {len(best_short)}")
    print(f"  Final configs: {len(final)}")
    return final


def build_symbols_dict(df):
    """Build YAML-ready symbols dict matching SymbolRRConfig format."""
    symbols = {}

    for sym in sorted(df["symbol"].unique()):
        sym_rows = df[df["symbol"] == sym]
        configs = []
        for _, row in sym_rows.iterrows():
            configs.append({
                "divergence_type": str(row["div_type"]),
                "rr": float(round(row["rr"], 1)),
                "atr_mult": float(round(row["atr"], 1)),
            })
        symbols[sym] = {"enabled": True, "configs": configs}

    return symbols


def write_config(symbols, dry_run=False):
    """Backup existing config.yaml, preserve non-symbol settings, write merged config."""
    with open(CONFIG_FILE) as f:
        existing = yaml.safe_load(f)

    # Stats
    total_symbols = len(symbols)
    dual_side = sum(1 for s in symbols.values() if len(s["configs"]) > 1)
    total_configs = sum(len(s["configs"]) for s in symbols.values())
    long_count = sum(
        1 for s in symbols.values() for c in s["configs"]
        if c["divergence_type"] in LONG_TYPES
    )
    short_count = sum(
        1 for s in symbols.values() for c in s["configs"]
        if c["divergence_type"] in SHORT_TYPES
    )

    # Update strategy metadata
    existing["strategy"] = {
        "name": "1h_merged_allweather_bear_bull",
        "description": f"{total_configs} configs across {total_symbols} symbols ({long_count} long + {short_count} short)",
        "version": "4.0_merged",
        "timeframe": "60",
    }
    existing["symbols"] = symbols

    if dry_run:
        print(f"\n[DRY RUN] Would write config.yaml with:")
    else:
        backup = f"{CONFIG_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(CONFIG_FILE, backup)
        print(f"\nBacked up to {backup}")

        with open(CONFIG_FILE, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Wrote {CONFIG_FILE}")

    print(f"  Symbols: {total_symbols} ({dual_side} dual-side, {total_symbols - dual_side} single-side)")
    print(f"  Total configs: {total_configs}")
    print(f"  Long: {long_count} | Short: {short_count}")


def save_audit(df):
    """Save audit trail CSV."""
    out = df[["symbol", "div_type", "atr", "rr", "avg_r_per_trade", "source", "tier", "test_trades_norm"]].copy()
    out = out.sort_values(["symbol", "div_type"]).reset_index(drop=True)
    out.to_csv(AUDIT_CSV, index=False)
    print(f"\nAudit trail saved to {AUDIT_CSV} ({len(out)} rows)")


def print_report(final_df, conflicts_df):
    """Print summary report."""
    print("\n" + "=" * 60)
    print("MERGE REPORT")
    print("=" * 60)

    # Source breakdown
    source_counts = final_df["source"].value_counts()
    print(f"\nFinal configs by source:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    # Side breakdown
    long_df = final_df[final_df["div_type"].isin(LONG_TYPES)]
    short_df = final_df[final_df["div_type"].isin(SHORT_TYPES)]
    print(f"\nBy side:")
    print(f"  Long:  {len(long_df)}")
    print(f"  Short: {len(short_df)}")

    # Divergence type breakdown
    print(f"\nBy divergence type:")
    for dt, cnt in final_df["div_type"].value_counts().items():
        print(f"  {dt}: {cnt}")

    # Source × side breakdown
    print(f"\nSource × side:")
    for src in final_df["source"].unique():
        src_df = final_df[final_df["source"] == src]
        src_long = src_df[src_df["div_type"].isin(LONG_TYPES)]
        src_short = src_df[src_df["div_type"].isin(SHORT_TYPES)]
        print(f"  {src}: {len(src_long)} long + {len(src_short)} short = {len(src_df)}")

    # Unique symbols
    total_syms = final_df["symbol"].nunique()
    dual = final_df.groupby("symbol").size()
    dual_count = (dual > 1).sum()
    print(f"\nSymbols: {total_syms} total ({dual_count} dual-side, {total_syms - dual_count} single-side)")

    # Avg R stats
    print(f"\nAvg R per trade stats:")
    print(f"  Mean:   {final_df['avg_r_per_trade'].mean():.3f}")
    print(f"  Median: {final_df['avg_r_per_trade'].median():.3f}")
    print(f"  Min:    {final_df['avg_r_per_trade'].min():.3f}")
    print(f"  Max:    {final_df['avg_r_per_trade'].max():.3f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge three config sources into unified config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Review without writing config.yaml")
    args = parser.parse_args()

    # Step 1: Load all sources
    combined = load_sources()

    # Step 2: Resolve conflicts
    resolved, conflicts_df = resolve_conflicts(combined)

    # Step 3: Best per side per symbol
    final = select_best_per_side(resolved)

    # Step 4: Build symbols dict and write config
    symbols = build_symbols_dict(final)
    write_config(symbols, dry_run=args.dry_run)

    # Step 5: Audit trail + report
    save_audit(final)
    if len(conflicts_df):
        conflicts_df.to_csv(CONFLICTS_CSV, index=False)
        print(f"Conflicts saved to {CONFLICTS_CSV} ({len(conflicts_df)} rows)")

    print_report(final, conflicts_df)


if __name__ == "__main__":
    main()
