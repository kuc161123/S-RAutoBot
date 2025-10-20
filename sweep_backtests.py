"""
Offline parameterized backtest runner for Trend Pullback.

Usage:
  python sweep_backtests.py

Outputs:
  backtests/results_by_variant.csv
  backtests/results_by_symbol_variant.csv
"""
import os
import csv
from typing import List, Dict

try:
    import yaml  # optional
except Exception:
    yaml = None

from enhanced_backtester import TrendVariantBacktester, Variant, summarize_results


def load_symbols_from_config(path: str = 'config.yaml', fallback: List[str] = None) -> List[str]:
    if fallback is None:
        fallback = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BCHUSDT', 'DOGEUSDT']
    if yaml is None:
        return fallback
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        syms = cfg.get('trade', {}).get('symbols') or []
        if not syms:
            return fallback
        return [str(s) for s in syms[:20]]  # cap for speed
    except Exception:
        return fallback


def ensure_dir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def main():
    symbols = load_symbols_from_config()
    print(f"Symbols: {symbols}")

    # Define a small pilot grid — expand as needed
    variants: List[Variant] = [
        Variant(name='opt_breakoutSL_030_hold5h', div_mode='optional', bos_hold_minutes=300, sl_mode='breakout', breakout_sl_buffer_atr=0.30, cancel_on_reentry=True),
        Variant(name='strict_or_breakoutSL_030_hold5h', div_mode='strict', bos_hold_minutes=300, sl_mode='breakout', breakout_sl_buffer_atr=0.30, cancel_on_reentry=True),
        Variant(name='strict_or_breakoutSL_020_hold5h', div_mode='strict', bos_hold_minutes=300, sl_mode='breakout', breakout_sl_buffer_atr=0.20, cancel_on_reentry=True),
    ]

    runner = TrendVariantBacktester()
    all_results: Dict[str, Dict[str, list]] = {}
    for v in variants:
        print(f"Running variant: {v.name}")
        sym_map: Dict[str, list] = {}
        for sym in symbols:
            res = runner.run_symbol(sym, v)
            sym_map[sym] = res
            print(f"  {sym}: {len(res)} trades")
        all_results[v.name] = sym_map

    # Summaries
    dfv, dfs = summarize_results(all_results)
    ensure_dir('backtests')
    dfv.to_csv('backtests/results_by_variant.csv', index=False)
    dfs.to_csv('backtests/results_by_symbol_variant.csv', index=False)
    print("Wrote backtests/results_by_variant.csv and results_by_symbol_variant.csv")


if __name__ == '__main__':
    main()

