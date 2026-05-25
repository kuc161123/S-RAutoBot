#!/usr/bin/env python3
"""
Walk-forward config optimizer.

For each enabled symbol, replay the bot's signal detection across the
*full* cached history, then on each divergence sweep a small grid of
(atr_mult, rr) combinations using a single 5m walk per atr_mult.

Splits chronologically into TRAIN and TEST, picks per (symbol, div_type)
the combination that:
  - has >= MIN_TRAIN_TRADES on train
  - has >= MIN_TEST_TRADES on test
  - has positive avg_R on BOTH halves (stability)
  - maximises TEST avg_R among survivors

Outputs:
  - grid_trades.csv          (every combo, one row per micro-trade)
  - optimized_trades.csv     (only selected combos)
  - config_optimized.yaml    (new bot config, both long and short where validated)
  - selection_report.csv     (per symbol+div decisions for transparency)
"""
import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'autobot'))

from autobot.core.divergence_detector import (
    prepare_dataframe, detect_divergences,
    DIV_REG_BULL, DIV_REG_BEAR, DIV_HID_BULL, DIV_HID_BEAR,
)

CACHE_1H = ROOT / 'cache_3yr_1h'
CACHE_5M = ROOT / 'cache_data' / '5m'
MAX_WAIT_CANDLES = 12
MAX_HOLD_DAYS = 30

ATR_MULT_GRID = [1.0, 1.5, 2.0]
RR_GRID = [3.0, 5.0, 8.0, 10.0]
ALL_DIV_TYPES = [DIV_REG_BULL, DIV_REG_BEAR, DIV_HID_BULL, DIV_HID_BEAR]

# Walk-forward boundary (inclusive lower bound for TEST)
TRAIN_END_EXCL = pd.Timestamp('2025-11-01')

MIN_TRAIN_TRADES = 15
MIN_TEST_TRADES = 5


def evaluate_signal(df5, entry_time, entry_price, atr, side):
    """Single 5m walk that returns, for each (atr_mult, rr) combo, the outcome.

    Returns list of dicts: {atr_mult, rr, exit_time, outcome, r_result}.
    """
    rows = []
    if df5 is None or df5.empty:
        return rows
    sub = df5[df5.index > entry_time]
    if sub.empty:
        return rows
    deadline = entry_time + pd.Timedelta(days=MAX_HOLD_DAYS)
    sub = sub[sub.index <= deadline]
    if sub.empty:
        return rows

    highs = sub['high'].values
    lows = sub['low'].values
    idx_arr = sub.index.values

    for am in ATR_MULT_GRID:
        sl_distance = atr * am
        if side == 'long':
            sl_price = entry_price - sl_distance
            tp_prices = [entry_price + sl_distance * rr for rr in RR_GRID]
        else:
            sl_price = entry_price + sl_distance
            tp_prices = [entry_price - sl_distance * rr for rr in RR_GRID]

        # Per-RR state: have we hit TP yet?
        tp_hit = [None] * len(RR_GRID)  # index k -> exit_time
        sl_hit = None

        for k in range(len(sub)):
            h, l = highs[k], lows[k]
            if side == 'long':
                this_sl = l <= sl_price
                # Update TP hits (pessimistic on same-candle both: SL first if also SL)
                for j, tp in enumerate(tp_prices):
                    if tp_hit[j] is None and h >= tp:
                        # Same-candle both -> SL wins
                        if this_sl:
                            pass  # don't mark TP; SL will be assigned below
                        else:
                            tp_hit[j] = idx_arr[k]
                if this_sl:
                    sl_hit = idx_arr[k]
                    break
            else:
                this_sl = h >= sl_price
                for j, tp in enumerate(tp_prices):
                    if tp_hit[j] is None and l <= tp:
                        if this_sl:
                            pass
                        else:
                            tp_hit[j] = idx_arr[k]
                if this_sl:
                    sl_hit = idx_arr[k]
                    break

        for j, rr in enumerate(RR_GRID):
            if tp_hit[j] is not None:
                rows.append({'atr_mult': am, 'rr': rr,
                             'exit_time': pd.Timestamp(tp_hit[j]),
                             'outcome': 'tp', 'r_result': rr})
            elif sl_hit is not None:
                rows.append({'atr_mult': am, 'rr': rr,
                             'exit_time': pd.Timestamp(sl_hit),
                             'outcome': 'sl', 'r_result': -1.0})
            # else: open/timeout -> skip (no row)
    return rows


def process_symbol(args):
    sym, start_iso, end_iso = args
    try:
        f1 = CACHE_1H / f'{sym}.parquet'
        if not f1.exists():
            return []
        df1 = pd.read_parquet(f1)
        if df1.empty or 'start' not in df1.columns:
            return []
        df1 = df1.sort_values('start').drop_duplicates(subset='start').reset_index(drop=True)
        df1 = df1.set_index('start')
        for c in ('open', 'high', 'low', 'close'):
            df1[c] = df1[c].astype(float)
        df1 = prepare_dataframe(df1)

        signals = detect_divergences(df1, sym, allowed_types=ALL_DIV_TYPES)
        if not signals:
            return []

        f5 = CACHE_5M / f'{sym}.parquet'
        df5 = None
        if f5.exists():
            try:
                tmp = pd.read_parquet(f5)
                if not tmp.empty and 'start' in tmp.columns:
                    tmp = tmp.sort_values('start').drop_duplicates(subset='start').reset_index(drop=True)
                    df5 = tmp.set_index('start')
                    for c in ('high', 'low', 'open', 'close'):
                        df5[c] = df5[c].astype(float)
            except Exception:
                df5 = None

        start_ts = pd.Timestamp(start_iso)
        end_ts = pd.Timestamp(end_iso)
        close = df1['close'].values
        opens = df1['open'].values
        atrs = df1['atr'].values
        emas = df1['daily_ema'].values
        idx = df1.index

        out = []
        for sig in signals:
            i = sig.divergence_idx
            confirmed_at = None
            for k in range(1, MAX_WAIT_CANDLES + 1):
                j = i + k
                if j >= len(df1) - 1:
                    break
                c = close[j]
                if sig.side == 'long' and c > sig.swing_level:
                    if not np.isnan(emas[j]) and c > emas[j]:
                        confirmed_at = j; break
                if sig.side == 'short' and c < sig.swing_level:
                    if not np.isnan(emas[j]) and c < emas[j]:
                        confirmed_at = j; break
            if confirmed_at is None:
                continue
            j = confirmed_at
            entry_j = j + 1
            entry_time = idx[entry_j]
            if entry_time < start_ts or entry_time > end_ts:
                continue
            entry_price = opens[entry_j]
            atr = atrs[j]
            if np.isnan(atr) or atr <= 0:
                continue

            combo_rows = evaluate_signal(df5, entry_time, entry_price, atr, sig.side)
            if not combo_rows:
                continue

            for r in combo_rows:
                # Compute sl_price for the chosen atr_mult so backtest sim works
                am = r['atr_mult']
                sld = atr * am
                if sig.side == 'long':
                    sl_p = entry_price - sld
                else:
                    sl_p = entry_price + sld
                out.append({
                    'entry_time': entry_time,
                    'exit_time': r['exit_time'],
                    'entry_price': entry_price,
                    'sl_price': sl_p,
                    'outcome': r['outcome'],
                    'r_result': r['r_result'],
                    'side': sig.side,
                    'fee_drag': 0.0,
                    'symbol': sym,
                    'div_type': sig.divergence_code,
                    'rr_setting': r['rr'],
                    'atr_mult': am,
                })
        return out
    except Exception as e:
        return [{'__error__': f'{sym}: {e}'}]


def load_enabled_symbols(config_path: Path):
    cfg = yaml.safe_load(open(config_path))
    return [s for s, v in (cfg.get('symbols') or {}).items() if v.get('enabled')]


def load_universe(config_path: Path, mode: str):
    """Return the symbol universe to scan.

    mode='enabled' -> only enabled symbols from config.yaml
    mode='all'     -> every symbol with a cached 1h parquet (lets the optimiser
                       discover symbols not currently in config). Validation rules
                       below still gate inclusion.
    """
    if mode == 'enabled':
        return load_enabled_symbols(config_path)
    if mode == 'all':
        cached = sorted({p.stem for p in CACHE_1H.glob('*.parquet')})
        enabled = set(load_enabled_symbols(config_path))
        new_pool = [s for s in cached if s not in enabled]
        print(f'Universe: {len(enabled)} enabled + {len(new_pool)} discovery candidates = {len(cached)} total')
        return cached
    raise ValueError(f'unknown mode {mode!r}')


def run_extraction(symbols, start, end, workers, out_csv):
    work = [(s, start, end) for s in symbols]
    t0 = time.time()
    all_rows = []
    errors = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_symbol, w): w[0] for w in work}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                rs = fut.result()
            except Exception as e:
                errors.append(f'{sym}: {e}')
                completed += 1
                continue
            for r in rs:
                if isinstance(r, dict) and '__error__' in r:
                    errors.append(r['__error__'])
                else:
                    all_rows.append(r)
            completed += 1
            if completed % 25 == 0 or completed == len(work):
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(work) - completed) / rate if rate > 0 else 0
                print(f'  {completed}/{len(work)}  rows={len(all_rows):,}  err={len(errors)}  {el:.0f}s eta {eta:.0f}s', flush=True)
    print(f'\nExtraction done: {len(all_rows):,} rows in {time.time()-t0:.0f}s ({len(errors)} errors)')
    if errors:
        for e in errors[:5]:
            print('  ', e)
    df = pd.DataFrame(all_rows)
    df = df.sort_values('entry_time').reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}')
    return df


def select_configs(grid_df):
    """Per (symbol, div_type), pick best (atr_mult, rr) using walk-forward."""
    train_mask = grid_df['entry_time'] < TRAIN_END_EXCL
    train = grid_df[train_mask]
    test = grid_df[~train_mask]

    decisions = []  # rows for selection report
    chosen = {}     # (sym, div_type) -> {atr_mult, rr, ...}

    keys = grid_df[['symbol', 'div_type']].drop_duplicates().itertuples(index=False)
    for sym, dt in keys:
        sub_tr = train[(train.symbol == sym) & (train.div_type == dt)]
        sub_te = test[(test.symbol == sym) & (test.div_type == dt)]
        best = None
        for am in ATR_MULT_GRID:
            for rr in RR_GRID:
                tr_combo = sub_tr[(sub_tr.atr_mult == am) & (sub_tr.rr_setting == rr)]
                te_combo = sub_te[(sub_te.atr_mult == am) & (sub_te.rr_setting == rr)]
                n_tr, n_te = len(tr_combo), len(te_combo)
                if n_tr < MIN_TRAIN_TRADES or n_te < MIN_TEST_TRADES:
                    continue
                avg_tr = tr_combo.r_result.mean()
                avg_te = te_combo.r_result.mean()
                if avg_tr <= 0 or avg_te <= 0:
                    continue
                # FIX: score on TRAIN only. TEST positivity is a robustness gate.
                # Scoring on TEST leaks the OOS window into selection, inflating
                # subsequent backtest results.
                score = avg_tr
                cand = {
                    'symbol': sym, 'div_type': dt, 'atr_mult': am, 'rr': rr,
                    'train_trades': n_tr, 'train_avg_r': avg_tr,
                    'test_trades': n_te, 'test_avg_r': avg_te,
                    'score': score,
                }
                if best is None or score > best['score']:
                    best = cand
        if best is not None:
            chosen[(sym, dt)] = best
            decisions.append({**best, 'selected': True})
        else:
            decisions.append({
                'symbol': sym, 'div_type': dt,
                'selected': False,
            })
    return chosen, pd.DataFrame(decisions)


def emit_config(chosen, template_path, out_path):
    tpl = yaml.safe_load(open(template_path))
    new_syms = {}
    side_long = {DIV_REG_BULL, DIV_HID_BULL}
    side_short = {DIV_REG_BEAR, DIV_HID_BEAR}

    # Build per-symbol configs from chosen
    per_sym = defaultdict(list)
    for (sym, dt), info in chosen.items():
        per_sym[sym].append({
            'divergence_type': dt,
            'rr': float(info['rr']),
            'atr_mult': float(info['atr_mult']),
        })

    # Keep only symbols that have at least one validated config
    for sym, configs in sorted(per_sym.items()):
        # Sort configs: long first, then short
        configs_sorted = sorted(configs, key=lambda c: (c['divergence_type'] not in side_long, c['divergence_type']))
        new_syms[sym] = {'enabled': True, 'configs': configs_sorted}

    out = dict(tpl)
    out['symbols'] = new_syms
    # Annotate strategy for traceability
    out['strategy'] = dict(out.get('strategy', {}))
    out['strategy']['description'] = (
        f"WF-optimized {len(new_syms)} symbols, {sum(len(v['configs']) for v in new_syms.values())} configs "
        f"(train<{TRAIN_END_EXCL.date()}, test through 2026-05-25)"
    )
    out['strategy']['version'] = (out['strategy'].get('version', '4.0') + '_wf')

    with open(out_path, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print(f'Wrote {out_path} with {len(new_syms)} symbols / {sum(len(v["configs"]) for v in new_syms.values())} configs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-start', default='2024-05-01')
    ap.add_argument('--full-end',   default='2026-05-25')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument('--skip-extract', action='store_true',
                    help='Reuse grid_trades.csv if present')
    ap.add_argument('--universe', choices=['enabled', 'all'], default='all',
                    help="'enabled' = currently-enabled symbols only; 'all' = every cached symbol (allows discovery)")
    args = ap.parse_args()

    grid_csv = ROOT / 'grid_trades.csv'
    if args.skip_extract and grid_csv.exists():
        print(f'Reusing {grid_csv}')
        grid_df = pd.read_csv(grid_csv, parse_dates=['entry_time', 'exit_time'])
    else:
        symbols = load_universe(ROOT / 'config.yaml', args.universe)
        print(f'Symbols: {len(symbols)}  grid: {len(ATR_MULT_GRID)} atr_mult x {len(RR_GRID)} rr = {len(ATR_MULT_GRID)*len(RR_GRID)} combos / signal / div')
        grid_df = run_extraction(symbols, args.full_start, args.full_end, args.workers, grid_csv)

    # Selection
    print('\n=== Walk-forward selection ===')
    chosen, decisions = select_configs(grid_df)
    print(f'Validated configs: {len(chosen)}')
    print(f'Symbols with at least one config: {len({k[0] for k in chosen})}')
    decisions.to_csv(ROOT / 'selection_report.csv', index=False)
    print(f'Saved selection_report.csv')

    # Emit config
    emit_config(chosen, ROOT / 'config.yaml', ROOT / 'config_optimized.yaml')

    # Build optimized trades CSV (filter grid to selected combos, test window only)
    test_mask = grid_df['entry_time'] >= TRAIN_END_EXCL
    test = grid_df[test_mask]
    sel_keys = {(s, d, info['atr_mult'], info['rr']) for (s, d), info in chosen.items()}
    test['key'] = list(zip(test.symbol, test.div_type, test.atr_mult, test.rr_setting))
    optimized = test[test['key'].isin(sel_keys)].drop(columns='key').reset_index(drop=True)
    optimized.to_csv(ROOT / 'optimized_trades.csv', index=False)
    print(f'optimized_trades.csv: {len(optimized):,} test-window trades under new config')


if __name__ == '__main__':
    main()
