#!/usr/bin/env python3
"""
End-to-end accurate trade extraction:
  - Uses the bot's actual divergence_detector for signal generation
  - Replays BOS confirmation + EMA gate the way bot.py does
  - Uses 5m candles to determine intra-candle SL/TP hit order
  - Reads per-symbol configs straight from config.yaml

Outputs a trades CSV in the same shape backtest_production_correct.py expects.
"""
import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'autobot'))

from autobot.core.divergence_detector import (
    prepare_dataframe, detect_divergences, check_bos,
    DIV_REG_BULL, DIV_REG_BEAR, DIV_HID_BULL, DIV_HID_BEAR,
)

CACHE_1H = ROOT / 'cache_3yr_1h'
CACHE_5M = ROOT / 'cache_data' / '5m'
MAX_WAIT_CANDLES = 12  # matches bot.py default


def load_symbol_configs(config_path: Path):
    """Map symbol -> {div_type: {rr, atr_mult}}."""
    cfg = yaml.safe_load(open(config_path))
    out = {}
    for sym, sv in (cfg.get('symbols') or {}).items():
        if not sv.get('enabled'):
            continue
        per_div = {}
        for c in sv.get('configs') or []:
            dt = c.get('divergence_type')
            if not dt:
                continue
            per_div[dt] = {
                'rr': float(c.get('rr', 2.0)),
                'atr_mult': float(c.get('atr_mult', 1.0)),
            }
        if per_div:
            out[sym] = per_div
    return out


def walk_5m_for_exit(df5: pd.DataFrame, entry_time: pd.Timestamp,
                    side: str, sl_price: float, tp_price: float,
                    max_hold_days: int = 30):
    """Walk 5m candles from entry_time onward, return (exit_time, outcome).

    outcome: 'tp' | 'sl' | 'timeout'
    If a single candle straddles both SL and TP, treat SL as hit (pessimistic).
    """
    if df5 is None or df5.empty:
        return None, 'no5m'
    # Slice starting at entry_time
    sub = df5[df5.index > entry_time]
    if sub.empty:
        return None, 'no5m'
    deadline = entry_time + pd.Timedelta(days=max_hold_days)
    sub = sub[sub.index <= deadline]
    if sub.empty:
        return None, 'timeout'

    highs = sub['high'].values
    lows = sub['low'].values
    idx = sub.index

    if side == 'long':
        for k in range(len(sub)):
            hit_sl = lows[k] <= sl_price
            hit_tp = highs[k] >= tp_price
            if hit_sl and hit_tp:
                return idx[k], 'sl'
            if hit_sl:
                return idx[k], 'sl'
            if hit_tp:
                return idx[k], 'tp'
    else:
        for k in range(len(sub)):
            hit_sl = highs[k] >= sl_price
            hit_tp = lows[k] <= tp_price
            if hit_sl and hit_tp:
                return idx[k], 'sl'
            if hit_sl:
                return idx[k], 'sl'
            if hit_tp:
                return idx[k], 'tp'
    return None, 'timeout'


def process_symbol(args):
    """Process one symbol; return list of trade dicts."""
    sym, per_div, start_iso, end_iso = args
    try:
        f1 = CACHE_1H / f'{sym}.parquet'
        if not f1.exists():
            return []
        df1 = pd.read_parquet(f1)
        if df1.empty or 'start' not in df1.columns:
            return []
        df1 = df1.sort_values('start').drop_duplicates(subset='start').reset_index(drop=True)
        df1 = df1.set_index('start')
        # Ensure proper dtype on close etc.
        for c in ('open', 'high', 'low', 'close'):
            df1[c] = df1[c].astype(float)

        df1 = prepare_dataframe(df1)

        allowed = list(per_div.keys())
        signals = detect_divergences(df1, sym, allowed_types=allowed)
        if not signals:
            return []

        # 5m data (optional but preferred)
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
        out = []

        close = df1['close'].values
        opens = df1['open'].values
        atrs = df1['atr'].values
        emas = df1['daily_ema'].values
        idx = df1.index

        for sig in signals:
            i = sig.divergence_idx
            # Scan up to MAX_WAIT_CANDLES forward looking for BOS confirmation
            confirmed_at = None
            for k in range(1, MAX_WAIT_CANDLES + 1):
                j = i + k
                if j >= len(df1) - 1:
                    break  # need j+1 for entry
                # BOS: close past swing_level
                c = close[j]
                if sig.side == 'long' and c > sig.swing_level:
                    if not np.isnan(emas[j]) and c > emas[j]:
                        confirmed_at = j
                        break
                if sig.side == 'short' and c < sig.swing_level:
                    if not np.isnan(emas[j]) and c < emas[j]:
                        confirmed_at = j
                        break
            if confirmed_at is None:
                continue

            j = confirmed_at
            entry_j = j + 1
            entry_time = idx[entry_j]
            # Only count trades whose entry is within the window
            if entry_time < start_ts or entry_time > end_ts:
                continue
            entry_price = opens[entry_j]
            atr = atrs[j]
            if np.isnan(atr) or atr <= 0:
                continue

            cfg = per_div.get(sig.divergence_code)
            if not cfg:
                continue
            rr = cfg['rr']
            atr_mult = cfg['atr_mult']
            sl_distance = atr * atr_mult
            if sig.side == 'long':
                sl_price = entry_price - sl_distance
                tp_price = entry_price + sl_distance * rr
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - sl_distance * rr

            exit_time, outcome = walk_5m_for_exit(df5, entry_time, sig.side, sl_price, tp_price)
            if exit_time is None or outcome == 'no5m':
                # Fall back to 1h walk if 5m unavailable
                # Use 1h candles after entry_time
                sub1 = df1.iloc[entry_j + 1:]
                hit_sl = hit_tp = None
                for r_pos, (ts, row) in enumerate(sub1.iterrows()):
                    high, low = row['high'], row['low']
                    if sig.side == 'long':
                        if low <= sl_price and high >= tp_price:
                            hit_sl = ts; break
                        if low <= sl_price:
                            hit_sl = ts; break
                        if high >= tp_price:
                            hit_tp = ts; break
                    else:
                        if high >= sl_price and low <= tp_price:
                            hit_sl = ts; break
                        if high >= sl_price:
                            hit_sl = ts; break
                        if low <= tp_price:
                            hit_tp = ts; break
                    if r_pos > 24 * 30:  # 30 days max
                        break
                if hit_sl is not None:
                    exit_time = hit_sl; outcome = 'sl'
                elif hit_tp is not None:
                    exit_time = hit_tp; outcome = 'tp'
                else:
                    continue
            elif outcome == 'timeout':
                continue

            r_result = rr if outcome == 'tp' else -1.0

            out.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'outcome': outcome,
                'r_result': r_result,
                'side': sig.side,
                'fee_drag': 0.0,
                'symbol': sym,
                'div_type': sig.divergence_code,
                'rr_setting': rr,
                'atr_mult': atr_mult,
            })
        return out
    except Exception as e:
        return [{'__error__': f'{sym}: {e}'}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2025-11-01')
    ap.add_argument('--end', default='2026-05-25')
    ap.add_argument('--out', default='trades_fresh_window.csv')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    configs = load_symbol_configs(ROOT / 'config.yaml')
    print(f'Loaded {len(configs)} enabled symbols from config.yaml')

    work = [(sym, per_div, args.start, args.end) for sym, per_div in configs.items()]

    t0 = time.time()
    all_trades = []
    errors = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_symbol, w): w[0] for w in work}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                trades = fut.result()
            except Exception as e:
                errors.append(f'{sym}: {e}')
                completed += 1
                continue
            for t in trades:
                if '__error__' in t:
                    errors.append(t['__error__'])
                else:
                    all_trades.append(t)
            completed += 1
            if completed % 20 == 0 or completed == len(work):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(work) - completed) / rate if rate > 0 else 0
                print(f'  {completed}/{len(work)}  trades={len(all_trades)}  err={len(errors)}  {elapsed:.0f}s eta {eta:.0f}s')

    print(f'\nGenerated {len(all_trades)} trades in {time.time()-t0:.0f}s')
    if errors:
        print(f'Errors ({len(errors)}):')
        for e in errors[:10]:
            print(f'  {e}')
        if len(errors) > 10:
            print(f'  ... {len(errors)-10} more')

    if not all_trades:
        print('No trades generated.')
        return

    df = pd.DataFrame(all_trades)
    df = df.sort_values('entry_time').reset_index(drop=True)
    out_path = ROOT / args.out
    df.to_csv(out_path, index=False)
    print(f'Saved -> {out_path}')
    print(f'Window: {df.entry_time.min()} -> {df.entry_time.max()}')
    wins = (df['r_result'] > 0).sum()
    print(f'WR: {wins/len(df)*100:.1f}%  Avg R: {df.r_result.mean():+.3f}')


if __name__ == '__main__':
    main()
