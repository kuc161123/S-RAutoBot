#!/usr/bin/env python3
"""
Realistic 400-Symbol Backtest
-----------------------------
Scope: All 400+ symbols from symbols_400.yaml
Logic: Realistic (Next Candle Open Entry, Slippage, Fees)
Filter: WR > 60% AND N >= 30 (Strict)
Output: symbol_overrides_400.yaml (Ready for Bot)
"""
import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime

# Import actual signal detector
sys.path.insert(0, os.path.dirname(__file__))
from autobot.brokers.bybit import Bybit, BybitConfig
from autobot.strategies.scalp.detector import detect_scalp_signal, ScalpSettings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

# Realistic trading costs
SLIPPAGE_PCT = 0.04  # 0.04% slippage
FEES_ROUND_TRIP_PCT = 0.11  # 0.055% * 2
TOTAL_COST_PCT = SLIPPAGE_PCT + FEES_ROUND_TRIP_PCT

async def backtest_symbol_realistic(bybit, sym: str, idx: int, total: int):
    """Backtest single symbol using ACTUAL bot signal logic + realistic costs"""
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch data (~10k candles)
        klines = []
        end_ts = None
        
        for i in range(50):
            k = bybit.get_klines(sym, '3', limit=200, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.03)
        
        if len(klines) < 2000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data ({len(klines)} candles)")
            return None
        
        # Build DataFrame
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        
        # Initialize Settings (MATCHING BOT DEFAULTS/BACKTEST LOGIC)
        settings = ScalpSettings(
            rr=2.1,
            bbw_pct_min=0.45,           
            vol_ratio_min=0.8,               
            atr_mult=2.0
        )

        # Scan for signals
        long_signals = []
        short_signals = []
        
        for i in range(100, len(df) - 100):
            try:
                hist_df = df.iloc[:i].copy()
                
                # Detect signal
                signal = detect_scalp_signal(hist_df, settings, sym)
                
                if signal is None:
                    continue
                
                # REALISTIC ENTRY: Next candle open
                next_candle = df.iloc[i]
                entry = float(next_candle['open'])
                
                # Apply SLIPPAGE
                if signal.side == 'long':
                    entry = entry * (1 + SLIPPAGE_PCT / 100)
                else:
                    entry = entry * (1 - SLIPPAGE_PCT / 100)
                
                # Adjust SL/TP to maintain R:R relative to REAL entry
                # Note: Detector gives SL/TP based on signal candle close. 
                # We recalculate based on actual entry to be fair to the strategy's intent (2.1R)
                atr = signal.meta.get('atr', 0.0)
                if atr == 0: continue

                dist = atr * settings.atr_mult
                
                if signal.side == 'long':
                    sl = entry - dist
                    tp = entry + (dist * settings.rr)
                else:
                    sl = entry + dist
                    tp = entry - (dist * settings.rr)
                
                # Validate
                if signal.side == 'long' and (sl >= entry or tp <= entry): continue
                if signal.side == 'short' and (sl <= entry or tp >= entry): continue
                
                # Forward test
                outcome = 'timeout'
                exit_price = entry
                future = df.iloc[i+1:i+101]  # Next 100 bars
                
                for _, f_row in future.iterrows():
                    if signal.side == 'long':
                        if f_row['low'] <= sl:
                            outcome = 'loss'
                            exit_price = sl * (1 - SLIPPAGE_PCT / 100)
                            break
                        if f_row['high'] >= tp:
                            outcome = 'win'
                            exit_price = tp * (1 - SLIPPAGE_PCT / 100)
                            break
                    else:  # short
                        if f_row['high'] >= sl:
                            outcome = 'loss'
                            exit_price = sl * (1 + SLIPPAGE_PCT / 100)
                            break
                        if f_row['low'] <= tp:
                            outcome = 'win'
                            exit_price = tp * (1 + SLIPPAGE_PCT / 100)
                            break
                
                # Calculate Net P&L
                if signal.side == 'long':
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_pct = (entry - exit_price) / entry * 100
                
                pnl_pct -= FEES_ROUND_TRIP_PCT
                
                if pnl_pct > 0:
                    final_outcome = 'win'
                elif pnl_pct < 0:
                    final_outcome = 'loss'
                else:
                    final_outcome = 'breakeven'
                
                combo = signal.meta.get('combo', 'unknown')
                
                if signal.side == 'long':
                    long_signals.append({'combo': combo, 'outcome': final_outcome})
                else:
                    short_signals.append({'combo': combo, 'outcome': final_outcome})
                    
            except Exception:
                continue
        
        # Compute stats (STRICT: WR > 60%, N >= 30)
        best_long = None
        best_short = None
        
        if long_signals:
            res_df = pd.DataFrame(long_signals)
            stats = res_df.groupby('combo')['outcome'].value_counts().unstack(fill_value=0)
            if 'win' not in stats.columns: stats['win'] = 0
            stats['total'] = stats.sum(axis=1)
            stats['wr'] = (stats['win'] / stats['total']) * 100
            
            best = stats[(stats['wr'] > 60) & (stats['total'] >= 30)].sort_values('wr', ascending=False)
            if not best.empty:
                top = best.iloc[0]
                best_long = {'combo': best.index[0], 'wr': top['wr'], 'n': int(top['total'])}
        
        if short_signals:
            res_df = pd.DataFrame(short_signals)
            stats = res_df.groupby('combo')['outcome'].value_counts().unstack(fill_value=0)
            if 'win' not in stats.columns: stats['win'] = 0
            stats['total'] = stats.sum(axis=1)
            stats['wr'] = (stats['win'] / stats['total']) * 100
            
            best = stats[(stats['wr'] > 60) & (stats['total'] >= 30)].sort_values('wr', ascending=False)
            if not best.empty:
                top = best.iloc[0]
                best_short = {'combo': best.index[0], 'wr': top['wr'], 'n': int(top['total'])}
        
        # Report
        if best_long or best_short:
            msg = []
            if best_long:
                msg.append(f"LONG: WR={best_long['wr']:.1f}%, N={best_long['n']}")
            if best_short:
                msg.append(f"SHORT: WR={best_short['wr']:.1f}%, N={best_short['n']}")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            
            return {
                'symbol': sym,
                'long': best_long,
                'short': best_short
            }
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No strict combo (WR>60%, N>=30)")
            return None
    
    except Exception as e:
        logger.error(f"[{idx}/{total}] {sym} ❌ Error: {e}")
        return None

async def run():
    # Load symbols
    try:
        with open('symbols_400.yaml', 'r') as f:
            data = yaml.safe_load(f)
        symbols = data['symbols'] if isinstance(data, dict) else data
    except FileNotFoundError:
        print("❌ symbols_400.yaml not found!")
        return

    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))

    print(f"{'='*80}")
    print(f"🔬 REALISTIC 400-SYMBOL BACKTEST")
    print(f"{'='*80}")
    print(f"Symbols: {len(symbols)}")
    print(f"Logic: Next Candle Open Entry + Slippage + Fees")
    print(f"Filter: WR > 60% AND N >= 30")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    for idx, sym in enumerate(symbols, 1):
        result = await backtest_symbol_realistic(bybit, sym, idx, len(symbols))
        if result:
            results.append(result)
        
        if idx % 20 == 0:
            print(f"\nProgress: {idx}/{len(symbols)} | Passed: {len(results)}\n")
    
    # Save results
    yaml_lines = [
        "# REALISTIC 400-Symbol Backtest Results",
        f"# Filter: WR > 60%, N >= 30",
        f"# Costs: {TOTAL_COST_PCT}% per trade",
        f"# Processed: {len(symbols)} symbols",
        f"# Passed: {len(results)} symbols",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    - \"{r['long']['combo']}\"")
            yaml_lines.append(f"    # WR={r['long']['wr']:.1f}%, N={r['long']['n']}")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"{r['short']['combo']}\"")
            yaml_lines.append(f"    # WR={r['short']['wr']:.1f}%, N={r['short']['n']}")
        yaml_lines.append("")
    
    with open('symbol_overrides_400.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
    
    print(f"\n{'='*80}")
    print(f"✅ DONE. Saved to symbol_overrides_400.yaml")
    print(f"Passed: {len(results)} symbols")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(run())
