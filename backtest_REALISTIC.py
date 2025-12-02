#!/usr/bin/env python3
"""
Realistic Backtest - Uses Actual Bot Signal Logic + Realistic Costs

Key improvements over backtest_400_symbols.py:
1. Uses ACTUAL scalp signal detection from detector.py
2. Entry on NEXT candle open (not signal candle close)
3. Models slippage (0.03% average)
4. Models trading fees (0.055% maker/taker = 0.11% round trip)
5. Models spread costs
6. More conservative TP/SL hit detection

This should give win rates 10-15% closer to real-world performance.
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
SLIPPAGE_PCT = 0.03  # 0.03% average slippage on entries/exits
FEES_ROUND_TRIP_PCT = 0.11  # 0.055% * 2 = 0.11% total
SPREAD_PCT = 0.02  # 0.02% bid/ask spread

async def backtest_symbol_realistic(bybit, sym: str, idx: int, total: int):
    """Backtest single symbol using ACTUAL bot signal logic + realistic costs"""
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch data (same as before - ~10k candles)
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
        
        # Initialize ACTUAL PRODUCTION bot settings from config.yaml
        settings = ScalpSettings(
            rr=2.1,
            vwap_window=50,
            min_bb_width_pct=0.45,           # From config
            vol_ratio_min=0.8,               # From config
            body_ratio_min=0.25,             # From config
            wick_delta_min=0.05,             # From config
            vwap_dist_atr_max=1.5,           # From config
            vwap_pattern='bounce',           # From config
            vwap_bounce_band_atr_min=0.1,    # From config
            vwap_bounce_band_atr_max=0.6,    # From config
            vwap_bounce_lookback_bars=3,     # From config
            vwap_require_alignment=True      # From config
        )

        
        # Scan for signals using ACTUAL detector
        long_signals = []
        short_signals = []
        
        for i in range(100, len(df) - 100):  # Leave room for forward testing
            try:
                # Get historical data up to current bar
                hist_df = df.iloc[:i].copy()
                
                # Detect signal using ACTUAL bot logic
                signal = detect_scalp_signal(hist_df, settings, sym)
                
                if signal is None:
                    continue
                
                # REALISTIC ENTRY: Next candle open (not current close!)
                next_candle = df.iloc[i]
                entry = float(next_candle['open'])
                
                # Apply SLIPPAGE to entry
                if signal.side == 'long':
                    entry = entry * (1 + SLIPPAGE_PCT / 100)  # Pay slippage going in
                else:
                    entry = entry * (1 - SLIPPAGE_PCT / 100)
                
                # Adjust SL/TP proportionally
                signal_r = abs(signal.entry - signal.sl)
                if signal.side == 'long':
                    sl = entry - signal_r  # Maintain same R distance
                    tp = entry + signal_r * settings.rr
                else:
                    sl = entry + signal_r
                    tp = entry - signal_r * settings.rr
                
                # Validate SL/TP
                if signal.side == 'long' and (sl >= entry or tp <= entry):
                    continue
                if signal.side == 'short' and (sl <= entry or tp >= entry):
                    continue
                
                # Forward test outcome
                outcome = 'timeout'
                exit_price = entry
                future = df.iloc[i+1:i+101]  # Next 100 bars
                
                for _, f_row in future.iterrows():
                    if signal.side == 'long':
                        # Check SL first (conservative)
                        if f_row['low'] <= sl:
                            outcome = 'loss'
                            exit_price = sl * (1 - SLIPPAGE_PCT / 100)  # Slippage on exit too
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
                
                # Calculate P&L including ALL costs
                if signal.side == 'long':
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_pct = (entry - exit_price) / entry * 100
                
                # Subtract fees and spread
                pnl_pct -= FEES_ROUND_TRIP_PCT
                pnl_pct -= SPREAD_PCT
                
                # Recompute outcome based on net P&L
                if pnl_pct > 0:
                    final_outcome = 'win'
                elif pnl_pct < 0:
                    final_outcome = 'loss'
                else:
                    final_outcome = 'breakeven'
                
                # Build combo signature (simplified - just use patterns from real detector)
                combo = f"VWAP_{signal.meta.get('acceptance_path', 'unknown')}"
                
                if signal.side == 'long':
                    long_signals.append({'combo': combo, 'outcome': final_outcome})
                else:
                    short_signals.append({'combo': combo, 'outcome': final_outcome})
                    
            except Exception as e:
                # logger.debug(f"Signal detection error at bar {i}: {e}")
                continue
        
        # Compute stats (REALISTIC criteria: WR > 45%, N >= 30)
        best_long = None
        best_short = None
        
        if long_signals:
            res_df = pd.DataFrame(long_signals)
            stats = res_df.groupby('combo')['outcome'].value_counts().unstack(fill_value=0)
            if 'win' not in stats.columns: stats['win'] = 0
            stats['total'] = stats.sum(axis=1)
            stats['wr'] = (stats['win'] / stats['total']) * 100
            
            best = stats[(stats['wr'] > 45) & (stats['total'] >= 30)].sort_values('wr', ascending=False)
            if not best.empty:
                top = best.iloc[0]
                best_long = {'combo': best.index[0], 'wr': top['wr'], 'n': int(top['total'])}
        
        if short_signals:
            res_df = pd.DataFrame(short_signals)
            stats = res_df.groupby('combo')['outcome'].value_counts().unstack(fill_value=0)
            if 'win' not in stats.columns: stats['win'] = 0
            stats['total'] = stats.sum(axis=1)
            stats['wr'] = (stats['win'] / stats['total']) * 100
            
            best = stats[(stats['wr'] > 45) & (stats['total'] >= 30)].sort_values('wr', ascending=False)
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
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No combo passing (WR>45%, N>=30)")
            return None
    
    except Exception as e:
        logger.error(f"[{idx}/{total}] {sym} ❌ Error: {e}")
        return None

async def run():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    # Use ALL 400 symbols for comprehensive scan
    with open('symbols_400.yaml', 'r') as f:
        data = yaml.safe_load(f)
    symbols = data['symbols'] if isinstance(data, dict) else data

    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))

    
    print(f"{'='*80}")
    print(f"🔬 REALISTIC BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Using ACTUAL bot signal detection logic")
    print(f"Slippage: {SLIPPAGE_PCT}% per trade")
    print(f"Fees: {FEES_ROUND_TRIP_PCT}% round trip")
    print(f"Spread: {SPREAD_PCT}%")
    print(f"Entry: Next candle open (realistic)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    for idx, sym in enumerate(symbols, 1):
        result = await backtest_symbol_realistic(bybit, sym, idx, len(symbols))
        if result:
            results.append(result)
        
        # Progress update every 50 symbols
        if idx % 50 == 0:
            print(f"\n{'='*80}")
            print(f"Progress: {idx}/{len(symbols)} | Passed: {len(results)}")
            print(f"{'='*80}\n")
    
    # Generate output
    print(f"\n{'='*80}")
    print(f"REALISTIC RESULTS: {len(results)} symbols validated")
    print(f"{'='*80}\n")
    
    # Save results
    yaml_lines = [
        "# REALISTIC Backtest Results (Actual Bot Logic + Costs)",
        f"# Slippage: {SLIPPAGE_PCT}%, Fees: {FEES_ROUND_TRIP_PCT}%, Spread: {SPREAD_PCT}%",
        f"# Entry: Next candle open (not signal close)",
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
            yaml_lines.append(f"    # Realistic BT: WR={r['long']['wr']:.1f}%, N={r['long']['n']}")
        
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"{r['short']['combo']}\"")
            yaml_lines.append(f"    # Realistic BT: WR={r['short']['wr']:.1f}%, N={r['short']['n']}")
        
        yaml_lines.append("")
    
    output = "\n".join(yaml_lines)
    
    with open('symbol_overrides_REALISTIC.yaml', 'w') as f:
        f.write(output)
    
    print(f"✅ Saved to symbol_overrides_REALISTIC.yaml")
    print(f"✅ {len(results)} symbols validated with realistic costs")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Comparison with original
    print(f"\n{'='*80}")
    print(f"COMPARISON TO ORIGINAL BACKTEST")
    print(f"{'='*80}")
    print(f"Original (optimistic): 196 symbols passed")
    print(f"Realistic (with costs): {len(results)} symbols passed")
    print(f"Difference: {196 - len(results)} symbols ({(196 - len(results))/196*100:.1f}% reduction)")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(run())
