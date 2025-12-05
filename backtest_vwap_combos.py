#!/usr/bin/env python3
"""
VWAP Bounce + Exhaustive Combo Backtest
Strategy: Find the best RSI/MACD/Fib combination for VWAP Bounces per symbol.
Logic:
1. Trigger: Price touches VWAP and bounces (Close > VWAP for Long).
2. Record State: RSI Bin, MACD State, Fib Bin.
3. Optimize: Find the Combo Key with highest PnL/WR in Train set.
4. Validate: Check if that specific Combo works in Test set.

Risk: Fixed TP (4ATR) / SL (2ATR) -> 1:2 R:R
Timeframe: 3m
"""
import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
import pandas_ta as ta
from autobot.brokers.bybit import Bybit, BybitConfig
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_vwap_combos.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
LOOKBACK_DAYS = 60   # 60 days - balanced
SLIPPAGE = 0.001   # 0.1% (Conservative/Realistic)
FEES = 0.0012      # 0.12% round trip (Standard Taker)
MIN_WR = 40.0      # Min Win Rate to consider a combo valid
MIN_TRADES = 10    # Min trades to consider a combo valid
TRAIN_PCT = 0.7
TIMEFRAME = '3'    # 3m Timeframe

# Strategy Params
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
ATR_TP_MULT = 4.0

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def get_rsi_bin(rsi):
    """Original RSI: 5 levels"""
    if rsi < 30: return '<30'
    if rsi < 40: return '30-40'
    if rsi < 60: return '40-60'
    if rsi < 70: return '60-70'
    return '70+'

def get_macd_bin(macd, signal):
    """Original MACD: 2 levels"""
    return 'bull' if macd > signal else 'bear'

def get_fib_bin(close, high_50, low_50):
    """Original Fib: 7 levels"""
    if high_50 == low_50: return '0-23'
    fib = (high_50 - close) / (high_50 - low_50) * 100
    if fib < 23.6: return '0-23'
    if fib < 38.2: return '23-38'
    if fib < 50.0: return '38-50'
    if fib < 61.8: return '50-61'
    if fib < 78.6: return '61-78'
    if fib < 100: return '78-100'
    return '100+'                      # Middle range # Should not happen if close within range

def calculate_indicators(df):
    """Calculate VWAP, RSI, MACD, Fib, ATR with direction data"""
    if len(df) < 200: return df
    
    # ATR
    df['atr'] = df.ta.atr(length=ATR_PERIOD)
    
    # RSI + Previous RSI for direction
    df['rsi'] = df.ta.rsi(length=14)
    df['prev_rsi'] = df['rsi'].shift(1)
    
    # MACD + Histogram + Previous Histogram for acceleration
    macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    df['prev_hist'] = df['macd_hist'].shift(1)
    
    # VWAP
    try:
        vwap = df.ta.vwap(high='high', low='low', close='close', volume='volume')
        if isinstance(vwap, pd.DataFrame):
            df['vwap'] = vwap.iloc[:, 0]
        else:
            df['vwap'] = vwap
    except Exception:
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).rolling(1440//3).sum() / df['volume'].rolling(1440//3).sum()

    # Fib (Rolling 50 High/Low)
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    return df.dropna()

def run_backtest(df, side):
    """
    Run backtest on a given dataframe for a given side.
    Returns results with win rate that NEVER dropped below MIN_WR (lower bound).
    If WR ever dips below 40%, that combo is invalidated.
    """
    results = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0.0, 'lower_bound_violated': False})
    
    # We iterate to find triggers
    # To simulate "holding", we skip candles while in trade? 
    # For exhaustive search, we want to evaluate ALL triggers. 
    # But overlapping trades distort results.
    # Simplified approach: We evaluate every trigger as if it were a standalone trade.
    # This might overcount, but it finds the "best condition".
    
    rows = list(df.itertuples())
    
    for i, row in enumerate(rows):
        if i == 0: continue
        prev = rows[i-1]
        
        # Trigger Logic: VWAP Bounce
        trigger = False
        if side == 'long':
            # Low touched VWAP, Close bounced above
            if row.low <= row.vwap and row.close > row.vwap:
                trigger = True
        else:
            # High touched VWAP, Close rejected below
            if row.high >= row.vwap and row.close < row.vwap:
                trigger = True
                
        if trigger:
            # Construct Combo Key (simplified: 18 combinations)
            rsi_bin = get_rsi_bin(row.rsi)
            macd_bin = get_macd_bin(row.macd, row.macd_signal)
            fib_bin = get_fib_bin(row.close, row.roll_high, row.roll_low)
            
            combo_key = f"RSI:{rsi_bin} MACD:{macd_bin} Fib:{fib_bin}"
            
            # Calculate Outcome
            entry_price = row.close
            atr = row.atr
            
            if side == 'long':
                sl = entry_price - (ATR_SL_MULT * atr)
                tp = entry_price + (ATR_TP_MULT * atr)
            else:
                sl = entry_price + (ATR_SL_MULT * atr)
                tp = entry_price - (ATR_TP_MULT * atr)
            
            # Forward scan for outcome
            outcome_pnl = 0.0
            for j in range(i+1, len(rows)):
                future = rows[j]
                
                if side == 'long':
                    if future.low <= sl:
                        # Loss
                        eff_entry = entry_price * (1 + SLIPPAGE)
                        eff_exit = sl * (1 - SLIPPAGE)
                        outcome_pnl = (eff_exit - eff_entry) / eff_entry - FEES
                        break
                    elif future.high >= tp:
                        # Win
                        eff_entry = entry_price * (1 + SLIPPAGE)
                        eff_exit = tp * (1 - SLIPPAGE)
                        outcome_pnl = (eff_exit - eff_entry) / eff_entry - FEES
                        break
                else:
                    if future.high >= sl:
                        # Loss
                        eff_entry = entry_price * (1 - SLIPPAGE)
                        eff_exit = sl * (1 + SLIPPAGE)
                        outcome_pnl = (eff_entry - eff_exit) / eff_entry - FEES
                        break
                    elif future.low <= tp:
                        # Win
                        eff_entry = entry_price * (1 - SLIPPAGE)
                        eff_exit = tp * (1 + SLIPPAGE)
                        outcome_pnl = (eff_entry - eff_exit) / eff_entry - FEES
                        break
                        
                # Timeout (100 bars max)
                if j - i > 100:
                    # Force close
                    exit_price = future.close
                    if side == 'long':
                        eff_entry = entry_price * (1 + SLIPPAGE)
                        eff_exit = exit_price * (1 - SLIPPAGE)
                        outcome_pnl = (eff_exit - eff_entry) / eff_entry - FEES
                    else:
                        eff_entry = entry_price * (1 - SLIPPAGE)
                        eff_exit = exit_price * (1 + SLIPPAGE)
                        outcome_pnl = (eff_entry - eff_exit) / eff_entry - FEES
                    break
            
            # Record result
            results[combo_key]['total'] += 1
            if outcome_pnl > 0: results[combo_key]['wins'] += 1
            results[combo_key]['pnl'] += outcome_pnl
            
            # Check ROLLING Win Rate (Lower Bound Check)
            wins = results[combo_key]['wins']
            total = results[combo_key]['total']
            if total >= 3:  # Only check after 3 trades to avoid early noise
                rolling_wr = (wins / total) * 100
                if rolling_wr < MIN_WR:
                    results[combo_key]['lower_bound_violated'] = True
            
    return results

async def backtest_symbol(bybit, sym, idx, total):
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch Data (Need enough for 3m)
        klines = []
        end_ts = None
        limit = 200
        reqs = 300 # ~60k candles (120 days)
        
        for _ in range(reqs):
            k = bybit.get_klines(sym, TIMEFRAME, limit=limit, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.1)
            
        if len(klines) < 2000:
            return None
        
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
        
        df = calculate_indicators(df)
        if df.empty: return None
        
        # Split
        split = int(len(df) * TRAIN_PCT)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        best_res = {'symbol': sym, 'long': [], 'short': []}
        
        for side in ['long', 'short']:
            # 1. Train: Find ALL combos that pass criteria
            train_results = run_backtest(train, side)
            test_results = run_backtest(test, side)
            
            passing_combos = []
            
            for combo, stats in train_results.items():
                if stats['total'] < MIN_TRADES: continue
                if stats.get('lower_bound_violated', False): continue  # LOWER BOUND CHECK
                wr = (stats['wins'] / stats['total']) * 100
                if wr < MIN_WR: continue
                if stats['pnl'] <= 0: continue  # Must be profitable in train
                
                # 2. Validate in Test
                test_stats = test_results.get(combo)
                if not test_stats: continue
                if test_stats['total'] < 1: continue
                if test_stats.get('lower_bound_violated', False): continue
                
                test_wr = (test_stats['wins'] / test_stats['total']) * 100
                if test_wr < MIN_WR: continue
                if test_stats['pnl'] <= 0: continue  # Must be profitable in test
                
                # Passed all checks!
                passing_combos.append({
                    'combo': combo,
                    'train_wr': wr,
                    'test_wr': test_wr,
                    'train_pnl': stats['pnl'],
                    'test_pnl': test_stats['pnl'],
                    'total_trades': stats['total'] + test_stats['total']
                })
            
            # Sort by test_wr (highest first)
            passing_combos.sort(key=lambda x: x['test_wr'], reverse=True)
            best_res[side] = passing_combos
        
        if best_res['long'] or best_res['short']:
            msg = []
            if best_res['long']: 
                msg.append(f"LONG: {len(best_res['long'])} combos")
            if best_res['short']: 
                msg.append(f"SHORT: {len(best_res['short'])} combos")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            
            # Incremental Save (ALL combos that pass)
            try:
                # Read existing
                existing = {}
                try:
                    with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
                        existing = yaml.safe_load(f) or {}
                except FileNotFoundError:
                    pass
                    
                # Update
                updated = False
                existing_sym = existing.get(sym, {})
                
                if best_res['long']:
                    existing_sym['long'] = [c['combo'] for c in best_res['long']]
                    updated = True
                    
                if best_res['short']:
                    existing_sym['short'] = [c['combo'] for c in best_res['short']]
                    updated = True
                
                if updated:
                    existing[sym] = existing_sym
                    # Write back
                    with open('symbol_overrides_VWAP_Combo.yaml', 'w') as f:
                        yaml.dump(existing, f)
                    
                    # Auto-push to git
                    import subprocess
                    try:
                        subprocess.run(['git', 'add', 'symbol_overrides_VWAP_Combo.yaml'], check=True, capture_output=True)
                        subprocess.run(['git', 'commit', '-m', f'Auto: New combo {sym}'], check=True, capture_output=True)
                        subprocess.run(['git', 'push'], check=True, capture_output=True)
                        logger.info(f"✅ Auto-pushed {sym} to git")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Git push failed (may be no changes): {e}")
            except Exception as e:
                logger.error(f"Failed to save incremental result: {e}")
                
            return best_res
        else:
            # Log max trades found to show "how close" it was
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No valid combo")
            return None

    except Exception as e:
        logger.error(f"[{idx}/{total}] {sym} ❌ Error: {e}")
        return None

async def run():
    # Load symbols
    try:
        with open('symbols_400.yaml', 'r') as f:
            sym_data = yaml.safe_load(f)
            symbols = sym_data['symbols']
    except FileNotFoundError:
        print("❌ symbols_400.yaml not found")
        return

    # Load already processed symbols (for resume)
    processed_symbols = set()
    try:
        with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
            existing = yaml.safe_load(f) or {}
            processed_symbols = set(existing.keys())
        print(f"📂 Found {len(processed_symbols)} already processed symbols")
    except FileNotFoundError:
        pass

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
    print(f"🧬 VWAP COMBO EXHAUSTIVE SEARCH: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Logic: VWAP Bounce + Best RSI/MACD/Fib Combo")
    print(f"Timeframe: {TIMEFRAME}m")
    print(f"Risk: TP=4ATR, SL=2ATR (1:2 R:R)")
    print(f"Filters: WR > {MIN_WR}% + Positive PnL (Train & Test)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Filter out already processed symbols (resume mode)
    remaining_symbols = [s for s in symbols if s not in processed_symbols]
    if processed_symbols:
        print(f"⏭️ Skipping {len(processed_symbols)} already processed, {len(remaining_symbols)} remaining\n")
    
    results = []
    BATCH_SIZE = 1 
    total = len(symbols)
    start_idx = len(processed_symbols)
    
    for batch_start in range(0, len(remaining_symbols), BATCH_SIZE):
        batch_symbols = remaining_symbols[batch_start:batch_start + BATCH_SIZE]
        current_idx = start_idx + batch_start + 1
        tasks = [backtest_symbol(bybit, sym, current_idx, total) for i, sym in enumerate(batch_symbols)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in batch_results:
            if r and not isinstance(r, Exception):
                results.append(r)
                
        print(f"Progress: {min(batch_start + BATCH_SIZE, len(symbols))}/{len(symbols)} | Found: {len(results)}")
        await asyncio.sleep(1)

    # Save Results
    yaml_lines = [
        "# VWAP Combo Results",
        f"# Logic: VWAP Bounce + Optimized Combo",
        f"# Risk: TP=4ATR, SL=2ATR",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            for c in r['long']:  # Iterate through list
                yaml_lines.append(f"  - {c['combo']}")
        if r['short']:
            yaml_lines.append(f"  short:")
            for c in r['short']:  # Iterate through list
                yaml_lines.append(f"  - {c['combo']}")
        yaml_lines.append("")
        
    with open('symbol_overrides_VWAP_Combo.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
        
    print(f"✅ Saved to symbol_overrides_VWAP_Combo.yaml")

if __name__ == "__main__":
    asyncio.run(run())
