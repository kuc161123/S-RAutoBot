#!/usr/bin/env python3
"""
Boom Hunter Pro Backtest
Logic: Ehlers Early Onset Trend (EOT) Oscillator
Features:
- Walk-Forward Analysis (Train 70% / Test 30%)
- Sharpe Ratio filtering (>1.0)
- 5m Timeframe
"""
import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
import math
from autobot.brokers.bybit import Bybit, BybitConfig
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_boom.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SLIPPAGE = 0.0005  # 0.05%
FEES = 0.0012      # 0.12% round trip
WR_THRESHOLD = 60.0
SHARPE_THRESHOLD = 1.0
MIN_TRADES = 30
TRAIN_PCT = 0.7
TIMEFRAME = '5'    # 5m for Boom Hunter

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def calculate_eot(df, length=25):
    """
    Calculate Ehlers Early Onset Trend (EOT) Oscillator
    Approximation of Boom Hunter Pro logic
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)
    
    # Initialize arrays
    filt = np.zeros(n)
    peak = np.zeros(n)
    osc = np.zeros(n)
    
    # Constants for Super Smoother
    # a1 = exp(-1.414 * 3.14159 / length)
    # b1 = 2 * a1 * cos(1.414 * 180 / length) -> radians
    # c2 = b1
    # c3 = -a1 * a1
    # c1 = 1 - c2 - c3
    
    # Using a simplified Roofing Filter approach for stability
    # High Pass + Super Smoother
    
    # 1. High Pass Filter (removes DC component / trend)
    alpha1 = (math.cos(0.707 * 2 * math.pi / 100) + math.sin(0.707 * 2 * math.pi / 100) - 1) / math.cos(0.707 * 2 * math.pi / 100)
    hp = np.zeros(n)
    
    for i in range(2, n):
        hp[i] = (1 - alpha1 / 2)**2 * (close[i] - 2 * close[i-1] + close[i-2]) + 2 * (1 - alpha1) * hp[i-1] - (1 - alpha1)**2 * hp[i-2]
        
    # 2. Super Smoother (removes aliasing noise)
    a1 = math.exp(-1.414 * math.pi / 10)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    for i in range(2, n):
        filt[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
        
    # 3. Automatic Gain Control / Normalization (Quotient Transform)
    # Fast Attack - Slow Decay
    for i in range(1, n):
        peak[i] = peak[i-1] * 0.991 # Decay
        if abs(filt[i]) > peak[i]:
            peak[i] = abs(filt[i])
            
        if peak[i] != 0:
            norm = filt[i] / peak[i]
        else:
            norm = 0
            
        # Quotient Transform to normalize to -1..1 range roughly
        # (x + k) / (k*x + 1)
        k = 0.85
        osc[i] = (norm + k) / (k * norm + 1)
        
    df['eot_osc'] = osc
    return df

def calculate_sharpe(returns):
    if len(returns) < 2: return 0.0
    arr = np.array(returns)
    if np.std(arr) == 0: return 0.0
    return np.mean(arr) / np.std(arr)

def backtest_strategy(df, side):
    """
    Backtest Boom Hunter Logic
    Long: Oscillator crosses UP from oversold (-0.8)
    Short: Oscillator crosses DOWN from overbought (0.8)
    """
    # Signal Generation
    # Using 0.6 as threshold for "Boom" spikes based on Quotient Transform characteristics
    THRESHOLD = 0.6
    
    if side == 'long':
        # Cross up from below -THRESHOLD
        signals = (df['eot_osc'] > -THRESHOLD) & (df['eot_osc'].shift(1) <= -THRESHOLD)
    else:
        # Cross down from above THRESHOLD
        signals = (df['eot_osc'] < THRESHOLD) & (df['eot_osc'].shift(1) >= THRESHOLD)
        
    wins = 0
    total = 0
    returns = []
    
    for ridx in df[signals].index:
        try:
            sig_iloc = df.index.get_loc(ridx)
            if sig_iloc >= len(df) - 1: continue
            
            # Entry on NEXT candle open
            entry_raw = df.iloc[sig_iloc + 1]['open']
            atr = (df.iloc[sig_iloc]['high'] - df.iloc[sig_iloc]['low']) # Simple ATR proxy
            
            # Risk Management (2:1 RR)
            RISK_MULT = 2.0
            
            if side == 'long':
                entry_real = entry_raw * (1 + SLIPPAGE)
                sl_raw = entry_raw - (atr * 1.5) # 1.5 ATR Stop
                tp_raw = entry_raw + (atr * 1.5 * RISK_MULT)
            else:
                entry_real = entry_raw * (1 - SLIPPAGE)
                sl_raw = entry_raw + (atr * 1.5)
                tp_raw = entry_raw - (atr * 1.5 * RISK_MULT)
                
            # Check outcome
            outcome = 'loss'
            pnl_pct = -(SLIPPAGE + FEES)
            
            future = df.iloc[sig_iloc + 1:].iloc[1:100] # Max 100 bars
            
            for _, f_row in future.iterrows():
                if side == 'long':
                    if f_row['low'] <= sl_raw:
                        outcome = 'loss'
                        pnl_pct = (sl_raw - entry_real) / entry_real - FEES
                        break
                    if f_row['high'] >= tp_raw:
                        outcome = 'win'
                        exit_real = tp_raw * (1 - SLIPPAGE)
                        pnl_pct = (exit_real - entry_real) / entry_real - FEES
                        break
                else:
                    if f_row['high'] >= sl_raw:
                        outcome = 'loss'
                        pnl_pct = (entry_real - sl_raw) / entry_real - FEES
                        break
                    if f_row['low'] <= tp_raw:
                        outcome = 'win'
                        exit_real = tp_raw * (1 + SLIPPAGE)
                        pnl_pct = (entry_real - exit_real) / entry_real - FEES
                        break
            
            total += 1
            returns.append(pnl_pct * 100)
            if outcome == 'win': wins += 1
            
        except Exception:
            continue
            
    if total < MIN_TRADES:
        return None
        
    wr = (wins / total) * 100
    sharpe = calculate_sharpe(returns)
    
    if wr >= WR_THRESHOLD and sharpe >= SHARPE_THRESHOLD:
        return {
            'wr': wr,
            'n': total,
            'sharpe': sharpe
        }
    return None

async def backtest_symbol(bybit, sym, idx, total):
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch Data (5m timeframe)
        klines = []
        end_ts = None
        for i in range(40): # ~8000 candles
            k = bybit.get_klines(sym, TIMEFRAME, limit=200, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.1)  # Increased from 0.03 to prevent rate limits
            
        if len(klines) < 2000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data")
            return None
            
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
        
        # Calculate Indicator
        df = calculate_eot(df)
        df = df.dropna()
        
        # Split Train/Test
        split_idx = int(len(df) * TRAIN_PCT)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        result = {'symbol': sym, 'long': None, 'short': None}
        
        # Test Long
        train_long = backtest_strategy(train_df, 'long')
        if train_long:
            test_long = backtest_strategy(test_df, 'long')
            if test_long:
                result['long'] = test_long
                
        # Test Short
        train_short = backtest_strategy(train_df, 'short')
        if train_short:
            test_short = backtest_strategy(test_df, 'short')
            if test_short:
                result['short'] = test_short
                
        if result['long'] or result['short']:
            msg = []
            if result['long']: msg.append(f"LONG: WR={result['long']['wr']:.1f}%")
            if result['short']: msg.append(f"SHORT: WR={result['short']['wr']:.1f}%")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            return result
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No valid strategy")
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
    print(f"💥 BOOM HUNTER PRO BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Logic: Ehlers EOT Oscillator (High-Pass + Super Smoother)")
    print(f"Timeframe: {TIMEFRAME}m")
    print(f"Filters: WR > {WR_THRESHOLD}%, Sharpe > {SHARPE_THRESHOLD}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    BATCH_SIZE = 1  # One symbol at a time as requested
    
    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch_symbols = symbols[batch_start:batch_start + BATCH_SIZE]
        tasks = [backtest_symbol(bybit, sym, batch_start + i + 1, len(symbols)) for i, sym in enumerate(batch_symbols)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in batch_results:
            if r and not isinstance(r, Exception):
                results.append(r)
                
        print(f"Progress: {min(batch_start + BATCH_SIZE, len(symbols))}/{len(symbols)} | Found: {len(results)}")
        await asyncio.sleep(1)  # Cooldown between batches

    # Save Results
    yaml_lines = [
        "# Boom Hunter Pro Results",
        f"# Logic: Ehlers EOT Oscillator (5m)",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    - \"BOOM_HUNTER_LONG\"")
            yaml_lines.append(f"      # WR={r['long']['wr']:.1f}%, Sharpe={r['long']['sharpe']:.2f}")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"BOOM_HUNTER_SHORT\"")
            yaml_lines.append(f"      # WR={r['short']['wr']:.1f}%, Sharpe={r['short']['sharpe']:.2f}")
        yaml_lines.append("")
        
    with open('symbol_overrides_BOOM.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
        
    print(f"✅ Saved to symbol_overrides_BOOM.yaml")

if __name__ == "__main__":
    asyncio.run(run())
