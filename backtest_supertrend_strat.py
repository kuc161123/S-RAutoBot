#!/usr/bin/env python3
"""
SuperTrend + MACD Backtest (Fixed TP/SL)
Strategy: Trend Following with Momentum Confirmation
Logic:
- Long: Close > EMA200 (Trend) AND SuperTrend Green AND MACD > Signal
  - SL: Entry - 2 * ATR
  - TP: Entry + 3 * ATR (1:1.5 Risk:Reward)
- Short: Close < EMA200 (Trend) AND SuperTrend Red AND MACD < Signal
  - SL: Entry + 2 * ATR
  - TP: Entry - 3 * ATR (1:1.5 Risk:Reward)
  
Validation: Walk-Forward (Train 70% / Test 30%)
Timeframe: 15m
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_supertrend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SLIPPAGE = 0.0005  # 0.05%
FEES = 0.0012      # 0.12% round trip
WR_THRESHOLD = 40.0 # Lower WR threshold for Trend Following (aiming for >1 Profit Factor)
MIN_TRADES = 10
TRAIN_PCT = 0.7
TIMEFRAME = '15'   # 15m Timeframe

# Strategy Params
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
ATR_TP_MULT = 3.0
SUPERTREND_LEN = 10
SUPERTREND_MUL = 3.0

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def calculate_indicators(df):
    """Calculate SuperTrend, MACD, EMA 200, ATR"""
    # Ensure we have enough data
    if len(df) < 200: return df
    
    # EMA 200
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # MACD (12, 26, 9)
    macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
    # pandas_ta returns columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    # We rename them for easier access
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    
    # ATR
    df['atr'] = df.ta.atr(length=ATR_PERIOD)
    
    # SuperTrend
    st = df.ta.supertrend(high='high', low='low', close='close', length=SUPERTREND_LEN, multiplier=SUPERTREND_MUL)
    # pandas_ta naming: SUPERT_{length}_{multiplier}
    # If multiplier is 3.0, it might be SUPERT_10_3.0
    # Let's dynamically find the columns
    
    # st.columns usually has: ['SUPERT_10_3.0', 'SUPERTd_10_3.0', 'SUPERTl_10_3.0', 'SUPERTs_10_3.0']
    # We look for the one starting with SUPERT_ and SUPERTd_
    
    st_val_col = None
    st_dir_col = None
    
    for col in st.columns:
        if col.startswith(f'SUPERT_{SUPERTREND_LEN}'):
            st_val_col = col
        elif col.startswith(f'SUPERTd_{SUPERTREND_LEN}'):
            st_dir_col = col
            
    if not st_val_col or not st_dir_col:
        # Fallback or error
        logger.error(f"SuperTrend columns not found. Available: {st.columns}")
        return df
    
    df['supertrend'] = st[st_val_col]
    df['supertrend_dir'] = st[st_dir_col] # 1 = Up (Green), -1 = Down (Red)
    
    return df.dropna()

def backtest_fold(df, side):
    """Backtest logic for a specific dataset and side"""
    # Signal detection
    if side == 'long':
        # Trend UP (Close > EMA), SuperTrend Green (1), MACD > Signal
        signals = (df['close'] > df['ema200']) & \
                  (df['supertrend_dir'] == 1) & \
                  (df['macd'] > df['macd_signal'])
    else:
        # Trend DOWN (Close < EMA), SuperTrend Red (-1), MACD < Signal
        signals = (df['close'] < df['ema200']) & \
                  (df['supertrend_dir'] == -1) & \
                  (df['macd'] < df['macd_signal'])
    
    trades = []
    in_position = False
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    
    rows = df.itertuples()
    
    wins = 0
    total = 0
    pnl_accum = 0.0
    
    for row in rows:
        if in_position:
            # Check Exit (Fixed TP/SL)
            # We assume we check Low/High of the candle to see if SL/TP was hit
            # Conservative: Check SL first, then TP
            
            exit_price = None
            pnl = 0.0
            
            if side == 'long':
                if row.low <= sl_price:
                    exit_price = sl_price # Stopped out
                elif row.high >= tp_price:
                    exit_price = tp_price # Take Profit
            else:
                if row.high >= sl_price:
                    exit_price = sl_price # Stopped out
                elif row.low <= tp_price:
                    exit_price = tp_price # Take Profit
            
            if exit_price:
                # Calculate PnL
                if side == 'long':
                    eff_entry = entry_price * (1 + SLIPPAGE)
                    eff_exit = exit_price * (1 - SLIPPAGE)
                    pnl = (eff_exit - eff_entry) / eff_entry - FEES
                else:
                    eff_entry = entry_price * (1 - SLIPPAGE)
                    eff_exit = exit_price * (1 + SLIPPAGE)
                    pnl = (eff_entry - eff_exit) / eff_entry - FEES
                
                total += 1
                if pnl > 0: wins += 1
                pnl_accum += pnl
                in_position = False
                
        else:
            # Check Entry
            # We need to check if the signal condition is met
            # Access signal by index logic or just re-eval condition (slower but safer)
            
            entry_signal = False
            if side == 'long':
                if row.close > row.ema200 and row.supertrend_dir == 1 and row.macd > row.macd_signal:
                    entry_signal = True
            else:
                if row.close < row.ema200 and row.supertrend_dir == -1 and row.macd < row.macd_signal:
                    entry_signal = True
            
            if entry_signal:
                entry_price = row.close
                atr = row.atr
                
                if side == 'long':
                    sl_price = entry_price - (ATR_SL_MULT * atr)
                    tp_price = entry_price + (ATR_TP_MULT * atr)
                else:
                    sl_price = entry_price + (ATR_SL_MULT * atr)
                    tp_price = entry_price - (ATR_TP_MULT * atr)
                    
                in_position = True
                
    if total >= MIN_TRADES:
        wr = (wins/total)*100
        return {'wr': wr, 'n': total, 'pnl': pnl_accum}
    return None

async def backtest_symbol(bybit, sym, idx, total):
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch ~5000 candles
        klines = []
        end_ts = None
        limit = 200
        reqs = 25 # 5000 candles
        
        for _ in range(reqs):
            k = bybit.get_klines(sym, TIMEFRAME, limit=limit, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.1)
            
        if len(klines) < 1000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data")
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
        
        res = {'symbol': sym, 'long': None, 'short': None}
        
        for side in ['long', 'short']:
            # Train
            train_res = backtest_fold(train, side)
            # For Trend Following, we accept lower WR if PnL is positive, but let's stick to WR > 40%
            if not train_res or train_res['wr'] < WR_THRESHOLD: continue
            
            # Validate in Test
            test_res = backtest_fold(test, side)
            if test_res and test_res['wr'] > WR_THRESHOLD and test_res['pnl'] > 0:
                res[side] = {
                    'train_wr': train_res['wr'],
                    'test_wr': test_res['wr'],
                    'n': test_res['n'],
                    'pnl': test_res['pnl']
                }
                
        if res['long'] or res['short']:
            msg = []
            if res['long']: msg.append(f"LONG: WR={res['long']['test_wr']:.1f}% PnL={res['long']['pnl']:.2f}R")
            if res['short']: msg.append(f"SHORT: WR={res['short']['test_wr']:.1f}% PnL={res['short']['pnl']:.2f}R")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            return res
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
    print(f"🚀 SUPER TREND + MACD BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Logic: EMA(200) + SuperTrend(10,3) + MACD")
    print(f"Exit: Fixed TP (3x ATR) / SL (2x ATR)")
    print(f"Filters: WR > {WR_THRESHOLD}% + Positive PnL (Train & Test)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    BATCH_SIZE = 1 # Single symbol for stability
    
    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch_symbols = symbols[batch_start:batch_start + BATCH_SIZE]
        tasks = [backtest_symbol(bybit, sym, batch_start + i + 1, len(symbols)) for i, sym in enumerate(batch_symbols)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in batch_results:
            if r and not isinstance(r, Exception):
                results.append(r)
                
        print(f"Progress: {min(batch_start + BATCH_SIZE, len(symbols))}/{len(symbols)} | Found: {len(results)}")
        await asyncio.sleep(1)

    # Save Results
    yaml_lines = [
        "# SuperTrend + MACD Results",
        f"# Logic: EMA(200) + SuperTrend + MACD",
        f"# Risk: TP=3ATR, SL=2ATR",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    - \"SuperTrend_MACD\"")
            yaml_lines.append(f"      # TrainWR={r['long']['train_wr']:.1f}%, TestWR={r['long']['test_wr']:.1f}%")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"SuperTrend_MACD\"")
            yaml_lines.append(f"      # TrainWR={r['short']['train_wr']:.1f}%, TestWR={r['short']['test_wr']:.1f}%")
        yaml_lines.append("")
        
    with open('symbol_overrides_SuperTrend.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
        
    print(f"✅ Saved to symbol_overrides_SuperTrend.yaml")

if __name__ == "__main__":
    asyncio.run(run())
