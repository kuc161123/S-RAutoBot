#!/usr/bin/env python3
"""
RSI 2 Mean Reversion Backtest
Strategy: Larry Connors' RSI 2 with Trend Filter
Logic:
- Long: Close > EMA200 (Trend) AND RSI(2) < 10 (Oversold)
  - Exit: Close > SMA(5)
- Short: Close < EMA200 (Trend) AND RSI(2) > 90 (Overbought)
  - Exit: Close < SMA(5)
  
Validation: Walk-Forward (Train 70% / Test 30%)
Timeframe: 15m
"""
import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
from autobot.brokers.bybit import Bybit, BybitConfig
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_rsi2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SLIPPAGE = 0.0005  # 0.05%
FEES = 0.0012      # 0.12% round trip
WR_THRESHOLD = 60.0
MIN_TRADES = 15    # Lower min trades as this is a swing/mean-rev strategy
TRAIN_PCT = 0.7
TIMEFRAME = '15'   # 15m Timeframe

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def calculate_indicators(df):
    """Vectorized calculation of RSI 2, EMA 200, SMA 5"""
    close = df['close']
    
    # RSI 2
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=1, adjust=False).mean() # RSI 2 -> com=1
    ma_down = down.ewm(com=1, adjust=False).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    df['rsi2'] = rsi
    
    # EMA 200 (Trend Filter)
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # SMA 5 (Exit)
    df['sma5'] = close.rolling(5).mean()
    
    return df.dropna()

def backtest_fold(df, side):
    """Backtest logic for a specific dataset and side"""
    # Signal detection
    if side == 'long':
        # Trend UP, Oversold
        signals = (df['close'] > df['ema200']) & (df['rsi2'] < 10)
    else:
        # Trend DOWN, Overbought
        signals = (df['close'] < df['ema200']) & (df['rsi2'] > 90)
    
    trades = []
    
    # Iterate through signals
    # We need to iterate carefully to handle "in position" state
    # But for vectorization speed, we can assume we take every signal if we are not in a trade?
    # Let's do a simple loop for accuracy
    
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    
    # Convert to list for speed
    closes = df['close'].values
    sma5s = df['sma5'].values
    sig_indices = np.where(signals)[0]
    
    # We need a full loop to handle exits correctly
    # Optimized loop: Jump to next signal if not in position
    
    # Re-do with full iteration only when needed
    # Actually, let's just iterate all bars, it's safer for state management
    
    rows = df.itertuples()
    
    wins = 0
    total = 0
    
    for row in rows:
        idx = row.Index
        i = df.index.get_loc(idx)
        
        if in_position:
            # Check Exit
            # Exit on Close > SMA5 (Long) or Close < SMA5 (Short)
            # We execute at the CLOSE of the bar (or Open of next? Strategy usually says Close)
            # Let's assume we exit on the Close of the bar where condition is met
            
            exit_signal = False
            if side == 'long':
                if row.close > row.sma5: exit_signal = True
            else:
                if row.close < row.sma5: exit_signal = True
                
            if exit_signal:
                exit_price = row.close
                
                # Apply costs
                if side == 'long':
                    # Buy Entry (already paid), Sell Exit
                    # PnL = (Exit - Entry) / Entry
                    # Costs: Entry Fee + Slippage, Exit Fee + Slippage
                    # Effective Entry = Entry * (1 + S)
                    # Effective Exit = Exit * (1 - S)
                    
                    eff_entry = entry_price * (1 + SLIPPAGE)
                    eff_exit = exit_price * (1 - SLIPPAGE)
                    pnl = (eff_exit - eff_entry) / eff_entry - FEES
                else:
                    # Sell Entry, Buy Exit
                    eff_entry = entry_price * (1 - SLIPPAGE)
                    eff_exit = exit_price * (1 + SLIPPAGE)
                    pnl = (eff_entry - eff_exit) / eff_entry - FEES
                
                total += 1
                if pnl > 0: wins += 1
                in_position = False
                
        else:
            # Check Entry
            entry_signal = False
            if side == 'long':
                if row.close > row.ema200 and row.rsi2 < 10: entry_signal = True
            else:
                if row.close < row.ema200 and row.rsi2 > 90: entry_signal = True
            
            if entry_signal:
                # Enter on Close
                entry_price = row.close
                in_position = True
                
    if total >= MIN_TRADES:
        return {'wr': (wins/total)*100, 'n': total}
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
            if not train_res or train_res['wr'] < WR_THRESHOLD: continue
            
            # Validate in Test
            test_res = backtest_fold(test, side)
            if test_res and test_res['wr'] > WR_THRESHOLD:
                res[side] = {
                    'train_wr': train_res['wr'],
                    'test_wr': test_res['wr'],
                    'n': test_res['n']
                }
                
        if res['long'] or res['short']:
            msg = []
            if res['long']: msg.append(f"LONG: WR={res['long']['test_wr']:.1f}%")
            if res['short']: msg.append(f"SHORT: WR={res['short']['test_wr']:.1f}%")
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
    print(f"📉 RSI 2 MEAN REVERSION BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Logic: RSI(2) < 10 / > 90 + EMA(200) Trend Filter")
    print(f"Timeframe: {TIMEFRAME}m")
    print(f"Filters: WR > {WR_THRESHOLD}% (Train AND Test)")
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
        "# RSI 2 Mean Reversion Results",
        f"# Logic: RSI(2) + EMA(200) + SMA(5)",
        f"# Filters: WR > {WR_THRESHOLD}% (Train & Test)",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    - \"RSI2_MeanRev\"")
            yaml_lines.append(f"      # TrainWR={r['long']['train_wr']:.1f}%, TestWR={r['long']['test_wr']:.1f}%")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"RSI2_MeanRev\"")
            yaml_lines.append(f"      # TrainWR={r['short']['train_wr']:.1f}%, TestWR={r['short']['test_wr']:.1f}%")
        yaml_lines.append("")
        
    with open('symbol_overrides_RSI2.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
        
    print(f"✅ Saved to symbol_overrides_RSI2.yaml")

if __name__ == "__main__":
    asyncio.run(run())
