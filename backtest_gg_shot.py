#!/usr/bin/env python3
"""
GG-Shot Backtest (Multi-Timeframe Optimization)
Logic: Donchian Channel Breakout (20-period) + ADX Filter
Features:
- Tests 5 Timeframes: 1m, 3m, 5m, 15m, 60m
- Finds BEST timeframe per Side per Symbol
- Walk-Forward Validation (Train 70% / Test 30%)
"""
import asyncio
import yaml
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import os
from autobot.brokers.bybit import Bybit, BybitConfig
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_gg.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SLIPPAGE = 0.0005  # 0.05%
FEES = 0.0012      # 0.12% round trip
WR_THRESHOLD = 50.0 # Lower threshold for trend following
SHARPE_THRESHOLD = 0.5 # Trend strategies often have lower Sharpe but high R:R
MIN_TRADES = 20
TRAIN_PCT = 0.7
TIMEFRAMES = ['1', '3', '5', '15', '60']

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def calculate_indicators(df):
    """Calculate Donchian Channels + ADX"""
    try:
        # Donchian Channels (20)
        df['dc_high'] = df['high'].rolling(20).max().shift(1)
        df['dc_low'] = df['low'].rolling(20).min().shift(1)
        
        # ADX (14)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and not adx.empty:
            df['adx'] = adx['ADX_14']
        else:
            df['adx'] = 0
            
        # ATR (14) for Stops
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def calculate_sharpe(returns):
    if len(returns) < 2: return 0.0
    arr = np.array(returns)
    if np.std(arr) == 0: return 0.0
    return np.mean(arr) / np.std(arr)

def backtest_strategy(df, side):
    """
    GG-Shot Logic:
    Long: Close > DC High AND ADX > 20
    Short: Close < DC Low AND ADX > 20
    """
    if side == 'long':
        signals = (df['close'] > df['dc_high']) & (df['adx'] > 20)
    else:
        signals = (df['close'] < df['dc_low']) & (df['adx'] > 20)
        
    wins = 0
    total = 0
    returns = []
    
    # Simple loop to avoid look-ahead bias
    # We jump forward when in a trade
    i = 0
    while i < len(df) - 1:
        if not signals.iloc[i]:
            i += 1
            continue
            
        # Entry
        entry_idx = i + 1
        if entry_idx >= len(df): break
        
        entry_price = df.iloc[entry_idx]['open']
        atr = df.iloc[i]['atr']
        
        # Risk Settings (Trend Following)
        # SL = 2 ATR, TP = 4 ATR (1:2 Risk/Reward)
        RR = 2.0
        SL_MULT = 2.0
        
        if side == 'long':
            entry_real = entry_price * (1 + SLIPPAGE)
            sl = entry_price - (atr * SL_MULT)
            tp = entry_price + (atr * SL_MULT * RR)
        else:
            entry_real = entry_price * (1 - SLIPPAGE)
            sl = entry_price + (atr * SL_MULT)
            tp = entry_price - (atr * SL_MULT * RR)
            
        # Simulate Trade
        outcome = 'loss'
        pnl_pct = -(SLIPPAGE + FEES)
        bars_held = 0
        
        for j in range(entry_idx, min(entry_idx + 100, len(df))):
            row = df.iloc[j]
            bars_held += 1
            
            if side == 'long':
                if row['low'] <= sl:
                    outcome = 'loss'
                    pnl_pct = (sl - entry_real) / entry_real - FEES
                    break
                if row['high'] >= tp:
                    outcome = 'win'
                    exit_real = tp * (1 - SLIPPAGE)
                    pnl_pct = (exit_real - entry_real) / entry_real - FEES
                    break
            else:
                if row['high'] >= sl:
                    outcome = 'loss'
                    pnl_pct = (entry_real - sl) / entry_real - FEES
                    break
                if row['low'] <= tp:
                    outcome = 'win'
                    exit_real = tp * (1 + SLIPPAGE)
                    pnl_pct = (entry_real - exit_real) / entry_real - FEES
                    break
        
        total += 1
        returns.append(pnl_pct * 100)
        if outcome == 'win': wins += 1
        
        # Skip bars we were in trade
        i += bars_held
        
    if total < MIN_TRADES:
        return None
        
    wr = (wins / total) * 100
    sharpe = calculate_sharpe(returns)
    net_profit = sum(returns)
    
    if wr >= WR_THRESHOLD and sharpe >= SHARPE_THRESHOLD:
        return {
            'wr': wr,
            'n': total,
            'sharpe': sharpe,
            'net_profit': net_profit
        }
    return None

async def test_timeframe(bybit, sym, tf):
    """Fetch data and test a specific timeframe"""
    try:
        # Fetch ~5000 candles
        klines = []
        end_ts = None
        limit = 200
        reqs = 25 # 5000 candles
        
        for _ in range(reqs):
            k = bybit.get_klines(sym, tf, limit=limit, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.1)
            
        if len(klines) < 1000: return None
        
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
        
        res = {'tf': tf, 'long': None, 'short': None}
        
        # Long
        train_l = backtest_strategy(train, 'long')
        if train_l:
            test_l = backtest_strategy(test, 'long')
            if test_l:
                res['long'] = test_l
                
        # Short
        train_s = backtest_strategy(train, 'short')
        if train_s:
            test_s = backtest_strategy(test, 'short')
            if test_s:
                res['short'] = test_s
                
        return res
        
    except Exception:
        return None

async def backtest_symbol(bybit, sym, idx, total):
    logger.info(f"[{idx}/{total}] {sym} - Optimizing Timeframe...")
    
    best_long = None
    best_short = None
    
    for tf in TIMEFRAMES:
        res = await test_timeframe(bybit, sym, tf)
        if not res: continue
        
        # Optimize Long (Maximize Net Profit)
        if res['long']:
            if not best_long or res['long']['net_profit'] > best_long['stats']['net_profit']:
                best_long = {'tf': tf, 'stats': res['long']}
                
        # Optimize Short
        if res['short']:
            if not best_short or res['short']['net_profit'] > best_short['stats']['net_profit']:
                best_short = {'tf': tf, 'stats': res['short']}
                
    result = {'symbol': sym, 'long': best_long, 'short': best_short}
    
    msg = []
    if best_long: msg.append(f"LONG: {best_long['tf']}m (WR={best_long['stats']['wr']:.1f}%)")
    if best_short: msg.append(f"SHORT: {best_short['tf']}m (WR={best_short['stats']['wr']:.1f}%)")
    
    if msg:
        logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
        return result
    else:
        logger.info(f"[{idx}/{total}] {sym} ⚠️ No valid strategy")
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
    print(f"🎯 GG-SHOT MULTI-TIMEFRAME BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Logic: Donchian Channel Breakout (20)")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Goal: Find BEST timeframe per side per symbol")
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
        "# GG-Shot Optimization Results",
        f"# Logic: Donchian Breakout",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    - \"GG_SHOT_{r['long']['tf']}m\"")
            yaml_lines.append(f"      # TF={r['long']['tf']}m, WR={r['long']['stats']['wr']:.1f}%, Profit={r['long']['stats']['net_profit']:.1f}%")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"GG_SHOT_{r['short']['tf']}m\"")
            yaml_lines.append(f"      # TF={r['short']['tf']}m, WR={r['short']['stats']['wr']:.1f}%, Profit={r['short']['stats']['net_profit']:.1f}%")
        yaml_lines.append("")
        
    with open('symbol_overrides_GG.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
        
    print(f"✅ Saved to symbol_overrides_GG.yaml")

if __name__ == "__main__":
    asyncio.run(run())
