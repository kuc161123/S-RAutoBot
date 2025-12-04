#!/usr/bin/env python3
"""
Hybrid Combo Backtest (Multi-Timeframe + Walk-Forward)
Logic: Original Combo Bins (RSI, MACD, VWAP, Fib, MTF)
Features:
- Tests 5 Timeframes: 1m, 3m, 5m, 15m, 60m
- Finds BEST Timeframe & Combo per Side per Symbol
- Walk-Forward Validation (Train 70% / Test 30%)
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
        logging.FileHandler("backtest_hybrid.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SLIPPAGE = 0.0005  # 0.05%
FEES = 0.0012      # 0.12% round trip
WR_THRESHOLD = 60.0
SHARPE_THRESHOLD = 1.0
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
    """Vectorized calculation of all features"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI 14
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    df['rsi'] = rsi
    
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    
    # VWAP (Rolling 20)
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()
    atr = (high - low).rolling(14).mean()
    df['atr'] = atr
    df['vwap_dist_atr'] = (close - vwap).abs() / atr
    
    # MTF (EMA 20 vs 50)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df['mtf_agree'] = ema20 > ema50
    
    # Fib Zone (50 bar lookback)
    roll_max = high.rolling(50).max()
    roll_min = low.rolling(50).min()
    fib_ret = (roll_max - close) / (roll_max - roll_min)
    df['fib_ret'] = fib_ret * 100
    
    # BB Width Pct
    std = close.rolling(20).std()
    ma = close.rolling(20).mean()
    upper = ma + 2*std
    lower = ma - 2*std
    bbw = (upper - lower) / close
    df['bbw_pct'] = bbw.rolling(100).rank(pct=True)
    
    # Volume Ratio
    vol_ma = volume.rolling(20).mean()
    df['vol_ratio'] = volume / vol_ma
    
    return df.dropna()

def get_combo(row):
    """Determine the combo string for a given row"""
    # RSI
    r = row['rsi']
    if r < 30: r_bin = '<30'
    elif r < 40: r_bin = '30-40'
    elif r < 60: r_bin = '40-60'
    elif r < 70: r_bin = '60-70'
    else: r_bin = '70+'
    
    # MACD
    m_bin = 'bull' if row['macd_hist'] > 0 else 'bear'
    
    # VWAP
    v = row['vwap_dist_atr']
    if v < 0.6: v_bin = '<0.6'
    elif v < 1.2: v_bin = '0.6-1.2'
    else: v_bin = '1.2+'
    
    # Fib
    f = row['fib_ret']
    if f < 23.6: f_bin = '0-23'
    elif f < 38.2: f_bin = '23-38'
    elif f < 50.0: f_bin = '38-50'
    elif f < 61.8: f_bin = '50-61'
    elif f < 78.6: f_bin = '61-78'
    else: f_bin = '78-100'
    
    # MTF
    mtf_bin = 'MTF' if row['mtf_agree'] else 'noMTF'
    
    return f"RSI:{r_bin} MACD:{m_bin} VWAP:{v_bin} Fib:{f_bin} {mtf_bin}"

def calculate_sharpe(returns):
    if len(returns) < 2: return 0.0
    arr = np.array(returns)
    if np.std(arr) == 0: return 0.0
    return np.mean(arr) / np.std(arr)

def backtest_fold(df, side):
    """Backtest logic for a specific dataset and side"""
    # Signal detection (Pre-filters)
    if side == 'long':
        signals = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] > df['open'])
    else:
        signals = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] < df['open'])
    
    combo_stats = {}
    
    for ridx in df[signals].index:
        try:
            sig_iloc = df.index.get_loc(ridx)
            if sig_iloc >= len(df) - 1: continue
            
            # Entry on NEXT candle open
            entry_raw = df.iloc[sig_iloc + 1]['open']
            atr = df.iloc[sig_iloc]['atr']
            
            if side == 'long':
                entry_real = entry_raw * (1 + SLIPPAGE)
                sl_raw = entry_raw - 2 * atr
                tp_raw = entry_raw + 2 * atr
            else:
                entry_real = entry_raw * (1 - SLIPPAGE)
                sl_raw = entry_raw + 2 * atr
                tp_raw = entry_raw - 2 * atr
            
            # Check future
            outcome = 'loss'
            pnl_pct = -(SLIPPAGE + FEES)
            future = df.iloc[sig_iloc + 1:].iloc[1:100]
            
            for _, f_row in future.iterrows():
                if side == 'long':
                    if f_row['low'] <= sl_raw:
                        outcome = 'loss'
                        pnl_pct = (sl_raw - entry_real) / entry_real - FEES
                        break
                    if f_row['high'] >= tp_raw:
                        exit_real = tp_raw * (1 - SLIPPAGE)
                        pnl_pct = (exit_real - entry_real) / entry_real - FEES
                        outcome = 'win' if pnl_pct > 0 else 'loss'
                        break
                else:
                    if f_row['high'] >= sl_raw:
                        outcome = 'loss'
                        pnl_pct = (entry_real - sl_raw) / entry_real - FEES
                        break
                    if f_row['low'] <= tp_raw:
                        exit_real = tp_raw * (1 + SLIPPAGE)
                        pnl_pct = (entry_real - exit_real) / entry_real - FEES
                        outcome = 'win' if pnl_pct > 0 else 'loss'
                        break
            
            combo = get_combo(df.loc[ridx])
            if combo not in combo_stats:
                combo_stats[combo] = {'wins': 0, 'total': 0, 'returns': []}
            
            combo_stats[combo]['total'] += 1
            combo_stats[combo]['returns'].append(pnl_pct * 100)
            if outcome == 'win':
                combo_stats[combo]['wins'] += 1
                
        except Exception:
            continue
            
    # Calculate stats
    results = []
    for combo, stats in combo_stats.items():
        if stats['total'] >= MIN_TRADES:
            wr = (stats['wins'] / stats['total']) * 100
            sharpe = calculate_sharpe(stats['returns'])
            results.append({
                'combo': combo,
                'wr': wr,
                'n': stats['total'],
                'sharpe': sharpe
            })
    return results

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
        
        for side in ['long', 'short']:
            # Train
            train_results = backtest_fold(train, side)
            if not train_results: continue
            
            # Find best in train
            best_train = max(train_results, key=lambda x: x['wr'] if x['wr'] > WR_THRESHOLD and x['sharpe'] > SHARPE_THRESHOLD else 0)
            if best_train['wr'] < WR_THRESHOLD or best_train['sharpe'] < SHARPE_THRESHOLD: continue
            
            # Validate in test
            test_results = backtest_fold(test, side)
            test_match = next((r for r in test_results if r['combo'] == best_train['combo']), None)
            
            if test_match and test_match['wr'] > WR_THRESHOLD and test_match['sharpe'] > SHARPE_THRESHOLD:
                res[side] = test_match
                
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
        
        # Optimize Long (Maximize WR)
        if res['long']:
            if not best_long or res['long']['wr'] > best_long['stats']['wr']:
                best_long = {'tf': tf, 'stats': res['long']}
                
        # Optimize Short
        if res['short']:
            if not best_short or res['short']['wr'] > best_short['stats']['wr']:
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
    print(f"🧬 HYBRID COMBO BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Logic: Original Combo Bins")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Filters: WR > {WR_THRESHOLD}%, Sharpe > {SHARPE_THRESHOLD}")
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
        "# Hybrid Combo Optimization Results",
        f"# Logic: Original Combo + Multi-TF",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    - \"{r['long']['stats']['combo']}\"")
            yaml_lines.append(f"      # TF={r['long']['tf']}m, WR={r['long']['stats']['wr']:.1f}%, Sharpe={r['long']['stats']['sharpe']:.2f}")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"{r['short']['stats']['combo']}\"")
            yaml_lines.append(f"      # TF={r['short']['tf']}m, WR={r['short']['stats']['wr']:.1f}%, Sharpe={r['short']['stats']['sharpe']:.2f}")
        yaml_lines.append("")
        
    with open('symbol_overrides_HYBRID.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
        
    print(f"✅ Saved to symbol_overrides_HYBRID.yaml")

if __name__ == "__main__":
    asyncio.run(run())
