#!/usr/bin/env python3
"""
Elite Backtest System - 400 Symbols
Features:
- Walk-Forward Analysis (3 folds)
- Sharpe Ratio filtering (>1.0)
- Hold-out validation (20%)
- Realistic execution assumptions
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
        logging.FileHandler("backtest_elite.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SLIPPAGE = 0.0005  # 0.05% (increased for safety margin)
FEES = 0.0012      # 0.12% round trip
WR_THRESHOLD = 60.0
SHARPE_THRESHOLD = 1.0
MIN_TRADES_PER_FOLD = 30

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
    
    return df

def get_combo(row):
    # RSI Bin
    r = row['rsi']
    if r < 30: rb = '<30'
    elif r < 40: rb = '30-40'
    elif r < 60: rb = '40-60'
    elif r < 70: rb = '60-70'
    else: rb = '70+'
    
    # MACD Bin
    mb = 'bull' if row['macd_hist'] > 0 else 'bear'
    
    # VWAP Bin
    v = row['vwap_dist_atr']
    if v < 0.6: vb = '<0.6'
    elif v < 1.2: vb = '0.6-1.2'
    else: vb = '1.2+'
    
    # Fib Bin
    f = row['fib_ret']
    if f < 23.6: fb = '0-23'
    elif f < 38.2: fb = '23-38'
    elif f < 50.0: fb = '38-50'
    elif f < 61.8: fb = '50-61'
    elif f < 78.6: fb = '61-78'
    else: fb = '78-100'
    
    # MTF
    ma = 'MTF' if row['mtf_agree'] else 'noMTF'
    
    return f"RSI:{rb} MACD:{mb} VWAP:{vb} Fib:{fb} {ma}"

def calculate_sharpe(returns):
    """Calculate Sharpe Ratio from list of returns"""
    if len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    if np.std(returns_array) == 0:
        return 0.0
    return np.mean(returns_array) / np.std(returns_array)

def backtest_fold(df, side):
    """Backtest a single fold for given side"""
    # Signal detection
    if side == 'long':
        signals = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] > df['open'])
    else:
        signals = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] < df['open'])
    
    combo_stats = {}
    
    for ridx in df[signals].index:
        try:
            sig_iloc = df.index.get_loc(ridx)
            if sig_iloc >= len(df) - 1:
                continue
            
            # Entry on NEXT candle open
            entry_raw = df.iloc[sig_iloc + 1]['open']
            atr = df.iloc[sig_iloc]['atr']
            
            # Apply slippage
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
            pnl_pct = -(SLIPPAGE + FEES)  # Default loss from costs
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
            combo_stats[combo]['returns'].append(pnl_pct * 100)  # Convert to percentage
            if outcome == 'win':
                combo_stats[combo]['wins'] += 1
                
        except Exception:
            continue
    
    # Calculate WR and Sharpe for each combo
    results = []
    for combo, stats in combo_stats.items():
        if stats['total'] >= MIN_TRADES_PER_FOLD:
            wr = (stats['wins'] / stats['total']) * 100
            sharpe = calculate_sharpe(stats['returns'])
            results.append({
                'combo': combo,
                'wr': wr,
                'n': stats['total'],
                'sharpe': sharpe
            })
    
    return results

async def backtest_symbol(bybit, sym, idx, total):
    """Backtest a single symbol with walk-forward validation"""
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch ~10,000 candles
        klines = []
        end_ts = None
        
        for i in range(50):
            k = bybit.get_klines(sym, '3', limit=200, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.03)
        
        if len(klines) < 3000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data ({len(klines)} candles)")
            return None
        
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
        
        # Calculate Indicators
        df = calculate_indicators(df)
        df = df.dropna()
        
        if len(df) < 2000:
            return None
        
        # Walk-Forward Analysis (3 folds + hold-out)
        # Fold 1: Train 0-40%, Test 40-60%
        # Fold 2: Train 40-60%, Test 60-70%
        # Fold 3: Train 60-70%, Test 70-80%  
        # Hold-out: 80-100%
        
        n = len(df)
        folds = [
            {'train': (0, int(n*0.4)), 'test': (int(n*0.4), int(n*0.6))},
            {'train': (int(n*0.4), int(n*0.6)), 'test': (int(n*0.6), int(n*0.7))},
            {'train': (int(n*0.6), int(n*0.7)), 'test': (int(n*0.7), int(n*0.8))}
        ]
        
        best_long = None
        best_short = None
        
        # Test both sides
        for side in ['long', 'short']:
            side_passed = False
            best_combo = None
            
            # Fold testing
            for fold_idx, fold in enumerate(folds):
                train_df = df.iloc[fold['train'][0]:fold['train'][1]]
                test_df = df.iloc[fold['test'][0]:fold['test'][1]]
                
                # Train
                train_results = backtest_fold(train_df, side)
                if not train_results:
                    break
                
                # Find best combo from training
                train_best = max(train_results, key=lambda x: x['wr'] if x['wr'] > WR_THRESHOLD and x['sharpe'] > SHARPE_THRESHOLD else 0)
                if train_best['wr'] < WR_THRESHOLD or train_best['sharpe'] < SHARPE_THRESHOLD:
                    break
                
                # Test on validation set
                test_results = backtest_fold(test_df, side)
                test_match = next((r for r in test_results if r['combo'] == train_best['combo']), None)
                
                if not test_match or test_match['wr'] < WR_THRESHOLD or test_match['sharpe'] < SHARPE_THRESHOLD:
                    break
                
                # If this is the last fold and we got here, strategy passed all folds
                if fold_idx == len(folds) - 1:
                    side_passed = True
                    best_combo = {
                        'combo': train_best['combo'],
                        'wr': test_match['wr'],
                        'n': test_match['n'],
                        'sharpe': test_match['sharpe']
                    }
            
            if side_passed and best_combo:
                # Final hold-out validation
                holdout_df = df.iloc[int(n*0.8):]
                holdout_results = backtest_fold(holdout_df, side)
                holdout_match = next((r for r in holdout_results if r['combo'] == best_combo['combo']), None)
                
                if holdout_match and holdout_match['wr'] > WR_THRESHOLD and holdout_match['sharpe'] > SHARPE_THRESHOLD:
                    if side == 'long':
                        best_long = holdout_match
                    else:
                        best_short = holdout_match
        
        # Report
        if best_long or best_short:
            msg = []
            if best_long:
                msg.append(f"LONG: WR={best_long['wr']:.1f}%, Sharpe={best_long['sharpe']:.2f}, N={best_long['n']}")
            if best_short:
                msg.append(f"SHORT: WR={best_short['wr']:.1f}%, Sharpe={best_short['sharpe']:.2f}, N={best_short['n']}")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            
            return {
                'symbol': sym,
                'long': best_long,
                'short': best_short
            }
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ Failed walk-forward validation")
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
        print("❌ symbols_400.yaml not found. Run fetch_symbols.py first.")
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
    print(f"🏆 ELITE BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Walk-Forward: 3 folds + 20% hold-out")
    print(f"Filters: WR > {WR_THRESHOLD}%, Sharpe > {SHARPE_THRESHOLD}, N >= {MIN_TRADES_PER_FOLD}")
    print(f"Costs: Slippage 0.05% + Fees 0.12% = 0.17% per trade")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Process in batches
    BATCH_SIZE = 10
    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch_symbols = symbols[batch_start:batch_start + BATCH_SIZE]
        
        tasks = []
        for i, sym in enumerate(batch_symbols):
            idx = batch_start + i + 1
            tasks.append(backtest_symbol(bybit, sym, idx, len(symbols)))
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if result and not isinstance(result, Exception):
                results.append(result)
        
        print(f"\n{'='*80}")
        print(f"Progress: {min(batch_start + BATCH_SIZE, len(symbols))}/{len(symbols)} | Elite: {len(results)}")
        print(f"{'='*80}\n")
    
    # Generate YAML
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(results)} elite symbols")
    print(f"{'='*80}\n")
    
    yaml_lines = [
        "# Elite Backtest Results (Walk-Forward + Sharpe)",
        f"# Filters: WR > {WR_THRESHOLD}%, Sharpe > {SHARPE_THRESHOLD}, N >= {MIN_TRADES_PER_FOLD}",
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
            yaml_lines.append(f"    # WR={r['long']['wr']:.1f}%, Sharpe={r['long']['sharpe']:.2f}, N={r['long']['n']}")
        
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"{r['short']['combo']}\"")
            yaml_lines.append(f"    # WR={r['short']['wr']:.1f}%, Sharpe={r['short']['sharpe']:.2f}, N={r['short']['n']}")
        
        yaml_lines.append("")
    
    output = "\n".join(yaml_lines)
    
    with open('symbol_overrides_ELITE.yaml', 'w') as f:
        f.write(output)
    
    print(f"✅ Saved to symbol_overrides_ELITE.yaml")
    print(f"✅ {len(results)} elite symbols with validated combos")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(run())
