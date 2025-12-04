#!/usr/bin/env python3
"""
Exhaustive Combo Backtest - Test ALL indicator combinations
Features:
- Tests all possible RSI/MACD/VWAP/Fib/MTF combinations
- Walk-forward validation (train 70% / test 30%)
- Keeps ALL combos with WR > 60% and Sharpe > 1.0
- Allows multiple combos per symbol/side
"""
import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
from autobot.brokers.bybit import Bybit, BybitConfig
from datetime import datetime
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_exhaustive.log"),
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

# All possible combo values
RSI_BINS = ['<30', '30-40', '40-60', '60-70', '70+']
MACD_BINS = ['bull', 'bear']
VWAP_BINS = ['<0.6', '0.6-1.2', '1.2+']
FIB_BINS = ['0-23', '23-38', '38-50', '50-61', '61-78', '78-100']
MTF_BINS = ['MTF', 'noMTF']

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

def get_combo_string(rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin):
    """Create combo string from bins"""
    return f"RSI:{rsi_bin} MACD:{macd_bin} VWAP:{vwap_bin} Fib:{fib_bin} {mtf_bin}"

def matches_combo(row, rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin):
    """Check if a row matches the combo criteria"""
    # RSI
    r = row['rsi']
    if rsi_bin == '<30' and r >= 30: return False
    if rsi_bin == '30-40' and not (30 <= r < 40): return False
    if rsi_bin == '40-60' and not (40 <= r < 60): return False
    if rsi_bin == '60-70' and not (60 <= r < 70): return False
    if rsi_bin == '70+' and r < 70: return False
    
    # MACD
    if macd_bin == 'bull' and row['macd_hist'] <= 0: return False
    if macd_bin == 'bear' and row['macd_hist'] > 0: return False
    
    # VWAP
    v = row['vwap_dist_atr']
    if vwap_bin == '<0.6' and v >= 0.6: return False
    if vwap_bin == '0.6-1.2' and not (0.6 <= v < 1.2): return False
    if vwap_bin == '1.2+' and v < 1.2: return False
    
    # Fib
    f = row['fib_ret']
    if fib_bin == '0-23' and not (f < 23.6): return False
    if fib_bin == '23-38' and not (23.6 <= f < 38.2): return False
    if fib_bin == '38-50' and not (38.2 <= f < 50.0): return False
    if fib_bin == '50-61' and not (50.0 <= f < 61.8): return False
    if fib_bin == '61-78' and not (61.8 <= f < 78.6): return False
    if fib_bin == '78-100' and f < 78.6: return False
    
    # MTF
    if mtf_bin == 'MTF' and not row['mtf_agree']: return False
    if mtf_bin == 'noMTF' and row['mtf_agree']: return False
    
    return True

def calculate_sharpe(returns):
    """Calculate Sharpe Ratio"""
    if len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    if np.std(returns_array) == 0:
        return 0.0
    return np.mean(returns_array) / np.std(returns_array)

def test_combo(df, side, rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin):
    """Test a specific combo on the data"""
    # Signal detection
    if side == 'long':
        base_signals = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] > df['open'])
    else:
        base_signals = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] < df['open'])
    
    # Filter by combo criteria
    combo_matches = df.apply(lambda row: matches_combo(row, rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin), axis=1)
    signals = base_signals & combo_matches
    
    wins = 0
    total = 0
    returns = []
    
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
            
            total += 1
            returns.append(pnl_pct * 100)
            if outcome == 'win':
                wins += 1
                
        except Exception:
            continue
    
    if total < MIN_TRADES:
        return None
    
    wr = (wins / total) * 100
    sharpe = calculate_sharpe(returns)
    
    if wr >= WR_THRESHOLD and sharpe >= SHARPE_THRESHOLD:
        return {
            'combo': get_combo_string(rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin),
            'wr': wr,
            'n': total,
            'sharpe': sharpe
        }
    
    return None

async def backtest_symbol(bybit, sym, idx, total):
    """Backtest a single symbol with ALL combo combinations"""
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
        
        if len(klines) < 2000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data ({len(klines)} candles)")
            return None
        
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
        
        # Calculate Indicators
        df = calculate_indicators(df)
        df = df.dropna()
        
        if len(df) < 1000:
            return None
        
        # Split train/test
        split_idx = int(len(df) * TRAIN_PCT)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Test ALL combinations
        all_combos = list(product(RSI_BINS, MACD_BINS, VWAP_BINS, FIB_BINS, MTF_BINS))
        logger.info(f"[{idx}/{total}] {sym} - Testing {len(all_combos)} combos per side...")
        
        long_combos = []
        short_combos = []
        
        # Test longs
        for rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin in all_combos:
            # Train
            train_result = test_combo(train_df, 'long', rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin)
            if train_result:
                # Validate on test
                test_result = test_combo(test_df, 'long', rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin)
                if test_result:
                    long_combos.append(test_result)
        
        # Test shorts
        for rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin in all_combos:
            # Train
            train_result = test_combo(train_df, 'short', rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin)
            if train_result:
                # Validate on test
                test_result = test_combo(test_df, 'short', rsi_bin, macd_bin, vwap_bin, fib_bin, mtf_bin)
                if test_result:
                    short_combos.append(test_result)
        
        # Report
        if long_combos or short_combos:
            msg = []
            if long_combos:
                avg_wr = sum(c['wr'] for c in long_combos) / len(long_combos)
                msg.append(f"LONG: {len(long_combos)} combos (avg WR={avg_wr:.1f}%)")
            if short_combos:
                avg_wr = sum(c['wr'] for c in short_combos) / len(short_combos)
                msg.append(f"SHORT: {len(short_combos)} combos (avg WR={avg_wr:.1f}%)")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            
            return {
                'symbol': sym,
                'long': long_combos,
                'short': short_combos
            }
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No combos passed validation")
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
    print(f"🔬 EXHAUSTIVE COMBO BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Testing: {len(list(product(RSI_BINS, MACD_BINS, VWAP_BINS, FIB_BINS, MTF_BINS)))} combos per side")
    print(f"Filters: WR > {WR_THRESHOLD}%, Sharpe > {SHARPE_THRESHOLD}, N >= {MIN_TRADES}")
    print(f"Split: {TRAIN_PCT*100}% train / {(1-TRAIN_PCT)*100}% test")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Process in batches
    BATCH_SIZE = 5  # Reduced batch size because testing all combos
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
        
        total_combos = sum(len(r.get('long', [])) + len(r.get('short', [])) for r in results)
        print(f"\n{'='*80}")
        print(f"Progress: {min(batch_start + BATCH_SIZE, len(symbols))}/{len(symbols)} | Symbols: {len(results)} | Total Combos: {total_combos}")
        print(f"{'='*80}\n")
    
    # Generate YAML
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(results)} symbols with combos")
    total_combos = sum(len(r.get('long', [])) + len(r.get('short', [])) for r in results)
    print(f"Total Combos: {total_combos}")
    print(f"{'='*80}\n")
    
    yaml_lines = [
        "# Exhaustive Combo Backtest Results",
        f"# Tested: {len(list(product(RSI_BINS, MACD_BINS, VWAP_BINS, FIB_BINS, MTF_BINS)))} combos per side per symbol",
        f"# Filters: WR > {WR_THRESHOLD}%, Sharpe > {SHARPE_THRESHOLD}, N >= {MIN_TRADES}",
        f"# Symbols: {len(results)}",
        f"# Total Combos: {total_combos}",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        
        if r.get('long'):
            yaml_lines.append(f"  long:")
            for combo in r['long']:
                yaml_lines.append(f"    - \"{combo['combo']}\"")
                yaml_lines.append(f"      # WR={combo['wr']:.1f}%, Sharpe={combo['sharpe']:.2f}, N={combo['n']}")
        
        if r.get('short'):
            yaml_lines.append(f"  short:")
            for combo in r['short']:
                yaml_lines.append(f"    - \"{combo['combo']}\"")
                yaml_lines.append(f"      # WR={combo['wr']:.1f}%, Sharpe={combo['sharpe']:.2f}, N={combo['n']}")
        
        yaml_lines.append("")
    
    output = "\n".join(yaml_lines)
    
    with open('symbol_overrides_EXHAUSTIVE.yaml', 'w') as f:
        f.write(output)
    
    print(f"✅ Saved to symbol_overrides_EXHAUSTIVE.yaml")
    print(f"✅ {len(results)} symbols with {total_combos} total combos")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(run())
