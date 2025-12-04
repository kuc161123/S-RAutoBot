#!/usr/bin/env python3
"""
Bollinger Band Squeeze Breakout Backtest
Based on forum research - one of the most popular scalping strategies.

Strategy Rules:
- Detect BB Squeeze (bands narrowing - low volatility)
- LONG: Price breaks above upper band after squeeze
- SHORT: Price breaks below lower band after squeeze
- TP: 1.5x ATR, SL: Middle band or 1x ATR
"""

import os
import sys
import time
import yaml
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
LOOKBACK_DAYS = 120
MIN_TRADES = 10
MIN_WR = 40.0
ATR_PERIOD = 14
ATR_SL_MULT = 1.0
ATR_TP_MULT = 1.5
BB_LENGTH = 20
BB_STD = 2.0
SQUEEZE_PERCENTILE = 20  # BB width in bottom 20% = squeeze

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def calculate_indicators(df):
    """Calculate BB, ATR, and squeeze detection"""
    if len(df) < 50: return df
    
    # ATR for SL/TP
    df['atr'] = df.ta.atr(length=ATR_PERIOD)
    
    # Bollinger Bands
    bb = df.ta.bbands(length=BB_LENGTH, std=BB_STD)
    df['bb_upper'] = bb[f'BBU_{BB_LENGTH}_{BB_STD}']
    df['bb_middle'] = bb[f'BBM_{BB_LENGTH}_{BB_STD}']
    df['bb_lower'] = bb[f'BBL_{BB_LENGTH}_{BB_STD}']
    
    # BB Width (for squeeze detection)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Rolling percentile of BB width (squeeze = narrow bands)
    df['bb_width_percentile'] = df['bb_width'].rolling(100).apply(
        lambda x: (x.iloc[-1] <= x.quantile(SQUEEZE_PERCENTILE/100)) * 1.0
    )
    
    # Previous values for breakout detection
    df['prev_close'] = df['close'].shift(1)
    df['prev_bb_upper'] = df['bb_upper'].shift(1)
    df['prev_bb_lower'] = df['bb_lower'].shift(1)
    
    # Was in squeeze recently (last 5 candles)
    df['recent_squeeze'] = df['bb_width_percentile'].rolling(5).max()
    
    return df.dropna()

def run_backtest(df):
    """Run BB squeeze breakout backtest"""
    results = {
        'long': {'wins': 0, 'losses': 0, 'total': 0, 'pnl': 0.0},
        'short': {'wins': 0, 'losses': 0, 'total': 0, 'pnl': 0.0}
    }
    
    rows = list(df.itertuples())
    
    for i, row in enumerate(rows[:-20]):
        # Skip if missing data
        if pd.isna(row.bb_upper) or pd.isna(row.atr):
            continue
            
        # Must have had a recent squeeze
        if row.recent_squeeze != 1.0:
            continue
            
        side = None
        
        # LONG: Price breaks above upper band
        if row.prev_close <= row.prev_bb_upper and row.close > row.bb_upper:
            side = 'long'
        
        # SHORT: Price breaks below lower band
        elif row.prev_close >= row.prev_bb_lower and row.close < row.bb_lower:
            side = 'short'
        
        if not side:
            continue
            
        # Calculate TP/SL
        entry_price = row.close
        atr = row.atr
        
        if side == 'long':
            sl = max(row.bb_middle, entry_price - (ATR_SL_MULT * atr))
            tp = entry_price + (ATR_TP_MULT * atr)
        else:
            sl = min(row.bb_middle, entry_price + (ATR_SL_MULT * atr))
            tp = entry_price - (ATR_TP_MULT * atr)
        
        # Forward scan for outcome
        outcome = None
        outcome_pnl = 0.0
        
        for j in range(i+1, min(i+20, len(rows))):
            future = rows[j]
            
            if side == 'long':
                if future.low <= sl:
                    outcome_pnl = -ATR_SL_MULT
                    outcome = 'loss'
                    break
                elif future.high >= tp:
                    outcome_pnl = ATR_TP_MULT
                    outcome = 'win'
                    break
            else:
                if future.high >= sl:
                    outcome_pnl = -ATR_SL_MULT
                    outcome = 'loss'
                    break
                elif future.low <= tp:
                    outcome_pnl = ATR_TP_MULT
                    outcome = 'win'
                    break
        
        if outcome:
            results[side]['total'] += 1
            results[side]['pnl'] += outcome_pnl
            if outcome == 'win':
                results[side]['wins'] += 1
            else:
                results[side]['losses'] += 1
    
    return results

def backtest_symbol(sym, idx, total, broker):
    """Backtest a single symbol"""
    try:
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
        
        klines = broker.get_klines(sym, '3', limit=1000, start=start_time, end=end_time)
        if not klines or len(klines) < 200:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ Insufficient data")
            return None
            
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        df.sort_index(inplace=True)
        
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        
        df = calculate_indicators(df)
        if len(df) < 100:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ Insufficient data after indicators")
            return None
        
        # Split: 70% train, 30% test
        split = int(len(df) * 0.7)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        train_results = run_backtest(train)
        test_results = run_backtest(test)
        
        valid_sides = {}
        
        for side in ['long', 'short']:
            train_stats = train_results[side]
            test_stats = test_results[side]
            
            if train_stats['total'] < MIN_TRADES:
                continue
            if test_stats['total'] < 3:
                continue
                
            train_wr = (train_stats['wins'] / train_stats['total']) * 100 if train_stats['total'] > 0 else 0
            test_wr = (test_stats['wins'] / test_stats['total']) * 100 if test_stats['total'] > 0 else 0
            
            if train_wr >= MIN_WR and test_wr >= MIN_WR:
                if train_stats['pnl'] > 0 and test_stats['pnl'] > 0:
                    valid_sides[side] = {
                        'train_wr': train_wr,
                        'test_wr': test_wr,
                        'train_trades': train_stats['total'],
                        'test_trades': test_stats['total'],
                        'total_trades': train_stats['total'] + test_stats['total']
                    }
        
        if valid_sides:
            msg_parts = []
            for side, stats in valid_sides.items():
                msg_parts.append(f"{side.upper()}: WR {stats['test_wr']:.1f}% (N={stats['total_trades']})")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg_parts)}")
            
            save_result(sym, valid_sides)
            return {'symbol': sym, 'results': valid_sides}
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No valid setup")
            return None
            
    except Exception as e:
        logger.error(f"[{idx}/{total}] {sym} ❌ Error: {e}")
        return None

def save_result(sym, valid_sides):
    """Save valid symbol to YAML"""
    yaml_file = 'symbol_overrides_BB_Squeeze.yaml'
    
    try:
        existing = {}
        try:
            with open(yaml_file, 'r') as f:
                existing = yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass
        
        existing[sym] = {
            'strategy': 'BB_Squeeze',
            'long': valid_sides.get('long') is not None,
            'short': valid_sides.get('short') is not None,
            'stats': valid_sides
        }
        
        with open(yaml_file, 'w') as f:
            yaml.dump(existing, f)
            
        # Auto-push
        import subprocess
        try:
            subprocess.run(['git', 'add', yaml_file], check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'Auto: BB Squeeze {sym}'], check=True, capture_output=True)
            subprocess.run(['git', 'push'], check=True, capture_output=True)
            logger.info(f"✅ Auto-pushed {sym}")
        except subprocess.CalledProcessError:
            pass
            
    except Exception as e:
        logger.error(f"Failed to save: {e}")

def main():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v
    except FileNotFoundError:
        pass
        
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from autobot.brokers.bybit import Bybit, BybitConfig
    
    broker = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    with open('symbols_400.yaml', 'r') as f:
        symbols = yaml.safe_load(f)['symbols']
    
    logger.info(f"🚀 Starting BB Squeeze Backtest on {len(symbols)} symbols")
    logger.info(f"📊 Strategy: Bollinger Band Squeeze Breakout")
    logger.info(f"⚙️ Settings: BB({BB_LENGTH},{BB_STD}), Squeeze<{SQUEEZE_PERCENTILE}%, MIN_WR={MIN_WR}%")
    
    winners = []
    
    for idx, sym in enumerate(symbols, 1):
        result = backtest_symbol(sym, idx, len(symbols), broker)
        if result:
            winners.append(result)
        time.sleep(0.3)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ BACKTEST COMPLETE")
    logger.info(f"📊 Winners: {len(winners)} / {len(symbols)}")
    for w in winners:
        logger.info(f"  - {w['symbol']}: {list(w['results'].keys())}")

if __name__ == '__main__':
    main()
