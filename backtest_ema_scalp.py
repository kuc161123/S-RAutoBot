#!/usr/bin/env python3
"""
EMA Scalp Backtest - 9/21 EMA Crossover + RSI + VWAP Strategy
Based on Reddit/forum research on profitable crypto scalping strategies.

Strategy Rules:
- LONG: 9 EMA crosses above 21 EMA, RSI 40-60, Price above VWAP
- SHORT: 9 EMA crosses below 21 EMA, RSI 40-60, Price below VWAP
- TP: 1.5x ATR, SL: 1x ATR
"""

import os
import sys
import time
import yaml
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
LOOKBACK_DAYS = 60  # Reduced for faster testing
MIN_TRADES = 5      # Lower threshold to find more setups
MIN_WR = 35.0       # Lower WR threshold for discovery
ATR_PERIOD = 14
ATR_SL_MULT = 1.0   # Tighter for scalping
ATR_TP_MULT = 1.5   # 1:1.5 R:R

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def calculate_indicators(df):
    """Calculate EMA, RSI, VWAP, ATR"""
    if len(df) < 50: return df
    
    # ATR for SL/TP
    df['atr'] = df.ta.atr(length=ATR_PERIOD)
    
    # EMAs
    df['ema_9'] = df.ta.ema(length=9)
    df['ema_21'] = df.ta.ema(length=21)
    
    # Previous EMAs for crossover detection
    df['prev_ema_9'] = df['ema_9'].shift(1)
    df['prev_ema_21'] = df['ema_21'].shift(1)
    
    # RSI
    df['rsi'] = df.ta.rsi(length=14)
    
    # VWAP
    try:
        vwap = df.ta.vwap(high='high', low='low', close='close', volume='volume')
        if isinstance(vwap, pd.DataFrame):
            df['vwap'] = vwap.iloc[:, 0]
        else:
            df['vwap'] = vwap
    except Exception:
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).rolling(480).sum() / df['volume'].rolling(480).sum()
    
    return df.dropna()

def run_backtest(df):
    """
    Run EMA scalp backtest.
    Returns results with stats for both long and short.
    """
    results = {
        'long': {'wins': 0, 'losses': 0, 'total': 0, 'pnl': 0.0, 'trades': []},
        'short': {'wins': 0, 'losses': 0, 'total': 0, 'pnl': 0.0, 'trades': []}
    }
    
    rows = list(df.itertuples())
    
    for i, row in enumerate(rows[:-20]):  # Leave room for forward scan
        # Skip if missing data
        if pd.isna(row.ema_9) or pd.isna(row.ema_21) or pd.isna(row.rsi) or pd.isna(row.vwap):
            continue
        if pd.isna(row.prev_ema_9) or pd.isna(row.prev_ema_21):
            continue
            
        # RSI Filter: Must be between 40-60 (neutral zone)
        if row.rsi < 40 or row.rsi > 60:
            continue
            
        side = None
        
        # LONG Signal: 9 EMA crosses above 21 EMA + Price above VWAP
        if row.prev_ema_9 <= row.prev_ema_21 and row.ema_9 > row.ema_21:
            if row.close > row.vwap:
                side = 'long'
        
        # SHORT Signal: 9 EMA crosses below 21 EMA + Price below VWAP
        elif row.prev_ema_9 >= row.prev_ema_21 and row.ema_9 < row.ema_21:
            if row.close < row.vwap:
                side = 'short'
        
        if not side:
            continue
            
        # Calculate TP/SL
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
        outcome = None
        
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
            results[side]['trades'].append({
                'entry': entry_price,
                'tp': tp,
                'sl': sl,
                'outcome': outcome,
                'pnl': outcome_pnl
            })
    
    return results

def backtest_symbol(sym, idx, total, broker):
    """Backtest a single symbol"""
    try:
        # Calculate start time
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
        
        # Get klines
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
        
        # Run backtest on both
        train_results = run_backtest(train)
        test_results = run_backtest(test)
        
        # Check if valid
        valid_sides = {}
        
        for side in ['long', 'short']:
            train_stats = train_results[side]
            test_stats = test_results[side]
            
            # Must have enough trades
            if train_stats['total'] < MIN_TRADES:
                continue
            if test_stats['total'] < 2:
                continue
                
            # Calculate WR
            train_wr = (train_stats['wins'] / train_stats['total']) * 100 if train_stats['total'] > 0 else 0
            test_wr = (test_stats['wins'] / test_stats['total']) * 100 if test_stats['total'] > 0 else 0
            
            # Must meet WR threshold in both
            if train_wr >= MIN_WR and test_wr >= MIN_WR:
                if train_stats['pnl'] > 0 and test_stats['pnl'] > 0:
                    valid_sides[side] = {
                        'train_wr': train_wr,
                        'test_wr': test_wr,
                        'train_trades': train_stats['total'],
                        'test_trades': test_stats['total'],
                        'train_pnl': train_stats['pnl'],
                        'test_pnl': test_stats['pnl']
                    }
        
        if valid_sides:
            msg_parts = []
            for side, stats in valid_sides.items():
                msg_parts.append(f"{side.upper()}: WR {stats['test_wr']:.1f}% (N={stats['train_trades']+stats['test_trades']})")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg_parts)}")
            
            # Save to YAML
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
    yaml_file = 'symbol_overrides_EMA_Scalp.yaml'
    
    try:
        existing = {}
        try:
            with open(yaml_file, 'r') as f:
                existing = yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass
        
        existing[sym] = {
            'strategy': 'EMA_Scalp',
            'long': valid_sides.get('long') is not None,
            'short': valid_sides.get('short') is not None,
            'stats': valid_sides
        }
        
        with open(yaml_file, 'w') as f:
            yaml.dump(existing, f)
            
        # Auto-push to git
        import subprocess
        try:
            subprocess.run(['git', 'add', yaml_file], check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'Auto: EMA Scalp {sym}'], check=True, capture_output=True)
            subprocess.run(['git', 'push'], check=True, capture_output=True)
            logger.info(f"✅ Auto-pushed {sym} to git")
        except subprocess.CalledProcessError:
            pass  # May fail if no changes
            
    except Exception as e:
        logger.error(f"Failed to save result: {e}")

def main():
    # Load config
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
    
    # Setup broker
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from autobot.brokers.bybit import Bybit, BybitConfig
    
    broker = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    # Load symbols
    with open('symbols_400.yaml', 'r') as f:
        symbols = yaml.safe_load(f)['symbols']
    
    logger.info(f"🚀 Starting EMA Scalp Backtest on {len(symbols)} symbols")
    logger.info(f"📊 Strategy: 9/21 EMA Crossover + RSI(40-60) + VWAP")
    logger.info(f"⚙️ Settings: {LOOKBACK_DAYS} days, MIN_TRADES={MIN_TRADES}, MIN_WR={MIN_WR}%")
    
    winners = []
    
    for idx, sym in enumerate(symbols, 1):
        result = backtest_symbol(sym, idx, len(symbols), broker)
        if result:
            winners.append(result)
        time.sleep(0.5)  # Rate limiting
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ BACKTEST COMPLETE")
    logger.info(f"📊 Winners: {len(winners)} / {len(symbols)}")
    for w in winners:
        logger.info(f"  - {w['symbol']}: {list(w['results'].keys())}")

if __name__ == '__main__':
    main()
