
import asyncio
import pandas as pd
import numpy as np
import yaml
import os
import sys
import logging
from datetime import datetime
from itertools import product

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Discovery")

sys.path.insert(0, os.path.dirname(__file__))
from autobot.brokers.bybit import Bybit, BybitConfig
from autobot.strategies.scalp.detector import detect_scalp_signal, ScalpSettings

# Configuration
SYMBOLS_TO_TEST = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'LTCUSDT', 'ETCUSDT',
    'APTUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'FILUSDT',
    'INJUSDT', 'SUIUSDT', 'SEIUSDT', 'TRBUSDT', 'TIAUSDT',
    'AAVEUSDT', 'ICPUSDT', 'RENDERUSDT', 'LDOUSDT', 'WLDUSDT'
]

# Parameter Grid
PARAM_GRID = {
    'timeframe': ['3', '5'],
    'rr': [1.5, 2.0, 2.5],
    'strategy_logic': ['vwap_bounce', 'ema_trend']
}

COSTS_PCT = 0.12  # Slippage + Fees (0.06 + 0.06)

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        return os.getenv(config[2:-1], config)
    return config

async def fetch_data(bybit, symbol, timeframe, limit=1000):
    """Fetch historical data"""
    try:
        klines = []
        end_ts = None
        
        # Fetch enough chunks
        chunks = limit // 200 + 1
        for _ in range(chunks):
            k = bybit.get_klines(symbol, timeframe, limit=200, end=end_ts)
            if not k:
                break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.05)
            
        if not klines:
            return None
            
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
            
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def run_backtest(df, settings, strategy_logic):
    """Run fast vectorised-style backtest (iterative for now for safety)"""
    trades = []
    
    # Pre-calculate indicators to speed up
    # (In a real optimized engine we'd do this once, but here we rely on detector)
    
    # We iterate through candles
    # To be efficient, we only check logic every candle
    
    # Note: This calls the actual detector which might re-calc indicators. 
    # For discovery, this might be slow. 
    # Optimization: We will just instantiate settings and call detector.
    
    # Override settings based on logic
    if strategy_logic == 'vwap_bounce':
        settings.vwap_only = True
        settings.ema_trend_filter = False
    elif strategy_logic == 'ema_trend':
        settings.vwap_only = False
        settings.ema_trend_filter = True
        
    for i in range(50, len(df)-1):
        hist = df.iloc[:i+1]
        
        # Detect
        signal = detect_scalp_signal(hist, settings, "TEST")
        
        if signal:
            # Check outcome
            entry = df.iloc[i+1]['open'] # Next candle open
            
            # Apply slippage
            if signal.side == 'long':
                entry_price = entry * (1 + 0.0004)
                tp = entry_price * (1 + (settings.min_r_pct * settings.rr))
                sl = entry_price * (1 - settings.min_r_pct)
            else:
                entry_price = entry * (1 - 0.0004)
                tp = entry_price * (1 - (settings.min_r_pct * settings.rr))
                sl = entry_price * (1 + settings.min_r_pct)
                
            # Simulate trade
            outcome = 'timeout'
            pnl = 0
            
            # Look ahead 20 candles
            future = df.iloc[i+1:i+21]
            for _, candle in future.iterrows():
                if signal.side == 'long':
                    if candle['low'] <= sl:
                        outcome = 'loss'
                        pnl = -settings.min_r_pct
                        break
                    if candle['high'] >= tp:
                        outcome = 'win'
                        pnl = settings.min_r_pct * settings.rr
                        break
                else:
                    if candle['high'] >= sl:
                        outcome = 'loss'
                        pnl = -settings.min_r_pct
                        break
                    if candle['low'] <= tp:
                        outcome = 'win'
                        pnl = settings.min_r_pct * settings.rr
                        break
            
            # Deduct fees
            pnl -= 0.0012 # 0.12% fees
            
            trades.append({
                'pnl': pnl,
                'outcome': outcome
            })
            
    return trades

async def main():
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    results = {}
    
    print(f"{'='*60}")
    print(f"🚀 ADAPTIVE STRATEGY DISCOVERY")
    print(f"{'='*60}")
    
    for symbol in SYMBOLS_TO_TEST:
        print(f"\nAnalyzing {symbol}...")
        best_score = -999
        best_config = None
        
        # 1. Fetch Data for both timeframes
        data_3m = await fetch_data(bybit, symbol, '3', limit=1000)
        data_5m = await fetch_data(bybit, symbol, '5', limit=1000)
        
        if data_3m is None or data_5m is None:
            continue
            
        # 2. Grid Search
        combinations = list(product(PARAM_GRID['timeframe'], PARAM_GRID['rr'], PARAM_GRID['strategy_logic']))
        
        for tf, rr, logic in combinations:
            df = data_3m if tf == '3' else data_5m
            
            # Create settings
            settings = ScalpSettings(
                rr=rr,
                min_r_pct=0.005, # Fixed 0.5% base risk
                ema_fast=8,
                ema_slow=21
            )
            
            trades = run_backtest(df, settings, logic)
            
            if len(trades) < 10:
                continue
                
            wins = sum(1 for t in trades if t['outcome'] == 'win')
            wr = (wins / len(trades)) * 100
            total_pnl = sum(t['pnl'] for t in trades)
            
            # Score: PnL * log(trades) -> favors profitable strategies with sample size
            score = total_pnl * np.log(len(trades))
            
            if score > best_score:
                best_score = score
                best_config = {
                    'timeframe': tf,
                    'rr': rr,
                    'logic': logic,
                    'wr': wr,
                    'trades': len(trades),
                    'pnl': total_pnl
                }
        
        if best_config:
            print(f"  Best for {symbol}: {best_config['timeframe']}m | {best_config['logic']} | R:R {best_config['rr']} | WR {best_config['wr']:.1f}% ({best_config['trades']} tr) | PnL {best_config['pnl']*100:.1f}%")
            
            # Threshold check (Lowered to 40% for visibility)
            if best_config['wr'] > 40 and best_config['pnl'] > 0:
                results[symbol] = best_config
                print(f"  ✅ SELECTED")
            else:
                print(f"  ❌ REJECTED (Low WR or PnL)")
        else:
            print(f"  ❌ NO TRADES FOUND")
            
    # Output to YAML
    print(f"\n{'='*60}")
    print(f"✅ DISCOVERY COMPLETE")
    print(f"Generating symbol_overrides.yaml...")
    
    overrides = {}
    for sym, res in results.items():
        overrides[sym] = {
            'scalp': {
                'timeframe': int(res['timeframe']),
                'rr': float(res['rr']),
                'signal': {
                    'vwap_only': True if res['logic'] == 'vwap_bounce' else False,
                    'ema_trend_filter': True if res['logic'] == 'ema_trend' else False
                }
            }
        }
        
    with open('symbol_overrides.yaml', 'w') as f:
        yaml.dump(overrides, f)
        
    print(f"Saved {len(overrides)} symbol configurations.")

if __name__ == "__main__":
    asyncio.run(main())
