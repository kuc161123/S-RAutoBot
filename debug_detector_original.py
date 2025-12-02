
import asyncio
import pandas as pd
import yaml
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from autobot.brokers.bybit import Bybit, BybitConfig
from autobot.strategies.scalp.detector import detect_scalp_signal, ScalpSettings

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        return os.getenv(config[2:-1], config)
    return config

async def debug():
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    symbol = "BTCUSDT"
    print(f"Fetching data for {symbol}...")
    klines = bybit.get_klines(symbol, '3', limit=200)
    
    df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    df.set_index('start', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
        
    print(f"Data loaded: {len(df)} candles")
    
    # Settings from discovery script
    settings = ScalpSettings(
        rr=1.5,
        min_r_pct=0.005,
        ema_fast=8,
        ema_slow=21,
        vwap_only=True # Test VWAP logic first
    )
    
    print("\nRunning detection on last 50 candles...")
    
    signals = 0
    for i in range(50, len(df)):
        hist = df.iloc[:i+1]
        sig = detect_scalp_signal(hist, settings, symbol)
        
        if sig:
            signals += 1
            print(f"[{hist.index[-1]}] SIGNAL: {sig.side} @ {sig.entry}")
            print(f"  Reason: {sig.reason}")
            print(f"  SL: {sig.sl} | TP: {sig.tp}")
            
            # Check immediate outcome
            if i + 1 < len(df):
                next_c = df.iloc[i+1]
                print(f"  Next Candle: Open {next_c['open']} High {next_c['high']} Low {next_c['low']} Close {next_c['close']}")
                
                if sig.side == 'long':
                    if next_c['close'] > sig.entry:
                        print("  -> MOVED UP (Good)")
                    else:
                        print("  -> MOVED DOWN (Bad)")
                else:
                    if next_c['close'] < sig.entry:
                        print("  -> MOVED DOWN (Good)")
                    else:
                        print("  -> MOVED UP (Bad)")
                        
    print(f"\nTotal signals: {signals}")

if __name__ == "__main__":
    asyncio.run(debug())
