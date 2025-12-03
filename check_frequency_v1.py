import asyncio
import aiohttp
import pandas as pd
import yaml
import logging
from backtest_wf_v2 import BacktestEngine, fetch_klines

async def run():
    # Load V1 Strategies
    with open('symbol_overrides_400.yaml', 'r') as f:
        overrides = yaml.safe_load(f)
        
    # Filter out comments/metadata if any (yaml load handles this usually)
    # Structure is {symbol: {side: [combo, ...]}}
    
    engine = BacktestEngine()
    
    print(f"Analyzing frequency for {len(overrides)} symbols (V1 System)...")
    print(f"{'Symbol':<15} | {'Trades (30d)':<12} | {'Freq/Day':<8}")
    print("-" * 50)
    
    total_trades = 0
    total_days = 0
    processed = 0
    
    async with aiohttp.ClientSession() as session:
        for symbol, sides in overrides.items():
            if not isinstance(sides, dict): continue # Skip metadata keys
            
            # Fetch last 30 days only for speed
            # We need enough data for indicators (1000 candles is ~2 days of 3m, but we need 30 days)
            # 30 days * 24h * 20 candles/h = 14400 candles.
            # fetch_klines in backtest_wf_v2 fetches by chunks. We can reuse it but limit days.
            
            # Monkey patch DAYS for this run or just pass it if I modified fetch_klines?
            # I'll just use the existing fetch_klines but it uses global DAYS=180.
            # That's fine, it's robust. It might take a bit longer but it's accurate.
            
            df = await fetch_klines(session, symbol)
            if df.empty: continue
            
            df = engine.calculate_indicators(df)
            
            # Filter last 30 days
            last_ts = df.iloc[-1]['startTime']
            start_ts = last_ts - (30 * 24 * 60 * 60 * 1000)
            df_30d = df[df['startTime'] >= start_ts].copy()
            
            if df_30d.empty: continue
            
            days = (df_30d.iloc[-1]['startTime'] - df_30d.iloc[0]['startTime']) / (1000 * 60 * 60 * 24)
            if days < 1: days = 1
            
            sym_trades = 0
            
            for side, combos in sides.items():
                for combo_str in combos:
                    # Parse combo string "RSI:..."
                    # The stored format is "RSI:..."
                    
                    sigs = (df_30d['bbw_pct'] > 0.45) & (df_30d['vol_ratio'] > 0.8)
                    if side == 'long':
                        sigs = sigs & (df_30d['close'] > df_30d['open'])
                    else:
                        sigs = sigs & (df_30d['close'] < df_30d['open'])
                        
                    for idx in df_30d[sigs].index:
                        row = df_30d.loc[idx]
                        if engine.get_combo(row) == combo_str:
                            sym_trades += 1
                            
            freq = sym_trades / days
            # print(f"{symbol:<15} | {sym_trades:<12} | {freq:.2f}")
            
            total_trades += sym_trades
            total_days += days
            processed += 1
            print(".", end="", flush=True)
            
    avg_freq = total_trades / (total_days / processed) if processed > 0 else 0
    # Total trades per day across ALL symbols = sum of individual freqs
    # Actually, total_days is sum of days. 
    # We want "System Trades Per Day" = Sum(Freq_Symbol_i)
    
    # Let's re-sum properly
    system_freq = 0
    print("\n" + "-" * 50)
    
    # Re-loop to sum frequencies (I didn't store them, let's just use the total_trades / 30 roughly)
    # Wait, total_trades is sum of all trades. total_days is sum of all days (e.g. 30 * 115).
    # So Total Trades / (Total Days / Num Symbols) = Total Trades / 30 = System Trades/Day
    
    system_daily_trades = total_trades / 30 # Approx
    
    print(f"✅ Analyzed {processed} symbols.")
    print(f"📊 Total Trades (Last 30 Days): {total_trades}")
    print(f"🚀 System Frequency: ~{system_daily_trades:.1f} trades/day")

if __name__ == "__main__":
    asyncio.run(run())
