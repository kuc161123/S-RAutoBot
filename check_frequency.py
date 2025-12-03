import asyncio
import aiohttp
import pandas as pd
import logging
from backtest_wf_v2 import BacktestEngine, fetch_klines

# Survivors from CSV
SURVIVORS = {
    'AAVEUSDT': 'short RSI:40-60 MACD:bear VWAP:<0.6 Fib:0-23 MTF',
    'APRUSDT': 'short RSI:<30 MACD:bear VWAP:1.2+ Fib:78-100 noMTF',
    'ALUUSDT': 'long RSI:30-40 MACD:bear VWAP:0.6-1.2 Fib:78-100 noMTF',
    'B2USDT': 'short RSI:60-70 MACD:bull VWAP:0.6-1.2 Fib:0-23 MTF',
    'BABYUSDT': 'short RSI:40-60 MACD:bull VWAP:<0.6 Fib:23-38 MTF',
    'ROSEUSDT': 'short RSI:40-60 MACD:bull VWAP:<0.6 Fib:23-38 MTF'
}

async def run():
    engine = BacktestEngine()
    print(f"{'Symbol':<10} | {'Combo':<50} | {'Days':<5} | {'Trades':<6} | {'Freq/Day':<8}")
    print("-" * 100)
    
    total_trades = 0
    total_days = 0
    
    async with aiohttp.ClientSession() as session:
        for symbol, target_combo in SURVIVORS.items():
            df = await fetch_klines(session, symbol)
            if df.empty: continue
            
            # Recalculate indicators
            df = engine.calculate_indicators(df)
            
            # Split (same as backtest)
            n = len(df)
            train_end = int(n * 0.60)
            test_end = int(n * 0.80)
            
            # We care about Test + Holdout (The "Live" Simulation part)
            # Test: 60% -> 80%
            # Holdout: 80% -> 100%
            # Total: 60% -> 100% (Last 40% of data)
            
            live_df = df.iloc[train_end:]
            
            # Count trades for the specific combo
            trades = 0
            start_time = pd.to_datetime(live_df.iloc[0]['startTime'], unit='ms')
            end_time = pd.to_datetime(live_df.iloc[-1]['startTime'], unit='ms')
            days = (end_time - start_time).days
            
            target_side = target_combo.split(' ')[0]
            target_c_str = target_combo.split(' ', 1)[1]
            
            sigs = (live_df['bbw_pct'] > 0.45) & (live_df['vol_ratio'] > 0.8)
            if target_side == 'long':
                sigs = sigs & (live_df['close'] > live_df['open'])
            else:
                sigs = sigs & (live_df['close'] < live_df['open'])
                
            for idx in live_df[sigs].index:
                row = live_df.loc[idx]
                if engine.get_combo(row) == target_c_str:
                    trades += 1
            
            freq = trades / days if days > 0 else 0
            print(f"{symbol:<10} | {target_combo:<50} | {days:<5} | {trades:<6} | {freq:.2f}")
            
            total_trades += trades
            total_days = days # Should be roughly same for all
            
    print("-" * 100)
    print(f"TOTAL: {total_trades} trades over ~{total_days} days")
    print(f"AVERAGE: {total_trades/total_days:.2f} trades/day (Combined)")

if __name__ == "__main__":
    asyncio.run(run())
