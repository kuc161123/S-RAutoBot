#!/usr/bin/env python3
"""
Trailing SL Optimization Backtest
Tests 27 configurations to find optimal trailing settings for 3m divergence strategy.

V3: Self-contained for robust execution on current project structure.
Filters applied (Trio): VWAP, RSI Zones, Volume.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import yaml
import sys
import os
import requests
from dataclasses import dataclass
from typing import List, Dict

# Set path to include project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autobot.brokers.bybit import Bybit
from autobot.core.divergence_detector import detect_divergence, calculate_rsi, prepare_dataframe, DivergenceSignal

@dataclass
class TrailingConfig:
    """Configuration for trailing SL strategy"""
    be_threshold: float  # R to move to BE
    trail_distance: float  # R to trail behind
    max_tp: float  # Max R target
    
    def __str__(self):
        return f"BE:{self.be_threshold}R | Trail:{self.trail_distance}R | Max:{self.max_tp}R"

class OptimizationBacktest:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.broker = Bybit(self.cfg)
        self.slippage_pct = 0.02  # 0.02% slippage base
        
    async def get_data(self, symbols: List[str], limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Fetch historical candles"""
        data = {}
        print(f"📥 Loading data for {len(symbols)} symbols...")
        for sym in symbols:
            try:
                klines = self.broker.get_klines(sym, '3', limit=limit)
                if klines and len(klines) > 100:
                    df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    # Calculate VWAP manually for backtest reliability
                    # Simple VWAP (doesn't reset daily for simplicity in backtest window)
                    tp = (df['high'] + df['low'] + df['close']) / 3
                    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
                    
                    df = prepare_dataframe(df)
                    data[sym] = df
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"Error loading {sym}: {e}")
        return data

    def check_trio_filters(self, row, signal_type, side):
        """Standard High-Probability Trio checks"""
        # 1. VWAP Check
        price = row['close']
        vwap = row['vwap']
        if side == 'long' and price >= vwap: return False
        if side == 'short' and price <= vwap: return False
        
        # 2. RSI Zone Check
        rsi = row['rsi']
        if signal_type == 'regular_bullish' and rsi >= 30: return False
        if signal_type == 'regular_bearish' and rsi <= 70: return False
        if signal_type == 'hidden_bullish' and not (30 <= rsi <= 50): return False
        if signal_type == 'hidden_bearish' and not (50 <= rsi <= 70): return False
        
        # 3. Volume Check (already in prepare_dataframe as vol_ok)
        if not getattr(row, 'vol_ok', True): return False
        
        return True

    def simulate_trade(self, df, start_idx, side, entry_price, sl_dist, config):
        """Simulate a trade with trailing SL updates every candle"""
        max_favorable_r = 0.0
        trailing_active = False
        current_sl = entry_price - sl_dist if side == 'long' else entry_price + sl_dist
        
        for i in range(start_idx + 1, len(df)):
            row = df.iloc[i]
            
            # Update max favorable R
            if side == 'long':
                unrealized_r = (row['high'] - entry_price) / sl_dist
            else:
                unrealized_r = (entry_price - row['low']) / sl_dist
            
            if unrealized_r > max_favorable_r:
                max_favorable_r = unrealized_r
            
            # Check Stop Loss
            if side == 'long':
                if row['low'] <= current_sl:
                    pnl_r = (current_sl - entry_price) / sl_dist
                    return pnl_r, "SL", i - start_idx
            else:
                if row['high'] >= current_sl:
                    pnl_r = (entry_price - current_sl) / sl_dist
                    return pnl_r, "SL", i - start_idx

            # Check Max TP
            if max_favorable_r >= config.max_tp:
                return config.max_tp, "TP", i - start_idx
            
            # Update Trailing / BE
            if max_favorable_r >= config.be_threshold:
                trailing_active = True
                protected_r = max_favorable_r - config.trail_distance
                protected_r = max(protected_r, 0.4) # Min protect some profit once past BE
                
                if side == 'long':
                    new_sl = entry_price + (protected_r * sl_dist)
                    if new_sl > current_sl: current_sl = new_sl
                else:
                    new_sl = entry_price - (protected_r * sl_dist)
                    if new_sl < current_sl: current_sl = new_sl
                    
        return max_favorable_r, "END", len(df) - 1 - start_idx

    async def run(self):
        # 1. Get symbols
        resp = requests.get("https://api.bybit.com/v5/market/tickers?category=linear")
        tickers = resp.json()['result']['list']
        symbols = [t['symbol'] for t in tickers if t['symbol'].endswith('USDT')]
        symbols = sorted(symbols, key=lambda x: float([t['turnover24h'] for t in tickers if t['symbol']==x][0]), reverse=True)[:30]
        
        # 2. Fetch Data
        data = await self.get_data(symbols)
        if not data: return
        
        # 3. Generate Configs
        configs = []
        for be in [0.5, 0.7, 1.0]:
            for trail in [0.2, 0.3, 0.5]:
                for tp in [2.0, 3.0, 4.0]:
                    configs.append(TrailingConfig(be, trail, tp))
                    
        print(f"\n🔬 Processing {len(configs)} configurations...")
        
        final_results = []
        for cfg in configs:
            trades = []
            for sym, df in data.items():
                signals = detect_divergence(df, sym)
                for sig in signals:
                    # Find candle index
                    try:
                        idx = df.index.get_loc(sig.timestamp)
                    except: continue
                    
                    row = df.iloc[idx]
                    if self.check_trio_filters(row, sig.signal_type, sig.side):
                        sl_dist = row['atr']
                        entry = row['close'] * (1+self.slippage_pct/100 if sig.side=='long' else 1-self.slippage_pct/100)
                        
                        pnl, reason, duration = self.simulate_trade(df, idx, sig.side, entry, sl_dist, cfg)
                        trades.append({'pnl': pnl, 'reason': reason})
            
            if trades:
                df_t = pd.DataFrame(trades)
                wr = len(df_t[df_t['pnl'] > 0]) / len(df_t) * 100
                total_r = df_t['pnl'].sum()
                avg_r = df_t['pnl'].mean()
                final_results.append({
                    'Config': str(cfg),
                    'Trades': len(trades),
                    'WR': f"{wr:.1f}%",
                    'Total R': round(total_r, 2),
                    'Avg R': round(avg_r, 3)
                })
        
        # 4. Report
        df_res = pd.DataFrame(final_results).sort_values('Total R', ascending=False)
        print("\n" + "="*50)
        print("🏆 TOP 10 CONFIGURATIONS")
        print("="*50)
        print(df_res.head(10).to_string(index=False))
        
        winner = df_res.iloc[0]
        print(f"\n🥇 WINNER: {winner['Config']}")
        print(f"   Total Profit: {winner['Total R']}R")
        print(f"   Win Rate: {winner['WR']}")
        
        df_res.to_csv("trailing_optimization_results.csv", index=False)
        print("\n✅ Results saved to trailing_optimization_results.csv")

if __name__ == "__main__":
    asyncio.run(OptimizationBacktest().run())
