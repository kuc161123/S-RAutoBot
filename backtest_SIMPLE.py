#!/usr/bin/env python3
"""
Simplified Strategy Backtest - Uses simpler, more robust signals

Instead of complex VWAP bounce, uses:
- Volatility expansion (BB width)
- Volume confirmation
- Trend direction (EMA)
- Simple pullback pattern

Maintains 2.1:1 R:R with realistic costs
"""
import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from autobot.brokers.bybit import Bybit, BybitConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Realistic costs
SLIPPAGE_PCT = 0.03
FEES_ROUND_TRIP_PCT = 0.11
SPREAD_PCT = 0.02
TOTAL_COST_PCT = SLIPPAGE_PCT + FEES_ROUND_TRIP_PCT + SPREAD_PCT

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        return os.getenv(config[2:-1], config)
    return config

def calculate_indicators(df):
    """Calculate technical indicators"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ATR (14)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # EMAs for trend
    df['ema8'] = close.ewm(span=8, adjust=False).mean()
    df['ema21'] = close.ewm(span=21, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + 2*std20
    df['bb_lower'] = sma20 - 2*std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / close
    
    # BB width percentile
    df['bbw_pct'] = df['bb_width'].rolling(100).rank(pct=True)
    
    # Volume ratio
    vol_ma = volume.rolling(20).mean()
    df['vol_ratio'] = volume / vol_ma
    
    # RSI for additional confirmation
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

async def backtest_symbol_simple(bybit, sym: str, idx: int, total: int):
    """Backtest symbol with simplified strategy"""
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch data
        klines = []
        end_ts = None
        for i in range(50):
            k = bybit.get_klines(sym, '3', limit=200, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.03)
        
        if len(klines) < 2000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data")
            return None
        
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        
        # Calculate indicators
        df = calculate_indicators(df)
        df = df.dropna()
        
        # Simple strategy: Volatility + Volume + Trend + Pullback
        # LONG: Uptrend + volatility + volume + pullback (close > open)
        long_signals = (
            (df['ema8'] > df['ema21']) &  # Uptrend
            (df['bbw_pct'] > 0.3) &       # Some volatility (relaxed)
            (df['vol_ratio'] > 0.6) &     # Volume confirmation (relaxed)
            (df['close'] > df['open']) &  # Bullish candle
            (df['close'] < df['ema8'])    # Pullback to EMA
        )
        
        # SHORT: Downtrend + volatility + volume + pullback (close < open)
        short_signals = (
            (df['ema8'] < df['ema21']) &  # Downtrend
            (df['bbw_pct'] > 0.3) &       # Some volatility
            (df['vol_ratio'] > 0.6) &     # Volume confirmation
            (df['close'] < df['open']) &  # Bearish candle
            (df['close'] > df['ema8'])    # Pullback to EMA
        )
        
        long_trades = []
        short_trades = []
        
        # Test longs
        for i in df[long_signals].index:
            if df.index.get_loc(i) >= len(df) - 100:
                continue
            
            row = df.loc[i]
            idx_pos = df.index.get_loc(i)
            
            # Entry on next candle open
            next_candle = df.iloc[idx_pos + 1]
            entry = float(next_candle['open']) * (1 + SLIPPAGE_PCT / 100)
            
            # SL/TP based on ATR (2.1:1 R:R maintained)
            atr = row['atr']
            sl = entry - 1.5 * atr  # 1.5 ATR stop
            tp = entry + (entry - sl) * 2.1  # 2.1:1 R:R
            
            if sl >= entry or tp <= entry:
                continue
            
            # Test outcome
            outcome = 'timeout'
            future = df.iloc[idx_pos+1:idx_pos+101]
            
            for _, f_row in future.iterrows():
                if f_row['low'] <= sl:
                    outcome = 'loss'
                    break
                if f_row['high'] >= tp:
                    outcome = 'win'
                    break
            
            # Calculate P&L with costs
            if outcome == 'win':
                pnl_pct = ((tp * (1 - SLIPPAGE_PCT/100)) - entry) / entry * 100
            elif outcome == 'loss':
                pnl_pct = ((sl * (1 - SLIPPAGE_PCT/100)) - entry) / entry * 100
            else:
                pnl_pct = 0
            
            pnl_pct -= FEES_ROUND_TRIP_PCT + SPREAD_PCT
            
            final_outcome = 'win' if pnl_pct > 0 else ('loss' if pnl_pct < 0 else 'break even')
            long_trades.append(final_outcome)
        
        # Test shorts
        for i in df[short_signals].index:
            if df.index.get_loc(i) >= len(df) - 100:
                continue
            
            row = df.loc[i]
            idx_pos = df.index.get_loc(i)
            
            next_candle = df.iloc[idx_pos + 1]
            entry = float(next_candle['open']) * (1 - SLIPPAGE_PCT / 100)
            
            atr = row['atr']
            sl = entry + 1.5 * atr
            tp = entry - (sl - entry) * 2.1
            
            if sl <= entry or tp >= entry or tp <= 0:
                continue
            
            outcome = 'timeout'
            future = df.iloc[idx_pos+1:idx_pos+101]
            
            for _, f_row in future.iterrows():
                if f_row['high'] >= sl:
                    outcome = 'loss'
                    break
                if f_row['low'] <= tp:
                    outcome = 'win'
                    break
            
            if outcome == 'win':
                pnl_pct = (entry - (tp * (1 + SLIPPAGE_PCT/100))) / entry * 100
            elif outcome == 'loss':
                pnl_pct = (entry - (sl * (1 + SLIPPAGE_PCT/100))) / entry * 100
            else:
                pnl_pct = 0
            
            pnl_pct -= FEES_ROUND_TRIP_PCT + SPREAD_PCT
            
            final_outcome = 'win' if pnl_pct > 0 else ('loss' if pnl_pct < 0 else 'breakeven')
            short_trades.append(final_outcome)
        
        # Calculate stats
        best_long = None
        best_short = None
        
        if len(long_trades) >= 25:
            wr = (sum(1 for t in long_trades if t == 'win') / len(long_trades)) * 100
            if wr > 40:
                best_long = {'wr': wr, 'n': len(long_trades)}
        
        if len(short_trades) >= 25:
            wr = (sum(1 for t in short_trades if t == 'win') / len(short_trades)) * 100
            if wr > 40:
                best_short = {'wr': wr, 'n': len(short_trades)}
        
        if best_long or best_short:
            msg = []
            if best_long:
                msg.append(f"LONG: WR={best_long['wr']:.1f}% (N={best_long['n']})")
            if best_short:
                msg.append(f"SHORT: WR={best_short['wr']:.1f}% (N={best_short['n']})")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            
            return {'symbol': sym, 'long': best_long, 'short': best_short}
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No passing strategy")
            return None
            
    except Exception as e:
        logger.error(f"[{idx}/{total}] {sym} ❌ Error: {e}")
        return None

async def run():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    with open('symbols_400.yaml', 'r') as f:
        data = yaml.safe_load(f)
    symbols = data['symbols'] if isinstance(data, dict) else data
    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    print(f"{'='*80}")
    print(f"🔬 SIMPLIFIED STRATEGY BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Strategy: Trend + Volatility + Volume + Pullback")
    print(f"R:R: 2.1:1 (maintained)")
    print(f"Criteria: WR > 40%, N >= 25")
    print(f"Costs: {TOTAL_COST_PCT}% per trade")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    for idx, sym in enumerate(symbols, 1):
        result = await backtest_symbol_simple(bybit, sym, idx, len(symbols))
        if result:
            results.append(result)
        
        if idx % 50 == 0:
            print(f"\n{'='*80}")
            print(f"Progress: {idx}/{len(symbols)} | Passed: {len(results)}")
            print(f"{'='*80}\n")
    
    # Save results
    yaml_lines = [
        "# Simplified Strategy Backtest Results",
        f"# Strategy: Trend (EMA 8/21) + Volatility (BBW) + Volume + Pullback",
        f"# R:R: 2.1:1",
        f"# Costs: {TOTAL_COST_PCT}% per trade",
        f"# Processed: {len(symbols)} symbols",
        f"# Passed: {len(results)} symbols",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    wr: {r['long']['wr']:.1f}%")
            yaml_lines.append(f"    n: {r['long']['n']}")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    wr: {r['short']['wr']:.1f}%")
            yaml_lines.append(f"    n: {r['short']['n']}")
        yaml_lines.append("")
    
    with open('symbol_overrides_SIMPLE.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
    
    print(f"\n{'='*80}")
    print(f"✅ BACKTEST COMPLETE")
    print(f"{'='*80}")
    print(f"Symbols validated: {len(results)}")
    print(f"Output: symbol_overrides_SIMPLE.yaml")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(run())
