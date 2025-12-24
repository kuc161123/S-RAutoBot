#!/usr/bin/env python3
"""
3M FILTER OPTIMIZATION
Based on Round 1 winner: 0.5% SL + Trail_Tight_5R (+495R)

Testing various filters to improve performance:
1. Volume filters (1x, 1.5x, 2x average)
2. RSI extreme zones
3. Hidden divergence only
4. Trend alignment (EMA filter)
5. Session filtering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration from Round 1 winner
TIMEFRAME = '3'
SL_PCT = 0.005  # 0.5%
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1
MAX_R = 5.0
DAYS = 30
SYMBOLS_COUNT = 50

# Fee model
TAKER_FEE = 0.00055
MAKER_FEE = 0.0002

def load_symbols(n=50):
    """Load top symbols"""
    from pybit.unified_trading import HTTP
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    api_key = config.get('api_key', '')
    api_secret = config.get('api_secret', '')
    
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret
    )
    
    result = session.get_tickers(category="linear")
    tickers = result.get('result', {}).get('list', [])
    
    usdt_perps = [t for t in tickers if t['symbol'].endswith('USDT') 
                  and 'USDC' not in t['symbol']]
    
    sorted_tickers = sorted(usdt_perps, 
                           key=lambda x: float(x.get('turnover24h', 0)), 
                           reverse=True)
    
    return [t['symbol'] for t in sorted_tickers[:n]]

def fetch_data(symbol, timeframe, days):
    """Fetch OHLCV data"""
    from pybit.unified_trading import HTTP
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    session = HTTP(testnet=False, api_key=config.get('api_key', ''), api_secret=config.get('api_secret', ''))
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current_end = end_time
    
    while current_end > start_time:
        result = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            start=start_time,
            end=current_end,
            limit=1000
        )
        
        klines = result.get('result', {}).get('list', [])
        if not klines:
            break
            
        all_data.extend(klines)
        current_end = int(klines[-1][0]) - 1
        
        if len(klines) < 1000:
            break
    
    if not all_data:
        return None
        
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df

def calculate_indicators(df):
    """Calculate all indicators including filters"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # Volume filter
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']
    
    # VWAP (session-based approximation)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Session (UTC hours)
    df['hour'] = df['timestamp'].dt.hour
    
    return df

def detect_divergences(df):
    """Detect ALL divergence types"""
    signals = []
    
    # Lookback for divergence detection
    lookback = 20
    
    for i in range(lookback, len(df) - 1):
        # Current bar info
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        
        # Skip invalid RSI
        if pd.isna(rsi):
            continue
        
        # Look for swing points in lookback window
        price_window = df['close'].iloc[i-lookback:i]
        rsi_window = df['rsi'].iloc[i-lookback:i]
        low_window = df['low'].iloc[i-lookback:i]
        high_window = df['high'].iloc[i-lookback:i]
        
        # Get additional data for filters
        volume_ratio = df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1.0
        ema_20 = df['ema_20'].iloc[i] if 'ema_20' in df.columns else price
        ema_50 = df['ema_50'].iloc[i] if 'ema_50' in df.columns else price
        hour = df['hour'].iloc[i] if 'hour' in df.columns else 12
        vwap = df['vwap'].iloc[i] if 'vwap' in df.columns else price
        
        # Regular Bullish: Price makes lower low, RSI makes higher low
        prev_low_idx = low_window.idxmin()
        if prev_low_idx is not None and low < low_window[prev_low_idx]:
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({
                    'idx': i,
                    'type': 'regular_bullish',
                    'side': 'LONG',
                    'entry': price,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'above_vwap': price > vwap,
                    'trend_aligned': ema_20 > ema_50,
                    'rsi_extreme': rsi < 30,
                    'hour': hour
                })
        
        # Regular Bearish: Price makes higher high, RSI makes lower high
        prev_high_idx = high_window.idxmax()
        if prev_high_idx is not None and high > high_window[prev_high_idx]:
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({
                    'idx': i,
                    'type': 'regular_bearish',
                    'side': 'SHORT',
                    'entry': price,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'above_vwap': price > vwap,
                    'trend_aligned': ema_20 < ema_50,
                    'rsi_extreme': rsi > 70,
                    'hour': hour
                })
        
        # Hidden Bullish: Price makes higher low, RSI makes lower low (trend continuation)
        if low > low_window.min():
            prev_low_idx = low_window.idxmin()
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({
                    'idx': i,
                    'type': 'hidden_bullish',
                    'side': 'LONG',
                    'entry': price,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'above_vwap': price > vwap,
                    'trend_aligned': ema_20 > ema_50,
                    'rsi_extreme': rsi < 30,
                    'hour': hour
                })
        
        # Hidden Bearish: Price makes lower high, RSI makes higher high (trend continuation)
        if high < high_window.max():
            prev_high_idx = high_window.idxmax()
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({
                    'idx': i,
                    'type': 'hidden_bearish',
                    'side': 'SHORT',
                    'entry': price,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'above_vwap': price > vwap,
                    'trend_aligned': ema_20 < ema_50,
                    'rsi_extreme': rsi > 70,
                    'hour': hour
                })
    
    return signals

def simulate_trade(df, signal, sl_pct, be_threshold, trail_distance, max_r):
    """Simulate trade with trailing SL"""
    idx = signal['idx']
    entry = signal['entry']
    side = signal['side']
    
    sl_distance = entry * sl_pct
    
    if side == 'LONG':
        sl = entry - sl_distance
    else:
        sl = entry + sl_distance
    
    # Simulate bar by bar
    current_sl = sl
    best_r = 0
    be_moved = False
    
    for j in range(idx + 1, min(idx + 480, len(df))):  # Max 24 hours
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        
        if side == 'LONG':
            # Check SL hit
            if low <= current_sl:
                r = (current_sl - entry) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, best_r
            
            # Update best R
            current_r = (high - entry) / sl_distance
            if current_r > best_r:
                best_r = current_r
                
                # Check max R target
                if best_r >= max_r:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return max_r - fee_r, best_r
                
                # Trail SL
                if best_r >= be_threshold and not be_moved:
                    current_sl = entry + (sl_distance * 0.01)  # Tiny profit
                    be_moved = True
                
                if be_moved:
                    new_sl = entry + (best_r - trail_distance) * sl_distance
                    if new_sl > current_sl:
                        current_sl = new_sl
        
        else:  # SHORT
            # Check SL hit
            if high >= current_sl:
                r = (entry - current_sl) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, best_r
            
            # Update best R
            current_r = (entry - low) / sl_distance
            if current_r > best_r:
                best_r = current_r
                
                # Check max R target
                if best_r >= max_r:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return max_r - fee_r, best_r
                
                # Trail SL
                if best_r >= be_threshold and not be_moved:
                    current_sl = entry - (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry - (best_r - trail_distance) * sl_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    # Timeout - close at current price
    final_price = df['close'].iloc[min(idx + 479, len(df) - 1)]
    if side == 'LONG':
        r = (final_price - entry) / sl_distance
    else:
        r = (entry - final_price) / sl_distance
    
    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
    return r - fee_r, best_r

def apply_filter(signals, filter_name):
    """Apply a specific filter to signals"""
    if filter_name == 'no_filter':
        return signals
    
    elif filter_name == 'volume_1x':
        return [s for s in signals if s.get('volume_ratio', 0) >= 1.0]
    
    elif filter_name == 'volume_1.5x':
        return [s for s in signals if s.get('volume_ratio', 0) >= 1.5]
    
    elif filter_name == 'volume_2x':
        return [s for s in signals if s.get('volume_ratio', 0) >= 2.0]
    
    elif filter_name == 'rsi_extreme':
        return [s for s in signals if s.get('rsi_extreme', False)]
    
    elif filter_name == 'hidden_only':
        return [s for s in signals if 'hidden' in s.get('type', '')]
    
    elif filter_name == 'regular_only':
        return [s for s in signals if 'regular' in s.get('type', '')]
    
    elif filter_name == 'trend_aligned':
        return [s for s in signals if s.get('trend_aligned', False)]
    
    elif filter_name == 'avoid_asia':  # Avoid Asian session (0-8 UTC)
        return [s for s in signals if not (0 <= s.get('hour', 12) <= 8)]
    
    elif filter_name == 'london_ny':  # Only London/NY (8-20 UTC)
        return [s for s in signals if 8 <= s.get('hour', 12) <= 20]
    
    elif filter_name == 'volume_trend':  # Volume + trend aligned
        return [s for s in signals if s.get('volume_ratio', 0) >= 1.0 and s.get('trend_aligned', False)]
    
    elif filter_name == 'hidden_volume':  # Hidden + volume
        return [s for s in signals if 'hidden' in s.get('type', '') and s.get('volume_ratio', 0) >= 1.0]
    
    elif filter_name == 'rsi_volume':  # RSI extreme + volume
        return [s for s in signals if s.get('rsi_extreme', False) and s.get('volume_ratio', 0) >= 1.0]
    
    return signals

def main():
    print("=" * 100)
    print("🔬 3M FILTER OPTIMIZATION")
    print("=" * 100)
    print(f"Base Config: 0.5% SL | Trail Tight 5R | BE@0.3R | Trail 0.1R behind")
    print(f"Testing: 13 filter configurations")
    print()
    
    # Filters to test
    filters = [
        'no_filter',
        'volume_1x',
        'volume_1.5x', 
        'volume_2x',
        'rsi_extreme',
        'hidden_only',
        'regular_only',
        'trend_aligned',
        'avoid_asia',
        'london_ny',
        'volume_trend',
        'hidden_volume',
        'rsi_volume'
    ]
    
    # Load symbols
    print("📋 Loading symbols...")
    symbols = load_symbols(SYMBOLS_COUNT)
    print(f"  Loaded {len(symbols)} symbols")
    
    # Load data and detect all signals
    all_signals = []
    print("\n📥 Loading data and detecting signals...")
    
    for i, symbol in enumerate(symbols):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(symbols)}] {symbol}")
        
        try:
            df = fetch_data(symbol, TIMEFRAME, DAYS)
            if df is None or len(df) < 100:
                continue
            
            df = calculate_indicators(df)
            signals = detect_divergences(df)
            
            for sig in signals:
                sig['symbol'] = symbol
                sig['df'] = df  # Store df reference for simulation
            
            all_signals.extend(signals)
            
        except Exception as e:
            continue
    
    print(f"\n✅ Total signals detected: {len(all_signals)}")
    
    # Test each filter
    results = []
    
    print("\n🔄 Testing filters...")
    
    for filter_name in filters:
        filtered = apply_filter(all_signals, filter_name)
        
        if len(filtered) < 10:
            results.append({
                'Filter': filter_name,
                'N': len(filtered),
                'WR': 0,
                'Total_R': 0,
                'Avg_R': 0,
                'Status': '❌ (N<10)'
            })
            continue
        
        # Simulate trades
        wins = 0
        total_r = 0
        
        for sig in filtered:
            r, best_r = simulate_trade(
                sig['df'], sig, SL_PCT, BE_THRESHOLD, TRAIL_DISTANCE, MAX_R
            )
            total_r += r
            if r > 0:
                wins += 1
        
        wr = wins / len(filtered) * 100
        avg_r = total_r / len(filtered)
        
        results.append({
            'Filter': filter_name,
            'N': len(filtered),
            'WR': wr,
            'Total_R': total_r,
            'Avg_R': avg_r,
            'Status': '✅' if total_r > 0 else '❌'
        })
        
        print(f"  {filter_name}: N={len(filtered)}, WR={wr:.1f}%, R={total_r:+.0f}")
    
    # Sort by Total R
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Total_R', ascending=False)
    
    print("\n" + "=" * 100)
    print("📊 FILTER RESULTS (0.5% SL + Trail 5R)")
    print("=" * 100)
    print(f"{'Filter':<20} {'N':>8} {'WR':>8} {'Total_R':>10} {'Avg_R':>10} {'Status':>8}")
    
    for _, row in results_df.iterrows():
        print(f"{row['Filter']:<20} {row['N']:>8} {row['WR']:>7.1f}% {row['Total_R']:>+10.0f} {row['Avg_R']:>+10.4f} {row['Status']:>8}")
    
    # Save results
    results_df.to_csv('3m_filter_results.csv', index=False)
    print("\n✅ Saved to 3m_filter_results.csv")
    
    # Best result
    best = results_df.iloc[0]
    print("\n" + "=" * 80)
    print("🏆 BEST FILTER CONFIGURATION")
    print("=" * 80)
    print(f"Filter: {best['Filter']}")
    print(f"Trades: {best['N']}")
    print(f"Win Rate: {best['WR']:.1f}%")
    print(f"Total R: {best['Total_R']:+.0f}")
    print(f"Avg R/Trade: {best['Avg_R']:+.4f}")

if __name__ == "__main__":
    main()
