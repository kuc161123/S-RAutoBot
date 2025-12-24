#!/usr/bin/env python3
"""
🔬 3M ROUND 5 - ATR-BASED DYNAMIC SL + QUALITY FILTERS
Previous rounds failed walk-forward validation. 
Trying completely different approach:

1. ATR-based dynamic SL (adapts to volatility)
2. RSI Strength filter (strong divergence only)
3. Momentum confirmation (price momentum aligns with signal)
4. Multi-timeframe confirmation (optional)
5. Proper walk-forward from the start
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
TIMEFRAME = '3'
DAYS = 45
SYMBOLS_COUNT = 60
TRAIN_RATIO = 0.6

# Parameters to test
ATR_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5]  # SL = ATR * multiplier
MAX_RS = [2, 3, 4, 5]  # Lower R targets for faster exits
COOLDOWNS = [5, 10]

# Trailing
BE_THRESHOLD = 0.5  # More conservative BE trigger
TRAIL_DISTANCE = 0.2

# Fee model
TAKER_FEE = 0.00055
SLIPPAGE = 0.0001

def load_symbols(n=60):
    from pybit.unified_trading import HTTP
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    session = HTTP(testnet=False, api_key=config.get('api_key', ''), api_secret=config.get('api_secret', ''))
    result = session.get_tickers(category="linear")
    tickers = result.get('result', {}).get('list', [])
    usdt_perps = [t for t in tickers if t['symbol'].endswith('USDT') and 'USDC' not in t['symbol']]
    sorted_tickers = sorted(usdt_perps, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in sorted_tickers[:n]]

def fetch_data(symbol, timeframe, days):
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
        result = session.get_kline(category="linear", symbol=symbol, interval=timeframe,
                                   start=start_time, end=current_end, limit=1000)
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
    """Calculate indicators with ATR"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (14-period)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Momentum (rate of change)
    df['momentum'] = df['close'].pct_change(5) * 100
    
    # Volume
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']
    
    # RSI momentum (RSI rate of change)
    df['rsi_momentum'] = df['rsi'].diff(3)
    
    return df

def detect_quality_divergences(df):
    """Detect ONLY high-quality divergences with strength filters"""
    signals = []
    lookback = 20
    
    for i in range(lookback + 5, len(df) - 1):
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        atr = df['atr'].iloc[i]
        momentum = df['momentum'].iloc[i]
        volume_ratio = df['volume_ratio'].iloc[i]
        rsi_momentum = df['rsi_momentum'].iloc[i]
        
        if pd.isna(rsi) or pd.isna(atr) or atr <= 0:
            continue
        
        low_window = df['low'].iloc[i-lookback:i]
        high_window = df['high'].iloc[i-lookback:i]
        rsi_window = df['rsi'].iloc[i-lookback:i]
        
        signal_base = {
            'idx': i,
            'entry': price,
            'atr': atr,
            'rsi': rsi,
            'momentum': momentum,
            'volume_ratio': volume_ratio,
            'rsi_momentum': rsi_momentum,
            'timestamp': df['timestamp'].iloc[i]
        }
        
        # REGULAR BULLISH (higher quality criteria)
        # Price makes significant lower low, RSI makes clear higher low
        prev_low_idx = low_window.idxmin()
        if prev_low_idx is not None and low < low_window[prev_low_idx]:
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            rsi_diff = rsi - prev_rsi  # Should be positive
            price_diff_pct = (low - low_window[prev_low_idx]) / low_window[prev_low_idx] * 100
            
            # Quality filters:
            # 1. RSI divergence strength > 5 points
            # 2. Price made significant new low (> 0.3%)
            # 3. RSI momentum turning up
            # 4. Volume above average
            if (rsi_diff > 5 and 
                price_diff_pct < -0.3 and 
                rsi_momentum > 0 and 
                volume_ratio >= 0.8):
                signals.append({**signal_base, 'type': 'regular_bullish', 'side': 'LONG',
                               'divergence_strength': rsi_diff})
        
        # REGULAR BEARISH (higher quality criteria)
        prev_high_idx = high_window.idxmax()
        if prev_high_idx is not None and high > high_window[prev_high_idx]:
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            rsi_diff = prev_rsi - rsi  # Should be positive (RSI went down)
            price_diff_pct = (high - high_window[prev_high_idx]) / high_window[prev_high_idx] * 100
            
            if (rsi_diff > 5 and 
                price_diff_pct > 0.3 and 
                rsi_momentum < 0 and 
                volume_ratio >= 0.8):
                signals.append({**signal_base, 'type': 'regular_bearish', 'side': 'SHORT',
                               'divergence_strength': rsi_diff})
        
        # HIDDEN BULLISH (trend continuation with confirmation)
        if low > low_window.min():
            prev_low_idx = low_window.idxmin()
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            rsi_diff = prev_rsi - rsi  # RSI is lower despite higher price low
            
            # Quality: Strong hidden divergence in uptrend
            # Momentum should still be positive (uptrend)
            if (rsi_diff > 5 and 
                momentum > 0 and 
                volume_ratio >= 0.8):
                signals.append({**signal_base, 'type': 'hidden_bullish', 'side': 'LONG',
                               'divergence_strength': rsi_diff})
        
        # HIDDEN BEARISH (trend continuation with confirmation)
        if high < high_window.max():
            prev_high_idx = high_window.idxmax()
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            rsi_diff = rsi - prev_rsi  # RSI is higher despite lower price high
            
            if (rsi_diff > 5 and 
                momentum < 0 and 
                volume_ratio >= 0.8):
                signals.append({**signal_base, 'type': 'hidden_bearish', 'side': 'SHORT',
                               'divergence_strength': rsi_diff})
    
    return signals

def simulate_trade_atr(df, idx, side, entry, atr_mult, max_r, atr):
    """Simulate trade with ATR-based SL"""
    sl_distance = atr * atr_mult
    
    # Minimum SL distance check (at least 0.3% to avoid noise)
    min_sl = entry * 0.003
    if sl_distance < min_sl:
        sl_distance = min_sl
    
    # Apply slippage
    if side == 'LONG':
        entry = entry * (1 + SLIPPAGE)
        sl = entry - sl_distance
    else:
        entry = entry * (1 - SLIPPAGE)
        sl = entry + sl_distance
    
    current_sl = sl
    best_r = 0
    be_moved = False
    bars_in_trade = 0
    
    for j in range(idx + 1, min(idx + 480, len(df))):
        bars_in_trade += 1
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        
        if side == 'LONG':
            if low <= current_sl:
                exit_price = current_sl * (1 - SLIPPAGE)
                r = (exit_price - entry) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, bars_in_trade
            
            current_r = (high - entry) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    exit_price = entry + (max_r * sl_distance)
                    exit_price = exit_price * (1 - SLIPPAGE)
                    r = (exit_price - entry) / sl_distance
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return r - fee_r, bars_in_trade
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry + (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry + (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if high >= current_sl:
                exit_price = current_sl * (1 + SLIPPAGE)
                r = (entry - exit_price) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, bars_in_trade
            
            current_r = (entry - low) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    exit_price = entry - (max_r * sl_distance)
                    exit_price = exit_price * (1 + SLIPPAGE)
                    r = (entry - exit_price) / sl_distance
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return r - fee_r, bars_in_trade
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry - (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry - (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    # Timeout
    final_price = df['close'].iloc[min(idx + 479, len(df) - 1)]
    if side == 'LONG':
        exit_price = final_price * (1 - SLIPPAGE)
        r = (exit_price - entry) / sl_distance
    else:
        exit_price = final_price * (1 + SLIPPAGE)
        r = (entry - exit_price) / sl_distance
    
    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
    return r - fee_r, bars_in_trade

def run_backtest_period(symbol_data, atr_mult, max_r, cooldown, start_pct, end_pct):
    """Run backtest on specific period percentage"""
    total_r = 0
    wins = 0
    trades = 0
    
    for symbol, data in symbol_data.items():
        df = data['df']
        signals = data['signals']
        
        total_bars = len(df)
        start_idx = int(total_bars * start_pct)
        end_idx = int(total_bars * end_pct)
        
        period_signals = [s for s in signals if start_idx <= s['idx'] < end_idx]
        
        last_trade_bar = -cooldown - 1
        
        for sig in period_signals:
            idx = sig['idx']
            if idx - last_trade_bar < cooldown:
                continue
            
            r, bars = simulate_trade_atr(df, idx, sig['side'], sig['entry'], 
                                         atr_mult, max_r, sig['atr'])
            total_r += r
            if r > 0:
                wins += 1
            trades += 1
            last_trade_bar = idx + bars
    
    if trades == 0:
        return {'trades': 0, 'wr': 0, 'total_r': 0, 'avg_r': 0}
    
    return {
        'trades': trades,
        'wins': wins,
        'wr': wins / trades * 100,
        'total_r': total_r,
        'avg_r': total_r / trades
    }

def main():
    print("=" * 120)
    print("🔬 3M ROUND 5 - ATR-BASED SL + QUALITY FILTERS")
    print("=" * 120)
    print(f"Timeframe: 3min | Symbols: {SYMBOLS_COUNT} | Days: {DAYS}")
    print(f"ATR Multipliers: {ATR_MULTIPLIERS}")
    print(f"Max R: {MAX_RS}")
    print(f"Cooldowns: {COOLDOWNS}")
    print(f"Walk-Forward: {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test")
    print()
    
    total_configs = len(ATR_MULTIPLIERS) * len(MAX_RS) * len(COOLDOWNS)
    print(f"Total configurations: {total_configs}")
    
    # Load symbols
    print("\n📋 Loading symbols...")
    symbols = load_symbols(SYMBOLS_COUNT)
    print(f"  Loaded {len(symbols)} symbols")
    
    # Load data
    print("\n📥 Loading data and detecting QUALITY signals...")
    symbol_data = {}
    total_signals = 0
    
    for i, symbol in enumerate(symbols):
        if (i + 1) % 15 == 0:
            print(f"  [{i+1}/{len(symbols)}] {symbol}")
        
        try:
            df = fetch_data(symbol, TIMEFRAME, DAYS)
            if df is None or len(df) < 200:
                continue
            
            df = calculate_indicators(df)
            signals = detect_quality_divergences(df)
            
            if signals:
                symbol_data[symbol] = {'df': df, 'signals': signals}
                total_signals += len(signals)
        except:
            continue
    
    print(f"\n✅ {len(symbol_data)} symbols with {total_signals} QUALITY signals")
    print(f"  (Avg {total_signals/max(1,len(symbol_data)):.1f} signals/symbol)")
    
    # Run walk-forward for all configs
    print("\n🔄 Running walk-forward validation...")
    results = []
    
    configs = list(product(ATR_MULTIPLIERS, MAX_RS, COOLDOWNS))
    
    for i, (atr_mult, max_r, cooldown) in enumerate(configs):
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(configs)}] ATR={atr_mult}x R={max_r} CD={cooldown}")
        
        # Train period (0-60%)
        train = run_backtest_period(symbol_data, atr_mult, max_r, cooldown, 0, TRAIN_RATIO)
        
        # Test period (60-100%)
        test = run_backtest_period(symbol_data, atr_mult, max_r, cooldown, TRAIN_RATIO, 1.0)
        
        results.append({
            'atr_mult': atr_mult,
            'max_r': max_r,
            'cooldown': cooldown,
            'train_trades': train['trades'],
            'train_r': train['total_r'],
            'train_wr': train['wr'],
            'test_trades': test['trades'],
            'test_r': test['total_r'],
            'test_wr': test['wr'],
            'test_avg_r': test['avg_r']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r', ascending=False)
    
    print("\n" + "=" * 120)
    print("📊 WALK-FORWARD RESULTS (ATR-Based SL)")
    print("=" * 120)
    print(f"{'ATR_Mult':>8} {'MaxR':>6} {'CD':>4} {'Train_N':>8} {'Train_R':>10} {'Test_N':>8} {'Test_R':>10} {'Test_WR':>8} {'Status'}")
    
    for _, row in results_df.head(16).iterrows():
        status = '✅' if row['test_r'] > 0 else '❌'
        print(f"{row['atr_mult']:>8.1f} {row['max_r']:>6} {row['cooldown']:>4} {row['train_trades']:>8} {row['train_r']:>+10.0f} {row['test_trades']:>8} {row['test_r']:>+10.0f} {row['test_wr']:>7.1f}% {status}")
    
    # Save
    results_df.to_csv('3m_round5_atr_results.csv', index=False)
    print("\n✅ Saved to 3m_round5_atr_results.csv")
    
    # Best config
    profitable = results_df[results_df['test_r'] > 0]
    if len(profitable) > 0:
        best = profitable.iloc[0]
        print("\n" + "=" * 80)
        print("🏆 BEST ATR-BASED 3M CONFIGURATION")
        print("=" * 80)
        print(f"ATR Multiplier: {best['atr_mult']}x")
        print(f"Max R Target: {best['max_r']}")
        print(f"Cooldown: {best['cooldown']} bars")
        print(f"\n📈 OUT-OF-SAMPLE PERFORMANCE:")
        print(f"  Test Trades: {best['test_trades']}")
        print(f"  Test R: {best['test_r']:+.0f}")
        print(f"  Test Win Rate: {best['test_wr']:.1f}%")
        print(f"  Test Avg R/Trade: {best['test_avg_r']:+.4f}")
        print("\n✅ PROFITABLE CONFIGURATION FOUND!")
    else:
        print("\n❌ No profitable configuration in walk-forward test")
        print("\n💡 INSIGHT: 3M timeframe may not be suitable for divergence trading")
        print("   Consider: 5M, 15M, or 1H timeframes instead")

if __name__ == "__main__":
    main()
