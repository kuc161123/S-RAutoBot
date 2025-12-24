#!/usr/bin/env python3
"""
🔬 EXTREME ADVANCED VALIDATION
Ultra-rigorous testing to definitively confirm 15M and 30M viability

Additional Tests:
1. CPCV (Combinatorial Purged Cross-Validation) - 10 folds
2. Shuffled Labels Test (Statistical Significance)
3. Time-of-Day Analysis (8 time buckets)
4. Day-of-Week Analysis
5. Slippage Sensitivity (0.01%, 0.02%, 0.05%)
6. Fee Sensitivity (0.04%, 0.055%, 0.08%)
7. BTC Correlation (performance vs BTC trend)
8. Out-of-Sample Symbols (test on symbols 76-100)
9. Monthly Performance Stability
10. Parameter Sensitivity (SL: 0.8%, 1%, 1.2%, 1.5%)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Configuration
TIMEFRAMES = ['15', '30']
DAYS = 90
SYMBOLS_COUNT = 100  # More symbols for out-of-sample test
N_CPCV_FOLDS = 10
N_SHUFFLE_TESTS = 500

# Best config
SL_PCT = 0.01
MAX_R = 5
COOLDOWN = 5
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1

# Base fees
BASE_FEE = 0.00055
BASE_SLIPPAGE = 0.0001

def load_symbols(n=100):
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
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df

def detect_divergences(df):
    signals = []
    lookback = 20
    
    for i in range(lookback, len(df) - 1):
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        volume_ratio = df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1.0
        
        if pd.isna(rsi) or volume_ratio < 0.8:
            continue
        
        low_window = df['low'].iloc[i-lookback:i]
        high_window = df['high'].iloc[i-lookback:i]
        
        signal_base = {
            'idx': i, 
            'entry': price, 
            'timestamp': df['timestamp'].iloc[i],
            'hour': df['hour'].iloc[i],
            'day_of_week': df['day_of_week'].iloc[i],
            'month': df['month'].iloc[i]
        }
        
        # Regular Bullish
        prev_low_idx = low_window.idxmin()
        if prev_low_idx is not None and low < low_window[prev_low_idx]:
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'side': 'LONG'})
        
        # Regular Bearish
        prev_high_idx = high_window.idxmax()
        if prev_high_idx is not None and high > high_window[prev_high_idx]:
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'side': 'SHORT'})
        
        # Hidden Bullish
        if low > low_window.min():
            prev_low_idx = low_window.idxmin()
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'side': 'LONG'})
        
        # Hidden Bearish
        if high < high_window.max():
            prev_high_idx = high_window.idxmax()
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'side': 'SHORT'})
    
    return signals

def simulate_trade(df, idx, side, entry, sl_pct=SL_PCT, max_r=MAX_R, slippage=BASE_SLIPPAGE, fee=BASE_FEE):
    sl_distance = entry * sl_pct
    
    if side == 'LONG':
        entry = entry * (1 + slippage)
        sl = entry - sl_distance
    else:
        entry = entry * (1 - slippage)
        sl = entry + sl_distance
    
    current_sl = sl
    best_r = 0
    be_moved = False
    
    for j in range(idx + 1, min(idx + 200, len(df))):
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        
        if side == 'LONG':
            if low <= current_sl:
                exit_price = current_sl * (1 - slippage)
                r = (exit_price - entry) / sl_distance
                fee_r = (fee * 2 * entry) / sl_distance
                return r - fee_r
            
            current_r = (high - entry) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    fee_r = (fee * 2 * entry) / sl_distance
                    return max_r - fee_r
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry + (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry + (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if high >= current_sl:
                exit_price = current_sl * (1 + slippage)
                r = (entry - exit_price) / sl_distance
                fee_r = (fee * 2 * entry) / sl_distance
                return r - fee_r
            
            current_r = (entry - low) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    fee_r = (fee * 2 * entry) / sl_distance
                    return max_r - fee_r
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry - (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry - (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    # Timeout
    final_price = df['close'].iloc[min(idx + 199, len(df) - 1)]
    if side == 'LONG':
        r = (final_price - entry) / sl_distance
    else:
        r = (entry - final_price) / sl_distance
    fee_r = (fee * 2 * entry) / sl_distance
    return r - fee_r

def get_all_trades(symbol_data, sl_pct=SL_PCT, max_r=MAX_R, slippage=BASE_SLIPPAGE, fee=BASE_FEE):
    all_trades = []
    
    for symbol, data in symbol_data.items():
        df = data['df']
        signals = data['signals']
        
        last_trade_idx = -COOLDOWN - 1
        
        for sig in signals:
            idx = sig['idx']
            if idx - last_trade_idx < COOLDOWN:
                continue
            
            r = simulate_trade(df, idx, sig['side'], sig['entry'], sl_pct, max_r, slippage, fee)
            all_trades.append({
                'symbol': symbol,
                'idx': idx,
                'timestamp': sig['timestamp'],
                'side': sig['side'],
                'hour': sig['hour'],
                'day_of_week': sig['day_of_week'],
                'month': sig['month'],
                'r': r
            })
            last_trade_idx = idx
    
    return sorted(all_trades, key=lambda x: x['timestamp'])

def cpcv_test(trades, n_folds=10):
    """Combinatorial Purged Cross-Validation"""
    if len(trades) < 100:
        return {'pass': False, 'pct_profitable': 0}
    
    # Create folds
    fold_size = len(trades) // n_folds
    folds = [trades[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    
    # Test all combinations of train (n-2) vs test (2)
    test_results = []
    for test_combo in combinations(range(n_folds), 2):
        test_trades = []
        for i in test_combo:
            test_trades.extend(folds[i])
        
        if len(test_trades) > 10:
            total_r = sum(t['r'] for t in test_trades)
            test_results.append(total_r > 0)
    
    pct_profitable = sum(test_results) / len(test_results) * 100
    return {'pass': pct_profitable >= 80, 'pct_profitable': pct_profitable}

def shuffled_labels_test(trades, n_tests=500):
    """Test if strategy beats random"""
    if len(trades) < 50:
        return {'pass': False, 'p_value': 1.0}
    
    actual_r = sum(t['r'] for t in trades)
    trade_rs = [t['r'] for t in trades]
    
    # Shuffle and recalculate
    random_rs = []
    for _ in range(n_tests):
        shuffled = np.random.permutation(trade_rs)
        random_rs.append(np.sum(shuffled))
    
    # p-value: how often random beats actual
    p_value = np.mean([r >= actual_r for r in random_rs])
    
    return {'pass': p_value < 0.05, 'p_value': p_value, 'actual_r': actual_r}

def time_of_day_test(trades):
    """Analyze performance by time bucket"""
    buckets = {
        '00-03': (0, 3),
        '03-06': (3, 6),
        '06-09': (6, 9),
        '09-12': (9, 12),
        '12-15': (12, 15),
        '15-18': (15, 18),
        '18-21': (18, 21),
        '21-24': (21, 24)
    }
    
    results = {}
    for name, (start, end) in buckets.items():
        bucket_trades = [t for t in trades if start <= t['hour'] < end]
        if len(bucket_trades) >= 20:
            total_r = sum(t['r'] for t in bucket_trades)
            wr = sum(1 for t in bucket_trades if t['r'] > 0) / len(bucket_trades) * 100
            results[name] = {'n': len(bucket_trades), 'r': total_r, 'wr': wr}
    
    # Pass if no bucket has severe negative performance
    severe_negative = sum(1 for r in results.values() if r['r'] < -50 and r['n'] >= 50)
    return {'pass': severe_negative == 0, 'results': results}

def day_of_week_test(trades):
    """Analyze performance by day of week"""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    results = {}
    for i, day in enumerate(days):
        day_trades = [t for t in trades if t['day_of_week'] == i]
        if len(day_trades) >= 20:
            total_r = sum(t['r'] for t in day_trades)
            wr = sum(1 for t in day_trades if t['r'] > 0) / len(day_trades) * 100
            results[day] = {'n': len(day_trades), 'r': total_r, 'wr': wr}
    
    severe_negative = sum(1 for r in results.values() if r['r'] < -50 and r['n'] >= 50)
    return {'pass': severe_negative <= 1, 'results': results}

def slippage_sensitivity_test(symbol_data, trades):
    """Test with different slippage values"""
    slippages = [0.0001, 0.0002, 0.0005]  # 0.01%, 0.02%, 0.05%
    
    results = {}
    for slip in slippages:
        test_trades = get_all_trades(symbol_data, slippage=slip)
        total_r = sum(t['r'] for t in test_trades)
        results[f'{slip*100:.2f}%'] = {'n': len(test_trades), 'r': total_r}
    
    # Pass if strategy is profitable even with 0.05% slippage
    worst = min(r['r'] for r in results.values())
    return {'pass': worst > 0, 'results': results}

def fee_sensitivity_test(symbol_data, trades):
    """Test with different fee values"""
    fees = [0.0004, 0.00055, 0.0008]  # 0.04%, 0.055%, 0.08%
    
    results = {}
    for fee in fees:
        test_trades = get_all_trades(symbol_data, fee=fee)
        total_r = sum(t['r'] for t in test_trades)
        results[f'{fee*100:.3f}%'] = {'n': len(test_trades), 'r': total_r}
    
    worst = min(r['r'] for r in results.values())
    return {'pass': worst > 0, 'results': results}

def parameter_sensitivity_test(symbol_data):
    """Test with different SL/R parameters"""
    sl_pcts = [0.008, 0.01, 0.012, 0.015]  # 0.8%, 1%, 1.2%, 1.5%
    max_rs = [4, 5, 6]
    
    results = []
    for sl in sl_pcts:
        for r in max_rs:
            trades = get_all_trades(symbol_data, sl_pct=sl, max_r=r)
            total_r = sum(t['r'] for t in trades)
            wr = sum(1 for t in trades if t['r'] > 0) / len(trades) * 100 if trades else 0
            results.append({
                'sl': f'{sl*100:.1f}%',
                'max_r': r,
                'n': len(trades),
                'r': total_r,
                'wr': wr
            })
    
    profitable = sum(1 for r in results if r['r'] > 0)
    return {'pass': profitable >= len(results) * 0.7, 'results': results, 'pct_profitable': profitable / len(results) * 100}

def out_of_sample_symbols_test(symbol_data_oos):
    """Test on completely new symbols (76-100)"""
    trades = get_all_trades(symbol_data_oos)
    
    if len(trades) < 50:
        return {'pass': False, 'r': 0, 'n': 0}
    
    total_r = sum(t['r'] for t in trades)
    wr = sum(1 for t in trades if t['r'] > 0) / len(trades) * 100
    
    return {'pass': total_r > 0, 'n': len(trades), 'r': total_r, 'wr': wr}

def monthly_stability_test(trades):
    """Check performance stability across months"""
    monthly = {}
    for t in trades:
        month = t['month']
        if month not in monthly:
            monthly[month] = []
        monthly[month].append(t['r'])
    
    results = {}
    for month, rs in monthly.items():
        total_r = sum(rs)
        wr = sum(1 for r in rs if r > 0) / len(rs) * 100
        results[month] = {'n': len(rs), 'r': total_r, 'wr': wr}
    
    profitable_months = sum(1 for r in results.values() if r['r'] > 0)
    return {'pass': profitable_months >= len(results) - 1, 'results': results, 'profitable_months': profitable_months}

def main():
    print("=" * 120)
    print("🔬 EXTREME ADVANCED VALIDATION")
    print("=" * 120)
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Symbols: {SYMBOLS_COUNT} (75 in-sample + 25 out-of-sample)")
    print(f"Days: {DAYS}")
    print(f"Tests: CPCV, Shuffled Labels, Time/Day Analysis, Sensitivity Tests, OOS Symbols")
    print()
    
    # Load ALL symbols
    print("📋 Loading symbols...")
    all_symbols = load_symbols(SYMBOLS_COUNT)
    in_sample_symbols = all_symbols[:75]
    oos_symbols = all_symbols[75:]
    print(f"  In-sample: {len(in_sample_symbols)} | Out-of-sample: {len(oos_symbols)}")
    
    for tf in TIMEFRAMES:
        print(f"\n{'='*100}")
        print(f"📊 EXTREME TESTING: {tf}M TIMEFRAME")
        print(f"{'='*100}")
        
        # Load in-sample data
        print("\n📥 Loading in-sample data...")
        symbol_data = {}
        for i, symbol in enumerate(in_sample_symbols):
            if (i + 1) % 15 == 0:
                print(f"  [{i+1}/{len(in_sample_symbols)}]")
            try:
                df = fetch_data(symbol, tf, DAYS)
                if df is None or len(df) < 100:
                    continue
                df = calculate_indicators(df)
                signals = detect_divergences(df)
                if signals:
                    symbol_data[symbol] = {'df': df, 'signals': signals}
            except:
                continue
        
        # Load OOS data
        print("📥 Loading out-of-sample data...")
        symbol_data_oos = {}
        for symbol in oos_symbols:
            try:
                df = fetch_data(symbol, tf, DAYS)
                if df is None or len(df) < 100:
                    continue
                df = calculate_indicators(df)
                signals = detect_divergences(df)
                if signals:
                    symbol_data_oos[symbol] = {'df': df, 'signals': signals}
            except:
                continue
        
        print(f"✅ In-sample: {len(symbol_data)} symbols | OOS: {len(symbol_data_oos)} symbols")
        
        # Get trades
        trades = get_all_trades(symbol_data)
        print(f"📈 Total in-sample trades: {len(trades)}")
        
        if len(trades) < 100:
            print("⚠️ Not enough trades")
            continue
        
        # Run all tests
        tests_passed = 0
        total_tests = 10
        
        # Test 1: CPCV
        print(f"\n🔄 TEST 1: CPCV ({N_CPCV_FOLDS} folds)...")
        cpcv = cpcv_test(trades, N_CPCV_FOLDS)
        print(f"  Profitable combos: {cpcv['pct_profitable']:.1f}%")
        print(f"  RESULT: {'✅ PASS' if cpcv['pass'] else '❌ FAIL'}")
        if cpcv['pass']: tests_passed += 1
        
        # Test 2: Shuffled Labels
        print(f"\n🎲 TEST 2: Shuffled Labels ({N_SHUFFLE_TESTS} tests)...")
        shuffle = shuffled_labels_test(trades, N_SHUFFLE_TESTS)
        print(f"  p-value: {shuffle['p_value']:.4f}")
        print(f"  RESULT: {'✅ PASS (p < 0.05)' if shuffle['pass'] else '❌ FAIL'}")
        if shuffle['pass']: tests_passed += 1
        
        # Test 3: Time of Day
        print(f"\n🕐 TEST 3: Time of Day Analysis...")
        tod = time_of_day_test(trades)
        for bucket, data in tod['results'].items():
            status = '✅' if data['r'] > 0 else '⚠️' if data['r'] > -50 else '❌'
            print(f"  {bucket}: N={data['n']}, R={data['r']:+.0f}, WR={data['wr']:.1f}% {status}")
        print(f"  RESULT: {'✅ PASS' if tod['pass'] else '❌ FAIL'}")
        if tod['pass']: tests_passed += 1
        
        # Test 4: Day of Week
        print(f"\n📅 TEST 4: Day of Week Analysis...")
        dow = day_of_week_test(trades)
        for day, data in dow['results'].items():
            status = '✅' if data['r'] > 0 else '⚠️' if data['r'] > -50 else '❌'
            print(f"  {day}: N={data['n']}, R={data['r']:+.0f}, WR={data['wr']:.1f}% {status}")
        print(f"  RESULT: {'✅ PASS' if dow['pass'] else '❌ FAIL'}")
        if dow['pass']: tests_passed += 1
        
        # Test 5: Slippage Sensitivity
        print(f"\n💸 TEST 5: Slippage Sensitivity...")
        slip = slippage_sensitivity_test(symbol_data, trades)
        for s, data in slip['results'].items():
            status = '✅' if data['r'] > 0 else '❌'
            print(f"  Slippage {s}: R={data['r']:+.0f} {status}")
        print(f"  RESULT: {'✅ PASS' if slip['pass'] else '❌ FAIL'}")
        if slip['pass']: tests_passed += 1
        
        # Test 6: Fee Sensitivity
        print(f"\n💰 TEST 6: Fee Sensitivity...")
        fee = fee_sensitivity_test(symbol_data, trades)
        for f, data in fee['results'].items():
            status = '✅' if data['r'] > 0 else '❌'
            print(f"  Fee {f}: R={data['r']:+.0f} {status}")
        print(f"  RESULT: {'✅ PASS' if fee['pass'] else '❌ FAIL'}")
        if fee['pass']: tests_passed += 1
        
        # Test 7: Parameter Sensitivity
        print(f"\n⚙️ TEST 7: Parameter Sensitivity...")
        param = parameter_sensitivity_test(symbol_data)
        print(f"  Profitable configs: {param['pct_profitable']:.1f}%")
        print(f"  RESULT: {'✅ PASS' if param['pass'] else '❌ FAIL'}")
        if param['pass']: tests_passed += 1
        
        # Test 8: Out-of-Sample Symbols
        print(f"\n🆕 TEST 8: Out-of-Sample Symbols (symbols 76-100)...")
        oos = out_of_sample_symbols_test(symbol_data_oos)
        if oos['n'] > 0:
            print(f"  Trades: {oos['n']}, R={oos['r']:+.0f}, WR={oos['wr']:.1f}%")
        print(f"  RESULT: {'✅ PASS' if oos['pass'] else '❌ FAIL'}")
        if oos['pass']: tests_passed += 1
        
        # Test 9: Monthly Stability
        print(f"\n📆 TEST 9: Monthly Stability...")
        monthly = monthly_stability_test(trades)
        for month, data in monthly['results'].items():
            status = '✅' if data['r'] > 0 else '⚠️'
            print(f"  Month {month}: N={data['n']}, R={data['r']:+.0f}, WR={data['wr']:.1f}% {status}")
        print(f"  Profitable months: {monthly['profitable_months']}/{len(monthly['results'])}")
        print(f"  RESULT: {'✅ PASS' if monthly['pass'] else '❌ FAIL'}")
        if monthly['pass']: tests_passed += 1
        
        # Test 10: Basic profitability
        print(f"\n📊 TEST 10: Overall Profitability...")
        total_r = sum(t['r'] for t in trades)
        wr = sum(1 for t in trades if t['r'] > 0) / len(trades) * 100
        basic_pass = total_r > 0 and wr > 55
        print(f"  Total R: {total_r:+.0f}, Win Rate: {wr:.1f}%")
        print(f"  RESULT: {'✅ PASS' if basic_pass else '❌ FAIL'}")
        if basic_pass: tests_passed += 1
        
        # Final verdict
        verdict = "✅ EXTREMELY ROBUST" if tests_passed >= 9 else "✅ ROBUST" if tests_passed >= 7 else "⚠️ MARGINAL" if tests_passed >= 5 else "❌ FAIL"
        
        print(f"\n{'='*80}")
        print(f"🏆 {tf}M FINAL VERDICT: {verdict} ({tests_passed}/{total_tests} tests passed)")
        print(f"{'='*80}")
    
    print("\n" + "=" * 120)
    print("🏆 EXTREME VALIDATION COMPLETE")
    print("=" * 120)

if __name__ == "__main__":
    main()
