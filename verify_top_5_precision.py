import yaml
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
import concurrent.futures

# CONFIG
RISK_PCT = 0.001
INITIAL_CAPITAL = 700.0
BASE_URL = "https://api.bybit.com"

TEST_SYMBOLS = ['XAUTUSDT', '1000PEPEUSDT', 'WIFUSDT', 'AGLDUSDT', 'BTCUSDT']
# Use Start Date covering the "Boom" + "Bust"
START_DATE = "2025-09-01 00:00:00"
END_DATE = "2025-12-31 23:59:59"

def fetch_klines(symbol, interval):
    # Fetch ample data from July to Jan
    end_req = int(datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp() * 1000)
    start_req = int(datetime(2025, 8, 15, tzinfo=timezone.utc).timestamp() * 1000)
    
    current_end = end_req
    all_candles = []
    
    while current_end > start_req:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=5)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
            time.sleep(0.05)
        except: break
        
    if not all_candles: return None
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = df[c].astype(float)
    df = df.sort_values('start').reset_index(drop=True)
    return df

def prepare_1h(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['ema'] = df['close'].ewm(span=200, adjust=False).mean()
    h, l, c_prev = df['high'], df['low'], df['close'].shift()
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df

def find_pivots(data, left=3, right=3):
    pivot_highs = np.full(len(data), np.nan)
    pivot_lows = np.full(len(data), np.nan)
    for i in range(left, len(data) - right):
        window = data[i-left : i+right+1]
        center = data[i]
        if len(window) != (left + right + 1): continue
        if center == max(window) and list(window).count(center) == 1: pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1: pivot_lows[i] = center
    return pivot_highs, pivot_lows

def detect_signals(df):
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    rsi = df['rsi'].values; ema = df['ema'].values
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    used_pivots = set() 
    
    for i in range(205, len(df) - 3):
        curr_price = close[i]; curr_ema = ema[i]
        
        # BULLISH
        if curr_price > curr_ema:
            p_lows = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2: break
            if len(p_lows) == 2:
                curr_idx, curr_val = p_lows[0]; prev_idx, prev_val = p_lows[1]
                dedup_key = (curr_idx, prev_idx, 'BULL')
                if (i - curr_idx) <= 10:
                    if dedup_key not in used_pivots:
                         added = False
                         if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'long', 'type': 'REG_BULL', 'swing': max(high[curr_idx:i+1])})
                            added = True
                         if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'long', 'type': 'HID_BULL', 'swing': max(high[curr_idx:i+1])})
                            added = True
                         if added: used_pivots.add(dedup_key)

        # BEARISH
        if curr_price < curr_ema:
             p_highs = []
             for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_ph[j]):
                    p_highs.append((j, price_ph[j]))
                    if len(p_highs) >= 2: break
             if len(p_highs) == 2:
                curr_idx, curr_val = p_highs[0]; prev_idx, prev_val = p_highs[1]
                dedup_key = (curr_idx, prev_idx, 'BEAR')
                if (i - curr_idx) <= 10:
                    if dedup_key not in used_pivots:
                        added = False
                        if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'short', 'type': 'REG_BEAR', 'swing': min(low[curr_idx:i+1])})
                            added = True
                        if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'short', 'type': 'HID_BEAR', 'swing': min(low[curr_idx:i+1])})
                            added = True
                        if added: used_pivots.add(dedup_key)
    return signals

def process_symbol(sym):
    print(f"Checking {sym}...")
    
    with open('config_combined.yaml', 'r') as f:
        conf = yaml.safe_load(f)
    cfg = conf['symbols'].get(sym)
    if not cfg: return []

    # Fetch Data
    df_1h = fetch_klines(sym, '60')
    if df_1h is None: return []
    df_1h = prepare_1h(df_1h)
    
    df_5m = fetch_klines(sym, '5')
    if df_5m is None: return []
    
    signals = detect_signals(df_1h)
    
    trades = []
    last_exit_time = 0
    
    StartTS = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()*1000)
    EndTS = int(datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()*1000)
    
    for sig in signals:
        if sig['type'] != cfg['divergence_type']: continue
        
        candle = df_1h.iloc[sig['conf_idx']]
        start_ts = int(candle['start'].timestamp()*1000)
        
        if start_ts < StartTS or start_ts > EndTS: continue
        if start_ts < last_exit_time: continue
        
        # Execution Check
        conf_idx = sig['conf_idx']
        side = sig['side']
        bos = sig['swing']
        
        entry_data = None
        for i in range(1, 7):
            if conf_idx+i+1 >= len(df_1h): break
            c = df_1h.iloc[conf_idx+i]
            triggered = (c['close'] > bos) if side == 'long' else (c['close'] < bos)
            if triggered:
                entry_c = df_1h.iloc[conf_idx+i+1] # Breakout entry candle
                entry_data = {
                    'price': entry_c['open'], 
                    'ts': int(entry_c['start'].timestamp()*1000), 
                    'atr': c['atr']
                }
                break
        
        if not entry_data: continue
        if entry_data['ts'] < last_exit_time: continue
        
        # 5M PRECISION CHECK
        rr = cfg['rr']
        sl_dist = entry_data['atr'] * cfg['atr_mult']
        entry_price = entry_data['price']
        
        tp = entry_price + (sl_dist * rr) if side == 'long' else entry_price - (sl_dist * rr)
        sl = entry_price - sl_dist if side == 'long' else entry_price + sl_dist
        
        outcome = -1.0
        exit_ts = entry_data['ts'] + (24*3600000) # Fallback timeout
        
        # Scan 5M Data
        sub = df_5m[df_5m['start'] >= pd.to_datetime(entry_data['ts'], unit='ms', utc=True)]
        
        hit = False
        for row in sub.itertuples():
            curr_ts = int(row.start.timestamp()*1000)
            if side == 'long':
                if row.low <= sl: outcome=-1.0; exit_ts=curr_ts; hit=True; break
                if row.high >= tp: outcome=rr; exit_ts=curr_ts; hit=True; break
            else:
                if row.high >= sl: outcome=-1.0; exit_ts=curr_ts; hit=True; break
                if row.low <= tp: outcome=rr; exit_ts=curr_ts; hit=True; break
        
        if not hit: outcome = 0 # Timeout/Running
        
        last_exit_time = exit_ts
        duration_hours = (exit_ts - entry_data['ts']) / (1000 * 3600)
        trades.append({'sym': sym, 'time': entry_data['ts'], 'r': outcome, 'duration': duration_hours})
        
    return trades

def main():
    print(f"🔥 ANALYZING TRADE DURATION")
    print(f"Symbols: {TEST_SYMBOLS}")
    print(f"Period: {START_DATE} to {END_DATE}")
    
    all_trades = []
    
    # Process sequentially to avoid rate limits on 5M data
    for sym in TEST_SYMBOLS:
        res = process_symbol(sym)
        all_trades.extend(res)
        
    winning_trades = [t for t in all_trades if t['r'] > 0]
    
    if not winning_trades:
        print("No winning trades found in this sample.")
        return

    avg_duration = sum(t['duration'] for t in winning_trades) / len(winning_trades)
    min_duration = min(t['duration'] for t in winning_trades)
    max_duration = max(t['duration'] for t in winning_trades)
    
    print("\n" + "="*50)
    print(f"WINNING TRADE DURATION STATS ({len(winning_trades)} Wins)")
    print("="*50)
    print(f"AVERAGE HOLD TIME: {avg_duration:.1f} Hours")
    print(f"MINIMUM HOLD TIME: {min_duration:.1f} Hours")
    print(f"MAXIMUM HOLD TIME: {max_duration:.1f} Hours")
    print("="*50)
    
    # Monthly Breakdown
    print("\n" + "="*40)
    print(f"{'MONTH':<10} | {'NET R':<10}")
    print("="*40)
    
    months = [
        ('SEP 2025', '2025-09'),
        ('OCT 2025', '2025-10'),
        ('NOV 2025', '2025-11'),
        ('DEC 2025', '2025-12')
    ]
    
    for m_name, m_key in months:
        m_trades = [x for x in all_trades if m_key in str(pd.to_datetime(x['time'], unit='ms'))]
        m_r = sum(x['r'] for x in m_trades)
        print(f"{m_name:<10} | {m_r:<10.1f}")
    print("="*40)

if __name__ == "__main__":
    main()
