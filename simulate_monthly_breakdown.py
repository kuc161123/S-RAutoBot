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

# MONTH DEFINITIONS
MONTHS = [
    ("SEP", "2025-09-01 00:00:00", "2025-09-30 23:59:59"),
    ("OCT", "2025-10-01 00:00:00", "2025-10-31 23:59:59"),
    ("NOV", "2025-11-01 00:00:00", "2025-11-30 23:59:59"),
    ("DEC", "2025-12-01 00:00:00", "2025-12-31 23:59:59"),
]

def get_ts(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)

MONTH_RANGES = []
for name, s, e in MONTHS:
    MONTH_RANGES.append({
        'name': name,
        'start': get_ts(s),
        'end': get_ts(e),
        'trades': []
    })

def fetch_klines(symbol, interval):
    # Fetch ample data from July to Jan
    end_req = int(datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp() * 1000)
    start_req = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
    
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
    # Returns raw signals
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
                         if added: used_pivots.add(dedup_key) # Dedup Per Symbol

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

def process_symbol(args):
    sym, cfg = args
    df = fetch_klines(sym, '60')
    if df is None or len(df) < 300: return []
    df = prepare_1h(df)
    signals = detect_signals(df)
    
    trades = []
    last_exit_time = 0 
    
    # Process sequentially for THIS symbol
    for sig in signals:
        # Check Signal Type Match
        if sig['type'] != cfg['divergence_type']: continue

        candle = df.iloc[sig['conf_idx']]
        start_ts = int(candle['start'].timestamp()*1000)
        
        if start_ts < last_exit_time: continue # Sequential enforcement
        
        # Execution Check
        conf_idx = sig['conf_idx']
        side = sig['side']
        bos = sig['swing']
        
        entry_data = None
        for i in range(1, 7):
            if conf_idx+i+1 >= len(df): break
            c = df.iloc[conf_idx+i]
            triggered = (c['close'] > bos) if side == 'long' else (c['close'] < bos)
            if triggered:
                entry_c = df.iloc[conf_idx+i+1]
                entry_data = {
                    'price': entry_c['open'], 
                    'ts': int(entry_c['start'].timestamp()*1000), 
                    'atr': c['atr']
                }
                break
        
        if not entry_data: continue
        
        if entry_data['ts'] < last_exit_time: continue # Double check
        
        # Outcome
        rr = cfg['rr']
        sl_dist = entry_data['atr'] * cfg['atr_mult']
        entry_price = entry_data['price']
        
        outcome = -1.0
        exit_ts = entry_data['ts'] + (24*3600000)
        
        if side == 'long':
            tp = entry_price + (sl_dist * rr)
            sl = entry_price - sl_dist
            for k in range(conf_idx+i+2, min(len(df), conf_idx+i+120)):
                 row = df.iloc[k]
                 k_ts = int(row['start'].timestamp()*1000)
                 if row['low'] <= sl: outcome=-1.0; exit_ts=k_ts; break
                 if row['high'] >= tp: outcome=rr; exit_ts=k_ts; break
                 exit_ts = k_ts
        else:
            tp = entry_price - (sl_dist * rr)
            sl = entry_price + sl_dist
            for k in range(conf_idx+i+2, min(len(df), conf_idx+i+120)):
                 row = df.iloc[k]
                 k_ts = int(row['start'].timestamp()*1000)
                 if row['high'] >= sl: outcome=-1.0; exit_ts=k_ts; break
                 if row['low'] <= tp: outcome=rr; exit_ts=k_ts; break
                 exit_ts = k_ts
                 
        if sym == 'XAUTUSDT':
             print(f"DEBUG XAUT: {entry_data['ts']} Result: {outcome}")
             
        trades.append({'time': entry_data['ts'], 'r': outcome})
        last_exit_time = exit_ts
        
    return trades

def main():
    print(f"🔥 FULL MONTHLY BREAKDOWN SIMULATION (ALL SYMBOLS)")
    start_time = time.time()
    
    with open('config_combined.yaml', 'r') as f:
        conf = yaml.safe_load(f)
        
    # ALL SYMBOLS
    tasks = list(conf['symbols'].items()) 
    print(f"Processing {len(tasks)} symbols...")
    
    all_trades = []
    # Use higher thread count? Bybit limit is sensitive. 
    # But fetching 1000 candles takes time. 20 workers is usually safe.
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exc:
        results = exc.map(process_symbol, tasks)
        for r in results:
            all_trades.extend(r)
            
    print(f"Total Trades Generated: {len(all_trades)}")
    
    # Bucket into Months
    for t in all_trades:
        ts = t['time']
        for m in MONTH_RANGES:
            if m['start'] <= ts <= m['end']:
                m['trades'].append(t)
                break
                
    # Report per Month
    print("\n" + "="*60)
    print(f"{'MONTH':<10} | {'TRADES':<8} | {'WR%':<6} | {'NET R':<10} | {'P&L ($)':<12}")
    print("="*60)
    
    total_r_cumulative = 0
    balance = INITIAL_CAPITAL
    
    for m in MONTH_RANGES:
        trades = m['trades']
        wins = sum(1 for x in trades if x['r'] > 0)
        losses = sum(1 for x in trades if x['r'] <= 0)
        total = wins + losses
        wr = (wins/total*100) if total > 0 else 0
        
        # Net R with Fee Drag (-0.1R per trade)
        daily_r = sum(x['r'] for x in trades)
        fees_r = total * 0.1
        net_r = daily_r - fees_r
        
        # P&L Estimate (Non-compounding for clarity per month)
        # Using fixed risk $ based on initial capital for standardized comparison?
        # User asked "how the bot performs". 
        # Let's show Dollar Value assuming we started that month with Current Balance?
        # Actually, simpler: show stats.
        
        dollar_val = net_r * (balance * RISK_PCT) # Approximation
        
        # Update Balance for next month (Compounding)
        # To be accurate, we should simulate trade-by-trade balance update.
        # Let's do that for the dollar column.
        
        m_start_bal = balance
        m_trades = sorted(trades, key=lambda x: x['time'])
        for t in m_trades:
             risk_amt = balance * RISK_PCT
             realized_r = t['r'] - 0.1
             pnl = realized_r * risk_amt
             balance += pnl
             
        m_pnl = balance - m_start_bal
        
        print(f"{m['name']:<10} | {total:<8} | {wr:<6.1f} | {net_r:<10.1f} | ${m_pnl:<12.2f}")
        total_r_cumulative += net_r

    print("="*60)
    print(f"Total Net R: {total_r_cumulative:.1f}")
    print(f"Final Balance: ${balance:.2f} (Start: ${INITIAL_CAPITAL:.2f})")
    print(f"Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
