import yaml
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
import concurrent.futures

# CONFIG
START_DATE = "2025-09-01 00:00:00"
END_DATE = "2025-12-31 23:59:59"
INITIAL_CAPITAL = 700.0
RISK_PCT = 0.001
BASE_URL = "https://api.bybit.com"

# --- UTIL ---
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
    rsi = df['rsi'].values; ema = df['ema'].values; times = df['start'].values
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    used_pivots = set() 
    
    for i in range(205, len(df) - 3):
        curr_price = close[i]; curr_ema = ema[i]
        
        # BULLISH (REG + HID)
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

        # BEARISH (REG + HID)
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

def execute_logic_sequential(signal, df, cfg, StartTS, EndTS):
    # Returns {'time': ts, 'r': outcome, 'exit_time': ts_end} OR None
    
    # 1. Check Date
    candle = df.iloc[signal['conf_idx']]
    start_ts = int(candle['start'].timestamp()*1000)
    if start_ts < StartTS or start_ts > EndTS: return None
    
    if signal['type'] != cfg['divergence_type']: return None
    
    conf_idx = signal['conf_idx']
    side = signal['side']
    bos = signal['swing']
    
    entry_price = None; sl_dist = None; entry_time_ts = 0
    
    for i in range(1, 7):
        if conf_idx+i+1 >= len(df): break
        c = df.iloc[conf_idx+i]
        triggered = (c['close'] > bos) if side == 'long' else (c['close'] < bos)
        
        if triggered:
            entry_c = df.iloc[conf_idx+i+1] # Entry on next open
            entry_price = entry_c['open'] 
            sl_dist = c['atr'] * cfg['atr_mult']
            entry_time_ts = int(entry_c['start'].timestamp()*1000)
            break
            
    if not entry_price: return None
    
    # 4. Check Outcome (Using 1H proxy for speed, assumes conservative loss if checking OHLC)
    rr = cfg['rr']
    outcome = -1.0 # Default loss
    
    # We estimate holding time. 
    # If using 1H data, we scan forward.
    exit_ts = entry_time_ts + (24*3600000) # Fallback
    
    if side == 'long':
        tp = entry_price + (sl_dist * rr)
        sl = entry_price - sl_dist
        for k in range(conf_idx+i+1, min(len(df), conf_idx+i+120)): # 5 days max
             k_ts = int(df.iloc[k]['start'].timestamp()*1000)
             if df['low'].iloc[k] <= sl: outcome = -1.0; exit_ts = k_ts; break
             if df['high'].iloc[k] >= tp: outcome = rr; exit_ts = k_ts; break
             exit_ts = k_ts # Update 'last checked'
    else:
        tp = entry_price - (sl_dist * rr)
        sl = entry_price + sl_dist
        for k in range(conf_idx+i+1, min(len(df), conf_idx+i+120)):
             k_ts = int(df.iloc[k]['start'].timestamp()*1000)
             if df['high'].iloc[k] >= sl: outcome = -1.0; exit_ts = k_ts; break
             if df['low'].iloc[k] <= tp: outcome = rr; exit_ts = k_ts; break
             exit_ts = k_ts
             
    return {'time': start_ts, 'r': outcome, 'exit_time': exit_ts}


# --- MAIN ---

def get_timestamps():
    s = datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    e = datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(s.timestamp()*1000), int(e.timestamp()*1000)

StartTS, EndTS = get_timestamps()

def fetch_klines(symbol, interval):
    end_req = int(datetime(2025, 12, 10, tzinfo=timezone.utc).timestamp() * 1000)
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

def worker(args):
    sym, cfg = args
    df = fetch_klines(sym, '60')
    if df is None or len(df) < 200: return []
    df = prepare_1h(df)
    signals = detect_signals(df)
    
    trades = []
    last_exit_time = 0 # PER SYMBOL STATE
    
    # Sort signals by time? They are detected in order.
    for sig in signals:
        # Check start time
        idx = sig['conf_idx']
        start_ts = int(df.iloc[idx]['start'].timestamp()*1000)
        
        # BUSY CHECK
        if start_ts < last_exit_time: continue
        
        res = execute_logic_sequential(sig, df, cfg, StartTS, EndTS)
        if res: 
             trades.append(res)
             last_exit_time = res['exit_time'] # Update Busy
             
    return trades

# XAUT SINGLE TEST MAIN
def main():
    print(f"🔥 PRECISION TEST (XAUTUSDT - SEP-DEC 2025 Ref Curve)")
    
    with open('config_combined.yaml', 'r') as f:
        conf = yaml.safe_load(f)
        
    cfg = conf['symbols']['XAUTUSDT']
    
    # Needs valid start time for finding it
    StartTS_ms = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()*1000)
    EndTS_ms = int(datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()*1000)

    print("Fetching 1H Data...")
    df_1h = fetch_klines('XAUTUSDT', '60')
    if df_1h is None: return
    df_1h = prepare_1h(df_1h) # Re-use prepare_1h
    
    print("Detecting Signals...")
    c = df_1h['close'].values; ema = df_1h['ema'].values; rsi = df_1h['rsi'].values
    ph, pl = find_pivots(c)
    starts = df_1h['start'].view(np.int64) // 10**6 
    
    trades = []
    last_exit_time = 0
    
    for i in range(200, len(df_1h)-10):
        ts = starts[i]
        if ts < StartTS_ms: continue
        if ts > EndTS_ms: break
        if ts < last_exit_time: continue
        
        found = False
        if 'BULL' in cfg['divergence_type'] and c[i] > ema[i]:
             p_lows = []
             for j in range(i-3, i-50, -1):
                 if not np.isnan(pl[j]): p_lows.append((j, pl[j]))
                 if len(p_lows) >= 2: break
             if len(p_lows)==2:
                 curr_i, curr_v = p_lows[0]; prev_i, prev_v = p_lows[1]
                 if curr_v < prev_v and rsi[curr_i] > rsi[prev_i]: found=True

        if found:
            entry_c = df_1h.iloc[i+1] # Breakout candle
            entry_time_ts = int(entry_c['start'].timestamp() * 1000)
            if entry_time_ts < last_exit_time: continue # Double check
            
            sl_dist = df_1h.iloc[i]['atr'] * cfg['atr_mult']
            entry_price = entry_c['open']
            
            outcome = -1.0 # Default Outcome
            exit_ts = entry_time_ts + (24*3600*1000) 
            
            tp = entry_price + (sl_dist * cfg['rr'])
            sl = entry_price - sl_dist
            
            # Simple 1H check for speed (Valid for Ref Curve since XAUT verified closely)
            # Actually, let's just output the dates to show the distribution
            for k in range(i+2, min(len(df_1h), i+200)):
                 row = df_1h.iloc[k]
                 k_ts = int(row['start'].timestamp()*1000)
                 if row['low'] <= sl: outcome=-1.0; exit_ts=k_ts; break
                 if row['high'] >= tp: outcome=cfg['rr']; exit_ts=k_ts; break
            
            trades.append(outcome)
            last_exit_time = exit_ts
            print(f"  Trade {entry_c['start']}: {outcome}R")
            
    print(f"Total R: {sum(trades)}")

if __name__ == "__main__":
    main()
