import pandas as pd
import numpy as np
import quantedge.backtester_core as qe
import os

# Load Data
def load_research_data():
    df = pd.read_parquet("quantedge/BTCUSDT_15m_3y.parquet")
    # Technical Indicators
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    # Bollinger Bands
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['ma20'] + (df['std20'] * 2)
    df['bb_lower'] = df['ma20'] - (df['std20'] * 2)
    
    return df.dropna().reset_index(drop=True)

# CANDIDATE 1: Bollinger Mean Reversion
def strategy_mean_reversion(df, rsi_thresh=30, bb_std=2.0):
    # Signals: Buy if Low < Lower BB and RSI < rsi_thresh
    # Sell if High > Upper BB and RSI > (100 - rsi_thresh)
    upper = df['ma20'] + (df['std20'] * bb_std)
    lower = df['ma20'] - (df['std20'] * bb_std)
    
    buy_sig = (df['low'] < lower) & (df['rsi'] < rsi_thresh)
    sell_sig = (df['high'] > upper) & (df['rsi'] > (100 - rsi_thresh))
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

# CANDIDATE 2: Trend Following Breakout
def strategy_trend_follow(df, lookback=20, trend_ema=200):
    # Buy if Close > Highest(High, lookback) and Close > EMA(200)
    # Sell if Close < Lowest(Low, lookback) and Close < EMA(200)
    high_channel = df['high'].rolling(lookback).max().shift(1)
    low_channel = df['low'].rolling(lookback).min().shift(1)
    
    buy_sig = (df['close'] > high_channel) & (df['close'] > df['ema200'])
    sell_sig = (df['close'] < low_channel) & (df['close'] < df['ema200'])
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

# CANDIDATE 3: Squeeze Breakout (BB inside KC)
def strategy_squeeze(df, sq_mult=2.0, kc_mult=1.5):
    # KC = EMA(Close, 20) +/- KC_MULT * ATR(20)
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['kc_upper'] = df['ema20'] + (df['atr'] * kc_mult)
    df['kc_lower'] = df['ema20'] - (df['atr'] * kc_mult)
    
    # BB = MA(20) +/- BB_STD * STD(20)
    # Squeeze = BB_UPPER < KC_UPPER and BB_LOWER > KC_LOWER
    df['bb_up'] = df['ma20'] + (df['std20'] * sq_mult)
    df['bb_lo'] = df['ma20'] - (df['std20'] * sq_mult)
    
    df['squeeze'] = (df['bb_up'] < df['kc_upper']) & (df['bb_lo'] > df['kc_lower'])
    
    # Buy if close breaks KC_UPPER and WAS in squeeze
    buy_sig = (df['close'] > df['kc_upper']) & (df['squeeze'].shift(1))
    sell_sig = (df['close'] < df['kc_lower']) & (df['squeeze'].shift(1))
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

def print_wfo_summary(name, wfo_results, bt_engine):
    print(f"\n--- WFO Summary for {name} ---")
    all_oos_trades = pd.concat([f['trades'] for f in wfo_results])
    metrics = bt_engine.calculate_metrics(all_oos_trades)
    
    print(f"Aggregate OOS Stats:")
    print(f"  Total PnL: {metrics['total_pnl']:.4f}")
    print(f"  Win Rate:  {metrics['win_rate']:.2%}")
    print(f"  Sharpe:    {metrics['sharpe']:.2f}")
    print(f"  Max DD:    {metrics['max_dd']:.2%}")
    print(f"  N Trades:  {metrics['trades']}")
    
    print(f"Per Fold Detail:")
    folds_prof = 0
    for f in wfo_results:
        m = bt_engine.calculate_metrics(f['trades'])
        is_prof = m['total_pnl'] > 0
        folds_prof += 1 if is_prof else 0
        mark = "✅" if is_prof else "❌"
        print(f"  Fold {f['fold']} {mark} | PnL: {m['total_pnl']:+7.4f} | Sharpe: {m['sharpe']:5.2f} | trades: {m['trades']}")
    
    prof_rate = folds_prof / len(wfo_results)
    print(f"Fold Profitable Rate: {prof_rate:.0%} ({'PASS' if prof_rate >= 0.6 else 'FAIL'})")

def run_research():
    df = load_research_data()
    bt = qe.QuantEdgeBacktester(df)
    
    print("Testing Candidate 1: Mean Reversion (BB + RSI)")
    mr_grid = [{'rsi_thresh': 25, 'bb_std': 2.2, 'rr': 2.0, 'sl_atr_mult': 2.0}]
    wfo_mr = bt.walk_forward(strategy_mean_reversion, mr_grid)
    print_wfo_summary("Mean Reversion", wfo_mr, bt)

    print("\nTesting Candidate 2: Trend Following (Channel Breakout)")
    tf_grid = [
        {'lookback': 24, 'rr': 2.5, 'sl_atr_mult': 1.5},
        {'lookback': 48, 'rr': 3.0, 'sl_atr_mult': 2.5}
    ]
    wfo_tf = bt.walk_forward(strategy_trend_follow, tf_grid)
    print_wfo_summary("Trend Follow", wfo_tf, bt)

    print("\nTesting Candidate 3: Squeeze Breakout")
    sq_grid = [
        {'sq_mult': 2.0, 'kc_mult': 1.5, 'rr': 2.5, 'sl_atr_mult': 1.5},
        {'sq_mult': 2.2, 'kc_mult': 1.8, 'rr': 3.0, 'sl_atr_mult': 2.0}
    ]
    wfo_sq = bt.walk_forward(strategy_squeeze, sq_grid)
    print_wfo_summary("Squeeze Breakout", wfo_sq, bt)

if __name__ == "__main__":
    run_research()
