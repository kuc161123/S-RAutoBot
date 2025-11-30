import json
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

class ProRulesBacktester:
    def __init__(self, data_file='backtest_data.json'):
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
        self.dfs = {}
        self.prepare_data()

    def prepare_data(self):
        print("Preparing data and calculating indicators...")
        for symbol, candles in self.raw_data.items():
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Calculate Indicators
            # RSI 14
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (12, 26, 9)
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp12 - exp26
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['hist'] = df['macd'] - df['signal']
            
            # Bollinger Bands (20, 2)
            df['bb_mid'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
            
            # ATR 14
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # VWAP (Simple approximation for backtest: typical price * volume / cumulative volume)
            # Reset VWAP daily? For simplicity, we'll use a rolling VWAP or just typical price average
            # Let's use a rolling VWAP approximation for simplicity in this script
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['tpv'] = df['tp'] * df['volume']
            df['cum_tpv'] = df['tpv'].cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = df['cum_tpv'] / df['cum_vol']
            
            # Volume Ratio (vs 20 SMA)
            df['vol_ma'] = df['volume'].rolling(window=20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
            
            # Wick calculation
            df['body_top'] = np.maximum(df['open'], df['close'])
            df['body_bottom'] = np.minimum(df['open'], df['close'])
            df['upper_wick'] = df['high'] - df['body_top']
            df['lower_wick'] = df['body_bottom'] - df['low']
            df['candle_range'] = df['high'] - df['low']
            
            self.dfs[symbol] = df.dropna()
            print(f"  {symbol}: {len(self.dfs[symbol])} candles ready")

    def check_signal(self, row, params):
        # Unpack params
        rsi_min_long = params['rsi_min_long']
        rsi_max_long = params['rsi_max_long']
        rsi_min_short = params['rsi_min_short']
        rsi_max_short = params['rsi_max_short']
        vwap_dist_max = params['vwap_dist_max']
        vol_ratio_min = params['vol_ratio_min']
        macd_hist_min = params['macd_hist_min']
        bb_width_min = params['bb_width_min']
        
        signal = None
        
        # Common checks
        if row['vol_ratio'] < vol_ratio_min:
            return None
        if row['bb_width'] < bb_width_min:
            return None
        if abs(row['hist']) < macd_hist_min:
            return None
            
        # VWAP Distance (in ATRs)
        dist_to_vwap = abs(row['close'] - row['vwap'])
        dist_atr = dist_to_vwap / row['atr'] if row['atr'] > 0 else 0
        if dist_atr > vwap_dist_max:
            return None

        # Long Logic
        if rsi_min_long <= row['rsi'] <= rsi_max_long:
            if row['hist'] > 0: # MACD Bullish
                # Wick check (lower wick > 10% of range)
                if row['candle_range'] > 0 and (row['lower_wick'] / row['candle_range']) > 0.10:
                    signal = 'long'

        # Short Logic
        if rsi_min_short <= row['rsi'] <= rsi_max_short:
            if row['hist'] < 0: # MACD Bearish
                # Wick check (upper wick > 10% of range)
                if row['candle_range'] > 0 and (row['upper_wick'] / row['candle_range']) > 0.10:
                    signal = 'short'
                    
        return signal

    def run_backtest(self, params):
        total_trades = 0
        wins = 0
        
        # Simple simulation: 
        # TP = 1.0%
        # SL = 0.5%
        tp_pct = 0.01
        sl_pct = 0.005
        
        for symbol, df in self.dfs.items():
            in_position = False
            
            for i in range(len(df) - 1): # Stop 1 before end to check outcome
                if in_position:
                    # Skip if already in position (simple mode: 1 trade at a time per symbol)
                    # For this simple test, we assume trade closes within next few bars or hits SL/TP
                    # Let's just check the outcome of the signal immediately
                    pass
                
                row = df.iloc[i]
                signal = self.check_signal(row, params)
                
                if signal:
                    # Check outcome in next 24 hours (96 candles of 15m)
                    entry_price = row['close']
                    outcome = 'loss' # Default
                    
                    future_candles = df.iloc[i+1 : i+97]
                    
                    if signal == 'long':
                        tp_price = entry_price * (1 + tp_pct)
                        sl_price = entry_price * (1 - sl_pct)
                        
                        for _, future in future_candles.iterrows():
                            if future['low'] <= sl_price:
                                outcome = 'loss'
                                break
                            if future['high'] >= tp_price:
                                outcome = 'win'
                                break
                                
                    elif signal == 'short':
                        tp_price = entry_price * (1 - tp_pct)
                        sl_price = entry_price * (1 + sl_pct)
                        
                        for _, future in future_candles.iterrows():
                            if future['high'] >= sl_price:
                                outcome = 'loss'
                                break
                            if future['low'] <= tp_price:
                                outcome = 'win'
                                break
                    
                    total_trades += 1
                    if outcome == 'win':
                        wins += 1
                        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        return total_trades, win_rate

    def run_grid_search(self):
        # Expanded Parameter Grid
        # RSI: Test wider and tighter ranges
        rsi_long_opts = [(40, 60), (38, 62), (35, 65), (42, 58)]
        rsi_short_opts = [(35, 55), (33, 57), (30, 60), (38, 52)]
        
        # VWAP: Test standard and relaxed
        vwap_opts = [1.2, 1.3, 1.5, 1.8]
        
        # Volume: Test strict and relaxed
        vol_opts = [1.2, 1.5, 1.8, 2.0]
        
        # MACD: Test standard and relaxed
        macd_opts = [0.0003, 0.0005, 0.0010]
        
        combinations = list(itertools.product(rsi_long_opts, rsi_short_opts, vwap_opts, vol_opts, macd_opts))
        print(f"Testing {len(combinations)} combinations on {len(self.dfs)} symbols...")
        
        # Store results per symbol
        symbol_results = {sym: [] for sym in self.dfs.keys()}
        
        for i, (rsi_l, rsi_s, vwap, vol, macd) in enumerate(combinations):
            params = {
                'rsi_min_long': rsi_l[0], 'rsi_max_long': rsi_l[1],
                'rsi_min_short': rsi_s[0], 'rsi_max_short': rsi_s[1],
                'vwap_dist_max': vwap,
                'vol_ratio_min': vol,
                'macd_hist_min': macd,
                'bb_width_min': 0.008
            }
            
            # Run backtest per symbol
            for sym in self.dfs.keys():
                trades, wr = self.run_backtest_single(sym, params)
                if trades >= 10: # Min trades to be relevant
                    symbol_results[sym].append({
                        'params': params,
                        'trades': trades,
                        'win_rate': wr,
                        'score': trades * wr
                    })
            
            if i % 100 == 0:
                print(f"Progress: {i}/{len(combinations)}")

        print("\n" + "="*50)
        print("🏆 BEST SETTINGS PER SYMBOL (Max Score = Trades * WR)")
        print("="*50)
        
        for sym, results in symbol_results.items():
            if not results:
                print(f"⚠️  {sym}: No config met min trades requirement")
                continue
                
            # Sort by Score (Volume * Quality)
            results.sort(key=lambda x: x['score'], reverse=True)
            best = results[0]
            p = best['params']
            
            print(f"\n🔹 {sym}")
            print(f"   WR: {best['win_rate']:.1f}% | Trades: {best['trades']}")
            print(f"   Settings: Vol {p['vol_ratio_min']}x | RSI {p['rsi_min_long']}-{p['rsi_max_long']} | VWAP {p['vwap_dist_max']} | MACD {p['macd_hist_min']}")

    def run_backtest_single(self, symbol, params):
        df = self.dfs[symbol]
        total_trades = 0
        wins = 0
        tp_pct = 0.01
        sl_pct = 0.005
        
        for i in range(len(df) - 1):
            row = df.iloc[i]
            signal = self.check_signal(row, params)
            
            if signal:
                entry_price = row['close']
                outcome = 'loss'
                future_candles = df.iloc[i+1 : i+97]
                
                if signal == 'long':
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)
                    for _, future in future_candles.iterrows():
                        if future['low'] <= sl_price: break
                        if future['high'] >= tp_price:
                            outcome = 'win'; break
                elif signal == 'short':
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)
                    for _, future in future_candles.iterrows():
                        if future['high'] >= sl_price: break
                        if future['low'] <= tp_price:
                            outcome = 'win'; break
                
                total_trades += 1
                if outcome == 'win': wins += 1
                
        wr = (wins / total_trades * 100) if total_trades > 0 else 0
        return total_trades, wr

if __name__ == "__main__":
    tester = ProRulesBacktester(data_file='backtest_data_all50.json')
    tester.run_grid_search()
