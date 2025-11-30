import json
import pandas as pd
import numpy as np
import itertools
import sys
from datetime import datetime

class SingleSymbolOptimizer:
    def __init__(self, data_file='backtest_data_all50.json'):
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
    
    def prepare_symbol_data(self, symbol):
        """Prepare data and calculate indicators for a single symbol"""
        candles = self.raw_data[symbol]
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # VWAP
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['tpv'] = df['tp'] * df['volume']
        df['cum_tpv'] = df['tpv'].cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_tpv'] / df['cum_vol']
        
        # Volume Ratio
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        # Wicks
        df['body_top'] = np.maximum(df['open'], df['close'])
        df['body_bottom'] = np.minimum(df['open'], df['close'])
        df['upper_wick'] = df['high'] - df['body_top']
        df['lower_wick'] = df['body_bottom'] - df['low']
        df['candle_range'] = df['high'] - df['low']
        
        return df.dropna()
    
    def check_signal(self, row, params):
        rsi_min_long = params['rsi_min_long']
        rsi_max_long = params['rsi_max_long']
        rsi_min_short = params['rsi_min_short']
        rsi_max_short = params['rsi_max_short']
        vwap_dist_max = params['vwap_dist_max']
        vol_ratio_min = params['vol_ratio_min']
        macd_hist_min = params['macd_hist_min']
        bb_width_min = params['bb_width_min']
        
        if row['vol_ratio'] < vol_ratio_min: return None
        if row['bb_width'] < bb_width_min: return None
        if abs(row['hist']) < macd_hist_min: return None
        
        dist_to_vwap = abs(row['close'] - row['vwap'])
        dist_atr = dist_to_vwap / row['atr'] if row['atr'] > 0 else 0
        if dist_atr > vwap_dist_max: return None
        
        if rsi_min_long <= row['rsi'] <= rsi_max_long:
            if row['hist'] > 0:
                if row['candle_range'] > 0 and (row['lower_wick'] / row['candle_range']) > 0.10:
                    return 'long'
        
        if rsi_min_short <= row['rsi'] <= rsi_max_short:
            if row['hist'] < 0:
                if row['candle_range'] > 0 and (row['upper_wick'] / row['candle_range']) > 0.10:
                    return 'short'
        
        return None
    
    def run_backtest_for_params(self, df, params):
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
                            outcome = 'win'
                            break
                elif signal == 'short':
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)
                    for _, future in future_candles.iterrows():
                        if future['high'] >= sl_price: break
                        if future['low'] <= tp_price:
                            outcome = 'win'
                            break
                
                total_trades += 1
                if outcome == 'win': wins += 1
        
        wr = (wins / total_trades * 100) if total_trades > 0 else 0
        return total_trades, wr
    
    def optimize_symbol(self, symbol):
        print(f"\n{'='*60}")
        print(f"🔍 Optimizing: {symbol}")
        print(f"{'='*60}")
        
        # Prepare data
        df = self.prepare_symbol_data(symbol)
        print(f"✓ Loaded {len(df)} candles")
        
        # Parameter grid
        rsi_long_opts = [(40, 60), (38, 62), (35, 65), (42, 58)]
        rsi_short_opts = [(35, 55), (33, 57), (30, 60), (38, 52)]
        vwap_opts = [1.2, 1.3, 1.5, 1.8]
        vol_opts = [1.2, 1.5, 1.8, 2.0]
        macd_opts = [0.0003, 0.0005, 0.0010]
        
        combinations = list(itertools.product(rsi_long_opts, rsi_short_opts, vwap_opts, vol_opts, macd_opts))
        print(f"✓ Testing {len(combinations)} combinations...")
        
        results = []
        for i, (rsi_l, rsi_s, vwap, vol, macd) in enumerate(combinations):
            params = {
                'rsi_min_long': rsi_l[0], 'rsi_max_long': rsi_l[1],
                'rsi_min_short': rsi_s[0], 'rsi_max_short': rsi_s[1],
                'vwap_dist_max': vwap,
                'vol_ratio_min': vol,
                'macd_hist_min': macd,
                'bb_width_min': 0.008
            }
            
            trades, wr = self.run_backtest_for_params(df, params)
            
            if trades >= 10:  # Min trades threshold
                results.append({
                    'params': params,
                    'trades': trades,
                    'win_rate': wr,
                    'score': trades * wr
                })
            
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(combinations)}")
        
        # Find best
        if not results:
            print(f"⚠️  No configuration met minimum trades threshold for {symbol}")
            return None
        
        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]
        
        print(f"\n🏆 BEST CONFIGURATION FOR {symbol}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Trades: {best['trades']}")
        print(f"  Score: {best['score']:.1f}")
        p = best['params']
        print(f"\n  Settings:")
        print(f"    RSI Long:  {p['rsi_min_long']}-{p['rsi_max_long']}")
        print(f"    RSI Short: {p['rsi_min_short']}-{p['rsi_max_short']}")
        print(f"    VWAP: {p['vwap_dist_max']}")
        print(f"    Volume: {p['vol_ratio_min']}x")
        print(f"    MACD: {p['macd_hist_min']}")
        print(f"    BB Width: {p['bb_width_min']}")
        
        return best

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 backtest_single_symbol.py <SYMBOL>")
        print("Example: python3 backtest_single_symbol.py BTCUSDT")
        sys.exit(1)
    
    symbol = sys.argv[1]
    optimizer = SingleSymbolOptimizer()
    result = optimizer.optimize_symbol(symbol)
    
    if result:
        # Save result to JSON for automated processing
        output = {
            'symbol': symbol,
            'win_rate': result['win_rate'],
            'trades': result['trades'],
            'score': result['score'],
            'params': result['params']
        }
        
        with open(f'backtest_result_{symbol}.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: backtest_result_{symbol}.json")
