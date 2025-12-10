
import asyncio
import pandas as pd
import pandas_ta as ta
import yaml
import logging
import os
from autobot.brokers.bybit import Bybit, BybitConfig

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock Config
config = {
    'bybit': {
        'base_url': 'https://api.bybit.com',
        'api_key': os.getenv('BYBIT_API_KEY'),
        'api_secret': os.getenv('BYBIT_API_SECRET'),
        'use_alt_on_fail': True,
        'alt_base_url': 'https://api.bybit.com'
    }
}

class SignalDiagnoser:
    def __init__(self):
        self.broker = Bybit(BybitConfig(**config['bybit']))
        self.load_combos()
    
    def load_combos(self):
        try:
            with open('backtest_golden_combos.yaml', 'r') as f:
                self.vwap_combos = yaml.safe_load(f)
            logger.info(f"Loaded {len(self.vwap_combos)} symbol configs")
        except Exception as e:
            logger.error(f"Failed to load combos: {e}")
            self.vwap_combos = {}

    def calculate_indicators(self, df):
        if len(df) < 50: return df
        
        # TA-Lib / Pandas TA calculations - MUST MATCH BOT EXACTLY
        df['atr'] = df.ta.atr(length=14)
        df['rsi'] = df.ta.rsi(length=14)
        
        macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        # VWAP calculation (using typical price)
        # Verify this matches bot.py logic if it's custom or library
        # In bot.py, vwap is calculated inside calculate_indicators? 
        # WAIT - bot.py uses df.ta.vwap OR custom?
        # Let's check bot.py again. It wasn't shown in the snippet!
        # Assuming standard VWAP for now, but this is a critical check point.
        
        # Standard VWAP
        if 'vwap' not in df.columns:
             # Typical Price * Volume
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['tp_v'] = df['tp'] * df['volume']
            df['cum_tp_v'] = df['tp_v'].cumsum()
            df['cum_v'] = df['volume'].cumsum()
            df['vwap'] = df['cum_tp_v'] / df['cum_v']
            
        # Rolling high/low for fibs
        df['roll_high'] = df['high'].rolling(50).max()
        df['roll_low'] = df['low'].rolling(50).min()
        
        return df

    def get_combo(self, row):
        # RSI
        rsi = row.rsi
        if rsi < 30: r_bin = '<30'
        elif rsi < 40: r_bin = '30-40'
        elif rsi < 60: r_bin = '40-60'
        elif rsi < 70: r_bin = '60-70'
        else: r_bin = '70+'
        
        # MACD
        m_bin = 'bull' if row.macd > row.macd_signal else 'bear'
        
        # Fib
        high, low, close = row.roll_high, row.roll_low, row.close
        if high == low: f_bin = '0-23'
        else:
            fib = (high - close) / (high - low) * 100
            if fib < 23.6: f_bin = '0-23'
            elif fib < 38.2: f_bin = '23-38'
            elif fib < 50.0: f_bin = '38-50'
            elif fib < 61.8: f_bin = '50-61'
            elif fib < 78.6: f_bin = '61-78'
            elif fib < 100: f_bin = '78-100'
            else: f_bin = '100+'
            
        return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

    async def diagnose(self, symbol):
        print(f"\n🔍 Diagnosing {symbol}...")
        klines = self.broker.get_klines(symbol, '3', limit=200)
        if not klines:
            print("   ❌ No data fetched")
            return

        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        for c in ['open','high','low','close','volume']: 
            df[c] = df[c].astype(float)
        
        # Ensure date sorting
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        df.sort_index(inplace=True)

        df = self.calculate_indicators(df)
        
        # Look at the last FEW candles to see if we missed anything recently
        print(f"   Last candle time: {df.index[-1]}")
        
        for i in range(-5, 0): # Check last 5 candles
            try:
                row = df.iloc[i]
                
                # Manual VWAP check
                # Note: Bot check is: low <= vwap and close > vwap (LONG)
                #                    high >= vwap and close < vwap (SHORT)
                
                is_long = row.low <= row.vwap and row.close > row.vwap
                is_short = row.high >= row.vwap and row.close < row.vwap
                
                combo = self.get_combo(row)
                
                status = "⚪️"
                if is_long: status = "🟢 LONG SIGNAL"
                if is_short: status = "🔴 SHORT SIGNAL"
                
                allowed_long = self.vwap_combos.get(symbol, {}).get('allowed_combos_long', [])
                allowed_short = self.vwap_combos.get(symbol, {}).get('allowed_combos_short', [])
                
                match_str = ""
                if is_long and combo in allowed_long: match_str = "✅ ALLOWED!"
                if is_short and combo in allowed_short: match_str = "✅ ALLOWED!"
                
                print(f"   [{df.index[i]}] P:{row.close:.4f} V:{row.vwap:.4f} | {status} | {combo} {match_str}")
                
                if match_str:
                    logger.info(f"   🎉 FOUND VALID SIGNAL IN LAST 5 CANDLES: {symbol} {combo}")

            except IndexError:
                continue

async def main():
    diag = SignalDiagnoser()
    # Test top valid symbols from config
    symbols = list(diag.vwap_combos.keys())[:20] 
    
    print(f"Targeting {len(symbols)} symbols...")
    for sym in symbols:
        await diag.diagnose(sym)

if __name__ == "__main__":
    asyncio.run(main())
