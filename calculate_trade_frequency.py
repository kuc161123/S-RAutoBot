import yaml
import re

def calculate():
    filename = 'symbol_overrides_400.yaml.bak'
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    # Regex to find Test N
    # Matches lines like: # Train: 68.3%(41) -> Test: 81.8%(11)
    matches = re.findall(r'Test:.*?\((\d+)\)', content)
    
    total_trades = sum(int(n) for n in matches)
    total_strategies = len(matches)
    
    # Backtest duration: 10,000 candles * 3 mins = 30,000 mins
    # 30,000 / 60 / 24 = 20.83 days
    days = 20.83
    
    trades_per_day = total_trades / days
    
    print(f"--- Trade Frequency Analysis ---")
    print(f"Total Strategies: {total_strategies}")
    print(f"Total Backtest Trades (N): {total_trades}")
    print(f"Backtest Duration: ~{days:.1f} days")
    print(f"--------------------------------")
    print(f"🔥 ESTIMATED TRADES PER DAY: {trades_per_day:.1f}")
    print(f"--------------------------------")
    print(f"Average trades per symbol/day: {(trades_per_day/115):.2f}")

if __name__ == "__main__":
    calculate()
