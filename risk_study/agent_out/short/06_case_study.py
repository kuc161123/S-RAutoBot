import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)
btc = pd.read_parquet("/Users/lualakol/AutoTrading Bot/cache_3yr_1h/BTCUSDT.parquet")
btc = btc.set_index("start")

bad_days = ["2025-06-23", "2026-02-25", "2025-10-21", "2026-03-23"]
good_days = ["2025-10-10", "2026-01-31", "2024-08-05", "2025-11-21"]

for tag, days in [("BAD", bad_days), ("GOOD", good_days)]:
    for d in days:
        d = pd.Timestamp(d)
        window = btc.loc[d - pd.Timedelta(hours=6): d + pd.Timedelta(hours=30)]
        if len(window) == 0:
            continue
        day_open = window.iloc[0]["open"]
        day_close = window.iloc[-1]["close"]
        day_high = window["high"].max()
        day_low = window["low"].min()
        move_pct = (day_close/day_open - 1)*100
        range_pct = (day_high/day_low - 1)*100
        print(f"{tag} {d.date()}: BTC open={day_open:.0f} close={day_close:.0f} move={move_pct:+.2f}% "
              f"range(hi/lo)={range_pct:.2f}% high={day_high:.0f} low={day_low:.0f}")

print()
print("Per-day short trade count by entry HOUR on worst days (checking single-hour bursts):")
short_train = train[train.side=="short"]
for d in ["2025-06-23", "2026-02-25", "2025-10-21"]:
    d = pd.Timestamp(d)
    day_rows = short_train[short_train.entry_date == d]
    print(f"\n{d.date()}: n={len(day_rows)}")
    print(day_rows.groupby(day_rows.entry_time.dt.hour).size())
