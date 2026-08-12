import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)
short_train = train[train.side=="short"].copy()

# does density correlate with time (confound: more symbols/configs added later)?
short_train["t_ord"] = (short_train.entry_time - short_train.entry_time.min()).dt.days
print("corr(density_short_24h, t_ord):", short_train["density_short_24h"].corr(short_train["t_ord"]))
print("corr(short_concurrency, t_ord):", short_train["short_concurrency"].corr(short_train["t_ord"]))

# bucket by entry-time density_short_24h decile, look at mean net_r per trade in each bucket
short_train["dens_decile"] = pd.qcut(short_train["density_short_24h"], 10, labels=False, duplicates="drop")
g = short_train.groupby("dens_decile").agg(n=("net_r","size"), mean_r=("net_r","mean"),
                                            density_lo=("density_short_24h","min"), density_hi=("density_short_24h","max"))
print("\nmean net_r by density_short_24h decile (entry-time density):")
print(g)

short_train["conc_decile"] = pd.qcut(short_train["short_concurrency"], 10, labels=False, duplicates="drop")
g2 = short_train.groupby("conc_decile").agg(n=("net_r","size"), mean_r=("net_r","mean"),
                                             conc_lo=("short_concurrency","min"), conc_hi=("short_concurrency","max"))
print("\nmean net_r by short_concurrency decile:")
print(g2)

# daily view: for each calendar day (by entry_date), n shorts entered and same-day exit-date short R sum
daily_entries = short_train.groupby("entry_date").size().rename("n_entries")
daily_exit_r = short_train.groupby("exit_date")["net_r"].sum().rename("exit_r_sum")
print("\ncorr(daily n_short_entries, same-day exit short R sum) -- rough same-day proxy:")
merged = pd.concat([daily_entries, daily_exit_r], axis=1).dropna()
print(merged.corr())

# also: correlation between SAME exit-day trade count and that day's mean R per trade (direct)
day_stats = short_train.groupby("exit_date").agg(n=("net_r","size"), sum_r=("net_r","sum"), mean_r=("net_r","mean"))
print("\ncorr(day n trades, day mean_r):", day_stats["n"].corr(day_stats["mean_r"]))
print("corr(day n trades, day sum_r):", day_stats["n"].corr(day_stats["sum_r"]))
print(day_stats.sort_values("n", ascending=False).head(15))
