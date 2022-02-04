import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database

db = Database()

# %%
dates = db.trade_dates(start=db.date("MARKET_START"))
bad_dates, missing_flags = [], []
for date in tqdm(dates):
    date_fmt = date.strftime("%m/%d/%Y")
    db.load_market_data(date)
    stats_ix = db.build_market_index(in_hy_stats_index=True)
    rets_ix = db.build_market_index(in_hy_returns_index=True)
    bad_stats = len(stats_ix.df) < 1000
    bad_rets = len(rets_ix.df) < 1000
    if bad_rets or bad_stats:
        bad_dates.append(date)
    if bad_stats and bad_rets:
        print(f"  {date_fmt}: both")
        missing_flags.append("both")
    elif bad_rets:
        print(f"  {date_fmt}: returns")
        missing_flags.append("returns")
    elif bad_stats:
        print(f"  {date_fmt}: stats")
        missing_flags.append("stats")


df = pd.Series(
    missing_flags, index=bad_dates, name="missing HY flags"
).to_frame()
df.to_csv("missing_hy_flags.csv")

# %%
df = pd.read_csv("missing_hy_flags.csv", index_col=0)
df.index = pd.to_datetime(df.index)
[d.strftime("%m/%d/%Y") for d in df.index]
len(df)
