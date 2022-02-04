import pandas as pd

from lgimapy.data import Database

# %%
db = Database()
dates = db.date("YEAR_STARTS")
df_list = []
for date in dates:
    db.load_market_data(date)
    ix = db.build_market_index(in_returns_index=True)
    sector_df = (
        ix.df[["Sector", "MarketValue"]]
        .groupby("Sector", observed=True)
        .sum()
        .squeeze()
    )
    sector_df = sector_df[sector_df > 0]
    sector_df /= sector_df.sum()
    df_list.append(sector_df.rename(date))

# %%
df = pd.concat(df_list, axis=1).T
df.to_csv("sector_index_weight_history.csv")
