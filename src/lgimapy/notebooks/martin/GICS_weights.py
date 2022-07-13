import pandas as pd

from lgimapy.bloomberg import bdp
from lgimapy.data import Database, groupby


db = Database()
db.load_market_data()
ix = db.build_market_index(in_H0A0_index=True)
ratings_kwargs = {"BB": ("BB+", "BB-"), "B": ("B+", "B-")}
df_list = []
for rating, rating_kws in ratings_kwargs.items():
    rating_ix = ix.subset(rating=rating_kws)
    rating_ix.df["GICS_Sector"] = bdp(
        rating_ix.df["ISIN"], "CORP", "GICS_SECTOR_NAME"
    ).values
    rating_df = (
        rating_ix.df[["GICS_Sector", "MarketValue"]]
        .groupby("GICS_Sector")
        .sum()
        .squeeze()
        .rename(rating)
    )
    df_list.append(rating_df)

# %%
df = pd.concat(df_list, axis=1).rename_axis(None)
df = df.divide(df.sum()).round(3)
df.to_csv("HY_GICS_Sector_Market_Values_by_Rating.csv")
