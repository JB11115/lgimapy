from datetime import datetime as dt

import pandas as pd

from lgimapy.data import Database, Index
from lgimapy.utils import load_json, root, mkdir

# %%
# -------------------------------------------------------------------------- #
account = "P-LD"
n_largest = 10
constraints = None
path = root("reports/account_overweights")
# constraints = {"rating": ("BBB+", "BBB-")}
constraints = {"sector": "TECHNOLOGY"}
fid = f"{account}_OAD_difference"
# -------------------------------------------------------------------------- #
# %%
mkdir(path)
today = dt.today().strftime("%Y-%m-%d")

ratings = load_json("ratings")
db = Database()
df = db.load_portfolio(accounts="P-LD")
df = df[~df["Sector"].isin({"CASH", "TREASURIES"})]
df["NumericRating"] = df["Idx_Rtg"].map(ratings)
if constraints is not None:
    ix = Index(df)
    ix_sub = ix.subset(**constraints)
    df = ix_sub.df.copy()

ticker_df = (
    df.groupby(["Ticker"], observed=True)
    .sum()
    .reset_index()
    .set_index("Ticker")
)
ticker_df.sort_values("OAD_Diff", ascending=False, inplace=True)
oad = ticker_df["OAD_Diff"]
oad = pd.DataFrame(oad).round(3)
oad.columns = ["Overweight (OAD)"]
if n_largest is not None:
    oad = pd.concat([oad.head(n_largest), oad.tail(n_largest)], axis=0)
oad.to_csv(path / f"{fid}_{today}.csv")
