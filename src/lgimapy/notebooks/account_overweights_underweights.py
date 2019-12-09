from datetime import datetime as dt

import pandas as pd

from lgimapy.data import Database


# -------------------------------------------------------------------------- #
account = "P-LD"
# -------------------------------------------------------------------------- #

today = dt.today().strftime("%Y-%m-%d")
db = Database()
df = db.load_portfolio(accounts="P-LD")
df = df[~df["Sector"].isin({"CASH", "TREASURIES"})]
ticker_df = (
    df.groupby(["Ticker"], observed=True)
    .sum()
    .reset_index()
    .set_index("Ticker")
)
ticker_df.sort_values("OAD_Diff", ascending=False, inplace=True)
oad = ticker_df["OAD_Diff"]
oad.columns = ["Overweight (OAD)"]
oad = pd.DataFrame(oad)
oad.to_csv(f"{account}_overweight_{today}.csv")
