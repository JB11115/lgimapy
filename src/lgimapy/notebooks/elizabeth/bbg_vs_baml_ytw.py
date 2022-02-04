import pandas as pd
from lgimapy.data import Database

# %%

port = db.load_portfolio(account="PMCHY")
ix = db.build_market_index(isin=port.isins)

# %%
df = pd.concat(
    (
        port.df.set_index("CUSIP")["YTW"].rename("BAML YTW"),
        ix.df["YieldToWorst"].rename("BBG YTW"),
    ),
    axis=1,
).rename_axis(None)
df["Resid"] = df["BBG YTW"] - df["BAML YTW"]
df.dropna(subset=["Resid"], inplace=True)
df.sort_values("Resid", inplace=True)
df_bad = df[df["Resid"].abs() > 1].copy()
df_bad["Description"] = port.df[port.df["CUSIP"].isin(df_bad.index)].set_index(
    "CUSIP"
)["Description"]
df_bad.to_csv("Worst_YTW_differences.csv")
