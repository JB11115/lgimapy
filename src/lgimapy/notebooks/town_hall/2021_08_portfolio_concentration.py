import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import copy

vis.style()

# %%

db = Database()
db.load_market_data(start="2/1/2020")
dates = {"ytd": db.date("ytd"), "current": None}
ports = {
    key: db.load_portfolio(account="P-LD", date=date, universe="stats")
    for key, date in dates.items()
}

# %%
ow = {
    key: port.ticker_overweights(by="OAD")
    .abs()
    .sort_values(ascending=True)
    .reset_index(drop=True)
    .iloc[-80:]
    for key, port in ports.items()
}
for key, port in ports.items():
    ticker_df = port.ticker_df
    n_bm = len(ticker_df[ticker_df["BM_Weight"] > 0])
    n_p = len(ticker_df[ticker_df["P_Weight"] > 0])
    print(f"{key}: {n_p} / {n_bm}")

# %%
fig, ax = vis.subplots()
kwargs = {"width": 1, "alpha": 0.9, "edgecolor": "w"}
ow["current"].plot.barh(ax=ax, color="navy", label="Current", **kwargs)
ow["ytd"].plot.barh(ax=ax, color="skyblue", label="January", **kwargs)
ax.set_yticks([])
ax.set_ylabel("Issuers")
ax.set_xlabel("Active Position Size relative to Benchmark (CTD)")
ax.legend(fancybox=True, shadow=True)
vis.savefig("Issuer_Concentration")

ow["current"].sort_values(ascending=False).reset_index(drop=True)
ow_df = pd.concat(
    (
        ow["current"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
        .rename("Current"),
        ow["ytd"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
        .rename("January"),
    ),
    axis=1,
)
ow_df.index += 1
ow_df.to_csv("top_OAD_overweight_magnitudes.csv")

# %%
ow_tickers = {
    key: port.ticker_overweights(by="OAD")
    .abs()
    .sort_values(ascending=False)
    .iloc[:30]
    for key, port in ports.items()
}

new_ow = set(ow_tickers["current"].index) - set(ow_tickers["ytd"].index)
new_ow_ytd_df = (
    ports["ytd"].ticker_overweights(by="OAD").reindex(new_ow).dropna()
)
new_ow_tickers = list(new_ow_ytd_df[new_ow_ytd_df < 0].index)
new_ow_tickers += list(new_ow - set(new_ow_ytd_df.index))
copy(new_ow_tickers)
ow_df

# %%
df = pd.read_clipboard()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
# %%
start = pd.to_datetime("9/1/2020")
te = df["Relative Return"].abs().sort_index().rolling(2).mean()
vol = ix.OAS().rolling(20).std().dropna()
vol = vol[vol.index > start]
te = te[te.index > start]
vis.plot_double_y_axis_timeseries(
    te,
    vol,
    ylabel_left="Portfolio Tracking Error",
    ylabel_right="Benchmark volatility",
)
vis.savefig("Tracking_Error")

# %%
ports["current"].ticker_overweights(by="OAD").abs().sort_values(
    ascending=False
).head(30)
ports["ytd"].ticker_overweights(by="OAD").abs().sort_values(
    ascending=False
).head(30)

# %%
ix_new = db.build_market_index(
    ticker=new_ow_tickers, in_stats_index=True, maturity=(10, None)
)
for ticker in new_ow_tickers:
    print(ticker)
    print(ix_new.subset(ticker=ticker).df["IssueDate"].sort_values())
