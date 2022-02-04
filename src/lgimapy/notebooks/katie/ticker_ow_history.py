from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%
start_date = pd.to_datetime("1/1/2020")
tickers = ["BA", "ED"]
account = "CHLD"

db = Database()
dates = [d for d in db.date("MONTH_STARTS") if d >= start_date]
d = defaultdict(list)
for date in tqdm(dates):
    port = db.load_portfolio(account=account, date=date)
    oad_ow = port.ticker_overweights(by="OAD")
    dts_ow = port.ticker_overweights(by="DTS")
    for ticker in tickers:
        d[f"{ticker} OAD OW"].append(oad_ow.get(ticker))
        d[f"{ticker} DTS OW"].append(dts_ow.get(ticker))

df = pd.DataFrame(d, index=dates)
df.to_csv(f"{account}_ticker_OW_history.csv")

# %%
for ticker in tickers:
    ax_left, ax_right = vis.plot_double_y_axis_timeseries(
        df[f"{ticker} OAD OW"],
        df[f"{ticker} DTS OW"],
        ylabel_left="OAD Overweight",
        ylabel_right="DTS Overweight",
        ret_axes=True,
        title=ticker,
    )
    ax_left.fill_between(
        df.index, 0, df[f"{ticker} OAD OW"], color="navy", alpha=0.1
    )
    ax_right.fill_between(
        df.index, 0, df[f"{ticker} DTS OW"], color="darkorchid", alpha=0.1
    )
    ax_left.axhline(0, color="k", alpha=0.5, lw=2)

    vis.savefig(f"{account}_{ticker}_OW_history")
