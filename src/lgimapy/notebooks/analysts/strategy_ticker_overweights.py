import pandas as pd

from lgimapy.data import Database

strategies = ["US Credit", "US Long Credit"]
tickers = ["PEMEX"]

db = Database()
for strategy in strategies:
    df = db.ticker_overweights(strategy, tickers)
    if len(tickers) == 1:
        df = df.to_frame()
    df.to_csv(f"{db.strategy_fid(strategy)}_ticker_ow.csv")


# %%
db.load_market_data(start=df.index[0], local=True)
oas_list = []
for ticker in tickers:
    ix = db.build_market_index(ticker=ticker)
    oas_list.append(ix.market_value_weight("OAS").rename(ticker))
    ix_long = ix.subset(maturity=(10, None))
    oas_list.append(ix_long.market_value_weight("OAS").rename(f"{ticker}_10+"))

oas_df = pd.concat(oas_list, axis=1, sort=True)
oas_df.to_csv("fallen_angels_spread_timeseries.csv")


from lgimapy.utils import load_json

x = load_json("cusip_bloomberg_subsectors")
