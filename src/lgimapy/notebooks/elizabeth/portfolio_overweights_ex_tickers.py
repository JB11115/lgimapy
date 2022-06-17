from lgimapy.data import Database

db = Database()

accounts = ["PMCHY", "HITHY", "SICHY", "GHYUSA"]
excluded_tickers = ["OXY", "BHCCN", "F", "CNC"]

# %%
# Market Value Overweight
df_list = []
for account in accounts:
    port = db.load_portfolio(account=account)
    ticker_excluded_port = port.subset(
        ticker=excluded_tickers, special_rules="~Ticker"
    )
    ows = port.rating_overweights(by="Weight").loc[["BB", "B"]]
    df_list.append(ows.rename(account))

weight_df = pd.concat(df_list, axis=1)

# %%

# %%
# DTS Overweight
df_list = []
for account in accounts:
    port = db.load_portfolio(account=account)
    ticker_excluded_port = port.subset(
        ticker=excluded_tickers, special_rules="~Ticker"
    )
    ows = port.rating_overweights(by="DTS").loc[["BB", "B"]]
    df_list.append(ows.rename(account))

dts_df = pd.concat(df_list, axis=1)
# %%

print("By Market Value Weight:")
100 * weight_df.round(3)

print("By Beta (DTS):")
dts_df.round(2)

# %%
# Overweight just in ticker bucket
df_list = []
for account in accounts:
    port = db.load_portfolio(account=account)
    ticker_excluded_port = port.subset(ticker=excluded_tickers)
    ows = ticker_excluded_port.rating_overweights(by="Weight").loc[["BB", "B"]]
    df_list.append(ows.rename(account))

weight_df = pd.concat(df_list, axis=1)
100 * weight_df.round(3)
