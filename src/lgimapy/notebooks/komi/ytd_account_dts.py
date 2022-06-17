import pandas as pd

from lgimapy.data import Database

# %%
db = Database()
start = db.date("YTD")

accounts = ["FLD", "CHLD", "GTMC", "CITMC"]
strategies = ["US Credit", "US Long Credit"]

dts_df_list = []
for account in accounts:
    port = db.load_portfolio(account, empty=True)
    port_dts = port.stored_properties_history_df["dts_pct"].round(2)
    dts_df_list.append(port_dts.rename(account))
dts_df = pd.concat(dts_df_list, axis=1) / 100

oas_df_list = []
for strategy in strategies:
    port = db.load_portfolio(strategy, empty=True)
    bm_oas = port.stored_properties_history_df["bm_oas"].round(0)
    oas_df_list.append(bm_oas.rename(f"{strategy} OAS"))

oas_df = pd.concat(oas_df_list, axis=1)

df = pd.concat((dts_df, oas_df), axis=1)
df = df[df.index >= start]

df.to_csv("YTD_account_DTS.csv")
