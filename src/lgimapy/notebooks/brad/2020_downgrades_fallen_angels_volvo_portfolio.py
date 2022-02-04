from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database

# %%

db = Database()
start = "1/1/2020"
db.load_market_data(start=start)


# %%
ix_ret = db.build_market_index(in_returns_index=True)
maturity_d = {
    "IG": None,
    "IG_0-10yr": (0, 10),
    "IG_10+yr": (10, 999),
}

d = defaultdict(list)
for name, matuirty_bounds in maturity_d.items():
    if matuirty_bounds is None:
        ix = ix_ret.copy()
    else:
        ix = ix_ret.subset(maturity=matuirty_bounds)

    mv = ix.total_value()

    for analysis in ["fallen_angels", "downgrades"]:
        if analysis == "fallen_angels":
            df = db.rating_changes(start=start, fallen_angels=True)
        else:
            df = db.rating_changes(start=start)
            df = df[df["NumericRating_CHANGE"] < 0]
        df = df[df["USCreditReturnsFlag"] == True]
        if matuirty_bounds is not None:
            df = df[
                (df["MaturityYears"] >= matuirty_bounds[0])
                & (df["MaturityYears"] <= matuirty_bounds[1])
            ]
        if analysis == "fallen_angels":
            notional = np.sum(df["AmountOutstanding"])
            pct = np.sum(df.set_index("Date_PREV")["AmountOutstanding"] / mv)
        else:
            notional_df = df.set_index("Date_PREV")
            notional_df["downgrades"] = notional_df[
                "AmountOutstanding"
            ] * np.abs(notional_df["NumericRating_CHANGE"])
            notional_df["downgrades"]
            notional = np.sum(notional_df["downgrades"])
            pct = np.sum(notional_df["downgrades"] / mv)

        d["Universe"].append(f"{name}_{analysis}")
        d["# Issuers"].append(len(df["Ticker"].unique()))
        d["Notional ($B)"].append(int(notional / 1e3))
        d["Percent of Universe"].append(np.round(pct * 100, 2))


df = pd.DataFrame(d)
df.to_csv("Fallen_Angels_Downgrades_IG_Universe.csv")

# %%

account_name = "CITLD"

cols = [
    "Date",
    "CUSIP",
    "Description",
    "Ticker",
    "Issuer",
    "P_AssetValue",
    "P_Weight",
]
col_names = {
    "P_AssetValue": "Portfolio Asset Value ($)",
    "P_Weight": "Portfolio Weight (%)",
}

fallen_angel_df_list = []
downgrade_df_list = []
dates = db.trade_dates(start=start)
for date in tqdm(dates):
    acnt = db.load_portfolio(account=account_name, date=date)
    p_df = acnt.df[acnt.df["P_Weight"] > 0].copy()
    p_df["P_Weight"] *= 100

    all_fallen_angel_df = db.rating_changes(
        start=date, end=date, fallen_angels=True
    )
    rating_df = db.rating_changes(start=date, end=date)
    all_downgrade_df = rating_df[rating_df["NumericRating_CHANGE"] < 0]

    p_downgrade_df = p_df[p_df["ISIN"].isin(all_downgrade_df["ISIN"])]
    p_fallen_angel_df = p_df[p_df["ISIN"].isin(all_fallen_angel_df["ISIN"])]

    downgrade_df_list.append(p_downgrade_df[cols].rename(columns=col_names))
    fallen_angel_df_list.append(
        p_fallen_angel_df[cols].rename(columns=col_names)
    )

downgrades = pd.concat(downgrade_df_list).reset_index(drop=True)
fallen_angels = pd.concat(fallen_angel_df_list).reset_index(drop=True)
downgrades.to_csv(f"{account_name}_downgrades.csv")
fallen_angels.to_csv(f"{account_name}_fallen_angels.csv")
