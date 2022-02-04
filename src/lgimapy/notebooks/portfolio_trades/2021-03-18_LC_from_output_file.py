from collections import defaultdict

import numpy as np
import pandas as pd
from lgimapy.data import Database, Account
from lgimapy.latex import Document
from lgimapy.utils import load_json, root

# %%
pt_fid = root("data/portfolio_trades/20210318.csv")
pt_df = pd.read_csv(pt_fid)
pt_df = pt_df.iloc[110:, :].copy()
for col in ["Cusip", "Side", "Portfolio Manager"]:
    pt_df[col] = pt_df[col].fillna(method="ffill")
pt_df = pt_df[pt_df["Portfolio Manager"] != "JTABELLIONE"].copy()
pt_df = pt_df[pt_df["Side"] == "Sell"].copy()

pt_df = (
    pt_df[["Cusip", "Account Code", "Ordered (M).1"]]
    .dropna()
    .reset_index(drop=True)
)
pt_df["Notional"] = (
    pt_df["Ordered (M).1"].str.replace("M", "000", regex=True).astype(float)
    * 1e3
)

pt_df["Account"] = pt_df["Account Code"]
pt_df["CUSIP"] = pt_df["Cusip"]

# db.load_market_data()
# ix = db.build_market_index()
# tickers = ix.subset(cusip=cusips_to_sell_old).tickers
# cusips_to_sell = set(ix.subset(ticker=tickers).cusips)


# %%


date = Database().date("today").strftime("%Y-%m-%d")
fid = f"PT_{date}"


db = Database()
account_strategies = load_json("account_strategy")
accounts = defaultdict(dict)
for account in pt_df["Account"].unique():
    if account == "NOFUND":
        continue
    strategy = account_strategies[account]
    accounts[strategy][account] = db.load_portfolio(
        account=account, universe="returns"
    )

# %%
sorted(list(accounts.keys()))


# %%

# Find holdings accross all relevant accounts
sell_notional = defaultdict(dict)
for __, row in pt_df.iterrows():
    sell_notional[row["Account"]][row["CUSIP"]] = row["Notional"]

# %%


def add_sell_table(strategy, accounts, sell_notional, doc):
    # print(strategy)
    maturities = [1, 3, 5, 7, 10, 25]
    maturity_buckets = {}
    for i, rm in enumerate(maturities[1:]):
        lm = maturities[i]
        maturity_buckets[f"{lm}-{rm}y"] = (lm + 0.001, rm)
    lm = rm
    maturity_buckets[f"{lm}+"] = (lm + 0.001, None)

    account_d = accounts[strategy]
    d = defaultdict(list)
    sell_d = defaultdict(list)
    for account_name, acnt in account_d.items():
        # print(account_name)
        account_sell_notional = sell_notional[account_name]
        old_dts = acnt.dts("pct")
        old_oad = acnt.oad()

        # Simulate trades and find new DTS and OAD vs benchmark.
        df = acnt.df.copy()
        cols = [
            "P_AssetValue",
            "P_OAD",
            "BM_OAD",
            "P_DTS",
            "BM_DTS",
            "Quantity",
        ]
        df[cols] = df[cols].fillna(0)
        for _, row in df.iterrows():
            cusip = row["CUSIP"]
            if cusip in account_sell_notional:
                acnt_notional = row["Quantity"]
                acnt_mv = row["P_AssetValue"]
                try:
                    sell_pct = account_sell_notional[cusip] / acnt_notional
                except ZeroDivisionError:
                    sell_pct = 0
                keep_pct = 1 - sell_pct
                sell_d["OAS"].append(row["OAS"])
                sell_d["MV"].append(sell_pct * acnt_mv)
                df.loc[df["CUSIP"] == cusip, "P_OAD"] *= keep_pct
                df.loc[df["CUSIP"] == cusip, "P_DTS"] *= keep_pct

        df["OAD_Diff"] = df["P_OAD"] - df["BM_OAD"]
        df["DTS_Diff"] = df["P_DTS"] - df["BM_DTS"]

        new_acnt = Account(df, account_name, acnt.date)
        new_dts = new_acnt.dts("pct")
        new_oad = new_acnt.oad()

        # Find changes.
        chg_dts = new_dts - old_dts
        chg_oad = new_oad - old_oad

        d["Curr*DTS"].append(old_dts)
        d["New*DTS"].append(new_dts)
        d["$\Delta$*DTS"].append(chg_dts)
        d["Curr*OAD"].append(old_oad)
        d["New*OAD"].append(new_oad)
        d["$\Delta$*OAD"].append(chg_oad)

        # Find changes to part of curve.
        if "Long" in strategy or "LIB" in strategy:
            col_fmt = "l|rrr|rrr"
            adjust = False
        else:
            col_fmt = "l|rrr|rrr|rrrrrr"
            adjust = True
            for bucket, mat_kws in maturity_buckets.items():
                old_bucket_oad = acnt.subset(maturity=mat_kws).oad()
                new_bucket_oad = new_acnt.subset(maturity=mat_kws).oad()
                d[f"$\Delta$OAD*{bucket}"].append(
                    new_bucket_oad - old_bucket_oad
                )

    df = pd.DataFrame(d, index=account_d.keys()).sort_values("$\Delta$*OAD")
    prec = {}
    for col in df.columns:
        if "DTS" in col:
            prec[col] = "1%"
        elif "OAD" in col:
            prec[col] = "2f"

    sell_df = pd.DataFrame(sell_d)
    if not len(sell_df):
        return
    total_sale = sell_df["MV"].sum() / 1e6
    bm_df = acnt.df[acnt.df["BM_Weight"] > 0]
    try:
        bm_oas = (bm_df["BM_Weight"] * bm_df["OAS"]).sum() / bm_df[
            "BM_Weight"
        ].sum()
    except ZeroDivisionError:
        bm_oas = np.nan
    avg_oas = (sell_df["MV"] * sell_df["OAS"]).sum() / sell_df["MV"].sum()
    strategy_fmt = strategy.replace("%", "\%").replace("_", " ")
    cap = (
        f"{strategy_fmt} \\\\"
        f"Total Sale: \${total_sale:.2f}M \\\\"
        f"MV Weighted OAS of Sale: {avg_oas:.0f} bp "
        f"(vs {bm_oas:.0f} bp for BM)"
    )
    if len(df) % 2 == 1:
        row_colors = ("lightgray", None)
    else:
        row_colors = (None, "lightgray")
    doc.add_table(
        df,
        caption=cap,
        col_fmt=col_fmt,
        font_size="footnotesize",
        prec=prec,
        multi_row_header=True,
        alternating_colors=row_colors,
        adjust=adjust,
    )
    doc.add_pagebreak()


# %%
doc = Document(fid, path="reports/portfolio_trades")
doc.add_preamble(
    table_caption_justification="c",
    margin={"paperheight": 32, "top": 0.5, "bottom": 0.2},
)
ignored_strategies = {
    "US Strips 15+ Yr",
    "US Strips 20+ Yr",
    "US Treasury Long",
}
for strategy in sorted(accounts.keys()):
    if strategy in ignored_strategies:
        continue
    add_sell_table(strategy, accounts, sell_notional, doc)

doc.save()


# %%
# doc = Document(fid, path="reports/portfolio_trades")
# cols = ["Ticker", "CouponRate", "MaturityDate", "Issuer", "Sector"]
# sell_list_df = ix.subset(cusip=sell_mv.keys()).df[cols]
# sell_list_df["MV to Sell"] = pd.Series(sell_mv)
# sell_list_df["Notional to Sell"] = pd.Series(sell_notional)
# sell_list_df.sort_values("MV to Sell", inplace=True)
