from collections import defaultdict
from datetime import datetime as dt

import pandas as pd
from lgimapy.data import Database, Account
from lgimapy.latex import Document
from lgimapy.utils import load_json, root, to_list

# %%
fid_date = dt.today().strftime("%Y-%m-%d")
fid = f"PT_{fid_date}"

strategies = [
    "US Long Credit",
    "US Long Credit - Custom",
    "US Long Credit Ex Emerging Market",
    "US Long Credit Plus",
    "US Long Corporate",
    "US Long Corp 2% Cap",
    "US Long Government/Credit",
    "US Long GC 70/30",
    "US Long GC 75/25",
    "US Long GC 80/20",
    "US Credit",
    "US Credit Plus",
    "US Corporate IG",
    "US Corporate 1% Issuer Cap",
    "GM_Blend",
    "Global Agg USD Corp",
    "Custom RBS",
]
# strategies_for_thresh = ["US Long Credit", "US Long Corporate"]
strategies_for_thresh = strategies
ignored_accounts = [
    "LHNGC",
    "AILGC",
    "BEULGC",
    "SEIGC",
    "BAGC",
    "PPG",
]


# %%
strategy_accounts = load_json("strategy_accounts")
accounts = defaultdict(dict)
for strategy in strategies:
    for account in strategy_accounts[strategy]:
        if account in ignored_accounts:
            continue
        accounts[strategy][account] = db.load_portfolio(
            account=account, universe="returns"
        )

# %%
def read_tsox_df(fid_date):
    tsox_fid = root(f"data/portfolio_trades/{fid_date}/tsox.csv")
    df = pd.read_csv(tsox_fid)
    fill_cols = ["Cusip", "Side", "Long Note 1"]
    for col in fill_cols:
        df[col].fillna(method="ffill", inplace=True)
    df["Long Note 1"] = df["Long Note 1"].str.upper()
    df["Unallocated"] *= 1e3
    df = df[df["Long Note 1"].isin({"PT IN COMP", "?"})]
    cols = {
        "Account Code": "Account",
        "Cusip": "CUSIP",
        "Unallocated": "Size",
        "Side": "Side",
    }
    return (
        df.rename(columns=cols)[cols.values()]
        .dropna(subset=["Size"])
        .reset_index(drop=True)
    )


tsox_df = read_tsox_df(fid_date)
account_sell_list_d = {
    account: df.set_index("CUSIP")["Size"].to_dict()
    for account, df in tsox_df.groupby("Account")
}
account_name = "WRKLD"
acnt = accounts["US Long Credit"][account_name]
strategy = "US Credit"
# %%
def add_PT_table(
    strategy,
    accounts,
    account_sell_list_d,
    doc,
):
    account_d = accounts[strategy]
    if not account_d:
        return
    maturities = [1, 3, 5, 7, 10, 25]
    maturity_buckets = {}
    for i, rm in enumerate(maturities[1:]):
        lm = maturities[i]
        maturity_buckets[f"{lm}-{rm}y"] = (lm + 0.001, rm)
    lm = rm
    maturity_buckets[f"{lm}+"] = (lm + 0.001, None)
    d = defaultdict(list)
    sell_d = defaultdict(list)
    account_index = []
    for account_name, acnt in account_d.items():
        try:
            sell_list = account_sell_list_d[account_name]
        except KeyError:
            continue
        old_dts = acnt.dts("pct")
        old_oad = acnt.oad()

        # Simulate trades and find new DTS and OAD vs benchmark.
        df = acnt.df.copy()
        cols = ["P_MarketValue", "P_OAD", "BM_OAD", "P_DTS", "BM_DTS"]
        df[cols] = df[cols].fillna(0)
        for _, row in df.iterrows():
            cusip = row["CUSIP"]
            if cusip not in sell_list:
                continue
            sell_d["OAS"].append(row["OAS"])
            sell_pct = sell_list[cusip] / row["P_Notional"]
            keep_pct = 1 - sell_pct
            sell_d["MV"].append(sell_pct * row["P_MarketValue"])
            df.loc[df["CUSIP"] == cusip, "P_OAD"] *= keep_pct
            df.loc[df["CUSIP"] == cusip, "P_DTS"] *= keep_pct

        df["OAD_Diff"] = df["P_OAD"] - df["BM_OAD"]
        df["DTS_Diff"] = df["P_DTS"] - df["BM_DTS"]
        new_acnt = Account(df, account_name, acnt.date)
        new_dts = new_acnt.dts("pct")
        new_oad = new_acnt.oad()

        # Break down individual cusip influence on current account
        # oad_change_df = pd.DataFrame()
        # oad_change_df["Prev OAD OW"] = acnt.df.set_index("Description")[
        #     "OAD_Diff"
        # ]
        # oad_change_df["New OAD OW"] = df.set_index("Description")["OAD_Diff"]
        # oad_change_df["OAD OW chg"] = (
        #     oad_change_df["New OAD OW"] - oad_change_df["Prev OAD OW"]
        # )
        # oad_change_df.rename_axis(None).sort_values("OAD OW chg").round(3).head(
        #     10
        # )

        # Find changes.
        chg_dts = new_dts - old_dts
        chg_oad = new_oad - old_oad
        account_index.append(account_name)
        d["Curr*DTS"].append(old_dts)
        d["New*DTS"].append(new_dts)
        d["$\Delta$*DTS"].append(chg_dts)

        # Find changes to part of curve.
        for bucket, mat_kws in maturity_buckets.items():
            old_bucket_oad = acnt.subset(maturity=mat_kws).oad()
            new_bucket_oad = new_acnt.subset(maturity=mat_kws).oad()
            d[f"$\Delta$OAD*{bucket}"].append(new_bucket_oad - old_bucket_oad)

        d["Curr*OAD"].append(old_oad)
        d["New*OAD"].append(new_oad)
        d["$\Delta$*OAD"].append(chg_oad)

    df = pd.DataFrame(d, index=account_index).sort_values("$\Delta$*OAD")
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
    bm_oas = (bm_df["BM_Weight"] * bm_df["OAS"]).sum() / bm_df[
        "BM_Weight"
    ].sum()
    avg_oas = (sell_df["MV"] * sell_df["OAS"]).sum() / sell_df["MV"].sum()
    strategy_fmt = strategy.replace("%", "\%").replace("_", " ")
    cap = (
        f"{strategy_fmt} \\\\"
        f"Total Sale: \${total_sale:.0f}M \\\\"
        f"MV Weighted OAS of Sale: {avg_oas:.0f} bp "
        f"(vs {bm_oas:.0f} bp for BM)"
    )
    if len(df) % 2 == 1:
        row_colors = ("lightgray", None)
    else:
        row_colors = (None, "lightgray")
    strategy_fmt = strategy.replace("%", "\%").replace("_", " ")
    doc.add_section(strategy_fmt)
    doc.add_table(
        df,
        caption=cap,
        col_fmt="l|rrr|rrrrrr|rrr",
        font_size="footnotesize",
        prec=prec,
        multi_row_header=True,
        alternating_colors=row_colors,
        adjust=True,
    )
    doc.add_pagebreak()


# %%
pdf_fid = f"{fid}"
doc = Document(pdf_fid, path="reports/portfolio_trades")
doc.add_preamble(
    table_caption_justification="c",
    bookmarks=True,
    margin={"paperheight": 27, "top": 0.5, "bottom": 0.2},
)
for strategy in strategies:
    print(strategy)
    add_PT_table(
        strategy,
        accounts,
        account_sell_list_d,
        doc,
    )

doc.save()
