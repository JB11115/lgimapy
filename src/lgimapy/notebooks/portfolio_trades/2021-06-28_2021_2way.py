from collections import defaultdict
from datetime import datetime as dt

import pandas as pd
from lgimapy.data import Database, Account
from lgimapy.latex import Document
from lgimapy.utils import load_json, root

# %%
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
]
# strategies_for_thresh = ["US Long Credit", "US Long Corporate"]
strategies_for_thresh = strategies
ignored_accounts = ["AXPLD", "SEMLD"]
sell_amt = 1
mv_thresh = 0
fid_date = dt.today().strftime("%Y-%m-%d")
fid = f"PT_LD_{fid_date}"
buy_fid = f"PT_LD_Buys_{fid_date}.csv"


db = Database()
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
db.load_market_data()
ix = db.build_market_index()
tickers = [
    "APD",
    "BDX",
    "CNA",
    "CPB",
    "DGELN",
    "DISCA",
    "EOG",
    "ETN",
    "FISV",
    "FOXA",
    "JNJ",
    "LLY",
    "LOW",
    "MA",
    "MARS",
    "MCD",
    "MDLZ",
    "MDT",
    "MKL",
    "NKE",
    "NNN",
    "PEG",
    "PH",
    "PNC",
    "PXD",
    "SANTAN",
    "SBUX",
    "SYK",
    "TACHEM",
    "TELEFO",
    "TFC",
    "UNANA",
    "UPS",
    "VMC",
    "VTRS",
]

cusips_to_sell = set(ix.subset(ticker=tickers).cusips)
cusip_sell_notional = {
    "86765BAV1": 25,
    "55336VBP4": 20.766,
    "96950FAP9": 24.352,
    "125523CQ1": 25,
    "125523CK4": 25,
}
cusip_sell_notional = {k: v * 1e6 for k, v in cusip_sell_notional.items()}

buys_df = pd.read_csv(root(f"data/portfolio_trades/{buy_fid}"), index_col=0)
cusips_to_buy = buys_df.set_index("Cusip")[" MV to Buy "].to_dict()

# %%

# Find holdings accross all relevant accounts
p_mv = defaultdict(float)
p_notional = defaultdict(float)
total_credit_mv = 0
for strategy in strategies_for_thresh:
    account_d = accounts[strategy]
    for acnt in account_d.values():
        total_credit_mv += acnt.df["P_AssetValue"].sum()
        acnt_mv_df = acnt.df.set_index("CUSIP")["P_AssetValue"].fillna(0)
        acnt_notional = acnt.df.set_index("CUSIP")["Quantity"].fillna(0)
        for cusip_list in [cusips_to_sell, cusip_sell_notional]:
            for cusip in cusip_list:
                if cusip in acnt_mv_df.index:
                    p_mv[cusip] += acnt_mv_df.loc[cusip]
                    p_notional[cusip] += acnt_notional.loc[cusip]

cusips_to_sell |= set(cusip_sell_notional)

sell_mv = {}
sell_notional = {}
for cusip, port_mv in p_mv.items():
    if port_mv <= 0:
        continue
    if cusip in cusip_sell_notional:
        sell_notional[cusip] = cusip_sell_notional[cusip]
        sell_mv[cusip] = sell_notional[cusip] * p_mv[cusip] / p_notional[cusip]
    elif port_mv > mv_thresh:
        sell_mv[cusip] = sell_amt * port_mv
        sell_notional[cusip] = sell_amt * p_notional[cusip]


# %%
for c, n in cusip_sell_notional.items():
    p_n = p_notional[c]
    print(f"{c}: ${p_n/1e6:.3f}M, {n / p_n:.1%}")

# %%

# %%
def add_sell_table(
    strategy,
    accounts,
    total_credit_mv,
    p_mv,
    p_notional,
    sell_notional,
    sell_amt,
    mv_thresh,
    cusips_to_sell,
    cusip_sell_notional,
    cusips_to_buy,
    doc,
):
    account_d = accounts[strategy]
    keep_amt = 1 - sell_amt
    maturities = [1, 3, 5, 7, 10, 25]
    maturity_buckets = {}
    for i, rm in enumerate(maturities[1:]):
        lm = maturities[i]
        maturity_buckets[f"{lm}-{rm}y"] = (lm + 0.001, rm)
    lm = rm
    maturity_buckets[f"{lm}+"] = (lm + 0.001, None)
    d = defaultdict(list)
    buy_d = defaultdict(list)
    sell_d = defaultdict(list)
    for name, acnt in account_d.items():
        old_dts = acnt.dts("pct")
        old_oad = acnt.oad()

        # Simulate trades and find new DTS and OAD vs benchmark.
        df = acnt.df.copy()
        acnt_credit_mv = df["P_AssetValue"].sum()
        cols = ["P_AssetValue", "P_OAD", "BM_OAD", "P_DTS", "BM_DTS"]
        df[cols] = df[cols].fillna(0)
        for _, row in df.iterrows():
            cusip = row["CUSIP"]
            acnt_cusip_mv = row["P_AssetValue"]
            if cusip in cusips_to_buy:
                buy_d["OAS"].append(row["OAS"])
                buy_pct = acnt_credit_mv / total_credit_mv
                buy_amt = buy_pct * cusips_to_buy[cusip]
                buy_d["MV"].append(buy_amt)

            elif cusip in cusips_to_sell and acnt_cusip_mv > 0:
                sell_d["OAS"].append(row["OAS"])
                if cusip in cusip_sell_notional:
                    cusip_sell_amt = (
                        cusip_sell_notional[cusip] / p_notional[cusip]
                    )
                    cusip_keep_amt = 1 - cusip_sell_amt
                    sell_d["MV"].append(cusip_sell_amt * acnt_cusip_mv)
                    df.loc[df["CUSIP"] == cusip, "P_OAD"] *= cusip_keep_amt
                    df.loc[df["CUSIP"] == cusip, "P_DTS"] *= cusip_keep_amt

                elif p_mv[cusip] > mv_thresh:
                    sell_d["MV"].append(sell_amt * acnt_cusip_mv)
                    df.loc[df["CUSIP"] == cusip, "P_OAD"] *= keep_amt
                    df.loc[df["CUSIP"] == cusip, "P_DTS"] *= keep_amt
                else:
                    sell_d["MV"].append(acnt_cusip_mv)
                    df.loc[df["CUSIP"] == cusip, "P_OAD"] = 0
                    df.loc[df["CUSIP"] == cusip, "P_DTS"] = 0
        df["OAD_Diff"] = df["P_OAD"] - df["BM_OAD"]
        df["DTS_Diff"] = df["P_DTS"] - df["BM_DTS"]

        new_acnt = Account(df, name, acnt.date)
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
        for bucket, mat_kws in maturity_buckets.items():
            old_bucket_oad = acnt.subset(maturity=mat_kws).oad()
            new_bucket_oad = new_acnt.subset(maturity=mat_kws).oad()
            d[f"$\Delta$OAD*{bucket}"].append(new_bucket_oad - old_bucket_oad)

    df = pd.DataFrame(d, index=account_d.keys()).sort_values("$\Delta$*OAD")
    prec = {}
    for col in df.columns:
        if "DTS" in col:
            prec[col] = "1%"
        elif "OAD" in col:
            prec[col] = "2f"

    sell_df = pd.DataFrame(sell_d)
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
    doc.add_table(
        df,
        caption=cap,
        col_fmt="l|rrr|rrr|rrrrrr",
        font_size="footnotesize",
        prec=prec,
        multi_row_header=True,
        alternating_colors=row_colors,
        adjust=True,
    )
    doc.add_pagebreak()


# %%

doc = Document(fid, path="reports/portfolio_trades")
doc.add_preamble(
    table_caption_justification="c",
    bookmarks=True,
    margin={"paperheight": 27, "top": 0.5, "bottom": 0.2},
)
for strategy in strategies:
    strategy_fmt = strategy.replace("%", "\%").replace("_", " ")
    doc.add_section(strategy_fmt)
    add_sell_table(
        strategy,
        accounts,
        total_credit_mv,
        p_mv,
        p_notional,
        sell_notional,
        sell_amt,
        mv_thresh,
        cusips_to_sell,
        cusip_sell_notional,
        cusips_to_buy,
        doc,
    )

doc.save()


# %%
cols = ["Ticker", "CouponRate", "MaturityDate", "Issuer", "Sector"]
sell_list_df = ix.subset(cusip=sell_mv.keys()).df[cols]
sell_list_df["MV to Sell"] = pd.Series(sell_mv)
sell_list_df["Notional to Sell"] = pd.Series(sell_notional)
sell_list_df.sort_values("MV to Sell", inplace=True)
sell_list_df.to_csv(f"{fid}.csv")
print(f"Total Sale: ${sell_list_df['MV to Sell'].sum() / 1e6:.0f}")

# %%
df_list = []

for account, port in accounts["US Long Credit"].items():
    df_list.append(
        port.df[port.df["Ticker"].isin(tickers)][["Ticker", "P_AssetValue"]]
        .groupby("Ticker", observed=True)
        .sum()
        .squeeze()
        .rename(account)
    )
df_strat = pd.concat(df_list, axis=1).sum(axis=1) / 1e6
df_strat.round(1)
