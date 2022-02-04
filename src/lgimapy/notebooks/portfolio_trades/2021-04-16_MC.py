from collections import defaultdict
from datetime import datetime as dt

import pandas as pd
from lgimapy.data import Database, Account
from lgimapy.latex import Document
from lgimapy.utils import load_json, root

# %%
strategies = [
    "US Credit",
    "US Credit Plus",
    "US Corporate IG",
    "US Corporate 1% Issuer Cap",
    "GM_Blend",
    "Global Agg USD Corp",
    "Custom RBS",
]
strategies_for_thresh = ["US Long Credit", "US Long Corporate"]
strategies_for_thresh = strategies
ignored_accounts = []
sell_amt = 1
mv_thresh = 0
date_threshold = pd.to_datetime("1/1/2025")
date = Database().date("today").strftime("%Y-%m-%d")
fid_date = dt.today().strftime("%Y-%m-%d")
fid = f"PT_MC_{fid_date}"

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
    "ADBE",
    "AMXLMM",
    "APD",
    "BALN",
    "BDX",
    "BHI",
    "BMW",
    "BPCEGP",
    "CAT",
    "CB",
    "DE",
    "DISCA",
    "DT",
    "EMR",
    "FISV",
    "FOXA",
    "GD",
    "HYNMTR",
    "ICE",
    "KO",
    "MA",
    "MARS",
    "MCD",
    "MKC",
    "NEM",
    "NESNVX",
    "NKE",
    "NSC",
    "PAA",
    "PEAK",
    "PFG",
    "PLD",
    "PM",
    "PSX",
    "SJM",
    "SPRNTS",
    "TACHEM",
    "TELEFO",
    "TRPCN",
    "TSN",
    "UPS",
    "VMW",
    "VW",
    "WSTP",
]
ix_tickers = ix.subset(ticker=tickers)
ix_tickers.df = ix_tickers.df[ix_tickers.df["MaturityDate"] >= date_threshold]
cusips_to_sell = set(ix_tickers.cusips)

# %%

# Find holdings accross all relevant accounts
p_mv = defaultdict(float)
p_notional = defaultdict(float)


for strategy in strategies_for_thresh:
    account_d = accounts[strategy]
    for acnt in account_d.values():
        acnt_mv = acnt.df.set_index("CUSIP")["P_AssetValue"].fillna(0)
        acnt_notional = acnt.df.set_index("CUSIP")["Quantity"].fillna(0)
        for cusip in cusips_to_sell:
            if cusip in acnt_mv.index:
                p_mv[cusip] += acnt_mv.loc[cusip]
                p_notional[cusip] += acnt_notional.loc[cusip]


sell_mv = {}
sell_notional = {}
for cusip, port_mv in p_mv.items():
    if port_mv <= 0:
        continue
    if port_mv > mv_thresh:
        sell_mv[cusip] = sell_amt * port_mv
        sell_notional[cusip] = sell_amt * p_notional[cusip]


# %%


def add_sell_table(strategy, accounts, p_mv, sell_amt, mv_thresh, doc):
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
    sell_d = defaultdict(list)
    for name, acnt in account_d.items():
        old_dts = acnt.dts("pct")
        old_oad = acnt.oad()

        # Simulate trades and find new DTS and OAD vs benchmark.
        df = acnt.df.copy()
        cols = ["P_AssetValue", "P_OAD", "BM_OAD", "P_DTS", "BM_DTS"]
        df[cols] = df[cols].fillna(0)
        for _, row in df.iterrows():
            cusip = row["CUSIP"]
            acnt_mv = row["P_AssetValue"]
            if cusip in cusips_to_sell and acnt_mv > 0:
                sell_d["OAS"].append(row["OAS"])
                if p_mv[cusip] > mv_thresh:
                    sell_d["MV"].append(sell_amt * acnt_mv)
                    df.loc[df["CUSIP"] == cusip, "P_OAD"] *= keep_amt
                    df.loc[df["CUSIP"] == cusip, "P_DTS"] *= keep_amt
                else:
                    sell_d["MV"].append(acnt_mv)
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
    margin={"paperheight": 25, "top": 0.5, "bottom": 0.2},
)
for strategy in strategies:
    strategy_fmt = strategy.replace("%", "\%").replace("_", " ")
    doc.add_section(strategy_fmt)
    add_sell_table(strategy, accounts, p_mv, sell_amt, mv_thresh, doc)

doc.save()


# %%
cols = ["Ticker", "CouponRate", "MaturityDate", "Issuer", "Sector"]
sell_list_df = ix.subset(cusip=sell_mv.keys()).df[cols]
sell_list_df["MV to Sell"] = pd.Series(sell_mv)
sell_list_df["Notional to Sell"] = pd.Series(sell_notional)
sell_list_df.sort_values("MV to Sell", inplace=True)
sell_list_df.to_csv(f"{fid}.csv")
