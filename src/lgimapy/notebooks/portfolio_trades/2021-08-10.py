from collections import defaultdict
from datetime import datetime as dt

import pandas as pd
from lgimapy.data import Database, Account
from lgimapy.latex import Document
from lgimapy.utils import load_json, root, to_list

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
sell_amt = 1
mv_thresh = 0
sell_limit = None

fid_date = dt.today().strftime("%Y-%m-%d")
fid = f"PT_{fid_date}"


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
    "RCICN",
    "DT",
    "MLM",
    "FDX",
    "GD",
    "LMT",
    "NVDA",
    "SVELEV",
    "KDP",
    "MRK",
    "BIIB",
    "PM",
    "WSTP",
    "CBAAU",
    "WLTW",
    "WPC",
    "PSX",
    "MPC",
    "ABXCN",
    "INDOIS",
    "AIA",
]
cusips = [
    # "337358BA2",
    # "337358BH7",
    # "8447HBAE2",
    # "92976GAG6",
    # "92976GAJ0",
    # "929771AE3",
    # "929903AM4",
    # "949746RE3",
    # "949746RF0",
    # "94974BFJ4",
    # "94974BFN5",
    # "94974BFP0",
    # "94974BFY1",
    # "94974BGE4",
    # "94974BGL8",
    # "94974BGQ7",
    # "94974BGT1",
    # "94974BGU8",
    # "94980VAG3",
    # "95000U2M4",  # WFC 51
    "084664CR0",  # BRK
    # "68233JAT1",  # ONCRTX
]
account_specific_cusips = {
    "BACLGC": "68233JAT1",
    "BEULGC": "68233JAT1",
    "CMAGC": "68233JAT1",
    "HRLGC": "68233JAT1",
    "KOLGC": "68233JAT1",
    "P-LGC": "68233JAT1",
    "USBGC": "68233JAT1",
}
cusips_to_sell = set(ix.subset(ticker=tickers).cusips) | set(cusips)
specific_cusip_sell_notionals = {}
specific_cusip_sell_notionals = {
    k: v * 1e6 for k, v in specific_cusip_sell_notionals.items()
}

# %%

# Find holdings accross all relevant accounts
p_mv = defaultdict(float)
p_notional = defaultdict(float)
for strategy in strategies_for_thresh:
    account_d = accounts[strategy]
    for account_name, acnt in account_d.items():
        acnt_mv = acnt.df.set_index("CUSIP")["P_MarketValue"].fillna(0)
        acnt_notional = acnt.df.set_index("CUSIP")["P_Notional"].fillna(0)
        for cusip_list in [cusips_to_sell, specific_cusip_sell_notionals]:
            for cusip in cusip_list:
                if cusip in acnt_mv.index:
                    p_mv[cusip] += acnt_mv.loc[cusip]
                    p_notional[cusip] += acnt_notional.loc[cusip]
        if account_name in account_specific_cusips:
            current_acnt_specific_cusips = to_list(
                account_specific_cusips[account_name], dtype=str
            )
            for cusip in current_acnt_specific_cusips:
                if cusip in acnt_mv.index:
                    p_mv[cusip] += acnt_mv.loc[cusip]
                    p_notional[cusip] += acnt_notional.loc[cusip]

cusips_to_sell |= set(specific_cusip_sell_notionals)
sell_mv = {}
sell_notional = {}
for cusip, port_mv in p_mv.items():
    if port_mv <= 0:
        continue
    if cusip in specific_cusip_sell_notionals:
        sell_notional[cusip] = specific_cusip_sell_notionals[cusip]
        sell_mv[cusip] = sell_notional[cusip] * p_mv[cusip] / p_notional[cusip]
    elif port_mv > mv_thresh:
        sell_mv[cusip] = sell_amt * port_mv
        sell_notional[cusip] = sell_amt * p_notional[cusip]

if sell_limit is not None:
    sell_notional = {
        k: v for k, v in sell_notional.items() if v < sell_limit * 1e3
    }

# %%
# for c, n in specific_cusip_sell_notionals.items():
#     p_n = p_notional[c]
#     print(f"{c}: ${p_n/1e6:.3f}M, {n / p_n:.1%}")


# %%

# buys_df = pd.read_csv(root(f"data/portfolio_trades/{buy_fid}"), index_col=0)
# buys_df
# %%
def add_PT_table(
    strategy,
    accounts,
    p_mv,
    p_notional,
    sell_notional,
    sell_amt,
    mv_thresh,
    cusips_to_sell,
    cusip_sell_notional,
    account_specific_cusips,
    doc,
):

    account_d = accounts[strategy]
    if not account_d:
        return
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
    account_specific_cusips
    for account_name, acnt in account_d.items():
        old_dts = acnt.dts("pct")
        old_oad = acnt.oad()
        current_acnt_specific_cusips = to_list(
            account_specific_cusips.get(account_name, []), dtype=str
        )

        # Simulate trades and find new DTS and OAD vs benchmark.
        df = acnt.df.copy()
        cols = ["P_MarketValue", "P_OAD", "BM_OAD", "P_DTS", "BM_DTS"]
        df[cols] = df[cols].fillna(0)
        for _, row in df.iterrows():
            cusip = row["CUSIP"]
            acnt_mv = row["P_MarketValue"]
            if (
                cusip in cusips_to_sell or cusip in current_acnt_specific_cusips
            ) and acnt_mv > 0:

                sell_d["OAS"].append(row["OAS"])
                if cusip in cusip_sell_notional:
                    cusip_sell_amt = (
                        cusip_sell_notional[cusip] / p_notional[cusip]
                    )
                    cusip_keep_amt = 1 - cusip_sell_amt
                    sell_d["MV"].append(cusip_sell_amt * acnt_mv)
                    df.loc[df["CUSIP"] == cusip, "P_OAD"] *= cusip_keep_amt
                    df.loc[df["CUSIP"] == cusip, "P_DTS"] *= cusip_keep_amt

                elif p_mv[cusip] > mv_thresh:
                    sell_d["MV"].append(sell_amt * acnt_mv)
                    df.loc[df["CUSIP"] == cusip, "P_OAD"] *= keep_amt
                    df.loc[df["CUSIP"] == cusip, "P_DTS"] *= keep_amt
                else:
                    sell_d["MV"].append(acnt_mv)
                    df.loc[df["CUSIP"] == cusip, "P_OAD"] = 0
                    df.loc[df["CUSIP"] == cusip, "P_DTS"] = 0
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
pdf_fid = f"{fid}_ex_WFC"
doc = Document(pdf_fid, path="reports/portfolio_trades")
doc.add_preamble(
    table_caption_justification="c",
    bookmarks=True,
    margin={"paperheight": 27, "top": 0.5, "bottom": 0.2},
)
for strategy in strategies:
    strategy_fmt = strategy.replace("%", "\%").replace("_", " ")
    doc.add_section(strategy_fmt)
    print(strategy_fmt)
    add_PT_table(
        strategy,
        accounts,
        p_mv,
        p_notional,
        sell_notional,
        sell_amt,
        mv_thresh,
        cusips_to_sell,
        specific_cusip_sell_notionals,
        account_specific_cusips,
        doc,
    )

doc.save()


# %%
cols = ["Ticker", "CouponRate", "MaturityDate", "Issuer", "Sector"]
sell_list_df = ix.subset(cusip=sell_mv.keys()).df[cols]
sell_list_df["MV to Sell"] = pd.Series(sell_mv)
sell_list_df["Notional to Sell"] = pd.Series(sell_notional)
sell_list_df.sort_values("MV to Sell", inplace=True)
sell_list_df.to_csv(f"{pdf_fid}.csv")
print(f"Total MV Sale: ${sell_list_df['MV to Sell'].sum() / 1e6:.0f}")
print(
    f"Total Notional Sale: ${sell_list_df['Notional to Sell'].sum() / 1e6:.0f}"
)

# %%
(
    sell_list_df.groupby("Ticker", observed=True).sum()["MV to Sell"] / 1e6
).sort_index()
