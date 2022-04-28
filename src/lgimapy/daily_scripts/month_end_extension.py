from collections import defaultdict
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import load_json, root, pprint
from lgimapy.latex import Document, drop_consecutive_duplicates

# %%
def tendered_cusips():
    return None
    # return {
    #     "GE": {
    #         "36962G6F6": 0.25,
    #         "369604BD4": 0.35,
    #         "36962G6S8": 0.25,
    #         "369604BG7": 0.25,
    #         "36962G7K4": 0.25,
    #         "36166NAG8": 0.50,
    #         "36164Q6M5": 0.65,
    #         "36962GT95": 0.50,
    #         "369604BV4": 0.50,
    #         "36166NAH6": 0.50,
    #         "81413PAG0": 0.25,
    #         "369604BW2": 0.60,
    #         "36166NAJ2": 0.60,
    #         "36962GXZ2": 0.40,
    #         "36166NAK9": 0.60,
    #         "36962G3A0": 0.25,
    #         "36962G3P7": 0.35,
    #         "36962G4B7": 0.35,
    #         "369604BX0": 0.40,
    #         "369604BF9": 0.25,
    #         "369604BH5": 0.25,
    #         "369604BY8": 0.30,
    #         "36164QNA2": 0.50,
    #     },
    # }


# %%
def build_month_end_extensions_report():
    # %%
    db = Database()
    db.load_market_data()

    strategy_accounts = db._strategy_account_map()
    account_strategy = db._account_strategy_map()
    manager_accounts = get_PM_accounts(db)
    account_manager = {}
    for manager, accounts in manager_accounts.items():
        for account in accounts:
            account_manager[account] = manager

    strategies_with_no_benchmark = {
        "LDI No Benchmark",
        "Custom Management No Benchmark",
        "Custom Management No Benchmark - MDLZSTRP",
        "Custom BNM - MBRF",
        "Absolute Return - LIB150",
        "Absolute Return LIB350",
        "US High Yield",
        "Global Corporate IBoxx USD",
    }

    d = defaultdict(list)
    large_ticker_deltas_list = []
    for strategy, accounts in tqdm(strategy_accounts.items()):
        # if strategy == "US Long GC 75/25":
        #     break
        # else:
        #     continue
        if strategy in strategies_with_no_benchmark:
            continue

        strat = db.load_portfolio(strategy=strategy, empty=True)
        for account in accounts:
            kwargs = {"account": account, "ret_df": True, "market_cols": False}
            try:
                ret_df = db.load_portfolio(universe="returns", **kwargs)
                stats_df = db.load_portfolio(universe="stats", **kwargs)
            except ValueError:
                # except (ValueError, pd.io.sql.DatabaseError):
                continue
            else:
                pm = account_manager[account]
                next_oad, next_dts = port_metrics(stats_df, strategy)
                curr_oad, curr_dts = port_metrics(ret_df)

                d["PM"].append(pm)
                d["Strategy"].append(strategy.replace("_", " "))
                d["last_name"].append(pm.split()[1])
                d["$\Delta$OAD"].append(next_oad - curr_oad)
                d["$\Delta$DTS"].append(next_dts / curr_dts - 1)
                if strat.is_GC():
                    next_credit_oad, _ = port_metrics(
                        stats_df, strategy, credit=True
                    )
                    curr_credit_oad, _ = port_metrics(ret_df, credit=True)
                    d["Credit*$\Delta$OAD"].append(
                        next_credit_oad - curr_credit_oad
                    )
                else:
                    d["Credit*$\Delta$OAD"].append(np.nan)

                large_ticker_deltas_df = inspect_extension(ret_df, stats_df)
                if len(large_ticker_deltas_df):
                    large_ticker_deltas_df["Strategy"] = strategy
                    large_ticker_deltas_list.append(large_ticker_deltas_df)
                break

    ext_df = (
        pd.DataFrame(d)
        .sort_values(["last_name", "Strategy"])
        .drop("last_name", axis=1)
        .reset_index(drop=True)
    )

    # Dont duplicate names.
    vals = ext_df["PM"]
    prev = vals[0]
    unique_vals = [prev]
    for val in vals[1:]:
        if val == prev:
            unique_vals.append(" ")
        else:
            unique_vals.append(val)
            prev = val
    ext_df["PM"] = unique_vals

    midrule_locs = [i for i, v in enumerate(unique_vals) if v != " "][1:]
    today = dt.today()
    fid = f"month_end_extensions_{today:%Y-%m-%d}"
    # %%
    doc = Document(fid, path="reports/month_end_extensions")
    doc.add_preamble(
        ignore_bottom_margin=True,
        margin={"left": 0.5, "right": 0.5, "top": 0.5, "bottom": 0.2,},
        header=doc.header(
            left="Month End Extensions", right=today.strftime("%B %Y"),
        ),
        footer=doc.footer(logo="LG_umbrella"),
        table_caption_justification="c",
    )
    doc.add_table(
        ext_df,
        prec={
            "$\Delta$OAD": "+2f",
            "$\Delta$DTS": "+2%",
            "Credit*$\Delta$OAD": "+2f",
        },
        hide_index=True,
        col_fmt="llrr|r|r",
        multi_row_header=True,
        midrule_locs=midrule_locs,
        alternating_colors=(None, "lightgray"),
    )
    doc.add_pagebreak()
    # Format ticker change table.
    if len(large_ticker_deltas_list):
        large_ticker_deltas = pd.concat(large_ticker_deltas_list)
        large_ticker_deltas["Ticker"] = large_ticker_deltas.index
        cols = ["Strategy", "Ticker", "Returns*CTD", "Stats*CTD", "$\Delta$CTD"]
        ticker_deltas_table = large_ticker_deltas[cols]
        ticker_deltas_table["Strategy"] = drop_consecutive_duplicates(
            ticker_deltas_table["Strategy"]
        )
        ticker_deltas_table["Strategy"] = ticker_deltas_table[
            "Strategy"
        ].str.replace("_", " ")
        midrule_locs = [
            i for i, v in enumerate(ticker_deltas_table["Strategy"]) if v != " "
        ][1:]
        prec = {}
        for col in ticker_deltas_table.columns:
            if "Delta" in col:
                prec[col] = "+2f"
            elif "CTD" in col:
                prec[col] = "2f"

        doc.add_table(
            ticker_deltas_table,
            caption="Large Ticker changes in CTD",
            prec=prec,
            multi_row_header=True,
            midrule_locs=midrule_locs,
            hide_index=True,
            col_fmt="lll|cc|c",
        )
    doc.save()
    tenders = tendered_cusips()
    if tenders is not None:
        tendered_tickers = "_".join(list(tenders.keys()))
        save_as_fid = f"Month_End_Extensions_w_{tendered_tickers}_tenders"
    else:
        save_as_fid = "Month_End_Extensions"
    doc.save_as(save_as_fid, path="reports/current_reports")

    # %%


def get_PM_accounts(db):
    sql = """\
        SELECT\
            BloombergId,\
            PrimaryPortfolioManager,\
            s.[Name] as PMStrategy\
        FROM [LGIMADatamart].[dbo].[DimAccount] a (NOLOCK)\
        INNER JOIN LGIMADatamart.dbo.DimStrategy s (NOLOCK)\
            ON a.StrategyKey = s.StrategyKey\
        WHERE a.DateEnd = '9999-12-31'\
            AND a.DateClose IS NULL\
        ORDER BY BloombergID\
        """
    df = db.query_datamart(sql)
    df.columns = ["account", "manager", "strategy"]
    return {
        pm: list(df[df["manager"] == pm]["account"])
        for pm in df["manager"].unique()
    }


def port_metrics(df, strategy=None, credit=False):
    """
    Perform special rules for specifying return index
    for next month from current stats index, and return
    OAD and DTS for benchmark portfolio.
    """
    if strategy is None:
        if credit:
            df_credit = df[df["Sector"] != "TREASURIES"]
            bm_oad = df_credit["BM_OAD"].sum()
        else:
            bm_oad = df["BM_OAD"].sum()
        bm_dts = (df["BM_OASD"] * df["OAS"]).sum()
        return bm_oad, bm_dts

    minimum_tenors_d = {
        "US Credit ex Subordinated 5+": 5,
        "Liability Aware Long Duration Credit": 10,
        "US Government: Long": 10,
        "US Long Credit": 10,
        "P-LD": 10,
        "US Long Credit Plus": 10,
        "US Long Credit - Custom": 10,
        "US Government": 1,
        "US Strips 20+ Yr": 20,
        "US Credit": 1,
        "US Credit Plus": 1,
        "US Agg Securitized Passive": 1,
        "US Long GC 70/30": 10,
        "US Long Government/Credit": 10,
        "US Long GC 75/25": 10,
        "US Intermediate Credit A or better": 1,
        "US Long Corporate": 10,
        "US Government: Intermediate": 1,
        "US Long Corp 2% Cap": 10,
        "US Treasury TIPS 15+": 20,
        "US Strips 15+ Yr": 15,
        "US Long Credit Ex Emerging Market": 10,
        "US Long Corporate A or better": 10,
        "US Intermediate Credit": 1,
        "BNM - US Long A+ Credit": 10,
        "Barclays-Russell LDI Custom - DE": 1,
        "80% US Credit/20% 7-10 yr Treasury": 1,
        "US Treasury 9+ Yr Custom Weighted": 9,
        "GM_Blend": 5,
        "US Treasury Long": 11,
        "Global Agg USD Corp": 1,
        "INKA": 7,
        "80% US A or Better LC/20% US BBB LC": 10,
        "Global Agg USD Securitized Passive": 1,
        "US Corporate IG": 1,
        "US Corporate 10+ Yr >500MM Ex Insur": 10,
        "US Corporate 1% Issuer Cap": 1,
        "Intermediate TIPS": 1,
        "OLIN CUSTOM STRIPS - OPEB": 0,
        "OLIN CUSTOM STRIPS - Pension": 0,
        "US Long GC 80/20": 10,
        "RBS BNMs": 1,
        "Custom RBS": 1,
        "US Credit A or better": 1,
        "US Long A+ Credit": 10,
        "US Corporate 7+ Years": 7,
        "BofA ML 10+ Year AAA-A US Corp Const": 10,
        "Bloomberg LDI Custom - DE": 3,
        "BNM - US 1-5 Yr Credit": 1,
        "BNM - ICE BofA US Non-Financial Index": 1,
    }
    tenders = tendered_cusips()
    if tenders is not None:
        for ticker, ticker_tenders in tenders.items():
            for cusip, tender_amt in ticker_tenders.items():
                df.loc[df["CUSIP"] == cusip, "BM_Weight"] *= 1 - tender_amt

    df["DataMart_OAD"] = df["BM_OAD"] / df["BM_Weight"]
    df["DataMart_OASD"] = df["BM_OASD"] / df["BM_Weight"]
    df["MaturityMonth"] = pd.to_datetime(
        df["MaturityDate"], errors="coerce"
    ).dt.to_period("M")
    try:
        minimum_tenor = minimum_tenors_d[strategy]
    except KeyError:
        raise KeyError(
            f"New Strategy: {strategy}, "
            f"minimum tenor = {find_minimum_maturity(strategy):.2f} yrs)"
        )
    minimum_years = int(minimum_tenor)
    minimum_months = int(12 * (minimum_tenor - minimum_years))
    dropped_maturities = pd.to_datetime(
        dt.today() + relativedelta(years=minimum_years, months=minimum_months)
    ).to_period("M")
    next_month_ret_df = df[df["MaturityMonth"] > dropped_maturities].copy()
    # Re-Weight index with only bonds that will be in it.

    next_month_ret_df["Estimated_OAD"] = (
        next_month_ret_df["DataMart_OAD"]
        * next_month_ret_df["BM_Weight"]
        / next_month_ret_df["BM_Weight"].sum()
    )
    if credit:
        next_month_credit_ret_df = next_month_ret_df[
            next_month_ret_df["Sector"] != "TREASURIES"
        ]
        bm_oad = next_month_credit_ret_df["Estimated_OAD"].sum()
    else:
        bm_oad = next_month_ret_df["Estimated_OAD"].sum()
    next_month_ret_df["Estimated_DTS"] = (
        next_month_ret_df["OAS"]
        * next_month_ret_df["DataMart_OASD"]
        * next_month_ret_df["BM_Weight"]
        / next_month_ret_df["BM_Weight"].sum()
    )
    bm_dts = next_month_ret_df["Estimated_DTS"].sum()
    return bm_oad, bm_dts


def inspect_extension(ret_df, stats_df):
    ret_oad = ret_df.set_index("CUSIP")["BM_OAD"].rename("Returns*CTD")
    stats_oad = stats_df.set_index("CUSIP")["BM_OAD"].rename("Stats*CTD")
    map_cols = ["Ticker", "Sector"]
    d = defaultdict(dict)
    for df in [ret_df, stats_df]:
        for col in map_cols:
            map = dict(zip(df["CUSIP"], df[col]))
            d[col] = {**map, **d[col]}

    df = pd.concat((ret_oad, stats_oad), axis=1).fillna(0)
    df["$\Delta$CTD"] = df["Stats*CTD"] - df["Returns*CTD"]
    for col in map_cols:
        df[col] = df.index.map(d[col])
    bad_tickers = {"TII", "WIT", "SP", "S", "FNCL", "FACL"}
    bad_sectors = {"CASH", "Bond Futures", "TREASURIES"}
    df = df[~(df["Ticker"].isin(bad_tickers) | df["Sector"].isin(bad_sectors))]
    ticker_deltas = (
        df.groupby("Ticker").sum().sort_values("$\Delta$CTD").rename_axis(None)
    )
    return ticker_deltas[ticker_deltas["$\Delta$CTD"].abs() > 0.05]


def find_minimum_maturity(strategy):
    db = Database()
    port = db.load_portfolio(strategy=strategy)
    bm_df = port.df[port.df["BM_Weight"] > 0]
    return bm_df["MaturityYears"].min().round(2)


# %%
if __name__ == "__main__":
    build_month_end_extensions_report()
