from collections import defaultdict
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import load_json, root
from lgimapy.latex import Document

# %%
def main():
    db = Database()
    db.update_portfolio_account_data()

    strategy_accounts = load_json("strategy_accounts")
    manager_accounts = load_json("manager_accounts")
    account_manager = {}
    for manager, accounts in manager_accounts.items():
        for account in accounts:
            account_manager[account] = manager

    stragies_with_no_benchmark = {
        "LDI No Benchmark",
        "Custom Management No Benchmark",
        "Custom Management No Benchmark - MDLZSTRP",
        "Custom BNM - MBRF",
        "Absolute Return - LIB150",
        "Absolute Return LIB350",
        "US High Yield",
    }

    d = defaultdict(list)
    for strategy, accounts in tqdm(strategy_accounts.items()):
        if strategy in stragies_with_no_benchmark:
            continue
        for account in accounts:
            kwargs = {"account": account, "ret_df": True, "market_cols": False}
            try:
                df_ret = db.load_portfolio(universe="returns", **kwargs)
                df_stats = db.load_portfolio(universe="stats", **kwargs)
            except (ValueError):
                # except (ValueError, pd.io.sql.DatabaseError):
                continue
            else:
                pm = account_manager[account]
                next_oad, next_dts = port_metrics(df_stats, strategy)
                curr_oad, curr_dts = port_metrics(df_ret)

                d["PM"].append(pm)
                d["Strategy"].append(strategy.replace("_", " "))
                d["last_name"].append(pm.split()[1])
                d["OAD"].append(next_oad - curr_oad)
                d["DTS"].append(next_dts - curr_dts)
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
    fid = f"month_end_extensions_{today.strftime('%Y-%m-%d')}"
    doc = Document(fid, path="reports/month_end_extensions")
    doc.add_preamble(
        ignore_bottom_margin=True,
        margin={
            "left": 0.5,
            "right": 0.5,
            "top": 0.5,
            "bottom": 0.2,
        },
        header=doc.header(
            left="Month End Extensions",
            right=today.strftime("%B %Y"),
        ),
        footer=doc.footer(logo="LG_umbrella"),
    )
    doc.add_table(
        ext_df,
        prec={"OAD": "2f", "DTS": "0f"},
        hide_index=True,
        col_fmt="llrc|c",
        midrule_locs=midrule_locs,
        alternating_colors=(None, "lightgray"),
    )
    doc.save()


def port_metrics(df, strategy=None):
    """
    Perform special rules for specifying return index
    for next month from current stats index, and return
    OAD and DTS for benchmark portfolio.
    """
    if strategy is None:
        bm_oad = df["BM_OAD"].sum()
        bm_dts = (df["BM_OASD"] * df["OAS"]).sum()
        return bm_oad, bm_dts

    maturity_min = {
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
    }
    df["DataMart_OAD"] = df["BM_OAD"] / df["BM_Weight"]
    df["DataMart_OASD"] = df["BM_OASD"] / df["BM_Weight"]
    df["MaturityMonth"] = pd.to_datetime(
        df["MaturityDate"], errors="coerce"
    ).dt.to_period("M")
    dropped_years = maturity_min[strategy]
    dropped_maturities = pd.to_datetime(
        dt.today() + relativedelta(years=dropped_years)
    ).to_period("M")
    next_month_ret_df = df[df["MaturityMonth"] > dropped_maturities]
    # Re-Weight index with only bonds that will be in it.
    bm_oad = (
        next_month_ret_df["DataMart_OAD"]
        * next_month_ret_df["BM_Weight"]
        / next_month_ret_df["BM_Weight"].sum()
    ).sum()
    bm_dts = (
        next_month_ret_df["OAS"]
        * next_month_ret_df["DataMart_OASD"]
        * next_month_ret_df["BM_Weight"]
        / next_month_ret_df["BM_Weight"].sum()
    ).sum()
    return bm_oad, bm_dts


# %%

if __name__ == "__main__":
    main()
