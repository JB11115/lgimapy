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

    strategies = []
    extensions = []
    managers = []
    for strategy, accounts in tqdm(strategy_accounts.items()):
        if strategy in stragies_with_no_benchmark:
            continue
        for account in accounts:
            kwargs = {"account": account, "ret_df": True, "market_cols": False}
            try:
                df_ret = db.load_portfolio(universe="returns", **kwargs)
                df_stats = db.load_portfolio(universe="stats", **kwargs)
            except ValueError:
                continue
            else:
                strategies.append(strategy.replace("_", " "))
                managers.append(account_manager[account])
                extensions.append(oad(df_stats, strategy) - oad(df_ret))
                break

    last_names = [name.split()[1] for name in managers]
    ext_df = (
        pd.DataFrame(
            {
                "PM": managers,
                "Strategy": strategies,
                "Extension": extensions,
                "last_names": last_names,
            }
        )
        .sort_values(["last_names", "Strategy"])
        .drop("last_names", axis=1)
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
    fid = f"month_end_extensions_{dt.today().strftime('%Y-%m-%d')}"
    doc = Document(fid, path="reports/month_end_extensions")
    doc.add_preamble(ignore_bottom_margin=True)
    doc.add_table(
        ext_df,
        prec=3,
        hide_index=True,
        col_fmt="llrc",
        midrule_locs=midrule_locs,
        alternating_colors=(None, "lightgray"),
    )
    doc.save()


def oad(df, strategy=None):
    """
    Perform special rules for specifying return index
    for next month from current stats index.
    """
    if strategy is None:
        return np.sum(df["BM_OAD"])

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
    return bm_oad


# %%

if __name__ == "__main__":
    main()
