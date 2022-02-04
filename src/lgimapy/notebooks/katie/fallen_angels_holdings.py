from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database, groupby, Strategy, BondBasket
from lgimapy.utils import root, mkdir


# %%
def fallen_angel_report(strategy, year, month):
    # %%
    db = Database()
    date = pd.to_datetime(f"{month}/15/{year}")
    month_beg = db.date("mtd", reference_date=date)
    try:
        month_end = db.date("mtd", reference_date=db.date("+1m", date))
    except IndexError:
        month_end = db.date("today")

    # Load data from the end of the month.
    db.load_market_data(date=month_end, local=True)
    ix_stats = db.build_market_index(in_stats_index=True)
    ix_ret = db.build_market_index(in_returns_index=True)

    # Find CUSIPs that were in returns index but not in the
    # stats index, and do not have IG ratings.
    ix_diff = ix_ret.subset(
        cusip=ix_stats.cusips, rating=("BB+", "D"), special_rules="~CUSIP"
    )

    fallen_angel_df = (
        ix_diff.df[["Ticker", "Issuer", "CollateralType", "Sector"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # %%
    # Exit early if there were no fallen angels.
    if not len(fallen_angel_df):
        return

    # Find portfolio positions at beginning and end of the month for each.
    strat_beg = db.load_portfolio(strategy=strategy, date=month_beg)
    strat_end = db.load_portfolio(strategy=strategy, date=month_end)
    # Find positions for each risk entity.
    d = defaultdict(list)
    for _, row in fallen_angel_df.iterrows():
        re_strat_beg = strat_beg.subset(
            ticker=row["Ticker"],
            issuer=row["Issuer"],
            collateral_type=row["CollateralType"],
        )
        re_strat_end = strat_end.subset(
            ticker=row["Ticker"],
            issuer=row["Issuer"],
            collateral_type=row["CollateralType"],
        )
        try:
            L3 = re_strat_beg.df["L3"].mode()[0]
        except IndexError:
            continue
        L3_strat_beg = strat_beg.subset(L3=L3)
        L3_strat_end = strat_end.subset(L3=L3)

        try:
            L4 = re_strat_beg.df["Sector"].mode()[0]
        except IndexError:
            continue
        L4_strat_beg = strat_beg.subset(sector=L4)
        L4_strat_end = strat_end.subset(sector=L4)

        account_bm_weights = re_strat_beg.calculate_account_values(
            lambda x: x.df["BM_Weight"].sum()
        )
        bm_weight = re_strat_beg.account_value_weight(account_bm_weights)
        port_weight = re_strat_beg.credit_pct()
        if pd.isna(port_weight) and pd.isna(bm_weight):
            continue

        account_port_oad = re_strat_end.calculate_account_values(
            lambda x: x.df["P_OAD"].sum()
        )
        port_oad = re_strat_end.account_value_weight(account_port_oad)
        account_bm_oad = re_strat_end.calculate_account_values(
            lambda x: x.df["BM_OAD"].sum()
        )
        bm_oad = re_strat_end.account_value_weight(account_bm_oad)

        d["Year"].append(year)
        d["Month"].append(month)
        d["Ticker"].append(row["Ticker"])
        d["Issuer"].append(row["Issuer"])
        d["CollateralType"].append(row["CollateralType"])
        d["L3"].append(L3)
        d["L4"].append(L4)
        d["Benchmark Weight Month Start"].append(bm_weight)
        d["Portfolio Weight Month Start"].append(port_weight)
        d["Fraction of Accounts Holding at Month Start"].append(
            (re_strat_beg.account_credit_pct() > 0).mean()
        )
        d["Overweight (OAD) Month Start"].append(re_strat_beg.oad())
        d["L3 Overweight (OAD) Month Start"].append(L3_strat_beg.oad())
        d["L4 Overweight (OAD) Month Start"].append(L4_strat_beg.oad())
        d["Portfolio Weight Month End"].append(re_strat_end.credit_pct())
        d["Fraction of Accounts Holding at Month End"].append(
            (re_strat_end.account_credit_pct() > 0).mean()
        )
        d["Portfolio OAD Month End"].append(port_oad)
        d["Benchmark OAD Month End"].append(bm_oad)
        d["Overweight (OAD) Month End"].append(re_strat_end.oad())
        d["L3 Overweight (OAD) Month End"].append(L3_strat_end.oad())
        d["L4 Overweight (OAD) Month End"].append(L4_strat_end.oad())
    # Save results if there are any.
    fid_dir = root("data/fallen_angel_holdings")
    mkdir(fid_dir)
    date_fmt = f"{year}-{str(month).zfill(2)}"
    fid = fid_dir / f"{strategy.replace(' ', '_')}_{date_fmt}_.csv"
    df = pd.DataFrame(d)
    if len(df):
        df.to_csv(fid)


def save_current_files(strategies):
    fid_dir = root("data/fallen_angel_holdings")
    for strategy in strategies:
        fid = strategy.replace(" ", "_")
        fids = fid_dir.glob(f"{fid}_[0-9]*.csv")
        df = pd.concat(
            [pd.read_csv(fid, index_col=0) for fid in fids], sort=False
        )
        df.sort_values(["Year", "Month"]).to_csv(fid_dir / f"{fid}_All.csv")


def main():
    strategy = "US Credit"
    strategy = "US Government: Intermediate"
    strategies = ["US Credit", "US Long Credit"]
    # dates = {2018: range(9, 13), 2019: range(1, 13), 2020: range(1, 4)}
    dates = {2020: [4, 5, 6]}
    for strategy in strategies:
        for year, months in dates.items():
            for month in months:
                fallen_angel_report(strategy, year, month)

    save_current_files(strategies)


if __name__ == "__main__":
    main()
