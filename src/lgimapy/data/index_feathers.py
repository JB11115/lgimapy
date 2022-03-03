from dateutil.relativedelta import relativedelta

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database, groupby
from lgimapy.utils import mkdir, root, Time


# %%


def create_feather(fid, db, force, s3):
    """
    Create feather files specified month from SQL.

    Parameters
    ----------
    fid: str
        Filename for new feather.
    all_trade_dates:
        List of all dates available in DataMart.
    """
    # Find start and end date for feather.
    first_of_month = pd.to_datetime(fid, format="%Y_%m")
    last_of_month = first_of_month + relativedelta(day=31)
    try:
        prev_last_of_month = db.trade_dates(exclusive_end=first_of_month)[-1]
    except IndexError:
        prev_last_of_month = db.trade_dates()[0]

    # Load index and drop holidays.
    if db.market == "US" or force:
        # Load data directly from Datamart or BASys.
        db.load_market_data(
            start=prev_last_of_month, end=last_of_month, local=False
        )
        ix = db.build_market_index(drop_treasuries=False)
    else:
        # For BASys data, use local data when possible, only
        # load new data from BASys.
        db.load_market_data(start=prev_last_of_month, end=last_of_month)
        prev_ix = db.build_market_index(drop_treasuries=False)
        if prev_ix.end_date == last_of_month:
            ix = prev_ix  # no new data to load
        else:
            # Load new data directly from BASys and combine with local data.
            new_start = db.trade_dates(exclusive_start=prev_ix.end_date)[0]
            db.load_market_data(start=new_start, end=last_of_month, local=False)
            new_ix = db.build_market_index(drop_treasuries=False)
            ix = prev_ix + new_ix

    # Drop holidays and add previous market value column.
    ix.df = ix.df[ix.df["Date"].isin(db.trade_dates())]
    ix._get_prev_market_value_history(force=True)

    if db.market == "US":
        # Fill missing data with bloomberg data pre Oct-2018.
        if first_of_month < pd.to_datetime("11/1/2018"):
            ix._fill_missing_columns_with_bbg_data()

        # Fill missing HY index flags.
        ix = fill_missing_HY_index_flags(ix)
        # Compute MTD total and excess returns.
        ix.compute_excess_returns()

    # Remove previous month's last day and save feather file.
    df = ix.subset(start=ix.dates[1]).df.reset_index(drop=True)
    df.to_feather(root(f"data/{db.market}/feathers/{fid}.feather"))

    # Add daily files to s3 drop point.
    s3_dirs_d = {
        "prod": [
            "s3://lgima-prod-3pdh-data-bucket/qws-inbound/qws-rds",
        ],
        "dev": [
            "s3://lgima-dev-3pdh-data-bucket/qws-inbound/qws-rds",
            "s3://lgima-qa-3pdh-data-bucket/qws-inbound/qws-rds",
            "s3://lgima-qa-3pdh-data-bucket/qws-inbound/qws-rds",
        ],
    }
    if s3:
        mkt = db.market.lower()
        keys = db._passwords["AWS"]
        for stage, s3_dirs in s3_dirs_d.items():
            sess = boto3.Session(
                aws_access_key_id=keys[f"{stage}_access_key"],
                aws_secret_access_key=keys[f"{stage}_secret_access_key"],
            )
            n_dates = len(df["Date"].unique())
            for i, (date, date_df) in enumerate(df.groupby("Date")):
                if i < (n_dates - 5):
                    continue
                filename = f"security_analytics_{mkt}_{date:%Y%m%d}"
                for s3_dir in s3_dirs:
                    s3_fid = f"{s3_dir}/{mkt}/{filename}.parquet"
                    wr.s3.to_parquet(
                        date_df, path=s3_fid, index=False, boto3_session=sess
                    )


def fill_missing_HY_index_flags(ix):
    min_daily_flags = (
        ix.df[["Date", "USHYReturnsFlag"]].groupby("Date").sum().min().iloc[0]
    )
    if min_daily_flags < 1000:
        flags = ix.df[["ISIN", "USHYReturnsFlag"]].groupby("ISIN").sum()
        index_isins = flags[flags > 0].dropna().index
        index_isin_loc = ix.df["ISIN"].isin(index_isins)
        for col in ["USHYReturnsFlag", "USHYStatisticsFlag"]:
            ix.df.loc[index_isin_loc, col] = 1

    return ix


def find_missing_feathers(db, update_current=False):
    """
    Find missing feathers from data dir.

    Parameters
    ----------
    all_trade_dates:
        List of all dates available in DataMart.

    Returns
    -------
    list[str]:
        List of feather files which are not saved in the
        data directory.
    """
    # Compare saved feather files to feather files expected to exist.
    feather_dir = root(f"data/{db.market}/feathers")
    mkdir(feather_dir)
    saved_feathers = sorted(feather_dir.glob("*.feather"))
    saved_fids = set([fid.name.strip(".feather") for fid in saved_feathers])
    expected_fids = set(
        pd.date_range(
            db.date("market_start").replace(day=1), db.date("today"), freq="MS"
        ).strftime("%Y_%m")
    )
    missing_fids = expected_fids - saved_fids
    if update_current:
        current_fid = sorted(list(expected_fids))[-1]
        missing_fids.add(current_fid)

    # Check that all expected trade dates for past couple months are
    # in saved the feather files.
    try:
        recent_df = pd.concat(
            (pd.read_feather(fid) for fid in saved_feathers[-2:])
        )
    except ValueError:
        return sorted(missing_fids)
    else:
        saved_dates = sorted(pd.to_datetime(recent_df["Date"].unique()))
    expected_dates = db.trade_dates(start=saved_dates[0])
    missing_dates = set(expected_dates) - set(saved_dates)
    missing_date_fids = set([d.strftime("%Y_%m") for d in missing_dates])

    return sorted(missing_fids | missing_date_fids, reverse=True)


def update_market_data_feathers(update_current=False, force=False, s3=False):
    """
    Update market data feather files for each market.
    Finds and creates any missing feather files.
    Creates verbose progress bar if more
    than 2 files are to be created.
    """
    markets = ["US", "GBP", "EUR"]
    # markets = ["GBP", "EUR"]
    for market in markets:
        db = Database(market=market)
        missing_fids = find_missing_feathers(db, update_current=update_current)
        pbar = len(missing_fids) > 3
        if pbar:
            print(f"Updating {market} Feather Files")
        for fid in tqdm(missing_fids, disable=(not pbar)):
            try:
                create_feather(fid=fid, db=db, force=force, s3=s3)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print(fid, "\n", e)
                continue
        print(f"Updated {market} Feather Files")


# %%

if __name__ == "__main__":
    # update_current = True
    # market = "EUR"
    # db = Database(market=market)
    # missing_fids = find_missing_feathers(db, update_current=update_current)
    # fid = missing_fids[-1]
    update_market_data_feathers(update_current=True, force=True)
