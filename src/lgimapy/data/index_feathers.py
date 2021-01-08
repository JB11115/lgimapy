from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from pyarrow import ArrowIOError
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import mkdir, root, restart_program


# %%


def create_feather(fid, db):
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
    db.load_market_data(
        start=prev_last_of_month, end=last_of_month, local=False
    )
    ix = db.build_market_index()
    ix.df = ix.df[ix.df["Date"].isin(db.trade_dates())]

    # Add previous market value column.
    ix._get_prev_market_value_history()

    if db.market == "US":
        # Fill missing data with bloomberg data pre Oct-2018.
        if first_of_month < pd.to_datetime("11/1/2018"):
            ix._fill_missing_columns_with_bbg_data()

        # Add HY index Flags.
        ix._add_hy_index_flags()

        # Compute MTD total and excess returns.
        ix.compute_excess_returns()

    # Remove previous month's last day and save feather file.
    df = ix.subset(start=ix.dates[1]).df.reset_index(drop=True)
    df.to_feather(root(f"data/{db.market}/feathers/{fid}.feather"))


def find_missing_feathers(db):
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

    return sorted(missing_fids | missing_date_fids)


def update_market_data_feathers(limit=20):
    """
    Update market data feather files for each market.
    Finds and creates any missing feather files.
    Creates verbose progress bar if more
    than 2 files are to be created.
    """
    markets = ["US", "GBP", "EUR"]
    for market in markets:
        db = Database(market=market)
        missing_fids = find_missing_feathers(db)
        pbar = len(missing_fids) > 3
        if pbar:
            print(f"Updating {market} Feather Files")
        for fid in tqdm(missing_fids, disable=(not pbar)):
            try:
                create_feather(fid, db)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print(fid, "\n", e)
                continue
        print(f"Updated {market} Feather Files")


# %%

if __name__ == "__main__":
    update_market_data_feathers()
