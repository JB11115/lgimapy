from datetime import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from pyarrow import ArrowIOError
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import mkdir, root


# %%


def create_us_feather(fid, all_trade_dates):
    """
    Create feather files for all months in database.

    Parameters
    ----------
    fid: str
        Filename for new feather.
    all_trade_dates:
        List of all dates available in DataMart.
    """
    db = Database()

    # Find start and end date for feather.
    year, month = (int(d) for d in fid.split("_"))
    next_month = month + 1 if month != 12 else 1
    next_year = year if month != 12 else year + 1
    try:
        start = pd.to_datetime(f"{next_month}/1/{next_year}")
        end_date = db.trade_dates(start=start)[0]
    except IndexError:
        end_date = all_trade_dates[-1]
    ex_end = pd.to_datetime(f"{month}/1/{year}")
    start_date = db.trade_dates(exclusive_end=ex_end)[-1]

    # Load index and drop holidays.
    db.load_market_data(start=start_date, end=end_date, local=False)
    ix = db.build_market_index()
    ix.df = ix.df[ix.df["Date"].isin(db.trade_dates())]

    # Fill missing data with bloomberg data pre Oct-2018.
    if start_date < pd.to_datetime("11/1/2018"):
        ix._fill_missing_columns_with_bbg_data()
    # Add previous market value column.
    ix._get_prev_market_value_history()
    # Compute MTD total and excess returns.
    ix.compute_excess_returns()

    # Save feather file.
    df = ix.subset(start=f"{month}/1/{year}").df.reset_index(drop=True)
    df.to_feather(root(f"data/US/feathers/{fid}.feather"))


def find_missing_feathers(market, all_trade_dates):
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
    db = Database()
    feather_dir = root(f"data/{market}/feathers")
    feather_dir.mkdir(parents=True, exist_ok=True)
    # Compare saved feather files to all feathers that should exist.
    saved_feathers = [
        f.name.strip(".feather") for f in feather_dir.glob("*.feather")
    ]
    # %%
    start = db.date("MARKET_START", market=market).replace(day=1)
    all_feathers = pd.date_range(start, db.date("today"), freq="MS").strftime(
        "%Y_%m"
    )
    # %%
    missing_feathers = [f for f in all_feathers if f not in saved_feathers]

    # Check that all trade dates for past couple months are
    # in saved feather files.
    db = Database()
    start = dt.today() - timedelta(45)
    # Load local data, if local data doesn't exist create it.
    while True:
        try:
            db.load_market_data(start=start, local=True)
        except (FileNotFoundError, ArrowIOError) as e:
            missing = str(e).split(".feather")[0].rsplit("feathers/", 1)[1]
            create_us_feather(missing, all_trade_dates)
        else:
            break

    ix = db.build_market_index()
    saved_dates = ix.dates
    # Compare to all dates in database to find missing dates.
    dates_to_save = [
        d
        for d in all_trade_dates
        if (d > saved_dates[0]) & (d not in db.holiday_dates())
    ]
    missing_dates = [
        d.strftime("%Y_%m") for d in dates_to_save if d not in saved_dates
    ]
    missing_feathers.extend(missing_dates)
    return sorted(list(set(missing_feathers)))


def update_feathers():
    """
    Update feather files. Finds and creates any missing
    historical feathers and updates the feather for
    current day. Creates verbose progress bar if more
    than 2 files are to be created.

    Parameters
    ----------
    dates: List[datetime].
        List of all trade dates available in DataMart.
    """
    dates = Database().trade_dates()
    missing_fids = find_missing_feathers("US", dates)
    pbar = len(missing_fids) > 2
    if pbar:
        print("Updating Feather Files")
    for fid in tqdm(missing_fids, disable=(not pbar)):
        try:
            create_us_feather(fid, dates)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(fid, "\n", e)
            continue
    print("Updated Feather Files")


if __name__ == "__main__":
    dates = None
    all_trade_dates = dates
    update_feathers()
