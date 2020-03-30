from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
from pyarrow import ArrowIOError
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import mkdir, root


def create_parquet(fid, all_trade_dates):
    """
    Create parquet files for all months in database.

    Parameters
    ----------
    fid: str
        Filename for new parquet.
    all_trade_dates:
        List of all dates available in DataMart.
    """
    db = Database()

    # Find start and end date for parquet.
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

    # Load index, compute MTD excess returns, and save parquet.
    db.load_market_data(start=start_date, end=end_date)
    ix = db.build_market_index()
    ix.compute_excess_returns()
    df = ix.subset(start=f"{month}/1/{year}").df.reset_index(drop=True)
    df.to_parquet(root(f"data/parquets/{fid}.parquet"))


def find_missing_parquets(all_trade_dates):
    """
    Find missing parquets from data dir.

    Parameters
    ----------
    all_trade_dates:
        List of all dates available in DataMart.

    Returns
    -------
    list[str]:
        List of parquet files which are not saved in the
        data directory.
    """
    # Compare saved parquet files to all parquets that should exist.
    saved_parquets = [
        f.name.strip(".parquet")
        for f in root("data/parquets").glob("*.parquet")
    ]
    if not saved_parquets:
        mkdir(root("data/parquets"))

    all_parquets = pd.date_range(
        "2/1/1998", Database().date("today"), freq="MS"
    ).strftime("%Y_%m")
    missing_parquets = [f for f in all_parquets if f not in saved_parquets]

    # Check that all trade dates for past couple months are
    # in saved parquet files.
    db = Database()
    start = dt.today() - timedelta(45)
    # Load local data, if local data doesn't exist create it.
    while True:
        try:
            db.load_market_data(
                start=start, local=True, local_file_fmt="parquet"
            )
        except OSError as e:
            missing = str(e).split(".parquet")[0].rsplit("parquets\\", 1)[1]
            create_parquet(missing, all_trade_dates)
        else:
            break
    ix = db.build_market_index()
    saved_dates = ix.dates
    # Compare to all dates in database to find missing dates.
    dates_to_save = [
        d
        for d in all_trade_dates
        if (d > saved_dates[0]) & (d not in db.holiday_dates)
    ]
    missing_dates = [
        d.strftime("%Y_%m") for d in dates_to_save if d not in saved_dates
    ]
    missing_parquets.extend(missing_dates)
    return sorted(list(set(missing_parquets)))


def update_parquets(dates=None):
    """
    Update parquet files. Finds and creates any missing
    historical parquets and updates the parquet for
    current day. Creates verbose progress bar if more
    than 2 files are to be created.

    Parameters
    ----------
    dates: List[datetime].
        List of all trade dates available in DataMart.
    """
    dates = Database().load_trade_dates() if dates is None else dates
    missing_fids = find_missing_parquets(dates)
    pbar = len(missing_fids) > 2
    if pbar:
        print("Updating Parquet Files")
    for fid in tqdm(missing_fids, disable=(not pbar)):
        try:
            create_parquet(fid, dates)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(fid, "\n", e)
            continue
    print("Updated Parquet Files")


if __name__ == "__main__":
    update_parquets()
