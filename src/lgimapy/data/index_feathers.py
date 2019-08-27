from datetime import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import mkdir, root


def create_feather(fid):
    """
    Create feather files for all months in database.

    Parameters
    ----------
    fid: str
        Filename for new feather.
    """
    db = Database()
    dates = np.array(db.trade_dates)

    # Find start and end date for feather.
    year, month = (int(d) for d in fid.split("_"))
    next_month = month + 1 if month != 12 else 1
    next_year = year if month != 12 else year + 1
    try:
        end_date = dates[
            dates >= pd.to_datetime(f"{next_month}/1/{next_year}")
        ][0]
    except IndexError:
        end_date = dates[-1]
    start_date = dates[dates < pd.to_datetime(f"{month}/1/{year}")][-1]

    # Load index, compute MTD excess returns, and save feather.
    db.load_market_data(start=start_date, end=end_date)
    ix = db.build_market_index()
    # ix.compute_excess_returns()
    df = ix.subset(start=f"{month}/1/{year}").df.reset_index(drop=True)
    df.to_feather(root(f"data/feathers/{fid}.feather"))


def find_missing_feathers():
    """
    Find missing feathers from data dir.

    Returns
    -------
    list[str]:
        List of feather files which are not saved in the
        data directory.
    """

    # Compare saved feather files to all feathers that should exist.
    saved_feathers = [
        f.name.strip(".feather")
        for f in root("data/feathers").glob("*.feather")
    ]
    if not saved_feathers:
        mkdir(root("data/feathers"))

    all_feathers = pd.date_range(
        "1/1/2004", Database().trade_dates[-1], freq="MS"
    ).strftime("%Y_%m")
    return [f for f in all_feathers if f not in saved_feathers]


def update_feathers():
    """
    Update feather files. Finds and creates any missing
    historical feathers and updates the feather for
    current day.

    Parameters
    ----------
    verbose: bool, default=True
        If True print updating message and display progress bar.
    """
    missing_fids = find_missing_feathers()
    missing_fids.append(dt.today().strftime("%Y_%m"))
    pbar = len(missing_fids) > 2
    if pbar:
        print("Updating Feather Files")
    for fid in tqdm(missing_fids, disable=(not pbar)):
        try:
            create_feather(fid)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(fid, "\n", e)
            continue
    print("Updated Feather Files")


if __name__ == "__main__":
    update_feathers()
