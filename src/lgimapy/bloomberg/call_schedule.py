import numpy as np
import pandas as pd
from blpapi import NotFoundException

from lgimapy.utils import root, mkdir
from lgimapy.bloomberg import bdp, bds

# %%


def get_call_schedule(isin):
    """
    Get call schedule for specified isin.

    Parameters
    ----------
    isin: str
        ISIN to get call schedule for.

    Returns
    -------
    pd.Series:
        Call prices with datetime index, final date is maturity.
    """
    fid = root(f"data/call_schedules/{isin}.parquet")
    try:
        call_schedule_df = pd.read_parquet(fid)
    except (FileNotFoundError, OSError):
        # Scrape and save call schedule.
        call_schedule_df = scrape_call_schedule(isin)
        mkdir(fid.parent)
        call_schedule_df.to_parquet(fid)

    return call_schedule_df["Call Price"]


def scrape_call_schedule(isin):
    """
    Scrapes specified isin's call schedule and create
    a .parquet file in the data directory with the result.

    Parameters
    ----------
    isin: str
        ISIN to scrape coupon dates for.
    """
    # Try list of keys to match given security.
    yellow_keys = ["Corp", "Muni", "Govt"]
    for yellow_key in yellow_keys:
        try:
            call_df = bds(isin, yellow_key, field="CALL_SCHEDULE")
        except NotFoundException:
            continue
        else:
            break

    for yellow_key in yellow_keys:
        try:
            maturity_date_df = bdp(isin, yellow_key, "FINAL_MATURITY")
        except NotFoundException:
            continue
        else:
            break

    maturity_date = pd.to_datetime(maturity_date_df.iloc[0, 0] / 1e3, unit="s")
    matuirty_df = pd.Series({maturity_date: 100}, name="Call Price").to_frame()
    df = pd.concat((call_df, matuirty_df))
    df["Call Price"] = df["Call Price"].astype("float32")
    return df
