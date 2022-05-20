import numpy as np
import pandas as pd
from blpapi import NotFoundException

from lgimapy.utils import root, mkdir, X_drive
from lgimapy.bloomberg import bds, fmt_bbg_dt

# %%


def get_cashflows(cusip, maturity_date=None):
    """
    Get cash flows and respective dates for specified cusip.

    Parameters
    ----------
    cusip: str
        Cusip to get cash flows for.

    Returns
    -------
    pd.Series:
        Cash flows with datetime index.
    """
    fid = root(f"data/cashflows/{cusip}.parquet")
    try:
        cash_flows_df = pd.read_parquet(fid)
    except (FileNotFoundError, OSError):
        # Scrape and save cash flows.
        cash_flows_df = scrape_cash_flows(cusip, maturity_date)
        mkdir(fid.parent)
        cash_flows_df.to_parquet(fid)
        X_drive_fid = X_drive(
            f"Credit Strategy/lgimapy/data/cashflows/{cusip}.parquet"
        )
        cash_flows_df.to_parquet(X_drive_fid)

    return cash_flows_df["cash_flows"]


def scrape_cash_flows(cusip, maturity_date=None):
    """
    Scrapes specified cusip's cash flows and create
    a .csv file in the data directory with the result.

    Parameters
    ----------
    cusip: str
        Cusip to scrape coupon dates for.
    """
    # Override date to before data collection began.
    ovrd = {"USER_LOCAL_TRADE_DATE": "19500101"}
    if maturity_date is not None:
        # Ensure maturity date is actual maturity and not a call date.
        ovrd["WORKOUT_DT_BID"] = fmt_bbg_dt(maturity_date)
        ovrd["WORKOUT_PX_BID"] = 100

    # Try list of keys to match given security.
    for yellow_key in ["Corp", "Muni", "Govt"]:
        try:
            df = bds(cusip, yellow_key, field="DES_CASH_FLOW", ovrd=ovrd)
        except NotFoundException:
            continue
        else:
            break

    # Compute cash flows and save.
    return (
        (np.sum(df, axis=1) / 1e4)
        .to_frame()
        .rename_axis(None)
        .rename(columns={0: "cash_flows"})
    )
