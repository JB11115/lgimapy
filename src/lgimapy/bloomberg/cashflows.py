import numpy as np
import pandas as pd
from blpapi import NotFoundException

from lgimapy.utils import root
from lgimapy.bloomberg import bds


def get_cashflows(cusip):
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
    try:
        return pd.read_csv(
            root(f"data/coupons/{cusip}.csv"),
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
        )["cash_flow"]
    except FileNotFoundError:
        scrape_cash_flows(cusip)
        return pd.read_csv(
            root(f"data/coupons/{cusip}.csv"),
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
        )["cash_flow"]


def scrape_cash_flows(cusip):
    """
    Scrapes specified cusip's cash flows and create
    a .csv file in the data directory with the result.

    Parameters
    ----------
    cusip: str
        Cusip to scrape coupon dates for.
    """
    # Override date to before data collection began.
    ovrd = {"USER_LOCAL_TRADE_DATE": "19501010"}

    # Try list of keys to match given security.
    for yellow_key in ["Corp", "Muni", "Govt"]:
        try:
            df = bds(cusip, yellow_key, field="DES_CASH_FLOW", ovrd=ovrd)
        except NotFoundException:
            continue
        else:
            break

    # Compute cash flows and save.
    df["cash_flow"] = np.sum(df, axis=1) / 1e4
    df["cash_flow"].to_csv(root(f"data/coupons/{cusip}.csv"), header=True)
