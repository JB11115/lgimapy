import json

import numpy as np
from pandas import to_datetime

from lgimapy.utils import root
from lgimapy.bloomberg import bds


def get_coupon_dates(cusip):
    """
    Get coupon dates for a single cusip. First attempts
    to use saved `cusip_coupon_dates.json` file, scraping and
    updating the file if the cusip is missing.

    Parameters
    ----------
    cusip: str
        Cusip to get coupon dates for.

    Returns
    -------
    coupon_dates: List[datetime].
        List of coupon dates for specified cusip.
    """
    # Load coupon date json file.
    coupon_date_json = root("data/cusip_coupon_dates.json")
    try:
        with open(coupon_date_json, "r") as fid:
            coupon_dates_dict = json.load(fid)
    except FileNotFoundError:
        coupon_dates_dict = {}

    # If cusip is not in file, scrape coupon dates and reload.
    if cusip not in coupon_dates_dict:
        scrape_coupon_dates(cusip)
        with open(coupon_date_json, "r") as fid:
            coupon_dates_dict = json.load(fid)

    coupon_dates = to_datetime(coupon_dates_dict[cusip], format="%Y%m%d")
    return coupon_dates


def scrape_coupon_dates(cusip):
    """
    Scrapes specified cusip's coupon dates and updates
    `cusip_coupon_dates.json` with result.

    Parameters
    ----------
    cusip: str
        Cusip to scrape coupon dates for.
    """
    # Scrape coupon dates from Bloomberg and convnert to
    # json serializable ints.
    coupon_dates = bds(cusip, "Corp", field="DES_CASH_FLOW", column=2)
    coupon_dates = [int(cd) for cd in coupon_dates]

    # Load `cusip_coupon_dates.json`, add new cusips, and save.
    coupon_date_json = root("data/cusip_coupon_dates.json")
    try:
        with open(coupon_date_json, "r") as fid:
            coupon_dates_dict = json.load(fid)
    except FileNotFoundError:
        coupon_dates_dict = {}
    coupon_dates_dict[cusip] = coupon_dates
    with open(coupon_date_json, "w") as fid:
        json.dump(coupon_dates_dict, fid, indent=4)


if __name__ == "__main__":
    print(get_coupon_dates("UV495426"))
