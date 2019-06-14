import json

import numpy as np
from pandas import to_datetime

from lgimapy.utils import load_json, dump_json
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
    List[datetime]:
        List of coupon dates for specified cusip.
    """
    fid = "cusip_coupon_dates"
    coupon_dates = load_json(fid, empty_on_error=True)

    # If cusip is not in file, scrape coupon dates and reload.
    if cusip not in coupon_dates:
        scrape_coupon_dates(cusip)
        coupon_dates = load_json(fid)
    return to_datetime(coupon_dates[cusip], format="%Y%m%d")


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
    scraped_coupon_dates = bds(
        cusip, "Corp", field="DES_CASH_FLOW", column=2, date=True
    )
    scraped_coupon_dates = [int(scd) for scd in scraped_coupon_dates]

    # Load `cusip_coupon_dates.json`, add new cusips, and save.
    fid = "cusip_coupon_dates"
    coupon_dates = load_json(fid, empty_on_error=True)
    coupon_dates[cusip] = scraped_coupon_dates
    dump_json(coupon_dates, fid)


if __name__ == "__main__":
    print(get_coupon_dates("AL068984"))
