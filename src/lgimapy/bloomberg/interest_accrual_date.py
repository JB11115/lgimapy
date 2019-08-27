from pandas import to_datetime

from lgimapy.utils import load_json, dump_json
from lgimapy.bloomberg import bdp


def get_accrual_date(cusips):
    """
    Get first accrual date for a list of cusips. First attempts
    to use saved `cusip_accrual_date.json` file, updating the file
    for any cusip which is unsuccesful.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to retrieve issue price for.

    Returns
    -------
    List[str]:
        List of first accrual dates matching input cusips.
    """
    if isinstance(cusips, str):
        cusips = [cusips]  # convert to list

    # Find any cusips which are not in saved file.
    fid = "cusip_accrual_date"
    accrual_dates = load_json(fid, empty_on_error=True)
    missing = []
    for c in list(set(cusips)):
        if c not in accrual_dates:
            missing.append(c)

    # Scrape missing cusips and reload file.
    if missing:
        scrape_accrual_dates(missing)
        accrual_dates = load_json(fid)

    all_accrual_dates = [to_datetime(accrual_dates[c]) for c in cusips]
    if len(all_accrual_dates) == 1:
        return all_accrual_dates[0]
    else:
        return all_accrual_dates


def scrape_accrual_dates(cusips):
    """
    Scrapes specified cusips and updates
    `cusip_accrual_date.json` with scraped
    cusips and their first accrual dates.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to retrieve issue prices for.
    """
    # Build dict of cusip: scraped accrual_dates.
    field = "INT_ACC_DT"
    df = bdp(cusips, "Corp", fields=field)
    scraped_accrual_dates = {
        c: s.strftime("%m/%d/%Y") for (c, s) in zip(cusips, df[field])
    }

    # Load `cusip_issue_price.json`, add new cusips, and save.
    fid = "cusip_accrual_date"
    accrual_dates = load_json(fid, empty_on_error=True)
    dump_json({**accrual_dates, **scraped_accrual_dates}, fid)
