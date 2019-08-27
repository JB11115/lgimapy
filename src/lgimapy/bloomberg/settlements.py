from pandas import to_datetime

from lgimapy.bloomberg import bdp
from lgimapy.utils import load_json, dump_json


def get_settlement_date(date):
    """
    Get Treasury settlement date for specified date.

    Parameters
    ----------
    date: str
        Cusip to get cash flows for.

    Returns
    -------
    pd.Series:
        Cash flows with datetime index.
    """
    # Load file.
    fid = "settlement_dates"
    settlement_dates = load_json(fid, empty_on_error=True)

    # Get settlement date from file if it exists, otherwise scrape it.
    input_date = to_datetime(date).strftime("%Y%m%d")
    try:
        settle_date = settlement_dates[input_date]
    except KeyError:
        scrape_settlement_date(input_date)
        settlement_dates = load_json(fid, empty_on_error=True)
        settle_date = settlement_dates[input_date]

    return to_datetime(settle_date)


def scrape_settlement_date(date):
    """
    Scrapes specified date's settlement date and
    add it to `settlement_dates.json`.

    Uses a 30 year treasury bond issued in 1997 which has
    settlement dates from 11/17/1997-11/15/2027.

    Parameters
    ----------
    date: datetime str '%Y%m%d'
        Date formatted for Bloomberg.
    """
    # Scrape settlement date.
    field = "SETTLE_DT"
    cusip = "912810FB9"
    ovrd = {"USER_LOCAL_TRADE_DATE": date}

    temp_df = bdp(cusip, "Corp", fields=field, ovrd=ovrd)
    settle_date = temp_df[field][0]

    # Load `settlement_dates.json`, add new date, and save.
    fid = "settlement_dates"
    settlement_dates = load_json(fid, empty_on_error=True)
    settlement_dates[date] = settle_date.strftime("%m/%d/%Y")
    dump_json(settlement_dates, fid)
