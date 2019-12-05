import datetime as dt
import warnings
from collections.abc import Iterable

import pandas as pd
import pybbg

from lgimapy.utils import to_list


def bdh(securities, yellow_key, fields, start, end=None, ovrd=None):
    """
    Retrieve Bloomberg Data History (BDH) query results.

    Parameters
    ----------
    securities: str or List[str].
        Security name(s) or cusip(s).
    yellow_key: ``{'Govt', 'Corp', 'Equity', 'Index', etc.}`` or List[str].
        Bloomberg yellow key(s) for securities. Case insensitive.
    fields: str or List[str].
        Bloomberg field(s) to collect history.
    start: datetime object
        Start date for scrape.
    end: datetime object, default=None
        End date for scrape, if None most recent date is used.
    ovrd: dict, default=None
        Bloomberg overrides {field: value}.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with datetime index.

        * If single security and field is provided, column
            is field name.
        * If multiple securities are provided, columns are
            security names.
        * If mulitple fields are provided, columns are field names.
        * If both multiple fields and securities are provided,
            a multi-index header column is used.
    """
    # Convert inputs for pybbg.
    securities = to_list(securities, dtype=str)
    fields = to_list(fields, dtype=str)
    if isinstance(yellow_key, str):
        tickers = [f"{security} {yellow_key}" for security in securities]
    elif isinstance(yellow_key, Iterable):
        tickers = [f"{s} {yk}" for s, yk in zip(securities, yellow_key)]
    else:
        raise ValueError(f"{type(yellow_key)} is not valid for `yellow_key`.")

    start = pd.to_datetime(start).strftime("%Y%m%d")
    end = dt.date.today() if end is None else pd.to_datetime(end)
    end = end.strftime("%Y%m%d")

    # Scrape from Bloomberg.
    warnings.simplefilter(action="ignore", category=UserWarning)
    bbg = pybbg.Pybbg()
    df = bbg.bdh(tickers, fields, start, end, overrides=ovrd)
    bbg.session.stop()
    warnings.simplefilter(action="default", category=UserWarning)

    # Format DataFrame.
    if len(securities) == 1:
        df.columns = fields
    elif len(fields) == 1:
        df.columns = securities
    return df


def bdp(securities, yellow_key, fields, ovrd=None):
    """
    Retrieve Bloomberg Data Point (BDP) query results.

    Parameters
    ----------
    securities: str or List[str].
        Security name(s) or cusip(s).
    yellow_key: ``{'Govt', 'Corp', 'Equity', 'Index', etc.}`` or List[str].
        Bloomberg yellow key(s) for securities. Case insensitive.
    fields: str or List[str].
        Bloomberg field(s) to collect history.
    ovrd: dict, default=None
        Bloomberg overrides {field: value}.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with security index and field columns.
    """
    # Convert inputs for pybbg.
    securities = to_list(securities, dtype=str)
    fields = to_list(fields, dtype=str)
    if isinstance(yellow_key, str):
        tickers = [f"{security} {yellow_key}" for security in securities]
    elif isinstance(yellow_key, Iterable):
        tickers = [f"{s} {yk}" for s, yk in zip(securities, yellow_key)]
    else:
        raise ValueError(f"{type(yellow_key)} is not valid for `yellow_key`.")

    # Scrape from Bloomberg.
    warnings.simplefilter(action="ignore", category=UserWarning)
    bbg = pybbg.Pybbg()
    df = bbg.bdp(tickers, fields, overrides=ovrd).T
    bbg.session.stop()
    warnings.simplefilter(action="default", category=UserWarning)

    # Format DataFrame.
    df.index = [ix.strip(f" {yellow_key}") for ix in df.index]
    return df.reindex(securities)


def bds(security, yellow_key, field, ovrd=None):
    """
    Retrieve Bloomberg Data Set (BDS) query results.
    Convert date columns to datetime
    Parameters
    ----------
    security: str
        Security name or cusip.
    yellow_key: ``{'Govt', 'Corp', 'Equity', 'Index', etc.}``.
        Bloomberg yellow key for securities. Case insensitive.
    field: str
        Bloomberg field to collect history.
    ovrd: dict, default=None
        Bloomberg overrides {field: value}.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with numeric index and columns determined
        by choice of Bloomberg field.
    """
    # Scrape from Bloomberg.
    warnings.simplefilter(action="ignore", category=UserWarning)
    bbg = pybbg.Pybbg()
    df = bbg.bds(f"{security} {yellow_key}", field, overrides=ovrd)
    bbg.session.stop()
    warnings.simplefilter(action="default", category=UserWarning)

    # Format DataFrame.
    date_cols = []
    for col in df.columns:
        if "Date" in col:
            date_cols.append(col)
            df[col] = pd.to_datetime(df[col])

    # Make date column the index if there is only one date column.
    if len(date_cols) == 1:
        df.set_index(date_cols[0], inplace=True)
    return df
