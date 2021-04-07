import re
import warnings
from collections.abc import Iterable
from datetime import datetime as dt

import pandas as pd
import pybbg

from lgimapy.utils import to_list, to_datetime


def fmt_bbg_dt(date):
    return to_datetime(date).strftime("%Y%m%d")


class BBGInputConverter:
    """
    Convert inputs from ``lgimapy`` to be used
    by ``pybbg`` library.
    """

    def __init__(self, securities, yellow_key, fields, start=None, end=None):
        self.securities = to_list(securities, dtype=str)
        self.yellow_key = yellow_key
        self.fields = to_list(fields, dtype=str)

        self._start = start
        self._end = end

    @property
    def start(self):
        return None if self._start is None else fmt_bbg_dt(self._start)

    @property
    def end(self):
        return (
            fmt_bbg_dt(dt.today())
            if self._end is None
            else fmt_bbg_dt(self._end)
        )

    @property
    def tickers(self):
        if isinstance(self.yellow_key, str):
            return [
                f"{security} {self.yellow_key}" for security in self.securities
            ]
        elif isinstance(self.yellow_key, Iterable):
            return [
                f"{s} {yk}" for s, yk in zip(self.securities, self.yellow_key)
            ]
        else:
            raise ValueError(
                f"{type(self.yellow_key)} is not valid for `yellow_key`."
            )


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
    args = BBGInputConverter(securities, yellow_key, fields, start, end)

    # Scrape from Bloomberg.
    warnings.simplefilter(action="ignore", category=UserWarning)
    bbg = pybbg.Pybbg()
    df = bbg.bdh(
        args.tickers, args.fields, args.start, args.end, overrides=ovrd
    )
    bbg.session.stop()
    warnings.simplefilter(action="default", category=UserWarning)

    # Format DataFrame.
    if len(args.securities) == 1:
        df.columns = args.fields
    elif len(args.fields) == 1:
        df.columns = args.securities
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
    args = BBGInputConverter(securities, yellow_key, fields)

    # Scrape from Bloomberg.
    warnings.simplefilter(action="ignore", category=UserWarning)
    bbg = pybbg.Pybbg()
    df = bbg.bdp(args.tickers, args.fields, overrides=ovrd).T
    bbg.session.stop()
    warnings.simplefilter(action="default", category=UserWarning)

    # Format DataFrame.
    df.index = [re.sub(f"\ {args.yellow_key}$", "", ix) for ix in df.index]
    return df.reindex(args.securities)


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
