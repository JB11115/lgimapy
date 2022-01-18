import argparse
import json
import re
import sys
import warnings
from collections.abc import Iterable
from datetime import datetime as dt
from numbers import Number

import pandas as pd
import pybbg


def to_list(obj, dtype=None, sort=False):
    """
    Convert object to ``list`` if it is not already
    a ``list`` or ``None``.

    Parameters
    ----------
    obj:
        Object to convert to list if required.
    dtype: type, optional
        Data type of object to check against/ data type
        of object in output list. If None the dtype is checked
        against ```list``` and ```tuple``` and converted to
        ```list``` if it is neither.

    Returns
    -------
    list:
        ``list`` conversion of input object.
    """
    if obj is None:
        return None

    if dtype is not None:
        # Compare to specific dtype.
        if isinstance(obj, dtype):
            if dtype in {float, int, Number, str}:
                list_obj = [obj]
            else:
                list_obj = list(obj)
        else:
            list_obj = to_list(obj)
    else:
        list_obj = list(obj)

    return sorted(list_obj) if sort else list_obj


def to_datetime(date):
    """
    Convert a datetime object to pandas datetime only if necessary.

    Parameters
    ----------
    date: datetime
        Date in any datetime type e.g., str.

    Returns
    -------
    pd._libs.tslibs.timestamps.Timestamp:
        Pandas datetime object of input date.
    """
    if isinstance(date, pd._libs.tslibs.timestamps.Timestamp):
        return date
    else:
        return pd.to_datetime(date)


def fmt_bbg_dt(date):
    return to_datetime(date).strftime("%Y%m%d")


class BBGInputConverter:
    """
    Convert inputs to be used by ``pybbg`` library.

    Parameters
    ----------
    securities: str or List[str].
        Security name(s) or cusip(s).
    yellow_keys: ``{'Govt', 'Corp', 'Equity', 'Index', etc.}`` or List[str].
        Bloomberg yellow key(s) for securities. Case insensitive.
    fields: str or List[str].
        Bloomberg field(s) to collect history.
    start: datetime object
        Start date for scrape.
    end: datetime object, default=None
        End date for scrape, if None most recent date is used.
    ovrd: dict, default=None
        Bloomberg overrides {field: value}.
    """

    def __init__(
        self, securities, yellow_keys, fields, start=None, end=None, ovrd=None
    ):
        self._securities = to_list(securities, dtype=str)
        self._yellow_keys = yellow_keys
        self._fields = to_list(fields, dtype=str)
        self._start = start
        self._end = end
        self.ovrd = ovrd

    @property
    def securities(self):
        return (
            self._securities[0]
            if len(self._securities) == 1
            else self._securities
        )

    @property
    def yellow_keys(self):
        yellow_keys = to_list(self._yellow_keys, dtype=str)
        return yellow_keys[0] if len(yellow_keys) == 1 else yellow_keys

    @property
    def fields(self):
        return self._fields[0] if len(self._fields) == 1 else self._fields

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
        if isinstance(self.yellow_keys, str):
            return [
                f"{security} {self.yellow_keys}"
                for security in self._securities
            ]
        elif isinstance(self.yellow_keys, list):
            return [
                f"{s} {yk}" for s, yk in zip(self._securities, self.yellow_keys)
            ]
        else:
            raise ValueError(
                f"{type(self._yellow_keys)} is not valid for `yellow_keys`."
            )

    def bbgwinpy_args(self, func):
        if func == "bdp":
            args = {
                "securities": self.securities,
                "yellow_keys": self.yellow_keys,
                "fields": self.fields,
                "ovrd": self.ovrd,
            }
        elif func == "bdh":
            args = {
                "securities": self.securities,
                "yellow_keys": self.yellow_keys,
                "fields": self.fields,
                "start": self.start,
                "end": self.end,
                "ovrd": self.ovrd,
            }
        if func == "bds":
            args = {
                "security": self.securities,
                "yellow_key": self.yellow_keys,
                "field": self.fields,
                "ovrd": self.ovrd,
            }

        args.update({"func": func})
        return json.dumps(args)


def bdh(securities, yellow_keys, fields, start, end=None, ovrd=None):
    """
    Retrieve Bloomberg Data History (BDH) query results.

    Parameters
    ----------
    securities: str or List[str].
        Security name(s) or cusip(s).
    yellow_keys: ``{'Govt', 'Corp', 'Equity', 'Index', etc.}`` or List[str].
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
    args = BBGInputConverter(securities, yellow_keys, fields, start, end)

    # Scrape from Bloomberg.
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        bbg = pybbg.Pybbg()
        df = bbg.bdh(
            args.tickers, args.fields, args.start, args.end, overrides=args.ovrd
        )
        bbg.session.stop()

    # Format DataFrame.
    if isinstance(args.securities, str):
        df.columns = to_list(args.fields, dtype=str)
    elif isinstance(args.fields, str):
        df.columns = to_list(args.securities, dtype=str)
    return df


def bdp(securities, yellow_keys, fields, ovrd=None):
    """
    Retrieve Bloomberg Data Point (BDP) query results.

    Parameters
    ----------
    securities: str or List[str].
        Security name(s) or cusip(s).
    yellow_keys: ``{'Govt', 'Corp', 'Equity', 'Index', etc.}`` or List[str].
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
    args = BBGInputConverter(securities, yellow_keys, fields)

    # Scrape from Bloomberg.
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        bbg = pybbg.Pybbg()
        df = bbg.bdp(args.tickers, args.fields, overrides=args.ovrd).T
        bbg.session.stop()

    # Format DataFrame.
    df.index = [re.sub(f"\ {args.yellow_keys}$", "", ix) for ix in df.index]

    return df.reindex(to_list(args.securities, dtype=str))


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
    args = BBGInputConverter(security, yellow_key, field)
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        bbg = pybbg.Pybbg()
        df = bbg.bds(
            f"{args.securities} {args.yellow_keys}",
            args.fields,
            overrides=args.ovrd,
        )
        bbg.session.stop()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("function_signature", help="Function Signature json")
    input_args = parser.parse_args()
    input_d = json.loads(input_args.function_signature)
    func = input_d.pop("func")
    df = eval(func)(**input_d)
    print(df.to_json(), file=sys.stdout)


# %%

if __name__ == "__main__":
    main()
