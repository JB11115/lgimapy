import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime as dt
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    import pybbg

from lgimapy.utils import to_list, to_datetime, root, mkdir


# %%
def bbgwinpy_executable():
    return str(root("src/lgimapy/bloomberg/bbgwinpy.py"))


def fmt_bbg_dt(date):
    return f"{to_datetime(date):%Y%m%d}"


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
        return args


def _windows_bdh(securities, yellow_keys, fields, start, end=None, ovrd=None):
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


def _windows_bdp(securities, yellow_keys, fields, ovrd=None):
    args = BBGInputConverter(securities, yellow_keys, fields)

    # Scrape from Bloomberg.
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        bbg = pybbg.Pybbg()
        df = bbg.bdp(args.tickers, args.fields, overrides=args.ovrd).T
        bbg.session.stop()

    # Format DataFrame.
    df.index = [re.sub(f"\ {args.yellow_keys}$", "", ix) for ix in df.index]

    return df


def _windows_bds(security, yellow_key, field, ovrd=None):
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
    if sys.platform == "win32":
        return _windows_bdh(securities, yellow_keys, fields, start, end, ovrd)
    elif sys.platform == "linux":
        df = _linux_to_windows_bbg(
            "bdh",
            securities=securities,
            yellow_keys=yellow_keys,
            fields=fields,
            start=start,
            end=end,
            ovrd=ovrd,
        )
        return df.sort_index()


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
    if sys.platform == "win32":
        return _windows_bdp(securities, yellow_keys, fields, ovrd)
    elif sys.platform == "linux":
        df = _linux_to_windows_bbg(
            "bdp",
            securities=securities,
            yellow_keys=yellow_keys,
            fields=fields,
            ovrd=ovrd,
        )
        return df


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
    if sys.platform == "win32":
        return _windows_bds(security, yellow_key, field, ovrd)
    elif sys.platform == "linux":
        df = _linux_to_windows_bbg(
            "bds",
            securities=security,
            yellow_keys=yellow_key,
            fields=field,
            ovrd=ovrd,
        )
        return df.sort_index()


def _linux_to_windows_bbg(func, *args, **kwargs):
    """
    Run the specified Bloomberg Data call in
    Windows, save the result in a file, load
    back into memory in python and return the
    resulting pd.DataFrame.
    """
    bbg_input = BBGInputConverter(*args, **kwargs)

    # Create a file name current epoch.
    epoch = str(int(time.time() * 1e8))

    # Save the BBG command arguments to a temporary location.
    args_dir = Path("/tmp/bbgwinpy/args")
    args_fid = args_dir / f"{epoch}.json"
    mkdir(args_dir)
    with open(args_fid, "w") as f:
        json.dump(bbg_input.bbgwinpy_args(func), f, indent=4)

    # Ensure there is a directory to save the data to.
    data_dir = Path("/tmp/bbgwinpy/data")
    mkdir(data_dir)

    # Run the BBG command from Windows,saving the data to the above dir.
    bash_func = [
        "python.exe",
        bbgwinpy_executable(),
        epoch,
    ]
    # print(" ".join(bash_func))  # for debugging
    try:
        s = subprocess.run(
            bash_func,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except TimeoutError:
        raise TimeoutError(f"{func.upper()} call '{epoch}' failed")

    # Load the data back into python.
    data_fid = data_dir / f"{epoch}.json"
    try:
        df = pd.read_json(data_fid)
    except ValueError:
        raise FileNotFoundError(
            "BBG data file not found, error occurred "
            "scraping data from BBG API."
        )

    # Delete temporary files.
    args_fid.unlink()
    data_fid.unlink()

    return df
