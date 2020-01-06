import os
import json
import pprint as pp
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict, OrderedDict
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from types import GeneratorType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
from tabulate import tabulate


def root(join_path=None):
    """
    Return path from root directory of the project.

    Parameters
    ----------
    join_path: str, Path, default=None
        Path to join to root directoyr of project.

    Returns
    -------
    root_path: Path
        Path of root directory and joined path.
    """
    root_path = Path(pkg_resources.get_distribution("lgimapy").location).parent
    if join_path is None:
        return root_path
    else:
        return root_path.joinpath(join_path)


def load_json(filename, empty_on_error=False):
    """
    Load json file from the `./data/` directory.

    Parameters
    ----------
    filename: str
        Filename of json in `./data/` directory.

    Returns
    -------
    dict:
        Loaded json file.
    """
    json_fid = root(f"data/{filename}.json")
    try:
        with open(json_fid, "r") as fid:
            return json.load(fid)
    except FileNotFoundError:
        if empty_on_error:
            return {}
        else:
            msg = f"{filename}.json does no exist in `./data/` directory."
            raise FileNotFoundError(msg)


def dump_json(d, filename, **kwargs):
    """
    Write json file to the `./data/` directory.

    Parameters
    ----------
    d: dict
        Dictionary to store as json.
    filename: str
        Filename of json in `./data/` directory.
    **kwargs:
        Keyword arguments for ``json.dump()``
    """
    json_fid = root(f"data/{filename}.json")
    dump_kwargs = {"indent": 4}
    dump_kwargs.update(**kwargs)
    with open(json_fid, "w") as fid:
        json.dump(d, fid, **dump_kwargs)


def pprint(obj):
    """Print object to screen in nice formatting"""
    if isinstance(obj, pd.DataFrame):
        print(tabulate(obj, headers="keys", tablefmt="psql"))
    elif type(obj) in [dict, defaultdict, OrderedDict]:
        printer = pp.PrettyPrinter(indent=4)
        printer.pprint(obj)
    else:
        raise ValueError(f"Error: {type(obj)} not yet supported")


class Time:
    """
    Simple class for time profiling.

    Parameters
    ----------
    unit: {'s', 'm', 't', h'}, default='s'
        Unit to format time outputs. {second, minute, minute, hour}.
    name: str, default=''
        Name to include on all output statements.

    Methods
    -------
    split(name): Print time split between last call.
    run_time(name): Print time since start.
    checkpoint(name): Print split and time since start.
    """

    def __init__(self, unit="s", name=""):
        self._name = name
        self._start = perf_counter()
        self._last = perf_counter()
        self._unit_str = {"s": "sec", "m": "min", "t": "min", "h": "hr"}[unit]
        self._unit_div = {"s": 1, "m": 60, "t": 60, "h": 3600}[unit]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.run_time("Total")

    def split(self, name=""):
        split = (perf_counter() - self._last) / self._unit_div
        self._last = perf_counter()
        print(f"{self._name} {name}: {split:.3f} {self._unit_str}.")

    def run_time(self, name=""):
        elapsed = (perf_counter() - self._start) / self._unit_div
        self._last = perf_counter()
        print(f"{self._name} {name}: {elapsed:.3f} {self._unit_str}.")

    def checkpoint(self, name=""):
        elapsed = (perf_counter() - self._start) / self._unit_div
        split = (perf_counter() - self._last) / self._unit_div
        self._last = perf_counter()
        print(
            f"{self._name} {name} split: {split:.3f} {self._unit_str}, "
            f"run time: {elapsed:.3f} {self._unit_str}."
        )


def mkdir(directory):
    """Make directory if it does not already exist."""
    try:
        os.makedirs(directory)
    except OSError:
        pass


def to_list(obj, dtype=None):
    """
    Convert single object to list if it is not already
    a list-like object.

    Parameters
    ----------
    obj:
        Object to convert to list if required.
    dtype: type
        Data type of object to check against/ data type
        of object in output list. If None the dtype is checked
        against ```list``` and ```tuple``` and converted to
        ```list``` if it is neither.

    Returns
    -------
    list-like:
        List-like version of input object.
    """
    if dtype is not None:
        # Compare to specific dtype.
        if isinstance(obj, dtype):
            if dtype == str:
                return [obj]
            else:
                return list(obj)
        else:
            return obj
    else:
        # Compare to list and tuple only.
        if type(obj) in [list, tuple]:
            return obj
        else:
            return list(obj)


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


def first_unique(vals, fill=""):
    """
    Find first unique values in a sorted list.

    Parameters
    ----------
    vals: list-like
        Sorted input list.
    fill: scalar, default=""
        Fill value to replace non-unique values in input list.
    Returns
    -------
    unique_locs: List[int].
        List of unique value locations.
    unique_list: List[scalar].
        List of unique values with non-unique values replaced with fill.
    """
    unique_list, unique_locs = [], []
    for i, val in enumerate(vals):
        if val in unique_list:
            unique_list.append(fill)
        else:
            unique_list.append(val)
            unique_locs.append(i)
    return unique_locs, unique_list


def nearest(items, pivot):
    """Return nearest nearest item in list to pivot."""
    return min(items, key=lambda x: abs(x - pivot))


def nearest_date(date, date_list, inclusive=True, before=True, after=True):
    """
    Search list of dates for nearest date to reference.

    Parameters
    ----------
    date: datetime
        Reference date to find date near.
    date_list: List[datetime].
        List of dates to search.
    inclusive: bool, default=True
        Whether to include to specified reference date in
        the searchable list.
    before: bool, default=True
        Whether to include dates before the reference date.
    after: bool, default=True
        Whether to include dates after the reference date.
    """
    ref_date = to_datetime(date)
    if inclusive:
        if before and after:
            return min(date_list, key=lambda x: abs(x - ref_date))
        elif before:
            return date_list[bisect_right(date_list, ref_date) - 1]
        elif after:
            return date_list[bisect_left(date_list, ref_date)]
        else:
            raise ValueError("Either before or after must be True")
    else:
        if before and after:
            date_ex = date_list[date_list != ref_date]
            return min(date_ex, key=lambda x: abs(x - ref_date))
        elif before:
            return date_list[bisect_left(date_list, ref_date) - 1]
        elif after:
            return date_list[bisect_right(date_list, ref_date)]
        else:
            raise ValueError("Either before or after must be True")


def check_all_equal(lst):
    """Efficiently check if all items in list are equal."""
    return not lst or lst.count(lst[0]) == len(lst)


def smooth_weight(x, B):
    """Weight functcion for smoothing two overlapping curves."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 / (1 + (x / (1 - x)) ** (-B))


def sep_str_int(input_string):
    """
    Efficiently separate a string and integer.
    Only finds first integer and first partial string
    in inpute string.

    Parameters
    ----------
    string: str
        Input string with embedded integer.

    Returns
    -------
    num: int
        Integer from input string.
    output_string: str
        String portion of input string.
    """
    num = re.findall("(\d+)", input_string)
    num = num[0] if num else 1  # set number to 1 if no number present
    output_string = re.findall(r"[a-zA-Z]+", input_string)[0]
    return int(num), output_string


def floatftime(time_horizon, unit="d"):
    """
    Similar to `strftime`, format time to numeric float.

    Parameters
    ----------
    time_horizon: str
        String representation of time.
    unit: {'s', 't', 'h', 'd', 'w', 'm', 'y'}, default='d'
        Unit of numeric result, not case sensative.

    Returns
    -------
    num_time: float
        Numeric time in specified units.
    """
    day = 24 * 3600
    unit_conversion = {
        "s": 1,
        "t": 60,
        "h": 3600,
        "d": day,
        "w": 7 * day,
        "m": 30 * day,
        "y": 365 * day,
    }
    # Append leading 1 if needed.
    if len(time_horizon) == 1:
        time_horizon = f"1{time_horizon.lower()}"

    # Split numeric from input unit.
    num, in_unit = float(time_horizon[:-1]), time_horizon[-1].lower()
    num_time = num * unit_conversion[in_unit] / unit_conversion[unit]
    return num_time


def rolling_sum(a, n, pad=None):
    """
    Vectorized implementation of a rolling sum.

    Parameters
    ----------
    a: array_like
        Calculate rolling sum of these values.
    n: int
        Number of elements to include in rolling sum.
    pad: None, False, or int/float, default=None
        Type of padding to be applied to returned array.
        - None: No padding, partial sums are included for first n-1 elements.
        - False: First n-1 elements are skipped and not included.
        - int/float: First n-1 elements are replaced with any number (e.g., 0).
        - list/tuple: First n-1 elements are replaced by list/tuple of size n-1.
    Returns
    -------
    ret: ndarray
        Return a new array that is rolling sum of original.
    """

    ret = np.cumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]

    if pad is None:
        return ret

    if pad is False:
        return ret[n - 1 :]

    ret[: n - 1] = pad if type(pad) in [list, tuple] else (n - 1) * [pad]
    return ret


def rolling_mean(a, n, pad=None):
    """
    Vectorized implementation of a rolling mean.

    Parameters
    ----------
    a: array_like
        Calculate rolling mean of these values.
    n: int
        Number of elements to include in rolling mean.
    pad: None, False, or int/float, default=None
        Type of padding to be applied to returned array.
        - None: No padding, partial means are included for first n-1 elements.
        - False: First n-1 elemnts are skipped and not included.
        - int/float: First n-1 elements are replaced with any number (e.g., 0).
        - list/tuple: First n-1 elements are replaced by list/tuple of size n-1.
    Returns
    -------
    ret: ndarray
        Return a new array that is rolling mean of original.
    """
    ret = np.cumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n - 1 :] = ret[n - 1 :] / n

    if pad is None:
        ret[: n - 1] = ret[: n - 1] / np.arange(1, n)
        return ret

    if pad is False:
        return ret[n - 1 :]

    ret[: n - 1] = pad if type(pad) in [list, tuple] else (n - 1) * [pad]
    return ret


def custom_sort(str_list, alphabet):
    """
    Sort a list of strings by a specified alphabetical order.

    Parameters
    ----------
    str_list: List[str].
        List of strings to be sorted.
    alphabet: str
        Case sensitive alphabetical order for sort.

    Returns
    -------
    Sorted List[str] according to specified alphabet.

    Examples
    --------
    custom_sort(['black', 'blue', 'green', 'red', 'yellow'], 'rygblu')
    >>> ['red', 'yellow', 'green', 'blue', 'black']

    """
    # Add all unique characters in list to end of provided alphabet.
    alphabet = alphabet + "".join(list(set("".join(str_list))))

    # Sort list with alphabet.
    return sorted(str_list, key=lambda x: [alphabet.index(c) for c in x])


def find_max_locs(df):
    """Find iloc locations of maximum values in pd.DataFrame."""
    num_df = df.apply(pd.to_numeric, errors="coerce").values
    row_max = num_df.max(axis=1)
    ij_max = []

    for i, i_max in enumerate(row_max):
        if np.isnan(i_max):  # ignore nan values
            continue
        for j in range(num_df.shape[1]):
            if num_df[i, j] == i_max:
                ij_max.append((i, j))
    return ij_max


def find_min_locs(df):
    """Find iloc locations of minimum values in pd.DataFrame."""
    num_df = df.apply(pd.to_numeric, errors="coerce").values
    row_min = num_df.min(axis=1)
    ij_min = []

    for i, i_min in enumerate(row_min):
        if np.isnan(i_min):  # ignore nan values
            continue
        for j in range(num_df.shape[1]):
            if num_df[i, j] == i_min:
                ij_min.append((i, j))
    return ij_min


def replace_multiple(text, repl_dict):
    """
    Replace multiple text fragments in a string.

    Parameters
    ----------
    text: str
        Input str to perform replacements on.
    repl_dict: dict[str: str].
        Dictionary with original str values as keys and replacment
        str values as values.

    Returns
    -------
    text: str
        Output str with all replacments made.
    """
    for key, val in repl_dict.items():
        text = text.replace(key, val)
    return text


def find_threshold_crossings(a, threshold):
    """
    Find ilocs where an array crosses a specified threshold.

    a: array-like
        Array of values.
    threshold: float
        Value of threshold to find crossovers.

    Returns
    -------
    locs: [1 x N] ndarray
        List of ilocs where threshold crossovers occur.
    """
    a = np.array(a)
    a_thresh = a - threshold
    locs = np.where(np.diff(np.sign(a_thresh)))[0]
    return locs


class EventDistance:
    """
    Convert timeseries of values to indexed array of bins
    of days before or after the event.

    Parameters
    ----------
    s: pd.Series or Iterable
        Series of values with datetime index or Iterable of
        Series with datetime indexes.
    event_dates: datetime or List[datetime].
        Date(s) of event(s).
    lookback: str
        String representation of start relative to
        event date. Input to :function:`floatftime`.
    lookforward: str
        String representation of end relative to
        event date. Input to :function:`floatftime`.

    Attributes
    ----------
    values: ndarray
        2D array with raw values for each event in every row.
    index: ndarray
        Array of days or bins before/after event.
    """

    def __init__(self, s, event_dates, lookback, lookforward):
        self._start = floatftime(lookback)
        self._end = floatftime(lookforward)

        # Convert event dates to list of datetime objects.
        if isinstance(event_dates, list):
            pass
        elif isinstance(event_dates, pd.Timestamp):
            event_dates = [event_dates]  # convert to list
        else:
            msg = (
                f"Expected type of `event_dates` is datetime or "
                f"List[datetime], not {type(event_dates)}."
            )
            raise TypeError(msg)

        # Get timeseries values as either generator or Series.
        if isinstance(s, pd.Series):
            generator_flag = False
        elif isinstance(s, GeneratorType):
            generator_flag = True
            s_generator = s
        else:
            msg = (
                f"Expected type of `s` is Series or "
                f"Generator[Series], not {type(s)}."
            )
            raise TypeError(msg)

        # Solve raw values for each event date.
        self._n = int(self._end - self._start + 1)
        self.values = np.zeros([len(event_dates), self._n])
        for i, d in enumerate(event_dates):
            if generator_flag:
                s = next(s_generator)
            self.values[i, :] = self._single_event(s, d)

        # Store index and zero-index value.
        self.index = np.arange(self._n) + self._start

    @property
    def max(self):
        """ndarray: Maximum values for each bin before/after event."""
        return np.nanmax(self.values, axis=0)

    @property
    def min(self):
        """ndarray: Minimum values for each bin before/after event."""
        return np.nanmin(self.values, axis=0)

    @property
    def mean(self):
        """ndarray: Mean values for each bin before/after event."""
        return np.nanmean(self.values, axis=0)

    @property
    def std(self):
        """
        ndarray:
            Standard deviation of values for each bin before/after event.
        """
        return np.nanstd(self.values, axis=0)

    def bin(self, binsize=7, pad="drop"):
        """
        Bin :attr:`values` into bins of a specifed number of days
        on either side of the event date. Updates all attributes.

        Parameters
        ----------
        binsize: int, default=7
            Number of days to include in each bin.
        pad: {'drop', None}, default='drop'
            How to pad bins which are not full. By default bins are
            dropped.
        """
        w = binsize
        n = self.values.shape[0]
        arrays = OrderedDict()

        # Find event date.
        try:
            zero_ix = list(self.index).index(0)
        except ValueError:
            # 0 (event date) not in index.
            zero_ix = -1

        # Bin and store values before event date.
        if self._start < 0:
            neg_array = np.flip(self.values[:, :zero_ix], axis=1)
            m_mod = neg_array.shape[1] % w
            if pad == "drop" and m_mod != 0:
                neg_array = neg_array[:, :-m_mod]
            else:
                nan_array = np.full([n, w - m_mod], np.NaN)
                neg_array = np.concatenate([neg_array, nan_array], axis=1)
            dims = neg_array.shape[1] // w, int(neg_array.shape[0] * w)
            neg_array_binned = np.reshape(neg_array.T, (dims)).T
            arrays[-1] = np.flip(neg_array_binned, axis=1)

        # Store values on event date.
        if zero_ix != -1:
            # Add event date padded with NaNs to arrays dict.
            nan_array = np.full([int(n * w - n)], np.NaN)
            zero_array = np.concatenate([self.values[:, zero_ix], nan_array])
            arrays[0] = np.reshape(zero_array, (n * w, 1))

        # Bin and store values after event date.
        if self._end > 0:
            pos_array = self.values[:, zero_ix + 1 :]
            m_mod = pos_array.shape[1] % w
            if pad == "drop" and m_mod != 0:
                pos_array = pos_array[:, :-m_mod]
            else:
                nan_array = np.full([n, w - m_mod], np.NaN)
                pos_array = np.concatenate([pos_array, nan_array], axis=1)
            dims = pos_array.shape[1] // w, int(pos_array.shape[0] * w)
            pos_array_binned = np.reshape(pos_array.T, (dims)).T
            arrays[1] = pos_array_binned

        # Combine all binned arrays back to one array.
        self.values = np.concatenate([v for v in arrays.values()], axis=1)

        # Find new binned index.
        binned_ix = []
        for k, v in arrays.items():
            if k == -1:
                binned_ix.extend(list(-np.arange(v.shape[1] + 1)[1:][::-1]))
            elif k == 0:
                binned_ix.append(0)
            else:
                binned_ix.extend(list(np.arange(v.shape[1] + 1)[1:]))
        self.index = np.array(binned_ix) * w

    def _single_event(self, s, event_date):
        """
        Convert single event to array.

        Parameters
        ----------
        s: pd.Series
            Series of values with datetime index.
        event_date: datetime
            Date of event.

        Returns
        -------
        a: nd.array
            Array of values
        """
        start_date = event_date + timedelta(self._start)
        end_date = event_date + timedelta(self._end)
        s = s[(start_date <= s.index) & (s.index <= end_date)]
        # Find index positions where values exist.
        ix_mask = np.array(
            [(date - event_date).days for date in s.index]
        ) - int(self._start)
        a = np.full(self._n, np.NaN)
        a[ix_mask] = s.values
        return a
