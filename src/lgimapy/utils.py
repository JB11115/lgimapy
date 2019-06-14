import os
import json
import pprint
import re
from pathlib import Path
from time import perf_counter

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
    elif isinstance(obj, dict):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(obj)
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
            f"{self._name} {name} split: {elapsed:.3f} {self._unit_str}, "
            f"run time: {elapsed:.3f} {self._unit_str}."
        )


def mkdir(directory):
    """Make directory if it does not already exist."""
    try:
        os.makedirs(directory)
    except OSError:
        pass


def savefig(fid, fdir=None, dpi=300):
    """Save figure to specified location."""
    if fdir is not None:
        mkdir(fdir)
    full_fid = f"{fid}.png" if fdir is None else f"{fdir}/{fid}.png"
    plt.savefig(full_fid, dpi=dpi, bbox_inches="tight")


def nearest(items, pivot):
    """Return nearest nearest item in list to pivot."""
    return min(items, key=lambda x: abs(x - pivot))


def check_all_equal(lst):
    """Efficiently check if all items in list are equal."""
    return not lst or lst.count(lst[0]) == len(lst)


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
        time_horizon = f"1{time_horizon}"

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

    Params
    ------
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
