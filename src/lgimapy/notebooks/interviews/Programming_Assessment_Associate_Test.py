from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from Programming_Assessment_Solutions import (
    get_clean_treasuries,
    nearest_date,
    count_number_of_increasing_IDs,
    Bond,
    OLS,
)


class Timer:
    def __init__(self, title):
        print(f"{title}\n{'-'*len(title)}")
        self._start = self._last = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.run_time("Total")
        print("\n")

    def split_time(self, title):
        split = perf_counter() - self._last
        print(f"{title}: {split:.2f}s")
        self._last = perf_counter()

    def run_time(self, title):
        elapsed = perf_counter() - self._start
        print(f"{title}: {elapsed:.2f}s")
        self._last = perf_counter()


def check_expected_dtype(res, dtype, problem):
    error_msg = (
        f"Problem {problem}: Expected type {dtype}, got '{type(res)}' instead."
    )
    if dtype is None:
        if res is not None:
            raise TypeError(error_msg)
    else:
        if not isinstance(res, dtype):
            raise TypeError(error_msg)


def load_problem_1_data():
    df = pd.read_csv("test_data/Problem_1_data.csv", index_col=0)
    col_dtypes_dict = {
        "float64": ["OAS", "OASD", "OAD", "CouponRate"],
        "category": [
            "ISIN",
            "Ticker",
            "Issuer",
            "Sector",
            "CouponType",
            "CallType",
        ],
        "int8": ["OriginalTenor"],
        "datetime": ["CurrentDate", "MaturityDate"],
    }
    for dtype, col_names in col_dtypes_dict.items():
        for col in col_names:
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col])
            else:
                df[col] = df[col].astype(dtype)

    return df


def test_problem_1():
    p1_df = load_problem_1_data()
    with Timer("Problem 1"):
        p1_solution = get_clean_treasuries(p1_df)
        print(f"{len(p1_solution)} cleaned treasuries found")
        check_expected_dtype(p1_solution, pd.DataFrame, 1)
        if "DTS" not in p1_solution.columns:
            raise ValueError("'DTS' columns missing from returned DataFrame")


def load_problem_2_data():
    all_dates = pd.date_range("1/1/1990", "1/1/2022")
    np.random.seed(8675309)
    mask = np.random.uniform(size=len(all_dates)) > 0.1
    date_list = all_dates[mask]

    test_dict = {
        1: {"date": "7/18/1993", "dtype": pd.Timestamp},
        2: {"date": "6/10/2004", "dtype": pd.Timestamp},
        3: {"date": "6/10/2004", "dtype": pd.Timestamp, "before": False},
        4: {"date": "6/10/2004", "dtype": pd.Timestamp, "after": False},
        5: {"date": "1969", "dtype": None, "after": False},
        6: {"date": "2050", "dtype": None, "before": False},
    }
    return test_dict, date_list


def test_problem_2():
    p2_test_dict, p2_date_list = load_problem_2_data()
    with Timer("Problem 2") as t:
        for test_number, test_dict in p2_test_dict.items():
            reference_date = pd.to_datetime(test_dict.pop("date"))
            expected_dtype = test_dict.pop("dtype")
            p2_solution = nearest_date(
                reference_date, p2_date_list, **test_dict
            )
            check_expected_dtype(
                p2_solution, expected_dtype, f"2.{test_number}"
            )
            kwargs_fmt = f", {test_dict}" if test_dict else ""
            p2_solution_fmt = (
                None if p2_solution is None else f"{p2_solution:%m/%d/%Y}"
            )
            t.split_time(
                f"  2.{test_number}: "
                f"{reference_date:%m/%d/%Y}{kwargs_fmt} --> {p2_solution_fmt}"
            )

        test_dates = pd.date_range("1/1/2000", "11/25/2021")
        for reference_date in test_dates:
            nearest_date(reference_date, p2_date_list)
        t.split_time(
            f"  2.{test_number+1}: Running {len(test_dates):,} iterations"
        )


def test_problem_3():
    with Timer("Problem 3"):
        n = count_number_of_increasing_IDs("test_data/Problem_3_data.txt")
        check_expected_dtype(n, int, 3)
        print(f"{n:,} stricly increasing ID's counted")


def load_problem_4_data():
    data_fids = Path("test_data/Problem_4_data").glob("*.csv")
    for fid in data_fids:
        yield pd.read_csv(
            fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )["cash_flows"]


def test_problem_4():
    p4_data = list(load_problem_4_data())
    with Timer("Problem 4"):
        for i, cashflows in enumerate(p4_data):
            test_number = i + 1
            price = 100 - i / 10
            bond = Bond(price=price, cashflows=cashflows)
            ytm = bond.ytm
            check_expected_dtype(ytm, float, f"4.{test_number}")
            print(f"  2.{test_number}: Computed Yield = {ytm:.2%} ")


def load_problem_5_data():
    n = 200
    beta = 7
    alpha = 4

    np.random.seed(8675309)
    x = np.random.normal(10, 3, size=n)
    random_noise = np.random.normal(0, 10, size=n)
    y = x * beta + alpha + random_noise
    return x, y


def test_problem_5():
    x, y = load_problem_5_data()
    with Timer("Problem 5") as t:
        ols = OLS(x, y)
        t.split_time("  5.1: Class is initialized")

        ols = OLS(x, y)
        check_expected_dtype(ols.resid, np.ndarray, 5.2)
        check_expected_dtype(ols.predict(np.arange(4)), np.ndarray, 5.2)
        check_expected_dtype(ols.predict(4), float, 5.2)
        t.split_time("  5.2: User calls methods without calling `OLS.fit()`")

        ols = OLS(x, y)
        ols.fit()
        ols.fit()
        check_expected_dtype(ols.resid, np.ndarray, 5.3)
        check_expected_dtype(ols.predict(np.arange(4)), np.ndarray, 5.3)
        check_expected_dtype(ols.predict(4), float, 5.3)
        t.split_time("  5.3: User calls `OLS.fit()` directly")

        ols = OLS(x, y)
        ols.plot()
        t.split_time("  5.4: User only calls `OLS.plot()`")
        figure_fid = Path("OLS_plot.png")
        if not figure_fid.exists():
            raise FileNotFoundError(
                f"Problem 5.4: '{figure_fid.stem}.png' not found."
            )


if __name__ == "__main__":
    test_problem_1()
    test_problem_2()
    test_problem_3()
    test_problem_4()
    test_problem_5()
