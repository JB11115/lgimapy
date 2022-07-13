import importlib.util
from collections import defaultdict
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
    def __init__(self):
        self._start = self._last = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed_time = self.run_time()

    def split_time(self):
        split = perf_counter() - self._last
        self._last = perf_counter()
        return split

    def run_time(self):
        elapsed = perf_counter() - self._start
        self._last = perf_counter()
        return elapsed


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
    df = pd.read_csv("exam_data/Problem_1_data.csv", index_col=0)
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


def test_problem_1(func):
    p1_df = load_problem_1_data()
    with Timer() as t:
        p1_solution = set(func(p1_df)["ISIN"])
    return p1_solution, t.elapsed_time


def load_problem_2_data():
    all_dates = pd.date_range("1/1/1990", "1/1/2022")
    np.random.seed(345908)
    mask = np.random.uniform(size=len(all_dates)) > 0.1
    date_list = all_dates[mask]
    np.random.seed(1001)
    all_test_dates = pd.date_range("1/1/1988", "1/1/2030")
    test_mask = np.random.uniform(size=len(all_test_dates)) > 0.995
    test_dates = all_test_dates[test_mask]
    return test_dates, date_list


def test_problem_2(func):
    p2_test_dates, p2_date_list = load_problem_2_data()
    test_kwarg_list = [{}, {"before": False}, {"after": False}]
    p2_solutions = []
    with Timer() as t:
        for test_date in p2_test_dates:
            for kws in test_kwarg_list:
                p2_solutions.append(func(test_date, p2_date_list, **kws))
    return pd.Series(p2_solutions), t.elapsed_time


def test_problem_3(func):
    with Timer() as t:
        n = func("exam_data/Problem_3_data.txt")
    return n, t.elapsed_time


def load_problem_4_data():
    data_fids = Path("exam_data/Problem_4_data").glob("*.csv")
    for fid in data_fids:
        yield pd.read_csv(
            fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )["cash_flows"]


def test_problem_4(cls):
    p4_data = list(load_problem_4_data())
    n_iters = 20
    p4_solutions = []
    with Timer() as t:
        for cashflows in p4_data:
            for i in range(20):
                price = 100 - i / 100
                bond = cls(price=price, cashflows=cashflows)
                p4_solutions.append(bond.ytm)

    return pd.Series(p4_solutions).round(4), t.elapsed_time


def load_problem_5_data(seed):
    n = 200
    beta = 7
    alpha = 4

    np.random.seed(seed)
    x = np.random.normal(10, 3, size=n)
    random_noise = np.random.normal(0, 10, size=n)
    y = x * beta + alpha + random_noise
    return x, y


def test_problem_5(cls):
    seeds = range(3)
    p5_solutions = []
    with Timer() as t:
        for seed in seeds:
            x, y = load_problem_5_data(seed)
            ols = cls(x, y)
            ols = cls(x, y)
            res = ols.resid

            ols.fit()
            ols.fit()
            p5_solutions.append(ols.predict(np.arange(1, 4, 0.5)))
            p5_solutions.append(ols.predict(6))

            ols = cls(x, y)
            ols.plot()

    return p5_solutions, t.elapsed_time


def get_master_results():
    funcs = [
        "get_clean_treasuries",
        "nearest_date",
        "count_number_of_increasing_IDs",
        "Bond",
        "OLS",
    ]
    results = {}
    for i, func in enumerate(funcs):
        results[i + 1] = eval(f"test_problem_{i+1}({func})")
    return results


def get_submission_results(fid):
    funcs = [
        "get_clean_treasuries",
        "nearest_date",
        "count_number_of_increasing_IDs",
        "Bond",
        "OLS",
    ]
    results = {}
    for i, func in enumerate(funcs):
        spec = importlib.util.spec_from_file_location(func, fid)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            results[i + 1] = eval(f"test_problem_{i+1}(module.{func})")
        except Exception as e:
            results[i + 1] = (None, None)
    return results


def compare_problem_1(res, master_res):
    n_missing = len(master_res - res)
    n_extra = len(res - master_res)
    return 1 - ((n_missing + n_extra) / len(master_res))


def compare_problem_2(res, master_res):
    equivalence = master_res == res
    equivalence[pd.isnull(master_res) == pd.isnull(res)] = True
    df = pd.concat(
        (master_res.rename("master"), res.rename("submission")), axis=1
    )
    df["equal"] = df["master"] == df["submission"]
    nan_locs = df[["master", "submission"]].isna().sum(axis=1) == 2
    df.loc[nan_locs, "equal"] = True
    return df["equal"].sum() / len(df)


def compare_problem_3(res, master_res):
    return min(res / master_res, master_res / res)


def compare_problem_4(res, master_res):
    return compare_problem_2(res, master_res)


def compare_problem_5(res, master_res):
    num = 0
    denom = 0
    for i, (r, mr) in enumerate(zip(res, master_res)):
        if i % 2 == 0:
            num += (r.round(4) == mr.round(4)).sum()
            denom += len(mr)
        else:
            num += r == mr
            denom += 1
    return num / denom


def test_applicant_submissions():
    master_results = get_master_results()
    submissions = Path("interview_submissions").glob("*py")
    submission_results = {}
    for submission in submissions:
        name = submission.stem
        submission_results[name] = get_submission_results(submission)

    d = defaultdict(list)
    for applicant, submission_res_d in submission_results.items():
        d["Applicant"].append(applicant)
        for problem, (res, t) in submission_res_d.items():
            master_res, master_t = master_results[problem]
            if res is None:
                accuracy = np.nan
                efficiency = np.nan
            else:
                accuracy = eval(f"compare_problem_{problem}(res, master_res)")
                efficiency = t / master_t

            d[f"P{problem}_Accuracy"].append(np.round(accuracy, 2))
            d[f"P{problem}_Efficiency"].append(np.round(efficiency, 1))

    df = pd.DataFrame(d)
    df.to_csv("Interview_Exam_Results.csv")


if __name__ == "__main__":
    test_applicant_submissions()
