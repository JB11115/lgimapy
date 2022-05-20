import datetime as dt
import gc
import warnings
from calendar import monthrange
from collections import defaultdict, namedtuple
from dateutil.relativedelta import relativedelta
from functools import lru_cache

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize, OptimizeWarning
from tqdm import tqdm

import lgimapy.vis as vis
from lgimapy.bloomberg import get_bloomberg_ticker
from lgimapy.data import Database, TBond, SyntheticTBill, TreasuryCurve
from lgimapy.utils import nearest, mkdir, root, smooth_weight

vis.style()

# %%
def svensson(t, B):
    """Vectorized Svensson model of instaneous forward rate."""
    with np.errstate(divide="ignore", invalid="ignore"):
        gam1 = t / B[4]
        gam2 = t / B[5]
        aux1 = 1 - np.exp(-gam1)
        aux2 = 1 - np.exp(-gam2)
        return np.where(
            t == 0,
            B[0] + B[1],
            np.nan_to_num(
                B[0]
                + B[1] * aux1 / gam1
                + B[2] * (aux1 / gam1 + aux1 - 1)
                + B[3] * (aux2 / gam2 + aux2 - 1)
            ),
        )


def clean_treasuries(df):
    """
    Get clean treasuries bond data for building the model curve.

    Keep only non-callable treasuries. Additionally,
    remove bonds that have current maturities less than
    the original maturity of another tenor (e.g., remove
    30 year bonds once their matuirty is less than 10 years,
    10 and 7,7 and 5, etc.).

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Get Bloomberg ticker.
    df["BTicker"] = get_bloomberg_ticker(df["CUSIP"].values)
    treasury_tickers = {"US/T", "T"}
    strip_tickers = {"S", "SP", "SPX", "SPY"}
    bill_tickers = set("B")

    # Make maturity map for finding on-the-run bonds.
    otr_maturity_map = {
        30: 10,
        10: 7,
        8: 5,
        7: 5,
        5: 3,
        3: 2,
        2: 3 / 12,
    }
    next_og_tenor = df["OriginalMaturity"].map(otr_maturity_map)
    is_on_the_run = (df["MaturityYears"] - next_og_tenor) > 0

    is_a_treasury = df["BTicker"].isin(treasury_tickers)
    is_bill_or_strip = df["BTicker"].isin(bill_tickers | strip_tickers)
    is_not_treasury_strip = ~df["BTicker"].isin(strip_tickers)
    has_zero_coupon = (df["CouponType"] == "ZERO COUPON") | (
        (df["CouponType"] == "FIXED") & (df["CouponRate"] == 0)
    )
    has_USD_currency = df["Currency"] == "USD"
    is_US_bond = df["CountryOfRisk"] == "US"
    matures_on_the_15th = df["MaturityDate"].dt.day == 15
    is_not_callable = df["CallType"] == "NONCALL"
    oas_isnt_too_high = df["OAS"] < 40
    oas_isnt_too_low = df["OAS"] > -30
    has_standard_maturity = df["OriginalMaturity"].isin(otr_maturity_map.keys())
    matures_in_more_than_3_months = df["MaturityYears"] > (3 / 12)

    # Apply all rules.
    return df[
        (is_a_treasury | has_zero_coupon)
        & (is_not_treasury_strip | matures_on_the_15th)
        & (is_on_the_run | is_bill_or_strip)
        & is_US_bond
        & has_USD_currency
        & has_standard_maturity
        & is_not_callable
        & oas_isnt_too_high
        & oas_isnt_too_low
        & matures_in_more_than_3_months
    ].copy()


def _exit_opt(Xi):
    """Callback check that raises error if convergence fails."""
    if np.sum(np.isnan(Xi)) > 1:
        raise RuntimeError("Minimize convergence failed.")


def _error_function(self, B, M, C, p, y, method, bonds):
    """
    Calcualte squared error between market value ytm's and ytm's
    estimated using Svensson method with specified beta B.

    Parameters
    ----------
    B: [1 x 6] np.array
        Beta array [B_0, B_1, B_2, B_3, Tau_1, Tau_2].

    Returns
    -------
    squared_error: float
        Total squared error between specified B and market ytm's.
    """
    # Build discount factor matrix respective to cash flow matrix.
    D = np.exp(-M * svensson(M, B))
    p_hat = np.sum(C * D, axis=0)  # theoretical prices

    if method == "price":
        # Minimize sum of squared inverse duration weighted price error.
        inv_durations = 1 / np.array([b.OAD for b in bonds])
        inv_dur_weights = inv_durations / sum(inv_durations)
        squared_error = inv_dur_weights @ ((p - p_hat) ** 2)
    elif method == "yield":
        # Minimize sum of squared yield error.
        y_hat = np.array(  # theoretical yields
            [b.theoretical_ytm(p) for b, p in zip(bonds, p_hat)]
        )
        squared_error = np.sum((y - y_hat) ** 2)
    else:
        raise ValueError("method must be 'price' or 'yeild'.")
    return squared_error


def _minimize_func(params, bnds, error_func, solver):
    try:
        opt = minimize(
            error_func,
            x0=params,
            method=solver,
            bounds=bnds,
            tol=1e-4,
            options={"disp": False, "maxiter": 1000},
            callback=_exit_opt,
        )
    except RuntimeError:
        opt = namedtuple("opt_res", ["status"])
        opt.status = 1
    return opt


class TreasuryCurveBuilder:
    """
    Class for fitting treasury curve and visualizing curve.

    Parameters
    ----------
    ix: :class`:`Index`
        :class:`Index` containing treasuries and strips
        for a single date.
    date: datetime
        Date to build treasury curve.
    """

    def __init__(self, ix=None, date=None):
        self._curves_df = self._load("treasury_curves")
        self._params_df = self._load("treasury_curve_krd_params")
        self._trade_dates = list(self._curves_df.index)
        t_mats = [0.5, 2, 5, 10, 20, 30]
        self._t_mats = np.array(t_mats)
        self._KRD_cols = [f"KRD_{t}" for t in t_mats]
        self._coupon_cols = [f"c_{t}" for t in t_mats]
        self._tret_cols = [f"tret_{t}" for t in t_mats] + ["tret_cash"]

        if ix is not None:
            # Load all index features.
            ix = ix.copy()
            self._full_df = ix.df.copy()  # for debugging
            df = clean_treasuries(ix.df)

            tsy_tickers = {"US/T", "T"}
            strip_tickers = {"SP", "S", "SPY", "SPX"}
            self._treasuries_df = df[df["BTicker"].isin(tsy_tickers)].copy()
            self._strips_df = df[df["BTicker"].isin(strip_tickers)].copy()
            self._bills_df = df[df["BTicker"] == "B"].copy()
            self.df = df.copy()
            self.date = df["Date"].iloc[0]
            self._all_bonds = self._preprocess_bonds()
        else:
            # Load parameters from parquet file.
            self.date = pd.to_datetime(date)
            self.curve = self._curves_df.loc[self.date, :]

        # Find iloc of yesterday in historical data.
        if self._trade_dates and self.date > self._trade_dates[-1]:
            yesterday = self._trade_dates[len(self._trade_dates) - 1]
            self._yesterday_curve = TreasuryCurve(date=yesterday)
        else:
            if self._trade_dates:
                # File exists.
                dates = list(Database().trade_dates())
                yesterday = dates[dates.index(self.date) - 1]
                self._yesterday_curve = TreasuryCurve(date=yesterday)
            else:
                # File does not exist.
                self._yesterday_curve = False

    def _preprocess_bonds(self):
        """
        Preprocess bonds, removing bonds for which
        a ytm can not be solved.

        Returns
        -------
        bonds: List[:class:`TBond`].
            Bonds to pass to fitting routine.
        """
        # Convert DataFrames to treasury bonds.
        bonds = [TBond(bond) for _, bond in self._treasuries_df.iterrows()]
        strips = [TBond(strip) for _, strip in self._strips_df.iterrows()]
        bills = [TBond(bill) for _, bill in self._bills_df.iterrows()]

        # Add strips and bills with maturity less than 3 years or less
        # than shortest treasury if it is greater than 3 years.
        min_mat = max(3, min([b.MaturityYears for b in bonds]))
        bonds += [s for s in strips if s.MaturityYears < min_mat]
        bonds += [b for b in bills if b.MaturityYears < min_mat]

        # Remove bonds with default ytm of 0.02.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bonds = [b for b in bonds if b.ytm != 0.02 and b.ytm > 0]
        return bonds

    def _load(self, fid):
        """
        Load saved data as DataFrame if it exists, otherwise
        return a blank DataFrame.
        """
        try:
            return pd.read_parquet(root(f"data/{fid}.parquet"))
        except FileNotFoundError:
            return pd.DataFrame()

    @property
    @lru_cache(maxsize=None)
    def _fed_funds_rate(self):
        """float: fed funds rate as decimal number"""
        df = pd.read_csv(
            root("data/fed_funds.csv"),
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
        )
        return df.loc[self.date, "PX_MID"] / 100

    def fit(
        self,
        model="NSS",
        method="price",
        n=50,
        n_drop=20,
        threshold=12,
        fit_strips=True,
        fit_bills=True,
        solver="Nelder-Mead",
        verbose=0,
    ):
        """
        Solve zero coupon bond curve given all bonds by minimizing
        the total squared error between the theoretical yield curve
        and market observed yield to maturities. Parametrize
        several maturity ranges, and then smooth combine curves
        into smooth curve.

        Methodolgy and notation taken from
        - https://www.jstatsoft.org/article/view/v036i01/v36i01.pdf

        Parameters
        ----------
        model: {'NSS', 'NS'}, default='NSS'

        method: {'price', 'yield'}, default='price'
            Minimize error in inversion duration weighted price or yield.
        n: int, default=50
            Number of minimizations to run for final yield curve.
        n_drop: int, default=20
            Number of minimizations to run before dropping bonds
            with large errors.
        theshold: float, default=10
            Threshold (bp) that ytm errors must be within to not drop bonds
            from fitting routine.
        solver: str, default='SLSQP'
            Scipy solver to use when minimizing errors.
        verbose: int, default=0
            Verbosity level for optimization.

                * 0: No results.
                * 1: Updated each time bonds are dropped.
                * 2: Update each completed optimization simulation.
                * 3: Update each optimization iteration.
        """
        self.model = model
        self._method = method
        self._solver = solver
        self._verbose = verbose
        self._n_drop = n_drop
        self._n_iters = n
        self._fit_strips = fit_strips
        self._fit_bills = fit_bills
        self._threshold = threshold / 1e4
        self.bonds = []
        if self._verbose >= 1:
            print(
                f"Fitting Treasury Curve for {self.date.strftime('%m/%d/%Y')}"
            )

        # Get Svensson parameters for each maturity range of the curve.
        self._partial_curve_params = {}
        self._curve_fit_ranges = {
            (0, 0): (0, 0.5),
            (0, 3): (0, 3),
            (0, 7): (2, 6),
            (1, 12): (5, 11),
            (1, 18): (9, 16),
            (5, 25): (14, 22),
            (5, 30): (20, 30),
            (100, 100): (30, 100),
        }
        for fit_range in self._curve_fit_ranges.keys():
            if fit_range in {(0, 0), (100, 100)}:
                # Dont need to be fit as they use linear extrapolation.
                continue

            key = "{}-{}".format(*fit_range)
            if fit_range[1] == 30:
                self._30yr_key = key

            self._fit_range(fit_range)
            self._partial_curve_params[key] = self._B

        self.curve = self._combine_curves(self._partial_curve_params)
        if self._verbose >= 1:
            print("Complete\n")

    def _combine_curves(self, svensson_params):
        """
        Combine curves from all maturity ranges and smooth the
        result.

        Parameters
        ----------
        svensonn_params: Dict[str: ndarray].
            Dictionary of maturity ranges and associated Svensson
            parameters.

        Returns
        -------
        pd.Series
            Smoothed combined curve.
        """
        if self._verbose >= 1:
            print(f"  Combining Curves")

        step = 1 / 96

        yield_30_yr = svensson(30, svensson_params[self._30yr_key])
        yield_100_yr = yield_30_yr - 1e-3

        # Build DataFrame with curve values for different maturity
        # ranges in each column.
        df = pd.DataFrame(index=(np.arange(0, 100 + step / 2, step).round(5)))
        for fit_range, curve_range in self._curve_fit_ranges.items():
            mat_lbl = "{}-{}".format(*fit_range)
            t = np.arange(*curve_range, step).round(5)
            if fit_range == (0, 0):
                df.loc[t, mat_lbl] = np.zeros(len(t)) + self._fed_funds_rate
            elif fit_range == (100, 100):
                df.loc[t, mat_lbl] = np.zeros(len(t)) + yield_100_yr
            else:
                df.loc[t, mat_lbl] = svensson(t, svensson_params[mat_lbl])

        t_30_100 = np.arange(30, 100, step).round(5)
        df.loc[t_30_100, self._30yr_key] = np.zeros(len(t_30_100)) + yield_30_yr

        # Fill non-overlapping positions.
        no_overlap_df = df[df.isnull().sum(axis=1) == len(df.columns) - 1]
        for col in no_overlap_df.columns:
            no_overlap_col = no_overlap_df[col].dropna()
            df.loc[no_overlap_col.index, self.date] = no_overlap_col

        # Perform smoothed interpolation on overlapping positions.
        for i in range(len(df.columns) - 2):
            overlap_df = df.iloc[:, i : i + 2].dropna()
            weight = smooth_weight(np.linspace(0, 1, len(overlap_df)), B=2)
            df.loc[overlap_df.index, self.date] = overlap_df.iloc[:, 0] * (
                1 - weight
            ) + overlap_df.iloc[:, 1] * (weight)

        # Add 100 year yield.
        df.loc[100.00000, self.date] = yield_100_yr

        # Use few points for period beyond 30 years.
        mask = np.concatenate(
            [
                np.arange(0, 30, step).round(5),
                np.arange(30, 100, step * 24).round(5),
                [100.00000],
            ]
        )
        return df.loc[mask, self.date]

    def _fit_range(self, maturity_range):
        """
        Solve zero coupon bond for specific maturity range minimizing
        the total squared error between the theoretical yield curve
        and market observed yield to maturities.

        Parameters
        ----------
        maturity_range: Tuple[int].
            Start and end maturity for fitting routine.
        """
        self._maturity_range = maturity_range
        if self._verbose >= 1:
            print("  Fitting {} - {} Year Maturites".format(*maturity_range))

        # Subset bonds to specified bonds to fit.
        bonds_to_fit = ["T"]
        if self._fit_strips:
            bonds_to_fit += ["SP"]
        if self._fit_bills:
            bonds_to_fit += ["B"]
        self._bonds = [
            bond
            for bond in self._all_bonds
            if bond.BTicker in bonds_to_fit
            and maturity_range[0] <= bond.MaturityYears <= maturity_range[1]
        ]
        # Add synthetic TBills to fit front end of the curve.
        if maturity_range == (0, 3):
            synthetic_maturities = range(8, 26, 2)
            synthetic_bonds = [
                SyntheticTBill(self.date, self._fed_funds_rate, day)
                for day in synthetic_maturities
            ]
        else:
            synthetic_bonds = []
        self._bonds = self._bonds + synthetic_bonds

        # Repeatedly fit treasury curve until there are no
        # outlier bonds remaining.
        i = 1
        n_outlier_bonds = 100
        while n_outlier_bonds > 0:
            self._fit(self._n_drop)

            # Get actual TBonds, ignoring synthetic TBills.
            bonds, resids = [], []
            for bond, resid in zip(self._bonds, self._resid):
                if isinstance(bond, TBond):
                    bonds.append(bond)
                    resids.append(resid)
            resids = np.array(resids)
            # Drop up to 3 bonds with error greater than threshold
            # unless fitting the 0-3 year curve, then only drop
            # 1 at a time.
            n_max_drop = {(0, 3): 1}.get(maturity_range, 3)
            r_max_drop = np.argpartition(resids, -n_max_drop)[-n_max_drop:]
            n_outlier_bonds = min(
                n_max_drop, len(resids[resids > self._threshold])
            )
            keep_bonds, dropped_bonds = [], []
            for j, (bond, resid) in enumerate(zip(bonds, resids)):
                if not (j in r_max_drop and resid > self._threshold):
                    keep_bonds.append(bond)
                else:
                    dropped_bonds.append(bond)
            self._bonds = keep_bonds + synthetic_bonds  # update bond
            # Remove dropped bonds from 0-3 range for rest of fittings.
            if maturity_range == (0, 3):
                self._all_bonds = [
                    bond
                    for bond in self._all_bonds
                    if bond not in dropped_bonds
                ]

            if self._verbose >= 1:
                print(f"    Fit iter {i} | Bonds Dropped: {n_outlier_bonds}")
                i += 1
        self._fit(self._n_iters)

        # Save bonds used in fit.
        used_bond_names = set([str(b) for b in self.bonds])
        self.bonds += [b for b in self._bonds if str(b) not in used_bond_names]

    def _fit(self, n):
        """
        Solve zero coupon bond curve given all bonds by minimizing
        the total squared error between the theoretical yield curve
        and market observed yield to maturities.

        Methodolgy and notation taken from:
        - https://www.jstatsoft.org/article/view/v036i01/v36i01.pdf

        Parameters
        ----------
        n: int
            Number of minimizations to run.
        """

        # Build matrices with rows as coupon payments and columns as bonds.
        self._n = max(len(b.coupon_dates) for b in self._bonds)
        self._m = len(self._bonds)
        self._M = np.zeros([self._n, self._m])
        self._C = np.zeros([self._n, self._m])

        # Fill matrices, M: years until payment and C: cash flows.
        for j, b in enumerate(self._bonds):
            for i, (cf, cy) in enumerate(zip(b.cash_flows, b.coupon_years)):
                self._C[i, j] += cf
                self._M[i, j] += cy

        # Calculate market value ytm and get dirty price for each bond.
        self._y = np.array([b.ytm for b in self._bonds])
        self._p = np.array([b.DirtyPrice for b in self._bonds])

        # Set up inits for optimizatios.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", OptimizeWarning)

            if self.model == "NS" or self._maturity_range == (0, 3):
                bnds = [(0, 15), (-15, 30), (-30, 30), (0, 0), (0, 5), (0, 0)]
            elif self.model == "NSS":
                bnds = [
                    (0, 15),
                    (-15, 30),
                    (-30, 30),
                    (-30, 30),
                    (0, 5),
                    (5, 10),
                ]
            else:
                raise ValueError("Model must be either NSS or NS.")

        # Randomly sample bounds to create n initial parameter combinations.
        init_params = np.zeros([n, len(bnds)])
        for j, bnd in enumerate(bnds):
            init_params[:, j] = np.random.uniform(*bnds[j], size=n)

        # Perform n optimizations, storing all results.
        beta_res = np.zeros([n, len(bnds)])
        rmse_res = np.zeros([n])

        # pool = mp.Pool(processes=mp.cpu_count() - 2)
        # mp_res = [
        #     pool.apply_async(
        #         _minimize_func,
        #         args=(params, bnds, self._error_function, self._solver),
        #     )
        #     for params in init_params
        # ]
        # opts = [p.get() for p in mp_res]

        # opts = joblib.Parallel(n_jobs=6)(
        #     joblib.delayed(_minimize_func)(
        #         params,
        #         bnds,
        #         lambda x: _error_function(
        #             x,
        #             self._M,
        #             self._C,
        #             self._p,
        #             self._y,
        #             self._method,
        #             self._bonds,
        #         ),
        #         self._solver,
        #     )
        #     for params in init_params
        # )

        # for i, opt in enumerate(opts):
        #     if opt.status != 0:
        #         rmse_res[i] = 1e9  # arbitrary high error value
        #     else:
        #         # Succesful optimization.
        #         beta_res[i, :] = opt.x
        #         rmse_res[i] = opt.fun
        #     if self._verbose >= 2:
        #         print(f"      Iteration {i+1} | RMSE: {rmse_res[i]:.4f}")
        for i, params in enumerate(init_params):
            try:
                self._opt = minimize(
                    self._error_function,
                    x0=params,
                    method=self._solver,
                    bounds=bnds,
                    tol=1e-4,
                    options={"disp": False, "maxiter": 1000},
                    callback=_exit_opt,
                )
            except RuntimeError:
                self._opt = namedtuple("opt_res", ["status"])
                self._opt.status = 1

            if self._opt.status != 0:
                rmse_res[i] = 1e9  # arbitrary high error value
            else:
                # Succesful optimization.
                beta_res[i, :] = self._opt.x
                rmse_res[i] = self._opt.fun
            if self._verbose >= 2:
                print(f"      Iteration {i+1} | RMSE: {rmse_res[i]:.4f}")

        # Store best results.
        if min(rmse_res) == 1e9:  # no succesful optimizations.
            raise RuntimeError("Optimization failed to converge.")
        self._B = beta_res[np.argmin(rmse_res), :]

        # Re-calculate theoretical ytm's for each bond.
        D = np.exp(-self._M * svensson(self._M, self._B))
        p_hat = np.sum(self._C * D, axis=0)
        self._y_hat = np.array(  # theoretical yields
            [b.theoretical_ytm(p) for b, p in zip(self._bonds, p_hat)]
        )
        self._resid = np.abs(self._y - self._y_hat)

    def _error_function(self, B):
        """
        Calcualte squared error between market value ytm's and ytm's
        estimated using Svensson method with specified beta B.

        Parameters
        ----------
        B: [1 x 6] np.array
            Beta array [B_0, B_1, B_2, B_3, Tau_1, Tau_2].

        Returns
        -------
        squared_error: float
            Total squared error between specified B and market ytm's.
        """
        # Build discount factor matrix respective to cash flow matrix.
        D = np.exp(-self._M * svensson(self._M, B))
        p_hat = np.sum(self._C * D, axis=0)  # theoretical prices

        if self._method == "price":
            # Minimize sum of squared inverse duration weighted price error.
            inv_durations = 1 / np.array([b.OAD for b in self._bonds])
            inv_dur_weights = inv_durations / sum(inv_durations)
            squared_error = inv_dur_weights @ ((self._p - p_hat) ** 2)
        elif self._method == "yield":
            # Minimize sum of squared yield error.
            y_hat = np.array(  # theoretical yields
                [b.theoretical_ytm(p) for b, p in zip(self._bonds, p_hat)]
            )
            squared_error = np.sum((self._y - y_hat) ** 2)
        else:
            raise ValueError("method must be 'price' or 'yeild'.")
        return squared_error

    def save(self):
        """
        Save treasury curve values to `data/treasury_curves.parquet`
        and save key rate durations, coupons, and total returns to
        `data/treasury_curve_krd_params.parquet`.
        """
        # Get treasury curve parameters.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            krds, coupons = self._get_KRDs_and_coupons()

        if self._yesterday_curve:
            # Have data for yesterday, calculate total returns.
            trets = self._get_KRD_total_returns()
        else:
            # No data for yesterday, fill total returns with NaNs.
            trets = np.array([np.nan] * 7)

        # Create single row to append to curves DataFrame.
        new_row = pd.DataFrame(self.curve).T
        fid = root("data/treasury_curves.parquet")
        if len(self._curves_df):
            # File exists, append to it and sort.
            df = self._curves_df.copy()
            df.columns = df.columns.astype(float)
            df = df[df.index != self.date].copy()  # drop date if it exists.
            df = pd.concat((df, new_row))
            if df.index[-1] != np.max(df.index):
                # Date added is not last date in file, sort file.
                df.sort_index(inplace=True)
            df.columns = df.columns.astype(str)
            df.to_parquet(fid)
        else:
            # Create parquet with first row.
            new_row.to_parquet(fid)

        # Create single row to append to KRD params DataFrame.
        row_vals = np.concatenate([krds, coupons, trets])
        cols = [*self._KRD_cols, *self._coupon_cols, *self._tret_cols]
        new_row = pd.DataFrame(row_vals, index=cols, columns=[self.date]).T
        fid = root("data/treasury_curve_krd_params.parquet")
        if len(self._params_df):
            # File exists, append to it and sort.
            df = self._params_df.copy()
            df = df[df.index != self.date].copy()  # drop date if it exists.
            df = pd.concat((df, new_row))
            if df.index[-1] != np.max(df.index):
                # Date added is not last date in file, sort file.
                df.sort_index(inplace=True)
            df[cols].to_parquet(fid)
        else:
            # Create parquet with first row.
            new_row.to_parquet(fid)

        # Save figure.
        fig_dir = root("data/treasury_curves")
        mkdir(fig_dir)
        fig, ax = vis.subplots(1, 1, figsize=(8, 6))
        self.plot(ax=ax)
        vis.savefig(fig_dir / self.date.strftime("%Y_%m_%d"))
        vis.close(fig)

    def _get_yield(self, t):
        """
        Vectorized implementation to get yield(s) for a
        given date and maturities.

        Parameters
        ----------
        t: float, nd.array[float].
            Maturity or maturities (yrs) to return yields for.

        Returns
        -------
        float, ndarray[float].
            Yields for specified maturities.
        """
        return np.interp(t, self.curve.index, self.curve.values)

    def _get_KRDs_and_coupons(self):
        """
        Memoized key rate durations for specified date and
        maturities of [0.5, 2, 5, 10, 20, 30] years.

        -------
        krds: [1x6] ndarray
            KRDs for [0.5, 2, 5, 10, 20, 30] years.
        """
        krds = np.zeros(len(self._t_mats))
        coupons = np.zeros(len(self._t_mats))
        for i, t in enumerate(self._t_mats):
            krds[i], coupons[i] = self._solve_krd_coupon(t)
        return krds, coupons

    def _solve_krd_coupon(self, t):
        """
        Solve key rate duration for specified maturity.

        Parameters
        ----------
        t: float
            Maturity in years.

        Returns
        -------
        oad: float
            Key rate duration for specified maturity.
        coupon: float
            Coupon which makes treasury a par bond.
        """
        # Solve for coupon required to create par bond
        # and create cash flows.
        self._c_yrs = np.arange(0.5, t + 0.5, 0.5)
        self._yields = self._get_yield(self._c_yrs)
        coupon = fsolve(self._coupon_func, 0)[0]
        cash_flows = np.zeros(len(self._c_yrs)) + coupon
        cash_flows[-1] += 100  # par bond
        oad = (
            sum(
                t * cf * np.exp(-y * t)
                for cf, y, t in zip(cash_flows, self._yields, self._c_yrs)
            )
            / 100
        )
        # Note that using continuously compounding yields
        # DMAC = DMOD which = OAD for optionless treasuries.
        return oad, coupon

    def _get_KRD_total_returns(self):
        """
        Reprice hypothetical par bond used for
        KRDs with a new yield curve.

        Returns
        -------
        tot_rets: [1 x 7] ndarray
            Total returns for KRD times and cash.
        """
        # Time since last trade date in years.
        t_yrs = (self.date - self._yesterday_curve._date).days / 365

        # Solve price of all bonds given yesterdays coupons.
        tot_rets = np.zeros(len(self._t_mats) + 1)
        for i, (y_c, tmat) in enumerate(
            zip(self._yesterday_curve.coupons(), self._t_mats)
        ):
            c_yrs = np.arange(0.5, tmat + 0.5, 0.5) - t_yrs
            cash_flows = np.zeros(len(c_yrs)) + y_c
            cash_flows[-1] += 100  # par bond terminal payment
            tot_rets[i] = sum(
                cf * np.exp(-y * t)
                for cf, y, t in zip(cash_flows, self._get_yield(c_yrs), c_yrs)
            )
        # Solve next day cash price based on yesterday's yield.
        yield_t_yrs = float(self._yesterday_curve.yields(t_yrs))
        tot_rets[i + 1] = 100 * np.exp(yield_t_yrs * t_yrs)
        # Convert prices to total returns.
        tot_rets -= 100
        tot_rets /= 100
        return tot_rets

    def _coupon_func(self, coupon):
        """
        Return error in price of a par bond, with maturity
        from :meth:`TreasuryCurveBuilder._solve_krd`
        and specified coupon.
        """
        # Create cash flow and yield vectors.
        cash_flows = np.zeros(len(self._c_yrs)) + coupon
        cash_flows[-1] += 100  # par bond terminal payment

        # Find error in price difference.
        error = (
            sum(
                cf * np.exp(-y * t)
                for cf, y, t in zip(cash_flows, self._yields, self._c_yrs)
            )
            - 100
        )
        return error

    def plot_discount_curve(self, trange=(0.1, 30), ax=None, figsize=(8, 6)):
        """
        Plot discount curve.

        Parameters
        ----------
        trange: Tuple(float, float), default=(0.1, 30).
            Range (min, max) in years to show.
        indv_bonds: bool, default=False
            If true, show individual bond yields as data points.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6).
            Figure size.
        """
        if ax is None:
            fig, ax = vis.subplots(1, 1, figsize=figsize)

        # Plot discount curve.
        d = np.exp(-self.curve.index * self.curve.values)
        ax.plot(self.curve.index, d, color="steelblue", lw=2)
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Discount")

    def plot(
        self,
        trange=(0.1, 30),
        indv_bonds=True,
        strips=True,
        ax=None,
        figsize=(8, 6),
        **kwargs,
    ):
        """
        Plot yield curve.

        Parameters
        ----------
        trange: Tuple(float, float), default=(0.1, 30).
            Range (min, max) in years to show.
        indv_bonds: bool, default=True,
            If true plot the market value YTM and estimated YTM of
            all bonds used to fit curve.
        strips: bool, default=True
            If true plot strips YTM.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6).
            Figure size.
        kwargs: dict
            Kwargs for plotting curve in matplotlib.
        """
        if ax is None:
            fig, ax = vis.subplots(1, 1, figsize=figsize)

        plot_kwargs = {
            "color": "k",
            "alpha": 0.8,
            "lw": 1.5,
            "label": "Zero Curve",
        }
        plot_kwargs.update(**kwargs)

        # Plot yield curve.
        ax.plot(self.curve, **plot_kwargs)

        if indv_bonds:
            # Compute YTM and curve estimated YTM for each
            # bond used in fitting process.
            d = defaultdict(list)
            for b in self.bonds:
                d["mats"].append(b.MaturityYears)
                d["yields"].append(b.ytm)
                est_price = b.calculate_price_with_curve(self.curve)
                d["est_yields"].append(b.theoretical_ytm(est_price))

            # Plot market value and estimated YTMs.
            ax.plot(
                d["mats"],
                d["yields"],
                "o",
                color="firebrick",
                ms=4,
                alpha=0.5,
                label="Market Value YTM",
            )
            ax.plot(
                d["mats"],
                d["est_yields"],
                "o",
                color="darkgreen",
                ms=4,
                alpha=0.5,
                label="Model Implied YTM",
            )

        if strips:
            # Plot treasury strip yields.
            sdf = self._strips_df.copy()
            sdf["y"] = -np.log(sdf["DirtyPrice"] / 100) / sdf["MaturityYears"]
            ax.plot(
                sdf["MaturityYears"],
                sdf["y"],
                "o",
                c="steelblue",
                ms=4,
                alpha=0.5,
                label="Strips YTM",
            )

        ax.legend()
        ax.set_xlim(trange[0] - 0.2, trange[1] + 0.2)
        tick = mtick.StrMethodFormatter("{x:.2%}")
        ax.yaxis.set_major_formatter(tick)
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Yield")
        ax.set_title(self.date.strftime("%m/%d/%Y"))


def update_treasury_curve_dates(dates=None, verbose=True):
    """
    Update `.data/treasury_curve_params.parquet` file
    with most recent data.

    Parameters
    ----------
    dates: List[datetime].
        List of all trade dates available in DataMart.
    verbose: bool, default=True
        If True print progress on each fitting to screen.
    """
    # Get list of dates with treasury curves
    # dates = None
    # verbose = True
    db = Database()

    # Get list of previously scraped dates if it exists.
    try:
        scraped_dates = TreasuryCurve().trade_dates
    except FileNotFoundError:
        scraped_dates = []

    # Scrape all dates if they have not been scraped
    # or if the day immediately preceding has not been
    # scraped, saving the curve and KRD parameters.
    i = 0
    for date in db.trade_dates(start=db.date("MARKET_START")):
        if date in scraped_dates and i == 0:
            continue
        # else:
        # break
        db.load_market_data(date=date, local=False)
        treasury_ix = db.build_market_index(
            drop_treasuries=False, sector="TREASURIES"
        )
        tcb = TreasuryCurveBuilder(treasury_ix)
        tcb.fit(
            verbose=0,
            threshold=12,
            n=70,
            n_drop=30,
            solver="Nelder-Mead",
        )
        tcb.save()

        del tcb
        gc.collect()

        if date in scraped_dates:
            i -= 1
        else:
            i = 1


def update_specific_date(specified_date, plot=True, **kwargs):
    """
    Update specific date's treasury curve. Useful for correcting
    dates in which default parameters did not yield a good
    fit. Additionally updates the next day's curve using default
    parameters, as total returns used previous day's curve.

    specified_date: datetime
        Date to re-fit curve for.
    plot: bool, default=True
        If True plot the two re-fit curves over
        the full month of curves.
    kwargs: dict
        Kwargs for :meth:`TreasuryCurveBuilder.fit`.
    """

    db = Database()
    tc = TreasuryCurve()

    # Find date and next date.
    month, day, year = (int(x) for x in specified_date.split("/"))
    next_date = db.trade_dates(start=specified_date)[1]

    # Fit curve for both days.
    db.load_market_data(date=specified_date, local=False)
    tcb = TreasuryCurveBuilder(
        db.build_market_index(drop_treasuries=False, sector="TREASURIES")
    )
    tcb.fit(verbose=1)
    tcb.save()

    db.load_market_data(date=next_date, local=False)
    tcb = TreasuryCurveBuilder(
        db.build_market_index(drop_treasuries=False, sector="TREASURIES")
    )
    tcb.fit(verbose=1)
    tcb.save()

    if plot:
        # Plot full month of curves.
        next_month = 1 if month == 12 else month + 1
        next_year = year + 1 if month == 12 else year
        start = pd.to_datetime(f"{month}/1/{year}")
        end = pd.to_datetime(f"{next_month}/1/{next_year}")
        dates = tc.trade_dates(start=start, exclusive_end=end)

        t = np.linspace(0, 30, 400)
        sns.set_palette("viridis_r", len(dates))
        fig, ax = vis.subplots(1, 1, figsize=(12, 8))
        for date in dates:
            curve = tc.yields(t, date=date)
            ax.plot(t, curve.values, lw=1.5, alpha=0.5)

        # Plot specified date's curve.
        curve = tc.yields(t, date=specified_date)
        ax.plot(
            t,
            curve.values,
            lw=3,
            c="k",
            label=f"Specified Date\n{specified_date}",
        )

        # Plot next date's curve.
        curve = tc.yields(t, date=next_date)
        ax.plot(
            t,
            curve.values,
            lw=3,
            c="firebrick",
            label=f'Next Date\n{next_date.strftime("%m/%d/%Y")}',
        )

        tick = mtick.StrMethodFormatter("{x:.2%}")
        ax.yaxis.set_major_formatter(tick)
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Yield")
        ax.legend()
        vis.show()


# %%
def main():
    # %%

    # update_treasury_curve_dates()

    bad_dates = [
        "10/31/2005",
        "11/10/2005",
        "11/17/2005",
        "11/21/2005",
        "11/23/2005",
        "11/28/2005",
        "11/29/2005",
        "11/30/2005",
        "12/24/2007",
        "1/18/2008",
        "3/11/2008",
        "3/14/2008",
        "3/27/2008",
        "3/28/2008",
        "4/2/2008",
        "4/3/2008",
        "4/4/2008",
        "4/8/2008",
        "4/25/2008",
        "7/15/2008",
        "9/29/2008",
        "12/5/2008",
        "12/8/2008",
        "12/12/2008",
        "12/15/2008",
        "3/2/2009",
        "3/3/2009",
        "3/13/2009",
        "3/16/2009",
        "12/3/2009",
    ]
    for date in bad_dates:
        print(date)
        update_specific_date(date, plot=False)


def update_historical_period(start, end=None):
    from tqdm import tqdm

    db = Database()
    for date in tqdm(db.trade_dates(start=start, end=end)):
        db.load_market_data(date=date)
        treasury_ix = db.build_market_index(
            drop_treasuries=False, sector="TREASURIES"
        )
        tcb = TreasuryCurveBuilder(treasury_ix)
        tcb.fit(
            verbose=0,
            threshold=12,
            n=70,
            n_drop=30,
            solver="Nelder-Mead",
        )
        tcb.save()


if __name__ == "__main__":
    update_historical_period(start="3/30/2022")


# %%
