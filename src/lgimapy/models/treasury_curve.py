import datetime as dt
import warnings
from calendar import monthrange
from collections import namedtuple
from dateutil.relativedelta import relativedelta
from functools import lru_cache

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize

from lgimapy.utils import nearest, mkdir, root

plt.style.use("fivethirtyeight")
# %matplotlib qt

# %%
class TreasuryCurveBuilder:
    """
    Class for fitting treasury curve and visualizing curve.

    Parameters
    ----------
    ix: Index
        Index class form IndexBuilder containing treasuries
        for a single date.

    Methods
    -------
    fit(): Fit curve to input data.
    plot(): Plot fitted treasury curve.
    """

    def __init__(self, ix):
        self._strips_df = ix.df[ix.df["Ticker"] == "SP"]
        ix.clean_treasuries()
        self.df = ix.df
        self.date = ix.dates[0]
        self.bonds = [TBond(bond) for _, bond in self.df.iterrows()]

    def fit(
        self,
        method="price",
        n=25,
        n_drop=10,
        threshold=12,
        solver="SLSQP",
        verbose=0,
    ):
        """
        Solve zero coupon bond curve given all bonds by minimizing
        the total squared error between the theoretical yield curve
        and market observed yield to maturities.

        Methodolgy and notation taken from
        - https://www.jstatsoft.org/article/view/v036i01/v36i01.pdf

        Parameters
        ----------
        init_params: {'price', 'yield'}, default='price'
            Minimize error in inversion duration weighted price or yield.
        n: int, default=20
            Number of minimizations to run for final yield curve.
        n_drop: int, default=10
            Number of minimizations to run before dropping bonds
            with large errors.
        theshold: float, default=12
            Threshold (bp) that ytm errors must be within to not drop bonds
            from fitting routine.
        solver: str, default='SLSQP'
            Scipy solver to use when minimizing errors.
        verbose: int, default=0
            Verbosity level for optimization.
                - 0: No results.
                - 1: Updated each time bonds are dropped.
                - 2: Update each completed optimization simulation.
                - 3: Update each optimization iteration.
        """
        self._method = method
        self._n_iters = n
        self._solver = solver
        self._verbose = verbose

        n_bad_bonds = 100  # init
        threshold /= 1e4
        i = 1
        while n_bad_bonds > 0:
            self._fit(n_drop)
            r = self._resid
            # Drop bonds with error greater than 15 bips.
            n_bad_bonds = len(r[r > threshold])
            self.bonds = [b for b, e in zip(self.bonds, r) if e < threshold]
            if verbose >= 1:
                print(f"Fit iter {i} | Bonds Dropped: {n_bad_bonds}")
                i += 1
        self._fit(n)
        if verbose >= 1:
            print("Fitting complete")

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
        self._n = max(len(b.coupon_dates) for b in self.bonds)
        self._m = len(self.bonds)
        self._M = np.zeros([self._n, self._m])
        self._C = np.zeros([self._n, self._m])

        # Fill matrices, M: years until payment and C: cash flows.
        for j, b in enumerate(self.bonds):
            for i, (cf, cy) in enumerate(zip(b.cash_flows, b.coupon_years)):
                self._C[i, j] += cf
                self._M[i, j] += cy

        # Calculate market value ytm and get dirty price for each bond.
        self._y = np.array([b.ytm for b in self.bonds])
        self._p = np.array([b.DirtyPrice for b in self.bonds])

        def exit_opt(Xi):
            """Callback check that raises error if convergence fails."""
            if np.sum(np.isnan(Xi)) > 1:
                raise RuntimeError("Minimize convergence failed.")

        def printx(Xi):
            """Prints optimization updates each iteration."""
            print(Xi)
            exit_opt(Xi)

        # Set up inits for optimizatios.
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        bnds = [(0, 15), (-15, 30), (-30, 30), (-30, 30), (0, 2.5), (2.5, 5.5)]
        n = self._n_iters

        # Randomly sample bounds to create n initial parameter combinations.
        init_params = np.zeros([n, len(bnds)])
        for j, bnd in enumerate(bnds):
            init_params[:, j] = np.random.uniform(*bnds[0], size=n)

        # Perform n optimizations, storing all results.
        beta_res = np.zeros([n, len(bnds)])
        rmse_res = np.zeros([n])
        for i, params in enumerate(init_params):
            try:
                opt = minimize(
                    self._yield_error_func,
                    x0=params,
                    method=self._solver,
                    bounds=bnds,
                    tol=1e-4,
                    options={"disp": False, "maxiter": 1000},
                    callback=printx if self._verbose >= 3 else exit_opt,
                )
            except RuntimeError:
                opt = namedtuple("opt_res", ["status"])
                opt.status = 0

            if opt.status != 0:
                rmse_res[i] = 1e9  # arbitrary high error value
            else:
                # Succesful optimization.
                beta_res[i, :] = opt.x
                rmse_res[i] = opt.fun
            if self._verbose >= 2:
                print(f"Iteration {i+1} | RMSE: {rmse_res[i]:.4f}")
        warnings.simplefilter(action="default", category=RuntimeWarning)

        # Store best results.
        if min(rmse_res) == 1e9:  # no succesful optimizations.
            raise RuntimeError("Optimization failed to converge.")
        self._B = beta_res[np.argmin(rmse_res), :]

        # Re-calculate theoretical ytm's for each bond.
        D = np.exp(-self._M * self._svensson(self._M, self._B))
        p_hat = np.sum(self._C * D, axis=0)
        self._y_hat = np.array(  # theoretical yields
            [b.theoretical_ytm(p) for b, p in zip(self.bonds, p_hat)]
        )
        self._resid = np.abs(self._y - self._y_hat)

    def _yield_error_func(self, B):
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
        D = np.exp(-self._M * self._svensson(self._M, B))
        p_hat = np.sum(self._C * D, axis=0)  # theoretical prices

        if self._method == "price":
            # Minimize sum of squared inverse duration weighted price error.
            inv_durations = 1 / np.array([b.OAD for b in self.bonds])
            inv_dur_weights = inv_durations / sum(inv_durations)
            squared_error = inv_dur_weights @ (self._p - p_hat) ** 2
        elif self._method == "yield":
            # Minimize sum of squared yield error.
            y_hat = np.array(  # theoretical yields
                [b.theoretical_ytm(p) for b, p in zip(self.bonds, p_hat)]
            )
            squared_error = np.sum((self._y - y_hat) ** 2)
        else:
            raise ValueError("method must be 'price' or 'yeild'.")
        return squared_error

    @staticmethod
    def _svensson(t, B):
        """Vectorized Svensson model of instaneous forward rate."""
        with np.errstate(divide="ignore", invalid="ignore"):
            gam1 = t / B[4]
            gam2 = t / B[5]
            aux1 = 1 - np.exp(-gam1)
            aux2 = 1 - np.exp(-gam2)
            return np.nan_to_num(
                B[0]
                + B[1] * aux1 / gam1
                + B[2] * (aux1 / gam1 + aux1 - 1)
                + B[3] * (aux2 / gam2 + aux2 - 1)
            )

    def save(self):
        """Save beta values to `../data/treasury_curve_params.csv`."""
        fid = root("data/treasury_curve_params.csv")
        cols = ["B_0", "B_1", "B_2", "B_3", "Tau_1", "Tau_2"]
        new_row = pd.DataFrame(self._B, index=cols, columns=[self.date]).T

        try:
            df = pd.read_csv(
                fid, index_col=0, parse_dates=True, infer_datetime_format=True
            )
        except FileNotFoundError:
            # File does not exist, create a new one.
            new_row.to_csv(fid)
        else:
            # File exists, append to it and sort.
            df = df[df.index != self.date].copy()  # drop date if it exists.
            df = df.append(new_row)
            df.sort_index(inplace=True)
            df.to_csv(fid)

    def _yield_30_100(self, t):
        """
        Linearly calculate yield curve from 30 to 100 years.
        Use Svensson method for 30 years and subtract 10 bp
        to account for convexity at 100 years.

        Parameters
        ----------
        t: float
            Time (yrs) to calcualt zero yield.

        Returns
        -------
        y_t: float
            Yield at time t.
        """
        y_30 = self._svensson(30, self._B)
        y_100 = y_30 - 1e-3  # 10 bp
        return y_30 + (y_100 - y_30) * (t - 30) / (100 - 30)

    def plot(
        self,
        trange=(0.1, 30),
        indv_bonds=False,
        strips=False,
        ax=None,
        figsize=(8, 6),
    ):
        """
        Plot yield curve.

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
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot yield curve.
        t = np.linspace(0.00001, 30, 1000)
        t = t[(t >= trange[0]) & (t <= trange[1])]
        y = [self._svensson(t_, self._B) for t_ in t]
        if trange[1] > 30:
            ax.plot(
                [30, trange[1]],
                [y[-1], self._yield_30_100(trange[1])],
                color="steelblue",
                label="_nolegend_",
            )
        ax.plot(t, y, color="steelblue", label="_nolegend_")

        if indv_bonds:
            # Plot market value and theoretical YTM for T bonds.
            my = [b.MaturityYears for b in self.bonds]
            ax.plot(
                my,
                self._y,
                "o",
                color="firebrick",
                ms=3,
                label="Market Value YTM",
            )
            ax.plot(
                my,
                self._y_hat,
                "o",
                color="darkgreen",
                ms=3,
                label="Theoretical YTM",
            )
            ax.legend()

        if strips:
            # Plot treasury strip yields.
            sdf = self._strips_df.copy()
            sdf["y"] = -np.log(sdf["DirtyPrice"] / 100) / sdf["MaturityYears"]
            ax.plot(
                sdf["MaturityYears"],
                sdf["y"],
                "o",
                c="darkorange",
                ms=4,
                alpha=0.5,
                label="Strips YTM",
            )
            ax.legend()

        tick = mtick.StrMethodFormatter("{x:.2%}")
        ax.yaxis.set_major_formatter(tick)
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Yield")


# %%
class TBond:
    """
    Class for treasury bond math and manipulation given current
    state of the bond.

    Parameters
    ----------
    series: pd.Series
        Single bond row from `index_builder` DataFrame.


    Attributes
    ----------
    coupon_dates: All coupon dates for given treasury.
    coupon_years: Time to all coupons in years.

    Methods
    -------
    calculate_price(rfr): Calculate price of bond with given risk free rate.
    """

    def __init__(self, s):
        self.s = s
        self.__dict__.update({k: v for k, v in zip(s.index, s.values)})
        self.MaturityDays = (self.MaturityDate - self.Date).days

    @property
    @lru_cache(maxsize=None)
    def coupon_dates(self):
        """
        Return list[datetime object] of timestamps for all coupons.
        Note that all coupons are assumed to be on either the 15th or
        last day of the month, business days and holidays are ignored.
        """
        i_date = self.IssueDate
        dates = []
        if 14 <= i_date.day <= 19:
            # Mid month issue and coupons.
            # Make first first coupon 6 months out.
            temp = i_date.replace(day=15) + relativedelta(months=6)
            while temp <= self.MaturityDate + dt.timedelta(5):
                if temp > self.Date:
                    dates.append(temp)
                temp += relativedelta(months=6)
        else:
            # End of the month issue and coupons.
            # Make first coupon the first of the next month 6 months out.
            temp = (i_date + relativedelta(months=6, days=5)).replace(day=1)
            while temp <= self.MaturityDate + dt.timedelta(5):
                if temp > self.Date - dt.timedelta(1):
                    # Use last day of prev month.
                    dates.append(temp - dt.timedelta(1))
                temp += relativedelta(months=6)

        dates[-1] = self.MaturityDate  # use actual final maturity date
        return dates

    @property
    @lru_cache(maxsize=None)
    def coupon_years(self):
        """Return np.array of time in years for all coupons."""
        return np.array(
            [(cd - self.Date).days / 365 for cd in self.coupon_dates]
        )

    @property
    @lru_cache(maxsize=None)
    def coupon_days(self):
        """Return np.array of time in years for all coupons."""
        return np.array([(cd - self.Date).days for cd in self.coupon_dates])

    @property
    @lru_cache(maxsize=None)
    def cash_flows(self):
        """Return np.array of cash flows to be acquired on `coupon_dates`."""
        cash_flows = self.CouponRate / 2 + np.zeros(len(self.coupon_dates))
        cash_flows[-1] += 100
        return cash_flows

    def calculate_price(self, rfr):
        """
        Calculate theoreticla price of the bond with given
        risk free rate.

        Parameters
        ----------
        rfr: float
            Continuously compouned risk free rate used to discount
            cash flows.

        Returns
        -------
        price: float
            Theoretical price ($) of bond.
        """

        return sum(
            cf * np.exp(-rfr * t)
            for cf, t in zip(self.cash_flows, self.coupon_years)
        )

    def _ytm_func(self, y, price=None):
        """Yield to maturity error function for solver, see `ytm`."""
        error = (
            sum(
                cf * np.exp(-y * t)
                for cf, t in zip(self.cash_flows, self.coupon_years)
            )
            - price
        )
        return error

    @property
    @lru_cache(maxsize=None)
    def ytm(self):
        """
        Memoized yield to maturity of the bond using true coupon
        values, dates, and dirty price.

        Parameters
        ----------
        price: float, default=None
            Price of the bond to use when calculating yield to maturity.
            If no price is given, the market value dirty price is used.

        Returns
        -------
        ytm: float
            Yield to maturity of the bond at specified price.
        """
        x0 = 0.02  # initial guess
        return fsolve(self._ytm_func, x0, args=(self.DirtyPrice))[0]

    def theoretical_ytm(self, price):
        """
        Calculate yield to maturity of the bond using true coupon
        values and dates and specified price.

        Parameters
        ----------
        price: float
            Price of the bond to use when calculating yield to maturity.

        Returns
        -------
        ytm: float
            Yield to maturity of the bond at specified price.
        """
        x0 = 0.02  # initial guess
        return fsolve(self._ytm_func, x0, args=(price))[0]


# %%
class TreasuryCurve:
    """
    Class for loading treasury yield curve and calculating yields
    for specified dates and maturities.

    Attributes
    ----------
    trade_dates: List of traded dates.

    Methods
    -------
    get_yield(date, t): Get yield for specified date and maturities.
    """

    def __init__(self):
        self._df = self._load()
        self._t_mats = np.array([0.5, 2, 5, 10, 20, 30])
        self.trade_dates = list(self._df.index)

    def _load(self):
        """Load data as DataFrame."""
        fid = root("data/treasury_curve_params.csv")
        return pd.read_csv(
            fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )

    @staticmethod
    def _svensson(t, B):
        """Vectorized Svensson model of instaneous forward rate."""
        with np.errstate(divide="ignore", invalid="ignore"):
            gam1 = t / B[4]
            gam2 = t / B[5]
            aux1 = 1 - np.exp(-gam1)
            aux2 = 1 - np.exp(-gam2)
            return np.nan_to_num(
                B[0]
                + B[1] * aux1 / gam1
                + B[2] * (aux1 / gam1 + aux1 - 1)
                + B[3] * (aux2 / gam2 + aux2 - 1)
            )

    def _yield_30_100(self, t, B):
        """
        Linearly calculate yield curve from 30 to 100 years.
        Use Svensson method for 30 years and subtract 10 bp
        to account for convexity at 100 years.

        Parameters
        ----------
        t: float
            Time (yrs) to calcualt zero yield.
        B: [1 x 6] array
            Svensson parameters.

        Returns
        -------
        y_t: float
            Yield at time t.
        """
        y_30 = self._svensson(30, B)
        y_100 = y_30 - 1e-3  # 10 bp
        return y_30 + (y_100 - y_30) * (t - 30) / (100 - 30)

    def get_yield(self, date, t):
        """
        Vectorized implementation to get yield(s) for a
        given date and maturities.

        Parameters
        ----------
        date: datetime object
            Date of yield curve.
        t: float, nd.array[float].
            Maturity or maturities (yrs) to return yields for.

        Returns
        -------
        y: float, ndarray[float].
            Yields for specified maturities.
        """
        date = pd.to_datetime(date)
        try:
            B_params = self._df.loc[date, :].values
        except KeyError:
            msg = f'{date.strftime("%m/%d/%Y")} is not a traded date.'
            raise ValueError(msg)

        if isinstance(t, float) or isinstance(t, int):
            # Single time.
            if t <= 30:
                return self._svensson(t, B_params)
            else:
                return self._yield_30_100(t, B_params)
        elif isinstance(t, np.ndarray):
            t_low, t_high = t[t <= 30], t[t > 30]
            y_low = self._svensson(t_low, B_params)
            y_high = self._yield_30_100(t_high, B_params)
            return np.concatenate([y_low, y_high])

    @lru_cache(maxsize=None)
    def get_KRD_yields(self, date):
        """
        Memoized yields for specified date and
        maturities of [0.5, 2, 5, 10, 20, 30] years.

        Parameters
        ----------
        date: datetime object
            Date of yield curve.

        Returns
        -------
        yields: [1 x 6] Array[float].
            Yields for [0.5, 2, 5, 10, 20, 30] years.
        """
        return get_yield(date, self._t_mats)

    @lru_cache(maxsize=None)
    def get_KRDs_and_coupons(self, date):
        """
        Memoized key rate durations for specified date and
        maturities of [0.5, 2, 5, 10, 20, 30] years.

        Parameters
        ----------
        date: datetime object
            Date of yield curve.

        Returns
        -------
        krds: [1 x 6] Array[float].
            KRDs for [0.5, 2, 5, 10, 20, 30] years.
        """
        self._krd_date = pd.to_datetime(date)
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
        self._yields = self.get_yield(self._krd_date, self._c_yrs)
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

    def get_KRD_total_returns(self, original_date, new_date, t):
        """
        Reprice hypothetical par bond used for
        KRDs with a new yield curve.


        """
        c_yrs = np.arange(0.5, t + 0.5, 0.5) - 1 / 365
        # coupon =

    def _coupon_func(self, coupon):
        """
        Return error in price of a par bond, with maturity
        from `self._solve_krd` and specified coupon.
        """
        # Create cash flow and yield vectors.
        cash_flows = np.zeros(len(self._c_yrs)) + coupon
        cash_flows[-1] += 100  # par bond

        # Find error in price difference.
        error = (
            sum(
                cf * np.exp(-y * t)
                for cf, y, t in zip(cash_flows, self._yields, self._c_yrs)
            )
            - 100
        )
        return error


# %%


def main():
    from lgimapy.index import IndexBuilder

    # for year in [2010, 2009, 2007, 2006, 2005, 2004]:
    # start = f'1/1/{year}'
    # start = f'5/20/2019'
    # # end = f'1/1/{year+1}'
    # end = None
    # ixb = IndexBuilder()
    # ixb.load(start=start, end=end, dev=True)
    # t.split('load')
    # ix = ixb.build(treasuries=True)
    # t.split('build')
    # for ld in ix.dates:
    #     tc = TreasuryCurveBuilder(ix.day(ld, as_index=True))
    #     print(f'\n{ld.strftime("%m/%d/%Y")}')
    #     tc.fit(n=50, verbose=1)
    #     t.split()
    #     tc.save()

    tc = TreasuryCurve()
    day = tc._df.index[0]

    df_vals = tc._df.values
    a = np.zeros([df_vals.shape[0], df_vals.shape[1] * 3])
    a[:, :6] = df_vals
    a
    for i, day in enumerate(tc._df.index):
        a[i, 6:12], a[i, 12:18] = tc.get_KRDs_and_coupons(day)

    cols = list(tc._df)
    cols
    times = [0.5, 2, 5, 10, 20, 30]
    c1 = [f"KRD_{t}" for t in times]
    c2 = [f"c_{t}" for t in times]
    cols.extend(c1)
    cols.extend(c2)

    df = pd.DataFrame(a, columns=cols, index=tc._df.index)

    df.to_csv(root("data/treasury_curve_params_.csv"))

    # %%


if __name__ == "__main__":
    main()

# date = '5/1/2019'
# self = TreasuryCurve()
# self.get_KRDs(date)
# tc.get_yield('5/1/2019', 5)

# %%

# from index_functions import IndexBuilder
#


# %%
