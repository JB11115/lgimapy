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

from lgimapy.index import TBond, IndexBuilder
from lgimapy.utils import nearest, mkdir, root

plt.style.use("fivethirtyeight")
# %matplotlib qt

# %%
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
        self._historical_df = self._load()
        self._trade_dates = list(self._historical_df.index)
        self._t_mats = np.array([0.5, 2, 5, 10, 20, 30])
        self._B_cols = ["B_0", "B_1", "B_2", "B_3", "Tau_1", "Tau_2"]
        self._KRD_cols = [
            "KRD_0.5",
            "KRD_2",
            "KRD_5",
            "KRD_10",
            "KRD_20",
            "KRD_30",
        ]
        self._coupon_cols = ["c_0.5", "c_2", "c_5", "c_10", "c_20", "c_30"]
        if ix is not None:
            # Load all index features.
            self._strips_df = ix.df[ix.df["Ticker"] == "SP"]
            ix.clean_treasuries()
            self.df = ix.df
            self.date = ix.dates[0]
            self.bonds = [TBond(bond) for _, bond in self.df.iterrows()]
        else:
            # Load parameters from csv file.
            self.date = pd.to_datetime(date)
            self._B = self._historical_df.loc[self.date, self._B_cols]

        # Find iloc of yesterday in historical data.
        if self.date > self._trade_dates[-1]:
            self._yesterday_iloc = len(self._trade_dates) - 1
        else:
            self._yesterday_iloc = self._trade_dates.index(self.date) - 1

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

                * 0: No results.
                * 1: Updated each time bonds are dropped.
                * 2: Update each completed optimization simulation.
                * 3: Update each optimization iteration.
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
        """Save treasury curve values to `./data/treasury_curve_params.csv`."""
        # Get treasury curve parameters.
        krds, coupons = self._get_KRDs_and_coupons()
        if self._yesterday_iloc >= 0:
            # Have data for yesterday, calculate total returns.
            trets = self._get_KRD_total_returns()
        else:
            # No data for yesterday, fill total returns with NaNs.
            trets = np.array([np.nan] * 7)

        # Create single row to append to historical DataFrame.
        row_vals = np.concatenate([self._B, krds, coupons, trets])
        tret_cols = [
            "tret_0.5",
            "tret_2",
            "tret_5",
            "tret_10",
            "tret_20",
            "tret_30",
            "tret_cash",
        ]
        cols = [*self._B_cols, *self._KRD_cols, *self._coupon_cols, *tret_cols]
        new_row = pd.DataFrame(row_vals, index=cols, columns=[self.date]).T
        fid = root("data/treasury_curve_params.csv")
        if len(self._historical_df):
            # File exists, append to it and sort.
            df = self._historical_df.copy()
            df = df[df.index != self.date].copy()  # drop date if it exists.
            df = df.append(new_row)
            df.sort_index(inplace=True)
            df.to_csv(fid)
        else:
            # Create csv with first row.
            new_row.to_csv(fid)

    def _yield_30_100(self, t):
        """
        Linearly calculate yield curve from 30 to 100 years.
        Use Svensson method for 30 years and subtract 10 bp
        to account for convexity at 100 years.

        Parameters
        ----------
        t: float
            Time (yrs) to calculate zero yield.

        Returns
        -------
        float
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
        y: float, ndarray[float].
            Yields for specified maturities.
        """
        if isinstance(t, float) or isinstance(t, int):
            # Single time.
            if t <= 30:
                return self._svensson(t, self._B)
            else:
                return self._yield_30_100(t)
        elif isinstance(t, np.ndarray):
            t_low, t_high = t[t <= 30], t[t > 30]
            y_low = self._svensson(t_low, self._B)
            y_high = self._yield_30_100(t_high)
            return np.concatenate([y_low, y_high])

    def _get_KRD_yields(self):
        """
        Yields for specified date and
        maturities of [0.5, 2, 5, 10, 20, 30] years.

        Returns
        -------
        yields: [1 x 6] ndarray
            Yields for [0.5, 2, 5, 10, 20, 30] years.
        """
        return self._get_yield(self._t_mats)

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
        yesterday = self._historical_df.index[self._yesterday_iloc]
        yesterday_coupons = self._historical_df.loc[
            yesterday, self._coupon_cols
        ].values
        tot_rets = np.zeros(len(self._t_mats) + 1)
        n_yrs = (self.date - yesterday).days / 365

        # Solve price of all bonds given yesterdays coupons.
        for i, (y_c, tmat) in enumerate(zip(yesterday_coupons, self._t_mats)):
            c_yrs = np.arange(0.5, tmat + 0.5, 0.5) - n_yrs
            cash_flows = np.zeros(len(c_yrs)) + y_c
            cash_flows[-1] += 100  # par bond terminal payment
            tot_rets[i] = sum(
                cf * np.exp(-y * t)
                for cf, y, t in zip(cash_flows, self._get_yield(c_yrs), c_yrs)
            )
        # Solve next day cash price based on yesterday's yield.
        yesterday_B = self._historical_df.loc[yesterday, self._B_cols].values
        tot_rets[i + 1] = 100 * np.exp(
            self._svensson(n_yrs, yesterday_B) * n_yrs
        )
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

    def _load(self):
        """Load data as DataFrame."""
        fid = root("data/treasury_curve_params.csv")
        try:
            return pd.read_csv(
                fid, index_col=0, parse_dates=True, infer_datetime_format=True
            )
        except FileNotFoundError:
            return pd.DataFrame()


# %%
class TreasuryCurve:
    """
    Class for loading treasury yield curve and calculating yields
    for specified dates and maturities.
    """

    def __init__(self):
        self._df = self._load()
        self._t_mats = np.array([0.5, 2, 5, 10, 20, 30])
        self.trade_dates = list(self._df.index)

    @property
    def trade_dates(self):
        """List[datetime] : List of traded dates."""
        return self._df.index

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


# %%
def update_treasury_curves():
    """
    Update `.data/treasury_curve_params.csv` file
    with most recent data.
    """
    # Get list of dates with treasury curves
    temp_tcb = TreasuryCurveBuilder(date="1/5/2004")
    last_saved_date = temp_tcb._trade_dates[-1]

    ixb = IndexBuilder()
    ixb.load(dev=True, start=last_saved_date)
    ix = ixb.build(treasuries=True)

    for date in ix.dates[1:]:
        print(f"\nFitting Treasury Curve for {date.strftime('%m/%d/%Y')}")
        tcb = TreasuryCurveBuilder(ix.day(date, as_index=True))
        tcb.fit(verbose=1)
        tcb.save()


def main():
    # %%
    update_treasury_curves()

    # %%


if __name__ == "__main__":
    main()
