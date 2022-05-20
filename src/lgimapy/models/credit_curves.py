import warnings
from functools import lru_cache, cached_property

import numpy as np
from scipy.optimize import fsolve, minimize, LinearConstraint
import pandas as pd


from lgimapy.data import TreasuryCurve, Bond, SyntheticBond
from lgimapy.utils import to_list, Time

# %%

# from lgimapy.data import Database

# db = Database()
#
# date = db.date("today")
# date = "3/9/2021"
# db.load_market_data(date=date)
# treasury_curve = TreasuryCurve(date)
# __ = treasury_curve._curves_df

# %%
# ix = db.build_market_index(ticker="IBM").subset_on_the_runs()
# isins = ix.isins
# isins = [
#     "US037833EB24",
#     "US037833ED89",
#     "US037833DV96",
#     "US037833EF38",
#     "US037833EC07",
#     "US037833EE62",
# ]
# cols = [
#     "IssueDate",
#     "MaturityDate",
#     "MaturityYears",
#     "OriginalMaturity",
#     "DirtyPrice",
#     "YieldToWorst",
# ]
# ix = db.build_market_index(isin=isins)
# ix.df[cols].sort_values("OriginalMaturity")
# df = ix.df.copy()
# df[cols].sort_values("OriginalMaturity")
# %%


class CreditCurve:
    def __init__(self, df_or_bonds, treasury_curve):
        self._treasury_curve = treasury_curve
        self._define_bond_properties(df_or_bonds)

    def _define_bond_properties(self, df_or_bonds):
        """
        Store bond properties in class.

        Attributes
        ----------
        _n: int
            Number of bonds.
        _m: int
            Number of cash flows
        _PV: [1 x n] np.array
            Present value (dirty price) of each bond (par = 100).
        _w: [1 x n] np.array
            Weight vector for objective function, equal to the
            inverse of the spread duration for each bond.
        _t: [n x m] np.array
            Matrix of time to coupons (in years).
        _cf: [n x m] np.array
            Matrix of cash flow values (par principal = 100).
        """
        if isinstance(df_or_bonds, pd.DataFrame):
            self.df = df_or_bonds.copy()
            bonds = [Bond(row) for _, row in df_or_bonds.iterrows()]
        elif isinstance(df_or_bonds, Bond):
            self.df = df_or_bonds.df.copy()
            bonds = [df_or_bonds]
        else:
            # Input is already a list of bonds.
            bonds = df_or_bonds

        self.bonds = bonds
        self._n = len(bonds)
        self._m = max(len(bond.coupon_years) for bond in bonds)

        self._PV = np.zeros(self._n)
        self._w = np.zeros(self._n)
        self._cf = np.zeros([self._n, self._m])
        self._t = np.zeros([self._n, self._m])
        for i, b in enumerate(bonds):
            self._PV[i] = b.DirtyPrice
            self._w[i] = 1 / (b.OASD**2)
            for j, (cf, t) in enumerate(zip(b.cash_flows, b.coupon_years)):
                self._cf[i, j] = cf
                self._t[i, j] = t

    def _define_model_parameters(self, knot_points, recovery):
        """
        Store model parameters.

        Attributes
        ----------
        knot_points: [1 x k] np.array
            Maturity (in years) of knot points.
        _R: float
            Recovery assumption as float where par is 100
            (e.g., for a 30% recovery _R = 30).
        _k: int
            Number of free Beta parameters in model.
        """
        self._R = recovery

        if knot_points is None:
            self.knot_points = None
            self._k = 3
        else:
            self.knot_points = np.array(knot_points, ndmin=1, dtype="float32")
            self._k = 3 + len(self.knot_points)

    @cached_property
    def _Z_yields(self):
        """
        Memoized risk free yields matrix for cash flow times
        of each bond.

        Returns
        -------
        [n x m] np.array:
            Risk free discount function for cash flow times.
        """
        Z_yields = np.zeros([self._n, self._m])
        for i in range(self._n):
            t = self._t[i, :]
            Z_yields[i, :] = self._treasury_curve.yields(t)
        return Z_yields

    @cached_property
    def _Z(self):
        """
        Memoized discount function matrix for cash flow times
        of each bond (Z_base).

        Returns
        -------
        [n x m] np.array:
            Risk free discount function for cash flow times.
        """
        return np.exp(-self._Z_yields * self._t)

    def _array_to_tensor(self, array):
        """
        Duplicate an input array into a tensor.

        Returns
        -------
        [L x n x m] np.array:
            Tensor with first dimension ``L`` equal to length of
            input array.
        """
        matrix = np.tile(array, (self._m, 1))
        tensor_unordered = np.tile(matrix, (self._n, 1, 1))
        return np.transpose(tensor_unordered, [2, 0, 1])

    def _SSpline(self, alpha):
        """
        Survival probability tensor. Equations [40] and [41].

        Returns
        -------
        spline_tensor: [k x n x m] np.array:
            Tensor of SSpline(alpha).
        """
        tensor_shape = (self._k, self._n, self._m)
        spline_tensor = np.zeros(tensor_shape)

        # Make tensors of same dim [k x n x m] for k and t values.
        t_tensor = np.broadcast_to(self._t, tensor_shape)
        k_tensor = self._array_to_tensor(np.arange(1, 4))

        # Solve the no-knot factors part of the spline.
        no_knot_loc = np.index_exp[:3, :, :]
        # Equation [40].
        spline_tensor[no_knot_loc] = np.exp(
            -k_tensor * alpha * t_tensor[no_knot_loc]
        )

        # Solve for part of the spline after knot points if required.
        if self.knot_points is not None:
            knot_pts_loc = np.index_exp[3:, :, :]
            T_knots = self._array_to_tensor(self.knot_points)
            Theta = (t_tensor[knot_pts_loc] > T_knots).astype(int)
            # Equation [41].
            x = -alpha * (t_tensor[knot_pts_loc] - T_knots)
            spline_tensor[knot_pts_loc] = Theta * (
                1 / 3 + np.exp(2 * x) - np.exp(x) - np.exp(3 * x) / 3
            )
        return spline_tensor

    @cached_property
    def _dZ(self):
        """
        Memoized difference (Z_t - Z_{t+1}) of discount function
        matrix for cash flow times of each bond. Used for
        Equation [15].

        Returns
        -------
        [n x m-1] np.array:
            Difference in diccount function for cash flow times.
        """
        # Use indicator function to make all differences =0
        # For empty cash flows in the matrix. This makes
        # Z_t - Z_{t+1} = 0 after the final cash flow.
        # This is non-zero otherwise due to having a full
        # matrix of cash flows.
        indicator_function = (self._cf > 0).astype(int)[:, 1:]
        return -np.diff(self._Z, axis=1) * indicator_function

    @cached_property
    def _principal_locs(self):
        """
        Memoized vector of princpal locations in :attr:`_cf`.

        Returns
        -------
        [2 x n] np.array:
            Index location of principal payment for each bond.
        """
        return np.index_exp[np.arange(self._n), np.argmax(self._cf, axis=1)]

    @cached_property
    def _discounted_coupons_ex_recovery(self):
        """
        Memoized matrix of discounted coupons ex conditional
        recovery. Bracketed portion of top line in Equation [15].

        Returns
        -------
        [n x m-1] np.array:
            Discounted coupons ex recovery.
        """
        # Remove principal cash flow from each bond and discount
        # (left side of - sign in bracket).
        coupons = self._cf.copy()
        coupons[self._principal_locs] = 0
        discounted_coupons = (coupons * self._Z)[:, :-1]

        # Compute assosiated conditionally discounted recovery.
        cond_discounted_recovery = self._dZ * self._R
        return discounted_coupons - cond_discounted_recovery

    @cached_property
    def _discounted_principal_ex_recovery(self):
        """
        Memoized matrix of discounted principal payments ex
        recovery. Paranthesis portion of bottom line in
        Equation [15].

        Returns
        -------
        discounted_principalL: [n x m] np.array
            Discounted principal ex recovery in matrix form.
        """
        discounted_cash_flows = (self._cf - self._R) * self._Z
        discounted_principal = np.zeros([self._n, self._m])
        loc = self._principal_locs
        discounted_principal[loc] = discounted_cash_flows[loc]
        return discounted_principal

    def _U(self, params):
        """
        Explanatory variable from splines used to compute model
        implied present value. Equation [15].

        Returns
        -------
        [n x k] np.array:
            Model computed present values
        """
        alpha = params[0]
        SSpline = self._SSpline(alpha)
        einsum_notation = "ijk,jk->ji"
        U = np.einsum(
            einsum_notation, SSpline, self._discounted_principal_ex_recovery
        ) + np.einsum(
            einsum_notation,
            SSpline[:, :, :-1],
            self._discounted_coupons_ex_recovery,
        )
        return U

    @cached_property
    def _V(self):
        """
        Memoized adjusted present value for bonds based off
        market prices and expected recovery. Equation [16].

        Returns
        -------
        [1 x n] np.array:
            Model computed present values
        """
        return self._PV - (self._R * self._Z[:, 0])

    def _model_errors(self, params):
        """
        Model errors (eta) for current parameter set.
        Equation [13/14].

        [1 x n] np.array:
            Model errors for each bond.
        """
        beta = params[1:]
        self.bond_errors = self._V - (self._U(params) @ beta)
        return self.bond_errors

    def _objective_function(self, params):
        """float: Objective for minimization. Equation [20]."""
        return np.sum(self._w * self._model_errors(params) ** 2)

    def fit(
        self,
        params=None,
        knot_points=10,
        recovery=30,
        solver="Nelder-Mead",
    ):
        """
        Fit survival curve model.

        Parameters
        ----------
        knot_points: float or List[float], optional, default=10
            Knot points for curve fitting.
        recovery: float, default=30
            Recovery of par from 0 to 100.
        solver: str, default='Nelder-Mead'
            Solver to use for ``scipy.optimize.minimize`` optimization.

        Returns
        -------
        :class:`CreditCurveResult`:
            Results of fit optimziation.
        """
        self._define_model_parameters(knot_points, recovery)

        if params is not None:
            init_params = np.array(params)
        else:
            init_params = np.random.uniform(size=self._k + 1)

        # Make an indicator array for beta 1, 2, and 3 parameters
        # for dot product linear constraint.
        beta_1_2_3 = np.zeros(self._k + 1)
        for i in range(1, 4):
            beta_1_2_3[i] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.opt = minimize(
                self._objective_function,
                x0=init_params,
                method=solver,
                tol=1e-4,
                constraints=(LinearConstraint(beta_1_2_3, lb=1, ub=1)),
            )
        self.params = self.opt.x
        self.resid = self.opt.fun

        return CreditCurveResult(opt=self.opt, mod=self)


class CreditCurveResult:
    def __init__(self, opt, mod):
        self.params = opt.x
        self.alpha = opt.x[0]
        self.beta = opt.x[1:]
        self.resid = opt.fun

        self.knot_points = mod.knot_points
        self.bond_errors = mod.bond_errors
        self._R = mod._R
        self._k = mod._k
        self._treasury_curve = mod._treasury_curve

    def _process_input(self, df_or_bonds):
        if isinstance(df_or_bonds, pd.DataFrame):
            self.df = df_or_bonds.copy()
            self._init_Zspreads = self.df["OAS"].values
        elif isinstance(df_or_bonds, Bond):
            self.df = df_or_bonds.df.copy()
            self._init_Zspreads = self.df["OAS"].values
        else:
            self._init_Zspreads = np.zeros(len(df_or_bonds)) + 100
        self._mod = CreditCurve(df_or_bonds, self._treasury_curve)
        self._mod._define_model_parameters(self.knot_points, self._R)
        self.bonds = self._mod.bonds
        self._n = len(self.bonds)

    def _fmt_t(self, t):
        return np.array(t, ndmin=1)

    def _fmt_output(self, t, x):
        """
        Format output to be float for single element input
        or pd.Series for multiple elements.
        """
        if len(t) == 1:
            return x[0]
        else:
            return pd.Series(x, index=t)

    def fit_prices(self, df_or_bonds):
        """
        Price a bond with given cash flows. Equation [33].

        Parameters
        ----------
        df_or_bonds: pd.DataFrame, :class:`Bond`, or List[:class:`Bond`]
            Bonds to find price.
        """
        self._process_input(df_or_bonds)
        V = self._mod._U(self.params) @ self.beta
        return V + (self._R * self._mod._Z[:, 0])

    def fit_Zspreads(self, df_or_bonds, prices=None):
        """
        Compute Z-spreads to treasuries for given bonds.

        Parameters
        ----------
        df_or_bonds: pd.DataFrame, :class:`Bond`, or List[:class:`Bond`]
            Bonds to find Z-spreads for.
        prices: [n x 1] np.array, optional
            Prices to compute Z-spread from. If no prices are
            provided, the model is used to price the bond.
        """
        self._process_input(df_or_bonds)
        P = (
            self.fit_prices(df_or_bonds)
            if prices is None
            else np.array(prices, ndmin=1)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Zspreads = fsolve(
                self._Zspread_objective_function,
                self._init_Zspreads,
                args=(P, self._mod._t, self._mod._cf, self._mod._Z_yields),
            )
        return Zspreads

    def _Zspread_objective_function(self, z_spread, P, t, cf, z):
        z_spread_matrix = np.tile(z_spread, (self._mod._m, 1)).T
        Z = np.exp(-(z + z_spread_matrix / 1e4) * t)
        return P - (cf * Z).sum(axis=1)

    def fit_OAS(self, df_or_bonds, prices=None):
        """
        Compute OAS given bond.

        Parameters
        ----------
        df_or_bonds: pd.DataFrame, :class:`Bond`, or List[:class:`Bond`]
            Bonds to find OAS for.
        prices: [n x 1] np.array, optional
            Prices to compute OAS from. If no prices are
            provided, the model is used to price the bond.
        """
        P = (
            self.fit_prices(df_or_bonds)
            if prices is None
            else np.array(prices, ndmin=1)
        )
        model_Zspreads = self.fit_Zspreads(self.df, P)
        market_Zspreads = self.fit_Zspreads(self.df, self.df["DirtyPrice"])
        oas = (self.df["OAS"] - market_Zspreads + model_Zspreads).values
        return oas

    def spot_curve(self, bond=None, coupon=None, maturity=None):
        if bond is not None:
            bonds = [bond]
            coupon = bond.CouponRate
            maturity = bond.MaturityYears

        else:
            bonds = []

        # Create list of synthetic bonds with same coupon
        # across different maturities.
        maturities = np.arange(0.5, bond.MaturityYears, 0.5)
        for t in maturities:
            bonds.append(SyntheticBond(maturity=t, coupon=coupon))

        # Solve for price of each bond.
        prices = self.fit_prices(bonds)
        return pd.Series(
            {
                bond.MaturityYears: bond.theoretical_ytm(price)
                for bond, price in zip(bonds, prices)
            }
        ).sort_index()

    def par_yield_curve(self, maturities=None):
        maturities = (
            np.arange(0.5, 30.5, 0.5)
            if maturities is None
            else np.array(maturities)
        )
        init_coupons = np.zeros(len(maturities)) + 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            par_yields = fsolve(
                self._par_yield_curve_root_funct,
                init_coupons,
                args=(maturities),
            )
        return pd.Series(par_yields, index=maturities)

    def _par_yield_curve_root_funct(self, coupons, maturities):
        bonds = [
            SyntheticBond(maturity=t, coupon=c)
            for t, c in zip(maturities, coupons)
        ]
        return 100 - self.fit_prices(bonds)

    def zero_curve(self, maturities=None):
        maturities = (
            np.arange(0.5, 30.5, 0.5)
            if maturities is None
            else np.array(maturities)
        )
        bonds = [SyntheticBond(maturity=t, coupon=0) for t in maturities]
        prices = self.fit_prices(bonds)
        return pd.Series(
            {
                bond.MaturityYears: bond.theoretical_ytm(price)
                for bond, price in zip(bonds, prices)
            }
        ).sort_index()


# %%
# maturities = None
# mod = CreditCurve(df, treasury_curve)
# bonds = [Bond(row) for _, row in df.iterrows()]
# bond = bonds[-1]
# self = mod.fit()
# self.bond_errors
#
# par_yields = self.par_yield_curve()
# df_or_bonds = [
#     SyntheticBond(maturity=t, coupon=c) for t, c in par_yields.items()
# ]
#
# self.spot_curve(bond)
# self.zero_curve()
# # %%
# self.fit_prices(df_or_bonds)
# self.fit_Zspreads(df_or_bonds)
# self.fit_OAS(df) - df["OAS"]
# self.fit_prices(bond)
# self.fit_Zspreads(bond)
# self.fit_OAS(bond)
#
# self.fit_Zspreads(bond)
# self.fit_OAS(bond)
# bond.OAS
# # %%
# df_or_bonds = df.copy()
# df_or_bonds = bond
# bond
#
# # %%
# from lgimapy import vis
#
# vis.style()
# # # %%
# db.load_market_data(date=db.nearest_date("7/18/2020"))
# fit_ix = db.build_market_index(
#     ticker="BA", issue_years=(None, 0.5), maturity=(None, 31)
# )
# fit_df = fit_ix.df.sort_values("MaturityYears")
#
# # %%
# fig, ax = vis.subplots()
# ax.plot(fit_df["MaturityYears"], fit_df["YieldToWorst"], "-o", lw=2)
# vis.show()
