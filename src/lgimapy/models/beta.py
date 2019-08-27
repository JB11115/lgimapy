from collections import defaultdict, OrderedDict
from functools import lru_cache
import warnings

import bottleneck as bn
import numpy as np
import pandas as pd
from sklearn.linear_model import ransac
from statsmodels.api import OLS, RLM, add_constant

# %%
class Beta:
    """
    Beta calculation class.

    """

    def __init__(self):
        pass

    def fit(self, method, **kwargs):
        """
        Fit beta model withod and input.

        Parameters
        ----------
        method: {'bucket', 'dts', 'ols'}.
            Beta calculation method.
        kwargs: dict
            Model specific arguments.
        """
        self.method = method.lower()
        self._col = f"beta_{self.method}"
        eval(f"self._fit_{self.method}(**kwargs)")

    def transform(self, benchmark_ix, portfolio_ix):
        """
        Compute beta for all bonds in :class:`Index`
        using the previously :meth:`Beta.fit` model.

        Parameters
        ----------
        benchmark_ix: :class:`Index`
            Benchmark index to compute betas over.
        portfolio_ix: :class:`Index`
            Portfolio to compute betas for.

        Returns
        -------
        :class:`Index`
            Portfolio index with betas computed.
        """
        self._bm_ix = benchmark_ix
        self._port_ix = portfolio_ix

        return eval(f"self._transform_{self.method}()")

    def _fit_bucket(self, universe_ix, window, decay):
        """
        Fit Pat Dan's bucketing beta measure.

        This method creates 13 bins based on OAS and places bonds in
        each bin. Bonds which do not change bins are differenced on a
        daily basis, and aggregated by market weight to create a daily
        OAS change for each bin. The rolling covariance of each bucket
        is computed with the full benchmark index and divided by the
        rolling variance of the benchmark index to estimate bucket
        risk. The bucket risk for each bond within the buckets are then
        weighted by option adjusted spread duration and market weight
        for each day and summed to get the benchmark index beta. The
        bucket risks for each day are stored for use in the
        :meth:`Beta.transform` method.

        Parameters
        ----------
        universe_ix: :class:`Index`
            Index to perform beta calculations on.
        input_variable: {'OAS', 'excess_return'}.
            Input variable to use in beta calculation.
        window: int
            Window size in days for rolling covariance.
        decay: bool
            If True use exponentially weighted decay of rolling
            covariance. If False use standard sample covariance.
        """
        # Create bucket bins.
        self._bin_edges = [
            0,
            25,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            300,
            400,
            3000,
        ]
        bin_ixs = OrderedDict(
            (
                f"{b}-{self._bin_edges[i]}",
                universe_ix.subset(OAS=(b, self._bin_edges[i])),
            )
            for i, b in enumerate(self._bin_edges[:-1], start=1)
        )
        self._bins = bin_ixs.keys()

        # Create market value weighted DataFrames for change in
        # OAS, duration, and total market value for each bucket.
        oas_df = pd.DataFrame(
            np.diff(universe_ix.market_value_weight("OAS", synthetic=True)),
            index=universe_ix.dates[2:],
            columns=["full"],
        )
        for key, ix in bin_ixs.items():
            oas_df[key] = np.diff(ix.market_value_weight("OAS", synthetic=True))

        # For each bucket find rolling bucket risk:
        # = cov(bucket, full ix) / var(full ix).
        bucket_risk_df = pd.DataFrame()
        if decay:
            full_ix_var = oas_df["full"].ewm(span=window).var()
            for b in self._bins:
                bucket_risk_df[b] = (
                    oas_df["full"].ewm(span=window).cov(oas_df[b]) / full_ix_var
                )
        else:
            full_ix_var = oas_df["full"].rolling(window).var()
            for b in self._bins:
                bucket_risk_df[b] = (
                    oas_df["full"].rolling(window).cov(oas_df[b]) / full_ix_var
                )

        # Determine benchmark beta for each date.
        self._bucket_risk_df = bucket_risk_df[window - 1 :]

    def _transform_bucket(self):
        """
        Compute beta for all bonds in specified :class:`Index`
        using Pat Dan's bucketing methodology.
        """
        # Find aggregate beta of bonds benchmark index each day.
        self._bm_ix = self._compute_bucket_from_universe(self._bm_ix)
        bm_beta_d = self._bm_ix.market_value_weight(self._col).to_dict()

        # Find beta of bonds in portfolioo index and divide
        # by the benchmark beta of each day.
        self._port_ix = self._compute_bucket_from_universe(self._port_ix)
        bm_beta = [bm_beta_d[d] for d in self._port_ix.df["Date"]]
        self._port_ix.df[self._col] /= bm_beta
        return self._port_ix

    def _compute_bucket_from_universe(self, ix):
        """
        Compute beta for all bonds in :class:`Index`
        using the previously :meth:`Beta.fit` model.

        Parameters
        ----------
        ix: :class:`Index`
            Index to compute betas over.
        """
        bucket_risk_d = self._bucket_risk_df.to_dict()
        # Find bin categories and replace with bucket risk for
        # current date.
        bin_cat = pd.cut(
            ix.df["OAS"], bins=self._bin_edges, right=True, labels=self._bins
        )
        bucket_risk = [
            bucket_risk_d[b].get(d, np.NaN)
            for b, d in zip(bin_cat, ix.df["Date"])
        ]
        # Weight bucket risk by spread duration for each bond.
        ix.df[self._col] = ix.df["OASD"] * bucket_risk
        return ix

    def _fit_dts(self):
        """
        Fit duration time spread (DTS) method for computing beta.
        """
        pass  # No fitting requried

    def _transform_dts(self):
        """
        Compute DTS of benchmark index and use it to weight
        the DTS of each bond in a portfolio to estimate beta.
        """
        # Find aggregate beta of bonds benchmark index each day.
        self._bm_ix.df[self._col] = (
            self._bm_ix.df["OASD"] * self._bm_ix.df["OAS"]
        )
        bm_beta_d = self._bm_ix.market_value_weight(self._col).to_dict()

        # Find beta of bonds in portfolioo index and divide
        # by the benchmark beta of each day.
        self._port_ix.df[self._col] = (
            self._port_ix.df["OASD"] * self._port_ix.df["OAS"]
        )
        bm_beta = [bm_beta_d[d] for d in self._port_ix.df["Date"]]
        self._port_ix.df[self._col] /= bm_beta
        return self._port_ix

    def _fit_ols(self):
        """
        Fit lead-lag OLS method of computing beta.

        Parameters
        ----------
        """
        pass


# %%

with Time() as t:
    self = Beta()
    t.split("init")
    self.fit(method="bucket", universe_ix=universe_ix, decay=False, window=30)
    t.split("fit")
    transformed_ix = self.transform(
        benchmark_ix=benchmark_ix, portfolio_ix=portfolio_ix
    )
    t.split("transform")
    print(transformed_ix.market_value_weight("beta"))
    t.split("mvw")
# %%

with Time() as t:
    self = Beta()
    t.split("init")
    self.fit(method="dts")
    t.split("fit")
    transformed_ix = self.transform(
        benchmark_ix=benchmark_ix, portfolio_ix=portfolio_ix
    )
    t.split("transform")
    print(transformed_ix.market_value_weight("dts"))
    t.split("mvw")


# %%

self._transform_ix = ix.subset(start="6/13/2019", OAS=(1e-6, 3000))

self._transform_ix.df.head()
self._port_ix.market_value_weight("beta")
# %%


def main():
    # %%
    from lgimapy.utils import Time
    from lgimapy.data import Database
    from statsmodels.tsa.stattools import adfuller

    # Check for staionarity
    # db2 = Database()
    # db2.load_market_data(start="1/1/2019", local=True)
    # ix2 = db2.build_market_index()
    #
    # oas_diff_df = ix2.get_value_history('OAS')
    # oas_diff_df.dropna(axis=1, inplace=True)
    # oas_diff_df = oas_diff_df.diff()[1:]
    #
    #
    # n = 500
    # pvals = np.zeros(n)
    # for i, c in enumerate(list(oas_diff_df)[:n]):
    #     res = adfuller(oas_diff_df[c])
    #     pvals[i] = res[1]
    #
    # pvals = pd.DataFrame(pvals)
    # pvals.describe()
    #
    #
    # # %%
    # pvals_sorted = pvals.sort_values(0, ascending=False).reset_index(drop=True)
    # pvals_sorted['0.05'] = 0.05
    # pvals_sorted.index = pvals_sorted.index / len(pvals_sorted) * 100
    # pvals_sorted.plot(lw=2, alpha=0.5)
    # plt.show()
    # # %%

    # %%
    db2 = Database()
    db2.load_market_data(date="8/21/2019")
    ix_temp = db2.build_market_index(in_returns_index=True)
    ix_temp.df["USAggReturnsFlag"]
    print(np.sum(ix_temp.df["USAggReturnsFlag"]))

    # %%

    db = Database()
    db.load_market_data(start="1/1/2018", end="12/31/2018", local=True)
    universe_ix = db.build_market_index(rating="IG", currency="USD")

    benchmark_ix = db.build_market_index(
        in_returns_index=True, maturity=(10, None)
    )

    portfolio_ix = db.build_market_index(
        rating="IG", currency="USD", OAS=(1e-6, 3000), maturity=(10, None)
    )

    self = Beta()
    self._bm_ix = benchmark_ix.copy()
    self._portfolio_ix = portfolio_ix.copy()

    # %%
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    # %matplotlib qt
    plt.style.use("fivethirtyeight")
    # %%

    factor = "OAS"
    lags = 4
    n = 20

    def fit_ols(self, factor):
        df = get_value_history("OAS")

    def transform_ols(self, factor, lags, n):
        # %%
        # Get mkt and cusip factor levels for full period.
        mkt = self._bm_ix.market_value_weight(factor, synthetic=True)
        df = self._portfolio_ix.get_value_history(factor, synthetic=True)

        if factor.upper() == "OAS":
            # Difference OAS to make it stationary.
            mkt = np.log(mkt[1:] / mkt.values[:-1])
            df = np.log(df[1:] / df.values[:-1])

        # Convert pandas objects to numpy arrays for speed.
        factor_a = df.values[:, :200]
        mkt_a = mkt.values

        # %%
        lags = 10
        n = 80

        # Find locations with `m` consecutive non-nan values.
        m = 1 + 2 * lags
        nan_locs = np.isnan(df).values
        warnings.simplefilter("ignore", category=FutureWarning)
        good_locs = bn.move_mean(~nan_locs, n + m, axis=0) == 1
        warnings.simplefilter("default", category=FutureWarning)

        # Compute beta variables i for each cusip j at time t.
        with Time():
            beta_vars = np.full([*factor_a.shape, 4 * lags + 1], np.nan)
            beta_pvals = np.full([*factor_a.shape, 4 * lags + 1], np.nan)
            for j in range(factor_a.shape[1]):
                for t in range(m + n - 2, len(mkt_a)):
                    if good_locs[t, j] != 1:
                        # Skip due to nan values in regression.
                        continue
                    k = 0  ## tensor coordinate.
                    # Peform Beta_j regressions.
                    for i in range(m):
                        x = add_constant(mkt_a[t - lags - n : t - lags])
                        y = factor_a[t - i - n + 1 : t - i + 1, j]
                        res = RLM(y, x).fit()
                        res = OLS(y, x).fit()
                        beta_vars[t, j, k] = res.params[1]
                        beta_pvals[t, j, k] = res.pvalues[1]
                        k += 1
                    # Peform Beta_M regressions.
                    for i in range(m):
                        if i - 1 == lags:
                            # Repeate regression from Beta_j == Beta_M
                            continue
                        x = add_constant(factor_a[t - lags - n : t - lags, j])
                        y = mkt_a[t - i - n + 1 : t - i + 1]
                        res = RLM(y, x).fit()
                        res = OLS(y, x).fit()
                        beta_vars[t, j, k] = res.params[1]
                        beta_pvals[t, j, k] = res.pvalues[1]
                        k += 1

            num = np.sum(beta_vars[:, :, :m], axis=2)
            denom = 1 + np.sum(beta_vars[:, :, m:], axis=2)
            beta = num / denom
            beta_df = pd.DataFrame(
                beta, columns=df.columns[:200], index=df.index
            )

        # %%
        beta_df.dropna(how="all", axis=0).plot(lw=1, legend=False)
        plt.suptitle(f"# Lags: {lags}, Sample Size N: {n}")
        # plt.ylim(-1, 1)
        plt.show()

        # %%
        cols = []
        for i in range(m):
            cols.append(f"j{lags-i:+d}")
        for i in range(m):
            if i == lags:
                continue
            cols.append(f"m{lags-i:+d}")

        warnings.simplefilter("ignore", category=RuntimeWarning)
        sig_beta_pvals = np.where(beta_pvals < 0.1, 1, np.nan)
        beta_pvals_df = pd.DataFrame(
            {
                c: bn.nansum(sig_beta_pvals[:, :, i], axis=1)
                for i, c in enumerate(cols)
            },
            index=df.index,
        )
        beta_pvals_df = beta_pvals_df[np.sum(beta_pvals_df, axis=1) != 0]

        beta_pvals_df[cols[:m]].plot(lw=2, cmap="coolwarm_r")
        beta_pvals_df[cols[m:]].plot(lw=2, cmap="coolwarm_r")
        plt.show()

        # %%
        plt.figure()
        avg_beta = np.sum(beta_df, axis=1) / np.sum(~np.isnan(beta_df), axis=1)
        avg_beta.dropna(inplace=True)
        avg_beta.plot(color="steelblue", lw=2)
        plt.show()

        # %%
        beta_sig_vars = np.where(beta_pvals < 0.2, beta_vars, np.nan)
        num_sig = bn.nansum(beta_sig_vars[:, :, :m], axis=2)
        denom_sig = 1 + bn.nansum(beta_sig_vars[:, :, m:], axis=2)
        beta_sig = num_sig / denom_sig
        beta_sig_df = pd.DataFrame(
            beta_sig, columns=df.columns[:200], index=df.index
        )
        beta_sig_df = beta_sig_df[np.sum(beta_sig_df, axis=1) != 0]
        beta_sig_df.plot(lw=1, legend=False)
        plt.suptitle(f"# Lags: {lags}, Sample Size N: {n}")
        # plt.ylim(-1, 1)
        plt.show()
        # %%
