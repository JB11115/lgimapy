from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import BondBasket, groupby
from lgimapy.utils import replace_multiple, root, to_datetime, to_list


# %%


class Portfolio:
    def __init__(self, date):
        self.date = to_datetime(date)

    def __repr__(self):
        date = self.date.strftime("%m/%d/%Y")
        return f"{self.name} {self.__class__.__name__} {date}"

    @property
    @lru_cache(maxsize=None)
    def _long_credit_A_rated_OAS(self):
        snapshot_fid = root(
            f"data/credit_snapshots/{self.date.strftime('%Y-%m-%d')}"
            "_Long_Credit_Snapshot.csv"
        )
        df = pd.read_csv(snapshot_fid, index_col=0)
        return df.loc["A", "Close*OAS"]

    @property
    @lru_cache(maxsize=None)
    def _otr_tsy_s(self):
        fid = root("data/OTR_treasury_OAD_values.parquet")
        s = pd.read_parquet(fid).loc[self.date]
        s.index = s.index.astype(int)
        return s

    def otr_tsy_oad(self, maturity):
        return self._otr_tsy_s.loc[maturity]

    def _ow_col(self, col):
        if col[:2] == "P_":
            return col
        elif col[:3] == "BM_":
            return col
        else:
            return f"{col}_Diff"

    def plot_tsy_weights(self, ax=None, figsize=(8, 4)):
        """
        Plot yield curve.

        Parameters
        ----------
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6).
            Figure size.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        tsy_weights = self.tsy_weights()
        y = [self.cash_pct()] + list(tsy_weights)
        x = np.arange(len(y))
        x_labels = ["Cash"] + list(tsy_weights.index)
        y_labels = ["" if val == 0 else f"{val:.1%}" for val in y]
        colors = ["navy" if weight > 0 else "firebrick" for weight in y]

        ax.set_title(
            f"Total Treasury OAD: {self.tsy_oad():.2f} yrs, "
            f"Portfolio Weight: {self.tsy_pct():.1%}\n",
            fontsize=13,
        )
        ax.bar(x, y, width=0.8, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Maturity")
        ax.set_ylabel("Portfolio Weight")
        ax.grid(False, axis="x")
        vis.format_yaxis(ax, "{x:.1%}")
        vis.set_n_ticks(ax, 5)

        # Label bars.
        for rect in ax.patches:
            height = rect.get_height()
            label = "" if height == 0 else f"{height:.2%}"
            if height < 0:
                height = 0

            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height,
                label,
                fontsize=10,
                ha="center",
                va="bottom",
            )


class Account(BondBasket, Portfolio):
    def __init__(self, df, name, date, constraints=None):
        BondBasket.__init__(self, df, name, constraints)
        Portfolio.__init__(self, date)
        self._tsy_bins = [2, 3, 5, 7, 10, 20, 30]
        self.df, self.tsy_df, self.cash_df = self._split_credit_tsy_cash(df)

    def _split_credit_tsy_cash(self, df):
        """Split DataFrame into cash, treasury, and credit components."""
        cash_locs = df["Sector"] == "CASH"
        tsy_locs = (df["Sector"] == "TREASURIES") | (df["L3"] == "Futures")
        credit_locs = ~(cash_locs | tsy_locs)

        cash_df = df[cash_locs].copy()
        tsy_df = df[tsy_locs].copy()
        credit_df = df[credit_locs].copy()

        # Add Maturity bin column to treasuries.
        bin_map = {i: bin_ for i, bin_ in enumerate(self._tsy_bins)}
        tsy_df["MaturityBin"] = np.digitize(
            tsy_df["MaturityYears"], self._tsy_bins
        )
        tsy_df["MaturityBin"] = tsy_df["MaturityBin"].map(bin_map)

        return credit_df, tsy_df, cash_df

    @property
    def sector_df(self):
        return groupby(self.df, "LGIMASector")

    @property
    def top_level_sector_df(self):
        return groupby(self.df, "LGIMATopLevelSector")

    @property
    @lru_cache(maxsize=None)
    def market_value(self):
        fid = root("data/account_values.parquet")
        return pd.read_parquet(fid).loc[self.date, self.name]

    def dts(self, method="pct"):
        method = method.lower()
        if method == "pct":
            return self.dts("port") / self.dts("bm")
        elif method == "abs":
            return np.sum(self.df["DTS_Diff"])
        elif method == "gc":
            return np.sum(self.df["DTS_Diff"]) / self._long_credit_A_rated_OAS
        elif method in {"p", "port", "portfolio"}:
            return np.sum(self.df["P_DTS"])
        elif method in {"bm", "benchmark"}:
            return np.sum(self.df["BM_DTS"])
        elif method == "hy_abs":
            return self.subset(rating=("HY")).dts("abs")
        elif method == "hy_pct":
            return self.subset(rating=("HY")).dts("port") / self.dts("bm")
        elif method == "ig_abs":
            return self.subset(rating=("IG")).dts("abs")
        elif method == "ig_pct":
            return self.subset(rating=("IG")).dts("port") / self.dts("bm")
        else:
            raise NotImplementedError(f"'{method}' is not a proper `method`.")

    def cash(self, pct=False):
        """float: Cash in account ($M)."""
        return np.sum(self.cash_df["P_AssetValue"]) / 1e6

    def cash_pct(self):
        return np.sum(self.cash_df["P_Weight"])

    def credit_pct(self):
        return np.sum(self.df["P_Weight"])

    def bm_credit_pct(self):
        return np.sum(self.df["BM_Weight"])

    def HY_mv_pct(self):
        return self.subset(rating=("HY")).credit_pct()

    def IG_mv_pct(self):
        return self.subset(rating=("IG")).credit_pct()

    def tsy_pct(self):
        return np.sum(self.tsy_df["P_Weight"])

    def bm_tsy_pct(self):
        return np.sum(self.tsy_df["BM_Weight"])

    def oad(self):
        return np.sum(self.df["OAD_Diff"])

    def oasd(self):
        return np.sum(self.df["OASD_Diff"])

    def tsy_oad(self):
        return np.sum(self.tsy_df["OAD_Diff"])

    def tsy_weights(self):
        ix = pd.Series(index=self._tsy_bins)
        weights = groupby(self.tsy_df, "MaturityBin")["P_Weight"]
        return weights.add(ix, fill_value=0).fillna(0)

    def rating_overweights(self, by="OAD"):
        ratings = {
            "Total": (None, None),
            "AAA": "AAA",
            "AA": ("AA+", "AA-"),
            "A": ("A+", "A-"),
            "BBB": ("BBB+", "BBB-"),
            "BB": ("BB+", "BB-"),
            "B": ("B+", "B-"),
        }
        overweights = []
        for rating in ratings.values():
            rating_bucket = self.subset(rating=rating)
            if len(rating_bucket.df):
                # Not empty
                overweights.append(eval(f"rating_bucket.{by.lower()}()"))
            else:
                overweights.append(np.nan)
        return pd.Series(overweights, index=ratings.keys(), name=by)

    def bond_overweights(self, by="OAD", sort=True):
        ix = (
            self.df["Ticker"].astype(str)
            + " "
            + self.df["Coupon"].apply(lambda x: f"{x:.2f}")
            + " "
            + self.df["MaturityDate"].apply(lambda x: f"`{x.strftime('%y')}")
            + "#"
            + self.df["CUSIP"].astype(str)
        ).values
        if sort:
            return pd.Series(
                self.df[self._ow_col(by)].values, index=ix, name=by
            ).sort_values(ascending=False)
        else:
            return pd.Series(
                self.df[self._ow_col(by)].values, index=ix, name=by
            )

    def ticker_overweights(self, by="OAD", sort=True):
        if sort:
            s = self.ticker_df[self._ow_col(by)].sort_values(ascending=False)
        else:
            s = self.ticker_df[self._ow_col(by)]
        return s.rename(by)

    def HY_ticker_overweights(self, by="OAD", sort=True):
        hy_account = self.subset(rating="HY")
        return hy_account.ticker_overweights(by=by, sort=True)

    def sector_overweights(self, by="OAD", sort=True):
        if sort:
            s = self.sector_df[self._ow_col(by)].sort_values(ascending=False)
        else:
            s = self.sector_df[self._ow_col(by)]
        return s.rename(by)

    def top_level_sector_overweights(self, by="OAD", sort=True):
        sectors = ["Industrials", "Financials", "Utilities", "Non-Corp"]
        ow_val_dict = self.top_level_sector_df[self._ow_col(by)].to_dict()
        # Add nans for any missing sectors.
        s = pd.Series(
            {sector: ow_val_dict.get(sector, np.nan) for sector in sectors}
        )
        # Calcualte Corp overweight and sort such that it is first listed.
        return pd.Series({"Corp": np.sum(s[:-1])}).append(s).rename(by)

    @property
    @lru_cache(maxsize=None)
    def _otr_treasury_oad(self):
        fid = root("data/OTR_treasury_OAD_values.parquet")
        oad_s = pd.read_parquet(fid).loc[self.date]
        oad_s.index = s.index.astype(int)
        return oad_s

    def curve_duration(self, pivot_maturity, weights=(0.7, 0.3)):
        if np.sum(weights) != 1:
            raise ValueError("Sum of `weights` must equal 1.")

        # Get current on-the-run durations for treasuries.
        tsy_2yr = self.otr_tsy_oad(2)
        tsy_30yr = self.otr_tsy_oad(30)
        tsy_pivot = self.otr_tsy_oad(pivot_maturity)

        # Subset bonds to columns required for model and
        # get % of market value for each bond in portfolio
        cols = ["CUSIP", "OAD", "Weight_Diff"]
        df = pd.concat([self.df[cols], self.tsy_df[cols]], sort=False)

        # Get 2s/pivot contriubtion.
        df_leq_2 = df[df["OAD"] <= tsy_2yr]
        df_2_pivot = df[(tsy_2yr < df["OAD"]) & (df["OAD"] <= tsy_pivot)]
        contrib_leq_2 = np.sum(df_leq_2["OAD"] * df_leq_2["Weight_Diff"])
        contrib_2_pivot = np.sum(
            (tsy_pivot - df_2_pivot["OAD"])
            / (tsy_pivot - tsy_2yr)
            * df_2_pivot["OAD"]
            * df_2_pivot["Weight_Diff"]
        )

        # Get pivot/30s contriubtion.
        df_geq_30 = df[df["OAD"] >= tsy_30yr]
        df_pivot_30 = df[(tsy_pivot < df["OAD"]) & (df["OAD"] < tsy_30yr)]
        contrib_geq_30 = np.sum(df_geq_30["OAD"] * df_geq_30["Weight_Diff"])
        contrib_pivot_30 = np.sum(
            (df_pivot_30["OAD"] - tsy_pivot)
            / (tsy_30yr - tsy_pivot)
            * df_pivot_30["OAD"]
            * df_pivot_30["Weight_Diff"]
        )

        return (weights[0] * (contrib_leq_2 + contrib_2_pivot)) - (
            weights[1] * (contrib_geq_30 + contrib_pivot_30)
        )


class Strategy(BondBasket, Portfolio):
    def __init__(self, df, name, date, constraints=None):
        BondBasket.__init__(self, df, name, constraints)
        Portfolio.__init__(self, date)
        self.name = name
        self.df = df

        # Separate all accounts in the strategy.
        self.accounts = self._split_accounts()
        self.account_names = list(self.accounts.keys())
        self.df = self._process_input_df(df)

    @property
    def fid(self):
        repl = {" ": "_", "/": "_", "%": "pct"}
        return replace_multiple(self.name, repl)

    def _split_accounts(self):
        return {
            account_name: Account(df, account_name, self.date)
            for account_name, df in self.df.groupby("Account", observed=True)
        }

    def _process_input_df(self, df):
        df["AccountValue"] = (
            df["Account"]
            .map(self.account_market_values.to_dict())
            .astype("float64")
        )
        return df

    @property
    @lru_cache(maxsize=None)
    def account_market_values(self):
        return (
            pd.read_parquet(root("data/account_values.parquet"))
            .loc[self.date, self.account_names]
            .rename("MarketValue")
        )

    @property
    @lru_cache(maxsize=None)
    def market_value(self):
        return np.sum(self.account_market_values)

    def calculate_account_values(self, fun, name=None):
        # Init Rule for aggregating pd.Series and
        # returning a pd.DataFrame vs aggregating
        # scalars and returning a pd.Series
        ret_df = False
        # Compute values for each respective account.
        account_vals = []
        for i, account_name in enumerate(self.account_names):
            account_val = fun(self.accounts[account_name])
            if ret_df or isinstance(account_val, pd.Series):
                ret_df = True
                account_vals.append(account_val.rename(account_name))
            else:
                account_vals.append(account_val)

        if ret_df:
            return pd.concat(account_vals, axis=1, sort=True).T
        else:
            return pd.Series(account_vals, index=self.account_names, name=name)

    def account_value_weight(self, account_vals):
        if isinstance(account_vals, pd.DataFrame):
            return (
                np.sum(
                    account_vals.mul(self.account_market_values, axis=0), axis=0
                )
                / self.market_value
            )
        elif isinstance(account_vals, pd.Series):
            return (
                np.sum(account_vals.mul(self.account_market_values, axis=0))
                / self.market_value
            )
        else:
            raise NotImplementedError

    def _property_functions(self, fun_type, properties):
        if properties is not None:
            properties = [p.lower() for p in to_list(properties, str)]
        default_properties = [
            "dts_pct",
            "dts_abs",
            "credit_pct",
            "cash_pct",
            "tsy_pct",
            "tsy_oad",
            "market_value",
        ]
        if properties is None:
            properties = default_properties
        elif properties == ["GC"]:
            gc_properties = [
                "dts_gc",
                "curve_duration(5)",
                "curve_duration(7)",
                "curve_duration(10)",
            ]
            properties = default_properties + gc_properties

        account_funs = {
            "dts_pct": self.account_dts("pct"),
            "dts_abs": self.account_dts("abs"),
            "dts_gc": self.account_dts("gc"),
            "hy_dts_abs": self.account_dts("hy_abs"),
            "hy_dts_pct": self.account_dts("hy_pct"),
            "ig_dts_abs": self.account_dts("ig_abs"),
            "ig_dts_pct": self.account_dts("ig_pct"),
            "credit_pct": self.account_credit_pct(),
            "ig_mv_pct": self.account_IG_mv_pct(),
            "hy_mv_pct": self.account_HY_mv_pct(),
            "cash_pct": self.account_cash_pct(),
            "tsy_pct": self.account_tsy_pct(),
            "tsy_oad": self.account_tsy_oad(),
            "curve_duration(5)": self.account_curve_duration(5),
            "curve_duration(7)": self.account_curve_duration(7),
            "curve_duration(10)": self.account_curve_duration(10),
            "market_value": self.account_market_values,
        }
        strategy_funs = {
            "dts_pct": self.dts("pct"),
            "dts_abs": self.dts("abs"),
            "dts_gc": self.dts("gc"),
            "hy_dts_abs": self.dts("hy_abs"),
            "hy_dts_pct": self.dts("hy_pct"),
            "ig_dts_abs": self.dts("ig_abs"),
            "ig_dts_pct": self.dts("ig_pct"),
            "credit_pct": self.credit_pct(),
            "ig_mv_pct": self.IG_mv_pct(),
            "hy_mv_pct": self.HY_mv_pct(),
            "cash_pct": self.cash_pct(),
            "tsy_pct": self.tsy_pct(),
            "tsy_oad": self.tsy_oad(),
            "curve_duration(5)": self.curve_duration(5),
            "curve_duration(7)": self.curve_duration(7),
            "curve_duration(10)": self.curve_duration(10),
            "market_value": self.market_value,
        }
        column_names = {
            "dts_pct": "DTS (%)",
            "dts_abs": "DTS (abs)",
            "dts_gc": "DTS/A-Spreads",
            "hy_dts_abs": "HY DTS (abs)",
            "hy_dts_pct": "HY DTS (%)",
            "ig_dts_abs": "IG DTS (abs)",
            "ig_dts_pct": "IG DTS (%)",
            "credit_pct": "Credit (%)",
            "hy_mv_pct": "HY MV (%)",
            "ig_mv_pct": "IG MV (%)",
            "cash_pct": "Cash (%)",
            "tsy_pct": "Tsy (%)",
            "tsy_oad": "Tsy OAD",
            "curve_duration(5)": "Curve Dur (5yr)",
            "curve_duration(7)": "Curve Dur (7yr)",
            "curve_duration(10)": "Curve Dur (10yr)",
            "market_value": "Market Value",
        }
        if fun_type == "account":
            return [account_funs[prop] for prop in properties]
        elif fun_type == "strategy":
            ix = [column_names[prop] for prop in properties]
            vals = [strategy_funs[prop] for prop in properties]
            return pd.Series(vals, ix, name=self.name)

    def property_latex_formats(self, properties):
        fmt = {
            "dts_pct": "1%",
            "dts_abs": "1f",
            "dts_gc": "2f",
            "hy_dts_pct": "1%",
            "ig_dts_pct": "1%",
            "hy_dts_abs": "1f",
            "ig_dts_abs": "1f",
            "credit_pct": "1%",
            "hy_mv_pct": "1%",
            "ig_mv_pct": "1%",
            "cash_pct": "1%",
            "tsy_pct": "1%",
            "tsy_oad": "1f",
            "curve_duration(5)": "3f",
            "curve_duration(7)": "3f",
            "curve_duration(10)": "3f",
            "market_value": "0f",
        }
        return [fmt[prop.lower()] for prop in to_list(properties, str)]

    def account_properties(self, properties=None):
        return pd.concat(
            self._property_functions("account", properties), axis=1, sort=True
        )

    def properties(self, properties=None):
        return self._property_functions("strategy", properties)

    @lru_cache(maxsize=None)
    def account_dts(self, method="pct"):
        name = {
            "pct": "DTS (%)",
            "abs": "DTS (abs)",
            "gc": "DTS/A-Spreads",
            "p": "Portfolio DTS (bp*yr)",
            "port": "Portfolio DTS (bp*yr)",
            "portfolio": "Portfolio DTS (bp*yr)",
            "bm": "Benchmark DTS (bp*yr)",
            "benchmark": "Benchmark DTS (bp*yr)",
            "hy_pct": "HY DTS (%)",
            "hy_abs": "HY DTS (abs)",
            "ig_pct": "IG DTS (%)",
            "ig_abs": "IG DTS (abs)",
        }[method.lower()]
        return self.calculate_account_values(lambda x: x.dts(method), name)

    @lru_cache(maxsize=None)
    def dts(self, method="pct"):
        return self.account_value_weight(self.account_dts(method))

    @lru_cache(maxsize=None)
    def account_oad(self):
        name = "OAD"
        return self.calculate_account_values(lambda x: x.oad(), name)

    @lru_cache(maxsize=None)
    def oad(self):
        return self.account_value_weight(self.account_oad())

    @lru_cache(maxsize=None)
    def account_oasd(self):
        name = "OASD"
        return self.calculate_account_values(lambda x: x.oasd(), name)

    @lru_cache(maxsize=None)
    def oasd(self):
        return self.account_value_weight(self.account_oasd())

    @lru_cache(maxsize=None)
    def account_cash_pct(self):
        name = "Cash (%)"
        return self.calculate_account_values(lambda x: x.cash_pct(), name)

    @lru_cache(maxsize=None)
    def cash_pct(self):
        return self.account_value_weight(self.account_cash_pct())

    @lru_cache(maxsize=None)
    def account_credit_pct(self):
        name = "Credit (%)"
        return self.calculate_account_values(lambda x: x.credit_pct(), name)

    @lru_cache(maxsize=None)
    def credit_pct(self):
        return self.account_value_weight(self.account_credit_pct())

    @lru_cache(maxsize=None)
    def account_HY_mv_pct(self):
        name = "HY MV (%)"
        return self.calculate_account_values(lambda x: x.HY_mv_pct(), name)

    @lru_cache(maxsize=None)
    def HY_mv_pct(self):
        return self.account_value_weight(self.account_HY_mv_pct())

    @lru_cache(maxsize=None)
    def account_IG_mv_pct(self):
        name = "IG MV (%)"
        return self.calculate_account_values(lambda x: x.IG_mv_pct(), name)

    @lru_cache(maxsize=None)
    def IG_mv_pct(self):
        return self.account_value_weight(self.account_IG_mv_pct())

    @lru_cache(maxsize=None)
    def account_bm_credit_pct(self):
        name = "Credit (%)"
        return self.calculate_account_values(lambda x: x.bm_credit_pct(), name)

    @lru_cache(maxsize=None)
    def bm_credit_pct(self):
        return self.account_value_weight(self.account_bm_credit_pct())

    @lru_cache(maxsize=None)
    def account_tsy_pct(self):
        name = "Tsy (%)"
        return self.calculate_account_values(lambda x: x.tsy_pct(), name)

    @lru_cache(maxsize=None)
    def tsy_pct(self):
        return self.account_value_weight(self.account_tsy_pct())

    @lru_cache(maxsize=None)
    def account_tsy_oad(self):
        name = "Tsy OAD"
        return self.calculate_account_values(lambda x: x.tsy_oad(), name)

    @lru_cache(maxsize=None)
    def tsy_oad(self):
        return self.account_value_weight(self.account_tsy_oad())

    def account_curve_duration(self, pivot_maturity, weights=(0.7, 0.3)):
        return self.calculate_account_values(
            lambda x: x.curve_duration(pivot_maturity, weights)
        ).rename(f"Curve Dur ({pivot_maturity}yr)")

    def curve_duration(self, pivot_maturity, weights=(0.7, 0.3)):
        return self.account_value_weight(
            self.account_curve_duration(pivot_maturity, weights)
        )

    def ticker_overweights(self, by="OAD"):
        return (
            self.account_value_weight(self.account_ticker_overweights(by=by))
            .sort_values(ascending=False)
            .rename(by)
        )

    def HY_ticker_overweights(self, by="OAD"):
        hy_strategy = self.subset(rating="HY")
        return hy_strategy.ticker_overweights(by=by)

    def sector_overweights(self, by="OAD"):
        return (
            self.account_value_weight(self.account_sector_overweights(by=by))
            .sort_values(ascending=False)
            .rename(by)
        )

    def top_level_sector_overweights(self, by="OAD"):
        sectors = ["Corp", "Industrials", "Financials", "Utilities", "Non-Corp"]

        return (
            self.account_value_weight(
                self.account_top_level_sector_overweights(by=by)
            )
            .rename(by)
            .loc[sectors]
        )

    def rating_overweights(self, by="OAD"):
        sorted_order = ["Total", "AAA", "AA", "A", "BBB", "BB", "B"]
        rating_ow = (
            self.account_value_weight(self.account_rating_overweights(by))
            .reindex(sorted_order)
            .rename(by)
        )
        for rating in sorted_order[-2:]:
            # Remove HY ratings if they are not in portfolio.
            if rating_ow.loc[rating] == 0 or np.isnan(rating_ow.loc[rating]):
                rating_ow.drop(rating, inplace=True)
        return rating_ow

    def bond_overweights(self, by="OAD"):
        return (
            self.account_value_weight(self.account_bond_overweights(by=by))
            .sort_values(ascending=False)
            .rename(by)
        )

    @lru_cache(maxsize=None)
    def account_ticker_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.ticker_overweights(sort=False, by=by)
        )

    @lru_cache(maxsize=None)
    def account_sector_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.sector_overweights(sort=False, by=by)
        )

    @lru_cache(maxsize=None)
    def account_top_level_sector_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.top_level_sector_overweights(sort=False, by=by)
        )

    @lru_cache(maxsize=None)
    def account_rating_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.rating_overweights(by=by)
        )

    @lru_cache(maxsize=None)
    def account_bond_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.bond_overweights(sort=False, by=by)
        )

    def tsy_weights(self):
        tsy_df = self.df[self.df["Sector"] == "TREASURIES"].copy()

        # Add Maturity bin column to treasuries.
        self._tsy_bins = [2, 3, 5, 7, 10, 20, 30]
        bin_map = {i: bin_ for i, bin_ in enumerate(self._tsy_bins)}
        tsy_df["MaturityBin"] = np.digitize(
            tsy_df["MaturityYears"], self._tsy_bins
        )
        tsy_df["MaturityBin"] = tsy_df["MaturityBin"].map(bin_map)

        ix = pd.Series(index=self._tsy_bins)
        tsy_df["AV_P_Weight"] = tsy_df["P_Weight"] * tsy_df["AccountValue"]
        weights = (
            tsy_df.groupby("MaturityBin").sum()["AV_P_Weight"]
            / self.market_value
        )
        return weights.add(ix, fill_value=0).fillna(0)


# %%
def main():
    pass
    # %%
    from collections import defaultdict
    from tqdm import tqdm
    from lgimapy.data import Database
    from lgimapy.utils import Time, load_json

    db = Database()
    db.display_all_columns()

    # %%
    date = db.date("today")

    # date = pd.to_datetime("2/24/2020")
    # date = db.date("1w", reference_date=date)
    # date = "2/24/2020"
    # act_name = "LGAS-CB"
    act_name = "AEELA"
    # act_name = "NFLLA"
    # act_name = "SEIC"
    # act_name = "LIB150"
    # act_name = "P-LD"
    act_name = "FLD"
    act_df = db.load_portfolio(
        account=act_name,
        date=date,
        market_cols=True,
        ret_df=True,
        # universe="stats",
    )
    acnt = Account(act_df, name=act_name, date=date)
    acnt.ticker_overweights()

    acnt.HY_ticker_overweights()
    # %%
    acnt.rating_overweights("OAD")
    # df.to_csv("rep_account_ticker_overweights.csv")
    # strat_name = "US Credit"
    strat_name = "US Long Credit"
    strat_name = "Liability Aware Long Duration Credit"
    strat_name = "US Long A+ Credit"
    strat_name = "US Credit Plus"
    # strat_name = "US High Yield"

    # %%
    strat_df = db.load_portfolio(
        strategy=strat_name,
        date=date,
        market_cols=True,
        ret_df=True,
        # universe="stats",
    )
    strat = Strategy(strat_df, name=strat_name, date=date)

    # %%
    strat.HY_ticker_overweights("P_Weight")
