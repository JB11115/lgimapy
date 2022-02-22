import warnings
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from functools import lru_cache, reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import (
    BondBasket,
    groupby,
    HY_sectors,
    IG_sectors,
    IG_market_segments,
    index_kwargs,
)
from lgimapy.utils import (
    get_ordinal,
    replace_multiple,
    root,
    to_labelled_buckets,
    to_datetime,
    to_list,
    to_set,
    quantile,
)


# %%


class Portfolio:
    def __init__(self, date, name, class_name=None):
        self.date = to_datetime(date)
        self.name = name
        self._class_name = (
            self.__class__.__name__ if class_name is None else class_name
        )

    def __repr__(self):
        date = self.date.strftime("%m/%d/%Y")
        return f"{self.name} {self._class_name} {date}"

    @property
    def fid(self):
        """str: Filename safe version of strategy name."""
        repl = {" ": "_", "/": "_", "%": "pct"}
        return replace_multiple(self.name, repl)

    @property
    def _stored_properties_history_fid(self):
        data_dir = root(f"data/portfolios/history/{self._class_name}")
        return data_dir / f"{self.fid}.parquet"

    @property
    def stored_properties_history_df(self):
        try:
            return pd.read_parquet(self._stored_properties_history_fid)
        except FileNotFoundError:
            return None

    @property
    def latex_name(self):
        """str: LaTeX safe version of strategy name."""
        repl = {"%": "\%", "_": " "}
        return replace_multiple(self.name, repl)

    @property
    def _tsy_bins(self):
        return [2, 3, 5, 7, 10, 20, 30]

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

    def tracking_error(self, lookback_months=3):
        """float: Annualized tracking error"""
        df = self.stored_properties_history_df
        if df is None:
            return np.nan

        # If there is not enough data, return nan
        start_date = self.date - relativedelta(months=lookback_months)
        df_prev = df[df.index < start_date]
        if not len(df_prev):
            return np.nan

        # Get performance in the current lookback period.
        lookback_df = df[(df.index >= start_date) & (df.index <= self.date)]
        lookback_perf = list(lookback_df["performance"])
        # Add today's performance.
        lookback_perf.append(self.performance())
        return np.sqrt(252) * np.std(lookback_perf)

    def normalized_tracking_error(self, lookback_months=1):
        return self.tracking_error(lookback_months) / self.bm_oas()

    def stored_properties(self):
        return pd.Series(
            {
                "dts_pct": 100 * self.dts("pct"),
                "dts_abs": self.dts("abs"),
                "dts_duration": self.dts("duration"),
                "barbell": 100 * self.barbell(),
                "tracking_error": self.tracking_error(),
                "normalized_tracking_error": self.normalized_tracking_error(),
                "port_carry": self.carry(),
                "port_yield": self.port_yield(),
                "port_oasd": self.oasd(),
                "bm_oas": self.bm_oas(),
                "bm_oad": self.bm_oad(),
                "total_oad_ow": self.total_oad(),
                "tsy_oad_ow": self.tsy_oad(),
                "tsy_mv_pct": 100 * self.tsy_pct(),
                "curve_duration_5": self.curve_duration(5),
                "curve_duration_7": self.curve_duration(7),
                "curve_duration_10": self.curve_duration(10),
                "curve_duration_20": self.curve_duration(20),
                "AAA_AA_oasd_ow": self.rating_overweights("AAA/AA", by="OASD"),
                "A_oasd_ow": self.rating_overweights("A", by="OASD"),
                "BBB_oasd_ow": self.rating_overweights("BBB", by="OASD"),
                "HY_oasd_ow": self.rating_overweights("HY", by="OASD"),
                "corp_oasd_ow": self.top_level_sector_overweights(
                    "Corp", by="OASD"
                ),
                "noncorp_oasd_ow": self.top_level_sector_overweights(
                    "Non-Corp", by="OASD"
                ),
                "performance": self.performance(),
            },
            name=self.date,
        )

    def save_stored_properties(self):
        df = self.stored_properties_history_df
        properties = self.stored_properties().to_frame().T
        if df is None:
            properties.to_parquet(self._stored_properties_history_fid)
        else:
            updated_df = pd.concat(
                (df[~df.index.isin(properties.index)], properties)
            ).sort_index()
            updated_df.to_parquet(self._stored_properties_history_fid)

        if self._class_name == "Strategy":
            for acnt in self.accounts.values():
                acnt.save_stored_properties()

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

    @property
    def _stored_properties_idx_map(self):
        return {
            "dts_pct": "DTS (%)",
            "dts_abs": "DTS OW (abs)",
            "dts_duration": "DTS OW (dur)",
            "barbell": "Barbell (%)",
            "tracking_error": "Tracking Error",
            "normalized_tracking_error": "Normalized TE",
            "port_carry": "Carry (bp)",
            "port_yield": "Port Yield (%)",
            "port_oasd": "Port OASD",
            "bm_oas": "BM OAS",
            "bm_oad": "BM OAD",
            "total_oad_ow": "Total OAD OW",
            "tsy_oad_ow": "Tsy OAD OW",
            "tsy_mv_pct": "Tsy MV (%)",
            "curve_duration_5": "Curve Dur (5yr)",
            "curve_duration_7": "Curve Dur (7yr)",
            "curve_duration_10": "Curve Dur (10yr)",
            "curve_duration_20": "Curve Dur (20yr)",
            "AAA_AA_oasd_ow": "AAA-AA OW",
            "A_oasd_ow": "A OW",
            "BBB_oasd_ow": "BBB OW",
            "HY_oasd_ow": "HY OW",
            "corp_oasd_ow": "Corp OW",
            "noncorp_oasd_ow": "Non-Corp OW",
            "performance": "Performance",
        }

    @property
    def stored_properties_history_table(self):
        # Get a full history DataFrame with NaNs filled for dates
        # prior to the strategy existing if need be.
        empty_dates = pd.Series(self._trade_dates, self._trade_dates)
        empty_dates = empty_dates[
            (empty_dates.index >= pd.to_datetime("9/1/2018"))
            & (empty_dates < self.stored_properties_history_df.index[0])
        ]
        empty_df = pd.DataFrame(
            columns=self.stored_properties_history_df.columns, index=empty_dates
        )
        df = pd.concat((empty_df, self.stored_properties_history_df))
        df = df[df.index <= self.date]

        # Get a table of the middle of each month, but have performance
        # be for the entire month.
        monthly_performance_df = (
            df.groupby(pd.Grouper(freq="M")).sum().iloc[-13:]
        )
        monthly_df = (
            df.groupby(pd.Grouper(freq="M"))
            .apply(lambda gdf: gdf.iloc[int(len(gdf) / 2)])
            .iloc[-13:]
        )
        monthly_df["performance"] = monthly_performance_df["performance"]
        monthly_df.index = pd.Series(monthly_df.index).dt.strftime("%b*%Y")

        # Make a quarterly table for time between last 6-12 months.
        quarterly_performance = (
            monthly_df["performance"].iloc[::-1].rolling(window=3).sum()
        )
        quarter_months = {"Jan": "Q1", "Apr": "Q2", "Jul": "Q3", "Oct": "Q4"}
        pattern = "|".join(quarter_months.keys())
        quarterly_df = monthly_df.iloc[:-6]
        quarterly_df = quarterly_df[quarterly_df.index.str.contains(pattern)]
        quarterly_df["performance"] = quarterly_performance
        new_quarterly_idx = []
        for idx in quarterly_df.index:
            month, year = idx.split("*")
            new_quarterly_idx.append(f"{quarter_months[month]}*{year}")
        quarterly_df.index = new_quarterly_idx

        # Get a current snapshot.
        current_row = df.iloc[-1]
        current_row.loc["performance"] = monthly_df["performance"].iloc[-1]
        current_row.rename(f"{current_row.name:%b %#d * %Y}", inplace=True)

        # Combine into one historical table.
        n_monthly_columns = 8 - len(quarterly_df)
        historical_df = pd.concat(
            (
                quarterly_df,
                monthly_df.iloc[-n_monthly_columns:-1],
                current_row.to_frame().T,
            )
        ).T

        # Replace columns with no performance (will be 0 due to
        # Numpy's sum) with NaNs
        historical_df.loc["performance"].replace(0, np.nan, inplace=True)
        historical_df = historical_df.reindex(
            self._stored_properties_idx_map.keys()
        )
        historical_df.index = [
            self._stored_properties_idx_map[idx] for idx in historical_df.index
        ]
        return historical_df

    @property
    def stored_properties_percentile_table(self):
        df = self.stored_properties_history_df.copy()
        df = df[df.index <= self.date]

        # Add recent percentiles
        pctile_table = pd.DataFrame()
        recent_date = df.index[-1] - relativedelta(months=3)
        recent_df = df[df.index >= recent_date]
        pctile_table["3m*%tile"] = recent_df.rank(pct=True).iloc[-1]
        pctile_table["Entire*%tile"] = df.rank(pct=True).iloc[-1]

        # Remove performance.
        pctile_table.loc["performance"] = np.nan
        pctile_table = pctile_table.reindex(
            self._stored_properties_idx_map.keys()
        )
        pctile_table.index = [
            self._stored_properties_idx_map[idx] for idx in pctile_table.index
        ]
        return pctile_table

    @property
    def stored_properties_range_table(self):
        df = self.stored_properties_history_df.copy()
        df = df[df.index <= self.date]

        range_table = pd.DataFrame()
        range_table["Entire*Min"] = df.quantile(0.01)
        range_table["Entire*Median"] = df.median()
        range_table["Entire*Max"] = df.quantile(0.99)

        # Remove performance.
        range_table.loc["performance"] = np.nan
        range_table = range_table.reindex(
            self._stored_properties_idx_map.keys()
        )
        range_table.index = [
            self._stored_properties_idx_map[idx] for idx in range_table.index
        ]
        return range_table


class Account(BondBasket, Portfolio):
    def __init__(
        self, df, name, date, market="US", constraints=None, index="CUSIP"
    ):
        BondBasket.__init__(
            self,
            df=df,
            name=name,
            market=market,
            constraints=constraints,
            index=index,
        )
        Portfolio.__init__(self, date=date, name=name)
        self.full_df = df.copy()
        self.df, self.tsy_df, self.cash_df = self._split_credit_tsy_cash(df)

    def _add_BM_treasuries(self, df):
        """
        Add respective benchmark treasury tenor as a column
        to input DataFrame.
        """
        bm_treasury_d = {}
        for i in range(25):
            if i <= 2:
                bm_treasury_d[i] = 2
            elif i <= 3:
                bm_treasury_d[i] = 3
            elif i <= 6:
                bm_treasury_d[i] = 5
            elif i <= 15:
                bm_treasury_d[i] = 10
            elif i <= 23:
                bm_treasury_d[i] = 20
            else:
                # Everything greater than 23yr tenor will be
                # benchmarked to the 30yr. Leave as nan for now.
                bm_treasury_d[i] = np.nan

        # Find the tenor of every bond and map the tenor to
        # respective benchmark treasury.
        year = df["Date"].dt.year
        maturity_date = df["MaturityDate"].dt.year
        issue_date = df["IssueDate"].dt.year
        tenor = maturity_date - year
        df["BMTreasury"] = tenor.map(bm_treasury_d).fillna(30).astype("int8")

        # Only bonds issued as 7yr bonds are benchmarked to the 7.
        loc_7yr = (tenor == 7) & (issue_date == year)
        df.loc[loc_7yr, "BMTreasury"] = 7
        return df.copy()

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

    def n_cusips(self):
        return len(self.port_df)

    def dts(self, method="pct"):
        method = method.lower()
        if method == "pct":
            return self.dts("port") / self.dts("bm")
        elif method == "abs":
            return np.sum(self.df["DTS_Diff"])
        elif method == "duration":
            return np.sum(self.df["DTS_Diff"]) / self.bm_weight("OAS")
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
        elif method == "derivatives_abs":
            return np.sum(self.derivatives_df["DTS_Diff"])
        elif method == "derivatives_pct":
            return np.sum(self.derivatives_df["DTS_Diff"]) / self.dts("bm")
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

    def derivatives_mv_pct(self):
        return np.sum(self.derivatives_df["P_Weight"])

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

    def total_oad(self):
        return np.sum(self.full_df["OAD_Diff"])

    def port_oas(self):
        return self.port_weight("OAS")

    def bm_oas(self):
        return self.bm_weight("OAS")

    def port_oad(self):
        return self.bm_weight("OAD")

    def bm_oad(self):
        return self.bm_weight("OAD")

    def port_yield(self):
        return self.port_weight("YieldToWorst")

    def bm_yield(self):
        return self.bm_weight("YieldToWorst")

    def carry(self):
        return self.port_oas() - self.bm_oas()

    def barbell(self):
        tsy_implied_dts = 1 - (self.tsy_oad() / self.full_df["BM_OAD"].sum())
        return self.dts() - tsy_implied_dts

    @property
    def bm_df(self):
        return self.full_df[self.full_df["BM_Weight"] > 0].copy()

    @property
    def port_df(self):
        return self.full_df[self.full_df["P_Weight"] > 0].copy()

    @property
    def derivatives_df(self):
        return self.full_df[self.full_df["L1"] == "Derivatives"].copy()

    def bm_weight(self, col):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return (
                self.full_df["BM_Weight"] * self.full_df[col]
            ).sum() / self.full_df["BM_Weight"].sum()

    def port_weight(self, col):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return (
                self.full_df["P_Weight"] * self.full_df[col]
            ).sum() / self.full_df["P_Weight"].sum()

    def tsy_weights(self):
        ix = pd.Series(index=self._tsy_bins)
        weights = groupby(self.tsy_df, "MaturityBin")["P_Weight"]
        return weights.add(ix, fill_value=0).fillna(0)

    def rating_overweights(self, rating_buckets=None, by="OAD"):
        rating_kws = {
            "Total": (None, None),
            "AAA": "AAA",
            "AA": ("AA+", "AA-"),
            "AAA/AA": ("AAA", "AA-"),
            "A": ("A+", "A-"),
            "BBB": ("BBB+", "BBB-"),
            "HY": ("BB+", "CCC-"),
            "BB": ("BB+", "BB-"),
            "B": ("B+", "B-"),
        }
        default_rating_buckets = ["Total", "AAA", "AA", "A", "BBB", "BB", "B"]
        rating_buckets = (
            default_rating_buckets
            if rating_buckets is None
            else to_list(rating_buckets, dtype=str)
        )
        overweights = []
        for rating_bucket in rating_buckets:
            rating_bucket_port = self.subset(rating=rating_kws[rating_bucket])
            if len(rating_bucket_port.df):
                # Not empty
                overweights.append(getattr(rating_bucket_port, by.lower())())
            else:
                overweights.append(np.nan)
        s = pd.Series(overweights, index=rating_buckets, name=by)
        return s if len(s) > 1 else s.values[0]

    def bond_overweights(self, by="OAD", sort=True):
        ix = (
            self.df["Ticker"].astype(str)
            + " "
            + self.df["Coupon"].apply(lambda x: f"{x:.2f}")
            + " "
            + self.df["MaturityDate"]
            .apply(lambda x: f"`{x.strftime('%y')}")
            .astype(str)
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

    def issuer_overweights(self, by="OAD", sort=True):
        if sort:
            s = self.issuer_df[self._ow_col(by)].sort_values(ascending=False)
        else:
            s = self.issuer_df[self._ow_col(by)]
        return s.rename(by)

    def HY_ticker_overweights(self, by="OAD", sort=True):
        hy_account = self.subset(rating="HY")
        return hy_account.ticker_overweights(by=by, sort=True)

    def IG_sector_overweights(self, sectors=None, by="OAD", sort=True):
        sectors = IG_sectors() if sectors is None else sectors
        d = {}
        for sector in sectors:
            kwargs = index_kwargs(sector, unused_constraints="in_stats_index")
            df = self.subset(**kwargs).df
            d[kwargs["name"]] = df[f"{by}_Diff"].sum()
        s = pd.Series(d)
        if sort:
            return s.sort_values(ascending=False)
        else:
            return s

    def IG_market_segment_overweights(self, segments=None, by="OAD", sort=True):
        segments = IG_market_segments() if segments is None else segments
        d = {}
        for segment in segments:
            kwargs = index_kwargs(segment, unused_constraints="in_stats_index")
            df = self.subset(**kwargs).df
            d[kwargs["name"]] = df[f"{by}_Diff"].sum()
        s = pd.Series(d)
        if sort:
            return s.sort_values(ascending=False)
        else:
            return s

    def sector_overweights(self, by="OAD", sort=True):
        if sort:
            s = self.sector_df[self._ow_col(by)].sort_values(ascending=False)
        else:
            s = self.sector_df[self._ow_col(by)]
        return s.rename(by)

    def top_level_sector_overweights(self, sectors=None, by="OAD"):
        default_sectors = [
            "Corp",
            "Industrials",
            "Financials",
            "Utilities",
            "Non-Corp",
        ]
        sectors = (
            default_sectors if sectors is None else to_list(sectors, dtype=str)
        )
        sector_kws = {"Corp": ["Industrials", "Financials", "Utilities"]}
        overweights = []
        for sector in sectors:
            sector_port = self.subset(
                LGIMA_top_level_sector=sector_kws.get(sector, sector)
            )
            if len(sector_port.df):
                # Not empty
                overweights.append(getattr(sector_port, by.lower())())
            else:
                overweights.append(np.nan)
        s = pd.Series(overweights, index=sectors, name=by)
        return s if len(s) > 1 else s.values[0]

    @property
    @lru_cache(maxsize=None)
    def _port_total_return(self):
        return (self.full_df["TRet"] * self.full_df["P_Weight"]).sum()

    @property
    @lru_cache(maxsize=None)
    def _bm_total_return(self):
        return (self.full_df["TRet"] * self.full_df["BM_Weight"]).sum()

    @property
    @lru_cache(maxsize=None)
    def _port_credit_total_return(self):
        return (self.df["TRet"] * self.df["P_Weight"]).sum() / self.df[
            "P_Weight"
        ].sum()

    def total_return(self, return_type="relative"):
        return_type = return_type.lower()
        if return_type in {"bm", "benchmark"}:
            return self._bm_total_return
        elif return_type in {"p", "port", "portfolio", "abs", "absolute"}:
            return self._port_total_return
        elif return_type in {"diff", "rel", "relative"}:
            return self._port_total_return - self._bm_total_return
        elif return_type in {"tsy", "trsy", "treasury"}:
            return self._port_total_return - self._port_credit_total_return
        elif return_type in {"credit"}:
            return self._port_credit_total_return - self._bm_total_return
        else:
            raise ValueError(f"`{return_type}` is not a valid return_type.")

    def performance(self):
        return 1e4 * self.total_return("relative")

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

    def bm_tsy_bucket_table(self):
        bm_dts = self.dts("bm")
        port_dts = self.dts("port")
        port_dts_pct = self.dts()
        d = defaultdict(list)
        for tsy in self._tsy_bins:

            bucket_port = self.subset(
                benchmark_treasury=(tsy, tsy),
                drop_treasuries=False,
                df=self.full_df,
            )
            bucket_port_dts = port_dts_pct * bucket_port.dts("port") / port_dts
            bucket_bm_dts = bucket_port.dts("bm") / bm_dts
            bucket_corp_port = bucket_port.subset(
                LGIMA_top_level_sector=[
                    "Industrials",
                    "Financials",
                    "Utilities",
                ]
            )
            d["Total OAD OW"].append(bucket_port.total_oad())
            d["Tsy MV (%)"].append(100 * bucket_port.tsy_df["P_Weight"].sum())
            d["Tsy OAD OW"].append(bucket_port.tsy_oad())
            d["Credit OAD OW"].append(bucket_port.oad())
            d["Corp OAD OW"].append(bucket_corp_port.oad())
            d["DTS OW (%)"].append(100 * (bucket_port_dts - bucket_bm_dts))
            d["Port DTS (%)"].append(100 * bucket_port_dts)
            d["BM DTS (%)"].append(100 * bucket_bm_dts)

        table = pd.DataFrame(d, index=self._tsy_bins).T
        table["Total"] = table.sum(axis=1)
        cash = pd.Series({"Tsy MV (%)": 100 * self.cash_pct()})
        table["Cash"] = cash
        return table

    def maturity_bucket_table(self, buckets=None):
        # Get maturity bucket tuples and labels.
        default_buckets = [0, 2, 4, 6, 8, 10, 12, 15, 20, 24, 31]
        maturity_buckets = to_labelled_buckets(
            default_buckets if buckets is None else buckets,
            right_end_closed=False,
        )

        bm_dts = self.dts("bm")
        port_dts = self.dts("port")
        port_dts_pct = self.dts()
        d = defaultdict(list)
        for maturity_bucket in maturity_buckets.values():

            bucket_port = self.subset(
                maturity=maturity_bucket,
                drop_treasuries=False,
                df=self.full_df,
            )
            bucket_corp_port = bucket_port.subset(
                LGIMA_top_level_sector=["Industrials", "Financials"]
            )
            bucket_port_dts = port_dts_pct * bucket_port.dts("port") / port_dts
            bucket_bm_dts = bucket_port.dts("bm") / bm_dts
            d["Total OAD OW"].append(bucket_port.total_oad())
            d["Credit OAD OW"].append(bucket_port.oad())
            d["Corp OAD OW"].append(bucket_corp_port.oad())
            d["DTS OW (%)"].append(100 * (bucket_port_dts - bucket_bm_dts))
            d["Port DTS (%)"].append(100 * bucket_port_dts)
            d["BM DTS (%)"].append(100 * bucket_bm_dts)

        table = pd.DataFrame(d, index=maturity_buckets.keys()).T
        table["Total"] = table.sum(axis=1)
        return table

    def rating_risk_bucket_table(self):
        bm_dts = self.dts("bm")
        port_dts = self.dts("port")
        port_dts_pct = self.dts()
        d = defaultdict(list)
        for rating_bucket in self._rating_risk_buckets:

            bucket_port = self.subset(
                rating_risk_bucket=rating_bucket,
            )
            bucket_corp_port = bucket_port.subset(
                LGIMA_top_level_sector=["Industrials", "Financials"]
            )
            bucket_port_dts = port_dts_pct * bucket_port.dts("port") / port_dts
            bucket_bm_dts = bucket_port.dts("bm") / bm_dts

            d["BM OAS"].append(bucket_port.bm_weight("OAS"))
            d["Credit OAD OW"].append(bucket_port.oad())
            d["Corp OAD OW"].append(bucket_corp_port.oad())
            d["DTS OW (%)"].append(100 * (bucket_port_dts - bucket_bm_dts))
            d["Port DTS (%)"].append(100 * bucket_port_dts)
            d["BM DTS (%)"].append(100 * bucket_bm_dts)

        table = pd.DataFrame(d, index=self._rating_risk_buckets).T
        table["Total"] = table.sum(axis=1)
        table.loc["BM OAS", "Total"] = self.bm_weight("OAS")
        return table

    def maturity_spread_bucket_heatmap(
        self, maturity_buckets=None, n_oas_buckets=5
    ):
        default_buckets = [0, 2, 4, 6, 8, 10, 12, 15, 20, 24, 31]
        maturity_buckets = to_labelled_buckets(
            default_buckets if maturity_buckets is None else maturity_buckets,
            right_end_closed=False,
        )

        bm_dts = self.dts("bm")
        port_dts = self.dts("port")
        port_dts_pct = self.dts()
        d = defaultdict(list)
        bm__dts = []
        p__dts = []
        for label, maturity_bucket in maturity_buckets.items():
            maturity_bucket_port = self.subset(maturity=maturity_bucket)
            try:
                _, oas_bins = pd.qcut(
                    maturity_bucket_port.bm_df["OAS"],
                    q=n_oas_buckets,
                    retbins=True,
                )
            except ValueError:
                for _ in range(n_oas_buckets):
                    d[label].append(np.nan)
                continue
            oas_buckets = to_labelled_buckets(
                np.round(oas_bins, 5)[1:-1],
                interval=1e-5,
                left_end_closed=False,
                right_end_closed=False,
            )
            for oas_bucket in oas_buckets.values():
                maturity_oas_bucket_port = maturity_bucket_port.subset(
                    OAS=oas_bucket
                )
                maturity_oas_bucket_port_dts = (
                    port_dts_pct
                    * maturity_oas_bucket_port.dts("port")
                    / port_dts
                )
                maturity_oas_bucket_bm_dts = (
                    maturity_oas_bucket_port.dts("bm") / bm_dts
                )
                bm__dts.append(maturity_oas_bucket_bm_dts)
                p__dts.append(maturity_oas_bucket_port_dts)
                d[label].append(
                    maturity_oas_bucket_port_dts - maturity_oas_bucket_bm_dts
                )

        # Make nice row labels.
        q_labels = range(1, n_oas_buckets + 1)
        quantile_label = quantile(n_oas_buckets)
        row_labels = [f"{q}{get_ordinal(q)} {quantile_label}" for q in q_labels]
        row_labels[0] = f"Tightest {quantile_label}"
        row_labels[-1] = f"Widest {quantile_label}"

        table = pd.DataFrame(d, index=row_labels)
        table["Total"] = table.sum(axis=1)
        return table


class Strategy(BondBasket, Portfolio):
    def __init__(
        self,
        df,
        name,
        date,
        market="US",
        constraints=None,
        index="CUSIP",
        ignored_accounts=None,
    ):
        self._all_accounts = self._unique("Account", df)
        if ignored_accounts is not None:
            ignored_accounts = to_set(ignored_accounts, dtype=str)
            self.ignored_accounts = ignored_accounts
            df = df[~df["Account"].isin(ignored_accounts)].copy()
        else:
            self.ignored_accounts = set()
        BondBasket.__init__(
            self,
            df=df,
            name=name,
            market=market,
            constraints=constraints,
            index=index,
        )
        Portfolio.__init__(self, date=date, name=name)
        self.name = name

        # Separate all accounts in the strategy.
        self.df = df
        self.accounts = self._split_accounts()
        self.account_names = list(self.accounts.keys())
        self.df = self._process_input_df(df)

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

    def add_ignored_accounts(self, accounts):
        if not accounts:
            return self

        new_ignored_accounts = to_set(accounts, dtype=str)
        ignored_accounts = new_ignored_accounts | self.ignored_accounts
        return Strategy(
            self.df,
            name=self.name,
            date=self.date,
            market=self.market,
            constraints=self.constraints,
            index=self.index,
            ignored_accounts=ignored_accounts,
        )

    def drop_empty_accounts(self):
        return self.add_ignored_accounts(self.empty_accounts())

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
            return pd.concat(account_vals, axis=1).T
        else:
            return pd.Series(account_vals, index=self.account_names, name=name)

    def account_value_weight(self, account_vals, skipna=True):
        if isinstance(account_vals, pd.DataFrame):
            return (
                np.sum(
                    account_vals.mul(self.account_market_values, axis=0), axis=0
                )
                / self.market_value
            )
        elif isinstance(account_vals, pd.Series):
            return (account_vals.mul(self.account_market_values, axis=0)).sum(
                skipna=skipna
            ) / self.market_value
        elif isinstance(account_vals, dict):
            account_dfs = []
            for account_name, df in account_vals.items():
                account_mv = self.account_market_values.loc[account_name]
                account_dfs.append(account_mv * df)

            return (
                reduce(lambda x, y: x.add(y), account_dfs) / self.market_value
            )
        else:
            raise NotImplementedError

    def account_n_cusips(self):
        return self.calculate_account_values(lambda x: x.n_cusips()).rename(
            f"n_cusips"
        )

    def empty_accounts(self):
        n_cusips_df = self.account_n_cusips()
        credit_pct_df = self.account_credit_pct()
        no_cusip_accounts = to_set(n_cusips_df[n_cusips_df == 0].index)
        no_credit_accounts = to_set(credit_pct_df[credit_pct_df == 0].index)
        return no_cusip_accounts | no_credit_accounts

    def _property_functions(self, fun_type, properties):
        if properties is not None:
            properties = [p.lower() for p in to_list(properties, str)]
        else:
            properties = [
                "dts_pct",
                "dts_abs",
                "dts_duration",
                "credit_pct",
                "cash_pct",
                "tsy_pct",
                "tsy_oad",
                "market_value",
                "curve_duration(5)",
                "curve_duration(7)",
                "curve_duration(10)",
                "curve_duration(20)",
            ]

        account_funs = {
            "dts_pct": self.account_dts("pct"),
            "dts_abs": self.account_dts("abs"),
            "dts_duration": self.account_dts("duration"),
            "hy_dts_abs": self.account_dts("hy_abs"),
            "hy_dts_pct": self.account_dts("hy_pct"),
            "ig_dts_abs": self.account_dts("ig_abs"),
            "ig_dts_pct": self.account_dts("ig_pct"),
            "derivatives_dts_abs": self.account_dts("derivatives_abs"),
            "derivatives_dts_pct": self.account_dts("derivatives_pct"),
            "credit_pct": self.account_credit_pct(),
            "ig_mv_pct": self.account_IG_mv_pct(),
            "hy_mv_pct": self.account_HY_mv_pct(),
            "derivatives_mv_pct": self.account_derivatives_mv_pct(),
            "cash_pct": self.account_cash_pct(),
            "tsy_pct": self.account_tsy_pct(),
            "tsy_oad": self.account_tsy_oad(),
            "curve_duration(5)": self.account_curve_duration(5),
            "curve_duration(7)": self.account_curve_duration(7),
            "curve_duration(10)": self.account_curve_duration(10),
            "curve_duration(20)": self.account_curve_duration(20),
            "market_value": self.account_market_values,
        }
        strategy_funs = {
            "dts_pct": self.dts("pct"),
            "dts_abs": self.dts("abs"),
            "dts_duration": self.dts("duration"),
            "hy_dts_abs": self.dts("hy_abs"),
            "hy_dts_pct": self.dts("hy_pct"),
            "ig_dts_abs": self.dts("ig_abs"),
            "ig_dts_pct": self.dts("ig_pct"),
            "derivatives_dts_abs": self.dts("derivatives_abs"),
            "derivatives_dts_pct": self.dts("derivatives_pct"),
            "credit_pct": self.credit_pct(),
            "ig_mv_pct": self.IG_mv_pct(),
            "hy_mv_pct": self.HY_mv_pct(),
            "derivatives_mv_pct": self.derivatives_mv_pct(),
            "cash_pct": self.cash_pct(),
            "tsy_pct": self.tsy_pct(),
            "tsy_oad": self.tsy_oad(),
            "curve_duration(5)": self.curve_duration(5),
            "curve_duration(7)": self.curve_duration(7),
            "curve_duration(10)": self.curve_duration(10),
            "curve_duration(20)": self.curve_duration(20),
            "market_value": self.market_value,
        }
        column_names = {
            "dts_pct": "DTS (%)",
            "dts_abs": "DTS OW (abs)",
            "dts_duration": "DTS OW (dur)",
            "hy_dts_abs": "HY DTS (abs)",
            "hy_dts_pct": "HY DTS (%)",
            "ig_dts_abs": "IG DTS (abs)",
            "ig_dts_pct": "IG DTS (%)",
            "derivatives_dts_abs": "Derivatives DTS (abs)",
            "derivatives_dts_pct": "Derivatives DTS (%)",
            "credit_pct": "Credit (%)",
            "hy_mv_pct": "HY MV (%)",
            "ig_mv_pct": "IG MV (%)",
            "derivatives_mv_pct": "Derivatives MV (%)",
            "cash_pct": "Cash (%)",
            "tsy_pct": "Tsy (%)",
            "tsy_oad": "Tsy OAD",
            "curve_duration(5)": "Curve Dur (5yr)",
            "curve_duration(7)": "Curve Dur (7yr)",
            "curve_duration(10)": "Curve Dur (10yr)",
            "curve_duration(20)": "Curve Dur (20yr)",
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
            "dts_duration": "2f",
            "hy_dts_pct": "1%",
            "hy_dts_abs": "1f",
            "ig_dts_pct": "1%",
            "ig_dts_abs": "1f",
            "derivatives_dts_abs": "1f",
            "derivatives_dts_pct": "2%",
            "credit_pct": "1%",
            "hy_mv_pct": "1%",
            "ig_mv_pct": "1%",
            "derivatives_mv_pct": "2%",
            "cash_pct": "1%",
            "tsy_pct": "1%",
            "tsy_oad": "1f",
            "curve_duration(5)": "3f",
            "curve_duration(7)": "3f",
            "curve_duration(10)": "3f",
            "curve_duration(20)": "3f",
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
            "abs": "DTS OW (abs)",
            "duration": "DTS OW (dur)",
            "p": "Portfolio DTS (bp*yr)",
            "port": "Portfolio DTS (bp*yr)",
            "portfolio": "Portfolio DTS (bp*yr)",
            "bm": "Benchmark DTS (bp*yr)",
            "benchmark": "Benchmark DTS (bp*yr)",
            "hy_pct": "HY DTS (%)",
            "hy_abs": "HY DTS (abs)",
            "ig_pct": "IG DTS (%)",
            "ig_abs": "IG DTS (abs)",
            "derivatives_pct": "Derivatives DTS (%)",
            "derivatives_abs": "Derivatives DTS (abs)",
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
    def account_bm_oad(self):
        name = "OAD"
        return self.calculate_account_values(lambda x: x.bm_oad(), name)

    @lru_cache(maxsize=None)
    def bm_oad(self):
        return self.account_value_weight(self.account_bm_oad())

    @lru_cache(maxsize=None)
    def account_total_oad(self):
        name = "OAD"
        return self.calculate_account_values(lambda x: x.total_oad(), name)

    @lru_cache(maxsize=None)
    def total_oad(self):
        return self.account_value_weight(self.account_total_oad())

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
    def account_derivatives_mv_pct(self):
        name = "Derivatives MV (%)"
        return self.calculate_account_values(
            lambda x: x.derivatives_mv_pct(), name
        )

    @lru_cache(maxsize=None)
    def derivatives_mv_pct(self):
        return self.account_value_weight(self.account_derivatives_mv_pct())

    @lru_cache(maxsize=None)
    def account_bm_credit_pct(self):
        name = "Credit (%)"
        return self.calculate_account_values(lambda x: x.bm_credit_pct(), name)

    @lru_cache(maxsize=None)
    def bm_credit_pct(self):
        return self.account_value_weight(self.account_bm_credit_pct())

    @lru_cache(maxsize=None)
    def account_port_oas(self):
        name = "Port OAS"
        return self.calculate_account_values(lambda x: x.port_oas(), name)

    @lru_cache(maxsize=None)
    def port_oas(self):
        return self.account_value_weight(self.account_port_oas())

    @lru_cache(maxsize=None)
    def account_bm_oas(self):
        name = "BM OAS"
        return self.calculate_account_values(lambda x: x.bm_oas(), name)

    @lru_cache(maxsize=None)
    def bm_oas(self):
        return self.account_value_weight(self.account_bm_oas())

    @lru_cache(maxsize=None)
    def carry(self):
        return self.account_value_weight(self.account_carry())

    @lru_cache(maxsize=None)
    def account_carry(self):
        name = "Carry (bp)"
        return self.calculate_account_values(lambda x: x.carry(), name)

    @lru_cache(maxsize=None)
    def barbell(self):
        return self.account_value_weight(self.account_barbell())

    @lru_cache(maxsize=None)
    def account_barbell(self):
        name = "Barbell"
        return self.calculate_account_values(lambda x: x.barbell(), name)

    @lru_cache(maxsize=None)
    def port_yield(self):
        return self.account_value_weight(self.account_port_yield())

    @lru_cache(maxsize=None)
    def account_port_yield(self):
        name = "Port Yield (%)"
        return self.calculate_account_values(lambda x: x.port_yield(), name)

    @lru_cache(maxsize=None)
    def bm_yield(self):
        return self.account_value_weight(self.account_bm_yield())

    @lru_cache(maxsize=None)
    def account_bm_yield(self):
        name = "BM Yield (%)"
        return self.calculate_account_values(lambda x: x.bm_yield(), name)

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

    def issuer_overweights(self, by="OAD"):
        return (
            self.account_value_weight(self.account_issuer_overweights(by=by))
            .sort_values(ascending=False)
            .rename(by)
        )

    def sector_overweights(self, by="OAD"):
        return (
            self.account_value_weight(self.account_sector_overweights(by=by))
            .sort_values(ascending=False)
            .rename(by)
        )

    def top_level_sector_overweights(self, sectors=None, by="OAD"):
        default_sectors = (
            "Corp",
            "Industrials",
            "Financials",
            "Utilities",
            "Non-Corp",
        )
        sectors = (
            default_sectors
            if sectors is None
            else tuple(to_list(sectors, dtype=str))
        )
        s = self.account_value_weight(
            self.account_top_level_sector_overweights(sectors, by=by),
            skipna=False,
        )
        if isinstance(s, pd.Series):
            return s.rename(by)
        else:
            return s

    def rating_overweights(self, rating_buckets=None, by="OAD"):
        default_rating_buckets = ("Total", "AAA", "AA", "A", "BBB", "BB", "B")
        rating_buckets = (
            default_rating_buckets
            if rating_buckets is None
            else tuple(to_list(rating_buckets, dtype=str))
        )
        s = self.account_value_weight(
            self.account_rating_overweights(rating_buckets, by=by), skipna=False
        )
        if isinstance(s, pd.Series):
            return s.rename(by)
        else:
            return s

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
    def account_issuer_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.issuer_overweights(sort=False, by=by)
        )

    @lru_cache(maxsize=None)
    def account_sector_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.sector_overweights(sort=False, by=by)
        )

    @lru_cache(maxsize=None)
    def account_top_level_sector_overweights(self, sectors=None, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.top_level_sector_overweights(sectors, by=by)
        )

    @lru_cache(maxsize=None)
    def account_rating_overweights(self, rating_buckets=None, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.rating_overweights(rating_buckets, by=by)
        )

    @lru_cache(maxsize=None)
    def account_bond_overweights(self, by="OAD"):
        return self.calculate_account_values(
            lambda x: x.bond_overweights(sort=False, by=by)
        )

    def account_ticker_overweights_comp(self, by="OAD", n=20):
        df = self.calculate_account_values(
            lambda x: x.ticker_overweights(sort=False, by=by)
        ).T.rename_axis(None)
        return self._largest_variation(df, n)

    def account_IG_sector_overweights_comp(self, sectors=None, by="OAD", n=20):
        df = self.calculate_account_values(
            lambda x: x.IG_sector_overweights(
                sectors=sectors, sort=False, by=by
            )
        ).T
        return self._largest_variation(df, n)

    def account_IG_market_segments_overweights_comp(
        self, segments=None, by="OAD", n=20
    ):
        df = self.calculate_account_values(
            lambda x: x.IG_market_segment_overweights(
                segments=segments, sort=False, by=by
            )
        ).T
        return self._largest_variation(df, n)

    @lru_cache(maxsize=None)
    def account_performance(self):
        return self.calculate_account_values(lambda x: x.performance())

    @lru_cache(maxsize=None)
    def performance(self):
        return self.account_value_weight(self.account_performance())

    @lru_cache(maxsize=None)
    def account_total_return(self, return_type="relative"):
        return self.calculate_account_values(
            lambda x: x.total_return(return_type)
        )

    @lru_cache(maxsize=None)
    def total_return(self, return_type="relative"):
        return self.account_value_weight(self.account_total_return(return_type))

    def _largest_variation(self, df, n):
        """
        pd.DatFrame:
            Find top ``n`` rows with largest variation between portfolios.
        """
        # Find index of rows with most variation among portfolios.
        avg = df.fillna(0).mean(axis=1)
        std = df.fillna(0).std(axis=1).sort_values()
        if n is not None:
            idx = std[-n:].index[::-1]
        else:
            idx = std.index[::-1]
        table = df.loc[idx]
        table["Avg"] = avg
        cols = list(table.columns[-1:]) + list(table.columns[:-1])
        return table[cols]

    def tsy_weights(self):
        tsy_df = self.df[self.df["Sector"] == "TREASURIES"].copy()

        # Add Maturity bin column to treasuries.
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

    def rating_risk_bucket_table(self):
        df_d = {
            account_name: acnt.rating_risk_bucket_table()
            for account_name, acnt in self.accounts.items()
        }
        return self.account_value_weight(df_d)

    def bm_tsy_bucket_table(self):
        df_d = {
            account_name: acnt.bm_tsy_bucket_table()
            for account_name, acnt in self.accounts.items()
        }
        return self.account_value_weight(df_d)

    def maturity_bucket_table(self, buckets=None):
        df_d = {
            account_name: acnt.maturity_bucket_table(buckets)
            for account_name, acnt in self.accounts.items()
        }
        return self.account_value_weight(df_d)

    def maturity_spread_bucket_heatmap(
        self, maturity_buckets=None, n_oas_buckets=5
    ):
        df_d = {
            account_name: acnt.maturity_spread_bucket_heatmap(
                maturity_buckets=maturity_buckets, n_oas_buckets=n_oas_buckets
            )
            for account_name, acnt in self.accounts.items()
        }
        return self.account_value_weight(df_d)


# %%
def main():
    pass
    # %%
    from tqdm import tqdm
    from lgimapy.data import Database
    from lgimapy.utils import Time

    db = Database()
    db.display_all_columns()

    # %%

    # %%
    date = db.date("today")
    # date = db.date("1w")

    account_name = "XCELLC"
    act_df = db.load_portfolio(
        account=account_name,
        date=date,
        ret_df=True,
        # universe="stats",
    )

    self = acnt = Account(act_df, name=account_name, date=date)

    # %%

    strat_name = "US Long Credit"
    strat_name = "US Long Corporate A or better"

    strat_df = db.load_portfolio(
        strategy=strat_name,
        date=date,
        ret_df=True,
        # universe="stats",
    )
    self = strat = Strategy(strat_df, name=strat_name, date=date)
    # %%
