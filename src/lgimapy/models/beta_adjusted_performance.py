from collections import defaultdict
from functools import cached_property

import numpy as np
import pandas as pd
import statsmodels.api as sms

from tqdm import tqdm

from lgimapy.stats import percentile
from lgimapy.utils import to_list, to_datetime


# %%


class BetaAdjustedPerformance:
    """
    Model for training and forecasting beta adjusted returns for a
    given benchmark.

    Beta is approximated at a bond level as the DTS divided by the
    market value weighted median DTS of the specified benchmark index.

    The model uses some forward information in its methodology, as described
    below.

    *Excess returns are forecasted by multiplying the beta of each bond
    by the excess returns of index over the forecast horizon.
    *Total returns are forecasted first by forecasting excess returns
    as described above, and then converting the forecasted excess
    return into total return by using the realized move the rates
    curve over the horizon.

    Parameters
    ----------
    db: :class:`Database`, optional
        Database with data already loaded.
    """

    def __init__(self, db, universe="IG"):
        self._db = db
        self.universe = self._check_universe(universe)

    def _check_universe(self, universe):
        universes = {"IG", "HY"}
        if universe is None:
            return self.universe
        elif universe in universes:
            return universe
        else:
            raise ValueError(f"Universe must be in {universes}, got {universe}")

    @cached_property
    def _model_history_fid(self):
        return self._db.local("beta_adjusted_model_history.pickle")

    @property
    def _model_history_df(self):
        return pd.read_pickle(self._model_history_fid)

    @property
    def _universe_ratings(self):
        return {
            "IG": {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")},
            "HY": {"BB": ("BB+", "BB-"), "B": ("B+", "B-")},
        }[self.universe]

    @property
    def _universe_index_kwargs(self):
        return {
            "IG": {"in_stats_index": True},
            "HY": {"in_H0A0_index": True},
        }[self.universe]

    def train(
        self,
        forecast="1m",
        date=None,
        predict_from_date=None,
        universe=None,
        **kwargs,
    ):
        """
        Train the model over given period of time.

        Parameters
        ----------
        forecast: str, default='1m'
            Window to forecast returns over.
        start: str, default='3m'
            Period of time to train model.
        date: datetime, optional
            Date to train for.
        universe: Rating buckets and benchmark to include.

        kwargs:
            Keyword arguments to :meth:`Index.subset`
            for universe of the broader index to compute beta against.
        """
        self.date = (
            self._db.date("today") if date is None else to_datetime(date)
        )
        self.universe = self._check_universe(universe)
        self.forecast = forecast.upper()
        if predict_from_date is None:
            self.predict_from_date = self._db.date(
                forecast, reference_date=self.date
            )
        else:
            self.predict_from_date = to_datetime(predict_from_date)

        self.predict_from_fmt_date = f"{self.predict_from_date:%#m/%d}"
        required_dates = set(
            self._db.trade_dates(start=self.predict_from_date, end=self.date)
        )
        if not self._db.has_dates_loaded(required_dates):
            self._db.load_market_data(
                start=self.predict_from_date, end=self.date
            )

        self._model_kwargs = {**kwargs}
        self._model_ID = self.universe
        for key, val in self._model_kwargs.items():
            self._model_ID += f"_{key}={val}"

        self._fcast_df = self._beta_adjusted_fcast(**kwargs)

    def _get_index(self, **kwargs):
        """
        Use stas index to find all index eligible bonds over
        entire period. Then build an index of these bonds
        where they do not drop out for any reason.

        Returns
        -------
        :class:`Index`:
            Index of all bonds that were in stats index at
            any point during forecast.
        """
        stats_ix = self._db.build_market_index(
            start=self.predict_from_date,
            end=self.date,
            OAS=(-10, 3000),
            OASD=(0, 100),
            **self._universe_index_kwargs,
            **kwargs,
        )
        ix = self._db.build_market_index(
            start=self.predict_from_date,
            end=self.date,
            isin=stats_ix.isins,
        )

        ix.df["RatingBucket"] = np.NaN
        rating_buckets = {
            "A": ("AAA", "A-"),
            "BBB": ("BBB+", "BBB-"),
            "BB": ("BB+", "BB-"),
            "B": ("B+", "B-"),
        }
        for rating, rating_bounds in rating_buckets.items():
            nr_min, nr_max = self._db.convert_letter_ratings(rating_bounds)
            ix.df.loc[
                (ix.df["NumericRating"] >= nr_min)
                & (ix.df["NumericRating"] <= nr_max),
                "RatingBucket",
            ] = rating
        return ix

    def _beta_adjusted_fcast(self, **kwargs):
        """
        Get a forecast for excess returns of every bond when
        it is first seen in a given rating bucket (e.g., A or BBB)
        until the specified date. The forecast is simply the DTS
        of the given bond on this date divided by the market value
        weighted median DTS of all bonds on that date multiplied
        by the market value weighted median aggregated excess return
        of all bonds from this date until the date to be forecasted.

        Retuns
        ------
        pd.DataFrame:
            DataFrame with forecasted and realized excess returns
            as well as proper weighting value for aggregating
            individual bonds together.
        """
        self.ix = self._get_index(**kwargs)
        d = defaultdict(dict)
        for date in self.ix.dates:
            # Get the days data.
            date_df = self.ix.day(date)

            # Subset the bonds which have not been forecasted already
            # using the (CUSIP, RatingBucket) tuple, and ensure
            # bonds are in a rating bucket.
            already_been_forecasted = (
                pd.Series(date_df["RatingBucket"].items())
                .isin(d["FCast*XSRet"])
                .values
            )
            in_rating_bucket = ~date_df["RatingBucket"].isna()
            bonds_to_forecast = date_df[
                ~already_been_forecasted & in_rating_bucket
            ].index
            if not len(bonds_to_forecast):
                # No new bonds to forecast.
                continue

            # Calculate excess returns and weights from the current
            # date to the forecast date.
            ix_from_date = self.ix.subset(start=date)
            xsrets = ix_from_date.accumulate_individual_excess_returns()
            trets = ix_from_date.accumulate_individual_total_returns()
            weights = ix_from_date.get_value_history("MarketValue").sum()

            # Find median DTS on that day and XSRet of index over the forecast.
            # Normalized DTS by the median and multiply to get forecasted
            # excess returns.
            median_dts = percentile(date_df["DTS"], date_df["MarketValue"])
            median_xsret = percentile(xsrets, weights)
            normalized_dts = date_df["DTS"] / median_dts
            date_df["fcast_xsret"] = normalized_dts * median_xsret

            cols = [
                "fcast_xsret",
                "RatingBucket",
                "Ticker",
                "MaturityYears",
            ]
            fcast_df = pd.concat(
                (
                    date_df.loc[
                        date_df["CUSIP"].isin(bonds_to_forecast),
                        cols,
                    ],
                    xsrets,
                    trets,
                    weights.rename("weight"),
                ),
                axis=1,
                join="inner",
                sort=False,
            ).dropna()
            fcast_df = fcast_df[fcast_df["weight"] > 0]
            fcast_df["fcast_tret"] = (
                fcast_df["TRet"] + fcast_df["fcast_xsret"] - fcast_df["XSRet"]
            )

            # Store forecasted values.
            for cusip, row in fcast_df.iterrows():
                key = (cusip, row["RatingBucket"])
                d["FCast*XSRet"][key] = row["fcast_xsret"]
                d["Real*XSRet"][key] = row["XSRet"]
                d["XSRet*Out*Perform"][key] = row["XSRet"] - row["fcast_xsret"]
                d["FCast*TRet"][key] = row["fcast_tret"]
                d["Real*TRet"][key] = row["TRet"]
                d["TRet*Out*Perform"][key] = row["TRet"] - row["fcast_tret"]
                d["weight"][key] = row["weight"]
                d["Ticker"][key] = row["Ticker"]
                d["Maturity"][key] = row["MaturityYears"]

        return (
            pd.DataFrame(d)
            .reset_index()
            .set_index("level_0")
            .rename(columns={"level_1": "RatingBucket"})
            .rename_axis(None)
        )

    @property
    def _fcast_df_no_dupes(self):
        df = self._fcast_df.sort_values("Maturity", ascending=False)
        return df[~df.index.duplicated(keep="first")]

    def _WLS(self):
        y = self._fcast_df["Real*XSRet"]
        X = sms.add_constant(self._fcast_df["FCast*XSRet"])
        X = self._fcast_df["FCast*XSRet"]
        w = self._fcast_df["weight"]
        return sms.WLS(y, X, weights=w).fit()

    def rsquared(self, pctile=False):
        """
        Get rsquared for current model run.
        """
        # Perform weighted least squares on the model.
        current_wls = self._WLS()
        if not pctile:
            return current_wls.rsquared
        else:
            wls_history = self._model_history_df[self._model_ID].dropna()
            r_squared_history = [wls.rsquared for wls in wls_history]
            r_squared_history.append(current_wls.rsquared)
            r_squared_s = pd.Series(r_squared_history)
            return int(100 * r_squared_s.rank(pct=True).iloc[-1])

    def _weighted_average(self, col, df):
        return 1e4 * (df[col] @ df["weight"]) / np.sum(df["weight"])

    @property
    def _table_columns(self):
        if self._return_type == "XSRET":
            self._fcast_df["Out*Perform"] = self._fcast_df["XSRet*Out*Perform"]
            return ["FCast*XSRet", "Real*XSRet", "Out*Perform"]
        elif self._return_type == "TRET":
            self._fcast_df["Out*Perform"] = self._fcast_df["TRet*Out*Perform"]
            return ["FCast*TRet", "Real*TRet", "Out*Perform"]

    @property
    def _index_kwargs_source(self):
        return {
            "IG": "bloomberg",
            "HY": "baml",
        }[self.universe]

    def get_sector_table(
        self,
        add_index_row=False,
        index_name="Benchmark Index",
        index_oas=None,
        return_type="XSRet",
    ):
        """
        Aggregate forecast by sector and show performance
        over specified horizon.

        Returns
        -------
        pd.DataFrame:
            Table of sector performance vs forecast.
        """
        self._return_type = return_type.upper()
        _ = self._table_columns

        d = defaultdict(list)
        top_level_sector = "Industrials"
        for sector in self.sectors:
            if sector == "SIFI_BANKS_SR":
                top_level_sector = "Financials"
            elif sector == "UTILITY_OPCO":
                top_level_sector = "Non-Corp"
            for rating, rating_kws in self._universe_ratings.items():
                ix_sector = self.ix.subset(
                    **self._db.index_kwargs(
                        sector,
                        rating=rating_kws,
                        source=self._index_kwargs_source,
                    )
                )
                if not len(ix_sector.df):
                    continue

                d["Sector"].append(ix_sector.name)
                d["raw_sector"].append(sector)
                d["TopLevelSector"].append(top_level_sector)
                d["Rating"].append(rating)
                fcast_xsret_df = self._fcast_df[
                    self._fcast_df.index.isin(ix_sector.cusips)
                ]
                # Get spread change over past month.
                oas = ix_sector.OAS()
                d[f"OAS*{self.predict_from_fmt_date}"].append(oas.iloc[0])
                d[f"{self.forecast}*$\\Delta$OAS"].append(
                    oas.iloc[-1] - oas.iloc[0]
                )

                # Get excess return forecast and realized values in bp.
                for col in self._table_columns:
                    d[col].append(self._weighted_average(col, fcast_xsret_df))

        df = pd.DataFrame(d).reset_index(drop=True)

        if add_index_row:
            df = self._add_index_row(df, index_name, index_oas)

        table = df.sort_values("Out*Perform", ascending=False).reset_index(
            drop=True
        )
        if self._return_type == "TRET":
            for col in self._table_columns:
                table[col] = table[col] / 1e4

        return table

    def _add_index_row(self, df, name, oas=None):
        """Add Stats index row to table."""
        if oas is None:
            oas = self.ix.OAS()
        row_d = {
            "Sector": name,
            "TopLevelSector": "-",
            "raw_sector": "-",
            "Rating": "-",
            f"OAS*{self.predict_from_fmt_date}": oas.iloc[0],
            f"{self.forecast}*$\\Delta$OAS": oas.iloc[-1] - oas.iloc[0],
        }
        for col in self._table_columns:
            if col.startswith("FCast"):
                row_d[col] = np.nan
            elif col.startswith("Real"):
                row_d[col] = self._weighted_average(col, self._fcast_df)
            elif col.startswith("Out"):
                row_d[col] = 0
            else:
                raise ValueError(f"unknown column {col} in table")

        return pd.concat((df, pd.Series(row_d).to_frame().T))

    def get_full_issuer_table(self, return_type="XSRet"):
        """
        Get issuer outperformance for every issuer.
        """
        self._return_type = return_type.upper()

        def aggregate_issuers(df):
            d = {}
            for col in self._table_columns:
                d[col] = self._weighted_average(col, df)
            return pd.Series(d)

        ticker_df = (
            self._fcast_df.groupby("Ticker")
            .apply(aggregate_issuers)
            .rename_axis(None)
        )
        return ticker_df

    def get_issuer_table(
        self, n=8, add_rating_rows=True, return_type="XSRet", **kwargs
    ):
        """
        Get issuer outperformance for every specified subset of bonds.

        Parameters
        ----------
        n: int, default=8
            Number of issuers to show in table. Ranked by impact
            on specified subset of bonds.
        add_rating_rows: bool, default=True
            If True, add the performance of each rating category
            of the specified subset to the table.
        kwargs:
            Keyword arguments to define subset of bonds using
            `meth:Basket.subset`. Generally index kwargs for a sector.
        """
        self._return_type = return_type.upper()
        _ = self._table_columns  # add necessary columns

        cusips = self.ix.subset(**kwargs).cusips
        fcast_df = self._fcast_df_no_dupes[
            self._fcast_df_no_dupes.index.isin(cusips)
        ]
        df_rating = self._fcast_df[self._fcast_df.index.isin(cusips)]
        # Get dataframes for sector subset by rating, and find performance
        # of the sector subset by rating.
        rating_df_d = {
            rating: df_rating[df_rating["RatingBucket"] == rating]
            for rating in self._universe_ratings.keys()
        }

        sector_out_perform_d = {}
        for rating, df in rating_df_d.items():
            if len(df):
                # Find outperformance of sector.
                sector_out_perform_d[rating] = self._weighted_average(
                    "Out*Perform", df
                )
            else:
                sector_out_perform_d[rating] = np.nan

        tickers = fcast_df["Ticker"].unique()
        d = defaultdict(list)
        for ticker in tickers:
            df_ticker = fcast_df[fcast_df["Ticker"] == ticker]
            rating_label = len(df_ticker["RatingBucket"].unique()) > 1
            for rating, rating_kws in self._universe_ratings.items():
                df_ticker_rating = df_ticker[
                    df_ticker["RatingBucket"] == rating
                ]
                if not len(df_ticker_rating):
                    continue
                name = f"{ticker}-{rating}" if rating_label else ticker
                d["Issuer"].append(name)
                d["RatingBucket"].append(rating)
                ix_ticker = self.ix.subset(cusip=df_ticker_rating.index)
                df_ex_ticker = rating_df_d[rating][
                    rating_df_d[rating]["Ticker"] != ticker
                ]
                if len(df_ex_ticker):
                    # Calculate performance of sector without current
                    # ticker and record the difference with the
                    # sector overall to get impact of current ticker.
                    out_perform_ex = self._weighted_average(
                        "Out*Perform", df_ex_ticker
                    )
                    d["Impact*Factor"].append(
                        abs(sector_out_perform_d[rating] - out_perform_ex)
                    )
                else:
                    # Current ticker is only ticker in sector,
                    # so the impact is 100%.
                    d["Impact*Factor"].append(1)
                oas = ix_ticker.OAS()
                d[f"OAS*{self.predict_from_fmt_date}"].append(oas.iloc[0])
                d[f"{self.forecast}*$\\Delta$OAS"].append(
                    oas.iloc[-1] - oas.iloc[0]
                )
                for col in self._table_columns:
                    d[col].append(self._weighted_average(col, df_ticker_rating))

        table = pd.DataFrame(d).sort_values("Out*Perform", ascending=False)

        # Calculate impact factor for each issuer-rating combination
        # as a percent of each rating bucket.
        impact_rating_d = {
            rating: table.loc[
                table["RatingBucket"] == rating, "Impact*Factor"
            ].sum()
            for rating in self._universe_ratings.keys()
        }
        table["Impact*Factor"] = [
            row["Impact*Factor"] / impact_rating_d[row["RatingBucket"]]
            for __, row in table.iterrows()
        ]

        if n is not None:
            table.sort_values("Impact*Factor", ascending=False, inplace=True)
            table = table.iloc[:n, :].copy()

        if add_rating_rows:
            for rating, rating_kws in self._universe_ratings.items():
                if not len(rating_df_d[rating]):
                    continue
                ix_rating = self.ix.subset(rating=rating_kws, **kwargs)
                oas = ix_rating.OAS()
                row_d = {
                    "Issuer": f"{rating}-Rated",
                    "Impact*Factor": np.nan,
                    f"OAS*{self.predict_from_fmt_date}": oas.iloc[0],
                    f"{self.forecast}*$\\Delta$OAS": oas.iloc[-1] - oas.iloc[0],
                }
                for col in self._table_columns:
                    row_d[col] = self._weighted_average(
                        col, rating_df_d[rating]
                    )
                table = pd.concat((table, pd.Series(row_d).to_frame().T))

        table = table.sort_values("Out*Perform", ascending=False).reset_index(
            drop=True
        )
        if self._return_type == "TRET":
            for col in self._table_columns:
                table[col] = table[col] / 1e4
        table.index += 1
        return table

    def get_curve_table(self, window, n_points, return_type="XSRet"):
        """
        Get A rated and BBB rated out performance
        maturity curve over specified forecast.
        """
        self._return_type = return_type.upper()
        midpoints = np.linspace(1, 30, n_points)
        dfs = {
            rating: self._fcast_df[self._fcast_df["RatingBucket"] == rating]
            for rating in ["A", "BBB"]
        }
        d = defaultdict(list)
        for rating, rating_df in dfs.items():
            for midpoint in midpoints:
                low = max(0, midpoint - window / 2)
                high = midpoint + window / 2
                df = rating_df[
                    (low <= rating_df["Maturity"])
                    & (rating_df["Maturity"] <= high)
                ]
                out_performance = self._weighted_average("Out*Perform", df)
                d[rating].append(out_performance)

        return pd.DataFrame(d, index=midpoints)

    def table_prec(self, table, added_precs=None):
        all_precs = {
            f"Impact*Factor": "2f",
            f"OAS*{self.predict_from_fmt_date}": "0f",
            f"{self.forecast}*$\\Delta$OAS": "0f",
            "Real*XSRet": "+0f",
            "Real*TRet": "+2%",
            "FCast*XSRet": "+0f",
            "FCast*TRet": "+2%",
        }
        if self._return_type == "XSRET":
            all_precs["Out*Perform"] = "+0f"
        elif self._return_type == "TRET":
            all_precs["Out*Perform"] = "+2%"

        if added_precs is not None:
            all_precs = {**all_precs, **added_precs}

        prec = {}
        for col in table.columns:
            try:
                prec[col] = all_precs[col]
            except KeyError:
                continue
        return prec

    def _weighted_average(self, col, df):
        return 1e4 * (df[col] @ df["weight"]) / np.sum(df["weight"])

    @property
    def sectors(self):
        if self.universe == "IG":
            return [
                "CHEMICALS",
                "METALS_AND_MINING",
                "CAPITAL_GOODS",
                "CABLE_SATELLITE",
                "MEDIA_ENTERTAINMENT",
                "WIRELINES_WIRELESS",
                "AUTOMOTIVE",
                "RETAILERS",
                "FOOD_AND_BEVERAGE",
                "HEALTHCARE_EX_MANAGED_CARE",
                "MANAGED_CARE",
                "PHARMACEUTICALS",
                "TOBACCO",
                "INDEPENDENT",
                "INTEGRATED",
                "OIL_FIELD_SERVICES",
                "REFINING",
                "MIDSTREAM",
                "TECHNOLOGY",
                "TRANSPORTATION",
                "SIFI_BANKS_SR",
                "SIFI_BANKS_SUB",
                "US_REGIONAL_BANKS",
                "YANKEE_BANKS",
                "BROKERAGE_ASSETMANAGERS_EXCHANGES",
                "LIFE_SR",
                "LIFE_SUB",
                "P_AND_C",
                "REITS",
                "UTILITY_OPCO",
                "UTILITY_HOLDCO",
                "OWNED_NO_GUARANTEE",
                "GOVERNMENT_GUARANTEE",
                "HOSPITALS",
                "MUNIS",
                "SUPRANATIONAL",
                "SOVEREIGN",
                "UNIVERSITY",
            ]
        elif self.universe == "HY":
            return self._db.HY_sectors(unique=True)

    def update(self, pbar=False, limit=500):
        """
        Update model history data with model error
        for standard forecast horizon.
        """
        # Find last modeled date
        try:
            saved_df = self._model_history_df
        except (FileNotFoundError, OSError):
            # Model has not been run yet, start from scratch.
            saved_df = pd.DataFrame()
            last_saved_date = pd.to_datetime("1/1/2005")
        else:
            last_saved_date = saved_df.index[-1]

        dates = self._db.trade_dates(exclusive_start=last_saved_date)
        if not dates:
            return  # already up to date
        if limit is not None:
            dates = dates[:limit]

        IG_maturities = [(5, 10), (10, None)]
        d = defaultdict(list)
        for date in tqdm(dates):
            # Update IG forecasts.
            for maturity in IG_maturities:
                mod.train(
                    forecast="1m", date=date, maturity=maturity, universe="IG"
                )
                d[mod._model_ID].append(mod._WLS())

            # Update HY forecasts.
            mod.train(forecast="1m", date=date, universe="HY")
            d[mod._model_ID].append(mod._WLS())

        # Update model history file.
        new_df = pd.DataFrame(d, index=dates)
        updated_df = pd.concat((saved_df, new_df), axis=0, sort=True)
        updated_df.to_pickle(self._model_history_fid)


# %%
def main():
    pass
    # %%
    # Code for testing
    from lgimapy.data import Database

    db = Database()

    # %%
    forecast = "1M"
    maturity = (10, None)
    date = None
    # %%
    self = BetaAdjustedPerformance(db)
    print(f"{self._model_history_df.index[-1]:%b %d, %Y}\n")
    self._model_history_df.iloc[0, 0]
    # self._model_history_df.median().round(2)

    # %%
    self.train(forecast=forecast, maturity=maturity)

    # %%

    sector_table = self.get_sector_table(
        add_index_row=True,
        index_name="Long Credit Stats Index",
    )
    sector_table
    # %%
    kwargs = db.index_kwargs("SIFI_BANKS_SR")
    issuer_table = self.get_issuer_table(**kwargs)
    issuer_table
    df = issuer_table[issuer_table["RatingBucket"] == "A"].round(0)
    df.drop(["Impact*Factor", "RatingBucket"], axis=1).set_index(
        "Issuer"
    ).astype(int)

    # %%


if __name__ == "__main__":
    from lgimapy.data import Database

    mod = BetaAdjustedPerformance(Database())
    for i in range(200):
        mod.update(pbar=True, limit=50)
