from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lgimapy.models import weighted_percentile
from lgimapy.utils import to_list, root


# %%


class XSRETPerformance:
    """
    Model for training and forecasting excess returns
    vs realized excess returns to find outperformance
    amongst bonds.

    Parameters
    ----------
    db: :class:`Database`, optional
        Database with data already loaded.
    """

    def __init__(self, db):
        self.db = db

    @property
    @lru_cache(maxsize=None)
    def _model_history_df(self):
        fid = root("data/xsret_model_history.parquet")
        return pd.read_parquet(fid)

    def train(self, forecast="1m", date=None, **kwargs):
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
        kwargs:
            Keyword arguments to :meth:`Index.subset`
            for bonds to train on.
        """
        self.date = pd.to_datetime(date)
        self.forecast = forecast.upper()
        self.predict_from_date = self.db.date(
            forecast, reference_date=self.date
        )
        self.predict_from_fmt_date = self.predict_from_date.strftime("%#m/%d")
        required_dates = set(
            self.db.trade_dates(start=self.predict_from_date, end=date)
        )
        if not required_dates < set(self.db.loaded_dates):
            # Database is not loaded with all required dates.
            raise IndexError("Database does not have required dates loaded")

        self._fcast_df = self._get_dts_forecast(**kwargs)

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
        stats_ix = self.db.build_market_index(
            start=self.predict_from_date,
            end=self.date,
            in_stats_index=True,
            OAS=(-10, 3000),
            OASD=(0, 100),
            **kwargs,
        )
        ix = self.db.build_market_index(
            start=self.predict_from_date,
            end=self.date,
            cusip=stats_ix.cusips,
            **kwargs,
        )

        ix.df["DTS"] = ix.df["OAS"] * ix.df["OASD"]
        ix.df["RatingBucket"] = np.NaN
        ix.df.loc[ix.df["NumericRating"] <= 7, "RatingBucket"] = "A"
        ix.df.loc[
            ix.df["NumericRating"].isin((8, 9, 10)), "RatingBucket"
        ] = "BBB"
        return ix

    def _get_dts_forecast(self, **kwargs):
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
            weights = ix_from_date.get_value_history("MarketValue").sum()

            # Find median DTS on that day and XSRet of index over the forecast.
            # Normalized DTS by the median and multiply to get forecasted
            # excess returns.
            median_dts = weighted_percentile(
                date_df["DTS"], date_df["MarketValue"]
            )
            median_xsret = weighted_percentile(xsrets, weights)
            normalized_dts = date_df["DTS"] / median_dts
            date_df["fcast"] = normalized_dts * median_xsret

            cols = ["fcast", "RatingBucket", "Ticker", "MaturityYears"]
            fcast_df = pd.concat(
                (
                    date_df.loc[
                        date_df["CUSIP"].isin(bonds_to_forecast),
                        cols,
                    ],
                    xsrets,
                    weights.rename("weight"),
                ),
                axis=1,
                join="inner",
                sort=False,
            ).dropna()
            fcast_df = fcast_df[fcast_df["weight"] > 0]

            # Store forecasted values.
            for cusip, row in fcast_df.iterrows():
                key = (cusip, row["RatingBucket"])
                d["FCast*XSRet"][key] = row["fcast"]
                d["Real*XSRet"][key] = row["XSRet"]
                d["Out*Perform"][key] = row["XSRet"] - row["fcast"]
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

    weighted_percentile([1, 2, 3, 4], [1, 1, 1, 10], q=50)

    def MAE(self, pctile=None):
        """
        Get mean absolute model error from current run.
        """
        df = self._fcast_df.copy()
        df["abs_resid"] = np.abs(df["Out*Perform"])
        mae = self._weighted_average("abs_resid", df)
        if pctile is None:
            return mae
        else:
            mae_history = self._model_history_df[f"MAE_{pctile}"].dropna()
            updated_history = mae_history.append(
                pd.Series(mae), ignore_index=True
            )
            return int(100 * updated_history.rank(pct=True).iloc[-1])

        return mae

    def MAD(self, pctile=None):
        """
        Get median absolute deviation of model error from current run.
        """
        df = self._fcast_df.copy()
        df["abs_resid"] = np.abs(df["Out*Perform"])
        mad = 1e4 * weighted_percentile(df["abs_resid"], df["weight"], q=50)
        if pctile is None:
            return mad
        else:
            mad_history = self._model_history_df[f"MAD_{pctile}"].dropna()
            updated_history = mad_history.append(
                pd.Series(mad), ignore_index=True
            )
            return int(100 * updated_history.rank(pct=True).iloc[-1])

        return mad

    def _weighted_average(self, col, df):
        return 1e4 * (df[col] @ df["weight"]) / np.sum(df["weight"])

    def get_sector_table(self):
        """
        Aggregate forecast by sector and show performance
        over specified horizon.

        Returns
        -------
        pd.DataFrame:
            Table of sector performance vs forecast.
        """
        ratings = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
        d = defaultdict(list)
        top_level_sector = "Industrials"
        for sector in self.sectors:
            if sector == "SIFI_BANKS_SR":
                top_level_sector = "Financials"
            elif sector == "UTILITY":
                top_level_sector = "Non-Corp"
            for rating, rating_kws in ratings.items():
                ix_sector = self.ix.subset(
                    **self.db.index_kwargs(sector, rating=rating_kws)
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
                for col in ["FCast*XSRet", "Real*XSRet", "Out*Perform"]:
                    d[col].append(self._weighted_average(col, fcast_xsret_df))
        return (
            pd.DataFrame(d)
            .sort_values("Out*Perform", ascending=False)
            .reset_index(drop=True)
        )

    def get_full_issuer_table(self):
        """
        Get issuer outperformance for every issuer.
        """
        ratings = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
        fcast_df = self._fcast_df.copy()

        def aggregate_issuers(df):
            d = {}
            for col in ["FCast*XSRet", "Real*XSRet", "Out*Perform"]:
                d[col] = self._weighted_average(col, df)
            return pd.Series(d)

        ticker_df = (
            fcast_df.groupby("Ticker")
            .apply(aggregate_issuers)
            .rename_axis(None)
        )
        return ticker_df

    def get_issuer_table(self, n=8, add_rating_rows=True, **kwargs):
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
        cusips = self.ix.subset(**kwargs).cusips
        fcast_df = self._fcast_df_no_dupes[
            self._fcast_df_no_dupes.index.isin(cusips)
        ]
        df_rating = self._fcast_df[self._fcast_df.index.isin(cusips)]
        rating_kwargs = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
        # Get dataframes for sector subset by rating, and find performance
        # of the sector subset by rating.
        rating_df_d = {
            rating: df_rating[df_rating["RatingBucket"] == rating]
            for rating in rating_kwargs.keys()
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
            for rating, rating_kws in rating_kwargs.items():
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
                for col in ["FCast*XSRet", "Real*XSRet", "Out*Perform"]:
                    d[col].append(self._weighted_average(col, df_ticker_rating))

        table = pd.DataFrame(d).sort_values("Out*Perform", ascending=False)

        # Calculate impact factor for each issuer-rating combination
        # as a percent of each rating bucket.
        impact_rating_d = {
            rating: table.loc[
                table["RatingBucket"] == rating, "Impact*Factor"
            ].sum()
            for rating in rating_kwargs.keys()
        }
        table["Impact*Factor"] = [
            row["Impact*Factor"] / impact_rating_d[row["RatingBucket"]]
            for __, row in table.iterrows()
        ]

        if n is not None:
            table.sort_values("Impact*Factor", ascending=False, inplace=True)
            table = table.iloc[:n, :].copy()

        if add_rating_rows:
            for rating, rating_kws in rating_kwargs.items():
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
                for col in ["FCast*XSRet", "Real*XSRet", "Out*Perform"]:
                    row_d[col] = self._weighted_average(
                        col, rating_df_d[rating]
                    )
                row = pd.Series(row_d)
                table = table.append(row, ignore_index=True)

        table = table.sort_values("Out*Perform", ascending=False).reset_index(
            drop=True
        )
        table.index += 1
        return table

    def get_curve_table(self, window, n_points):
        """
        Get A rated and BBB rated out performance
        maturity curve over specified forecast.
        """
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

    def table_prec(self, table):
        all_precs = {
            f"Impact*Factor": "2f",
            f"OAS*{self.predict_from_fmt_date}": "0f",
            f"{self.forecast}*$\\Delta$OAS": "0f",
            "Real*XSRet": "0f",
            "FCast*XSRet": "0f",
            "Out*Perform": "0f",
        }
        prec = {}
        for col in table.columns:
            try:
                prec[col] = all_precs[col]
            except KeyError:
                continue
        return prec

    def _weighted_average(self, col, df):
        return 1e4 * (df[col] @ df["weight"]) / np.sum(df["weight"])

    def add_index_row(self, df, name):
        """Add Stats index row to table."""
        oas = self.ix.OAS()
        row_d = {
            "Sector": name,
            "TopLevelSector": "-",
            "raw_sector": "-",
            "Rating": "-",
            f"OAS*{self.predict_from_fmt_date}": oas.iloc[0],
            f"{self.forecast}*$\\Delta$OAS": oas.iloc[-1] - oas.iloc[0],
            "FCast*XSRet": np.nan,
            "Real*XSRet": self._weighted_average("Real*XSRet", self._fcast_df),
            "Out*Perform": 0,
        }
        row = pd.Series(row_d)
        return df.append(row, ignore_index=True)

    @property
    def sectors(self):
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
            "LIFE",
            "P&C",
            "REITS",
            "UTILITY",
            "OWNED_NO_GUARANTEE",
            "GOVERNMENT_GUARANTEE",
            "HOSPITALS",
            "MUNIS",
            "SUPRANATIONAL",
            "SOVEREIGN",
            "UNIVERSITY",
        ]


# %%
def main():
    pass
    # %%
    from lgimapy.data import Database

    forecast = "1m"
    db = Database()
    db.load_market_data(start=db.date("2m"))
    date = None
    self = XSRETPerformance(db)
    self.train(forecast=forecast, date=date, maturity=(5, 10))
    kwargs = db.index_kwargs("MIDSTREAM")
    curr = self.get_issuer_table(**kwargs)

    # %%
    # %%
    # n_points = 500
    # window = 4
    # curve_table = (
    #     self.get_curve_table(window=window, n_points=n_points)
    #     .rolling(10, min_periods=0)
    #     .mean()
    # )
    #
    # from lgimapy import vis
    # vis.style()
    #
    # # %%
    # fig, ax = vis.subplots(figsize=(10, 6))
    # kws = {"alpha": 0.6}
    # ax.fill_between(
    #     curve_table.index, 0, curve_table["A"], color="navy", label="A", **kws
    # )
    # ax.fill_between(
    #     curve_table.index,
    #     0,
    #     curve_table["BBB"],
    #     color="goldenrod",
    #     label="BBB",
    #     **kws,
    # )
    # ax.legend(fancybox=True, shadow=True)
    # ax.set_title(f"Curve Outperformance since {self.predict_from_fmt_date}")
    # ax.set_ylabel("Excess Return (bp)")
    # ax.set_
    # %%
