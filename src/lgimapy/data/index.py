import time
import warnings
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from inspect import getfullargspec

import numpy as np
import pandas as pd
from oslo_concurrency import lockutils
from pyarrow.lib import ArrowInvalid

from lgimapy.bloomberg import get_bloomberg_ticker
from lgimapy.data import (
    BondBasket,
    concat_index_dfs,
    groupby,
    new_issue_mask,
    TreasuryCurve,
)
from lgimapy.stats import mean, std
from lgimapy.utils import (
    check_all_equal,
    dump_json,
    load_json,
    mkdir,
    replace_multiple,
    root,
    to_datetime,
)

# %%
def get_unique_fid(fid_map):
    """
    Generate a unique filename given map of keys
    to filenames.

    Parameters
    ----------
    fid_map: dict
        Mapping of keys to filename values.

    Returns
    -------
    fid: str
        Unique filename not in input filename map.
    """

    def generate_random_fid():
        """str: Generate random filename."""
        n_digits_in_fid = 18
        epoch = int(time.time() * 1e8)
        time.sleep(0.0001)
        fid = f"{epoch}.parquet"
        return fid

    while True:
        fid = generate_random_fid()
        if fid not in fid_map.values():
            return fid


class Index(BondBasket):
    """
    Class for indexes built by :class:`Database`.

    Parameters
    ----------
    df: pd.DataForame
        Index DataFrame from :meth:`Database.build_market_index`.
    name: str, optional
        Optional name of index.
    market: ``{"US", "EUR", "GBP"}``, default="US"
        Market to get trade dates for.
    index: str or None, default='CUSIP'
        Column to use as index for :attr:`BondBasket.df`.
    constraints: dict, optional
        Key: value pairs of the constraints used in either
        :meth:`Database.build_market_index` or
        :meth:`Index.subset` to create current :class:`Index`.

    Attributes
    ----------
    df: pd.DataFrame
        DataFrame with each row containing a bond in
        the :class:`Index`.
    """

    def __init__(
        self, df, name=None, constraints=None, market="US", index="CUSIP"
    ):
        super().__init__(
            df=df,
            name=name,
            market=market,
            constraints=constraints,
            index=index,
        )
        # Initialize cache for storing daily DataFrames.
        self._day_cache = {}
        self._day_cache_key = "_".join(self.df.columns)

    def __repr__(self):
        start = self.dates[0].strftime("%m/%d/%Y")
        end = self.dates[-1].strftime("%m/%d/%Y")
        if start == end:
            return f"{self.name} Index {start}"
        else:
            return f"{self.name} Index {start} - {end}"

    def __add__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        else:
            raise TypeError(f"Right operand must be an {type(self).__name__}.")

    def __radd__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        elif other == 0:
            return self
        else:
            raise TypeError(f"Left operand must be an {type(self).__name__}.")

    def __iadd__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        else:
            raise TypeError(f"Right operand must be an {type(self).__name__}.")

    def copy(self):
        """Create copy of current :class:`Index`."""
        return Index(self.df.copy(), constraints=self.constraints)

    def total_value(self, col="MarketValue", synthetic=False):
        """float: Total value of index in $M."""
        if synthetic:
            dates = self.dates[1:] if synthetic else self.dates
            a = np.zeros(len(dates))
            for i, date in enumerate(dates):
                df = self.synthetic_day(date) if synthetic else self.day(date)
                a[i] = np.sum(df[col])
            return pd.Series(a, index=dates, name="total_value")
        else:
            return (
                self.df[["Date", col]]
                .groupby("Date")
                .sum()[col]
                .rename("total_value")
            )

    def _create_day_cache(self):
        self._day_cache = {date: df for date, df in self.df.groupby("Date")}
        self._day_cache_key = "_".join(self.df.columns)

    def day(self, date, as_index=False):
        """
        Memoized call to a dict of single day DataFrame
        with date and all computed columns as key. If new
        columns are added to :attr:`Index.df`, an updated
        single day DataFrame is loaded and saved to cache.

        Parameters
        ----------
        date: datetime object
            Date of daily DataFrame to return.
        as_index: bool, default=False
            If true, return an :class:`Index` for specified day instead
            of a DataFrame.

        Returns
        -------
        df or Index: pd.DataFrame or :class:`Index`
            DataFrame or Index for single specified date.
        """
        date = to_datetime(date)
        # Create cache key of all columns added to :attr:`Index.df`.
        current_cache_key = "_".join(self.df.columns)
        if current_cache_key == self._day_cache_key:
            # No new column changes since last accessed,
            # treat like normal cache.
            try:
                if as_index:
                    return Index(self._day_cache[date], self.name)
                else:
                    return self._day_cache[date]
            except KeyError:
                # Cache doesn't exist, create it.
                self._day_cache = {
                    date: df for date, df in self.df.groupby("Date")
                }
        else:
            # New columns added since last accessed, update cache.
            self._create_day_cache()

        if as_index:
            return Index(self._day_cache[date], self.name)
        else:
            return self._day_cache[date]

    def synthetic_day(self, date, as_index=False):
        """
        Creates :class:`Index` or DataFrames containing only the intersection
        of constituents which existed in the index both the day before
        and the specified day.

        Parameters
        ----------
        date: datetime object
            Date of daily DataFrame to return.
        as_index: bool, default=False
            If true, return an Index for specified day instead
            of a DataFrame.

        Returns
        -------
        df or Index: pd.DataFrame or :class:`Index`
            DataFrame or Index for specified date.
        """
        # Get data for current and previous day.
        current_date = to_datetime(date)
        prev_date = self.dates[bisect_left(self.dates, current_date) - 1]
        if prev_date == self.dates[-1]:
            msg = "First day in index history, synthetic day not possible."
            raise IndexError(msg)
        current_df = self.day(current_date)
        prev_df = self.day(prev_date)

        # Return only bond data for bonds on the current day
        # that were in the index on the previous day.
        intersection_ix = set(current_df.index).intersection(prev_df.index)
        df = current_df[current_df.index.isin(intersection_ix)]
        if as_index:
            return Index(df, self.name)
        else:
            return df

    def issuer_index(self):
        """
        Combine bonds by ticker into an index of daily issuers.

        Returns
        -------
        :class:`Index`:
            Index of issuers.
        """

        def daily_issuer_agg(df):
            """Aggregate bonds for single day into issuers."""
            return groupby(df, "Ticker")

        idf = self.df.groupby("Date").apply(daily_issuer_agg).reset_index()
        # Add a CUSIP column since a :class:`Index`
        # expects it as the pd.DataFrame.index.
        idf["CUSIP"] = idf["Ticker"]
        return Index(idf.set_index("CUSIP", drop=False))

    def clear_day_cache(self):
        """Clear cache of stored days."""
        self._day_cache = ""

    def clean_treasuries(self):
        """
        Clean treasury index for build model curve.

        Keep only non-callable treasuries. Additionally,
        remove bonds that have current maturities less than
        the original maturity of another tenor (e.g., remove
        30 year bonds once their matuirty is less than 10 years,
        10 and 7,7 and 5, etc.).
        """
        # Add Bloomberg ticker.
        self.df["BTicker"] = get_bloomberg_ticker(self.df["CUSIP"].values)

        # For each bond, determine if it meets all the rules
        # required to be used in the curve.
        is_treasury = self.df["BTicker"].isin({"US/T", "T"})
        is_zero_coupon = (self.df["CouponType"] == "ZERO COUPON") | (
            (self.df["CouponType"] == "FIXED") & (self.df["CouponRate"] == 0)
        )
        is_not_strip = ~self.df["BTicker"].isin({"S", "SP", "SPX", "SPY"})
        matures_on_15 = self.df["MaturityDate"].dt.day == 15
        is_noncall = self.df["CallType"] == "NONCALL"
        oas_not_strongly_negative = self.df["OAS"] > -30
        oas_not_strongly_positive = self.df["OAS"] < 40
        maturity_map = {2: 0, 3: 2, 5: 3, 8: 5, 7: 5, 10: 7, 30: 10}
        has_normal_mat = self.df["OriginalMaturity"].isin(maturity_map.keys())
        matures_in_more_than_3_months = self.df["MaturityYears"] > (3 / 12)
        bad_cusips = ["912820FL6"]
        is_not_bad_cusip = ~self.df["CUSIP"].isin(bad_cusips)

        # Apply all rules.
        df = self.df[
            (is_treasury | is_zero_coupon)
            & (is_not_strip | matures_on_15)
            & is_noncall
            & has_normal_mat
            & oas_not_strongly_negative
            & oas_not_strongly_positive
            & matures_in_more_than_3_months
            & is_not_bad_cusip
        ].copy()
        mask = [
            my > maturity_map[om] or bt in {"B", "S", "SP", "SPX", "SPY"}
            for my, om, bt in zip(
                df["MaturityYears"], df["OriginalMaturity"], df["BTicker"]
            )
        ]
        df = df[mask].copy()
        self.df = df.copy()

    def get_value_history(
        self,
        col,
        start=None,
        end=None,
        inclusive_end_date=True,
        synthetic=False,
    ):
        """
        Get history of any column for all cusips in Index.

        Parameters
        ----------
        col: str
            Column from :attr:`Index.df` to build history for (e.g., 'OAS').
        start: datetime, optional
            Start date for value history.
        end: datetime, optional
            End date for value history.
        inclusive_end_date: bool, default=True
            If True include end date in returned DataFrame.
            If False do not include end date.
        synthethic: bool, default=False
            If True, use synthethic day data.

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index, CUSIP columns, and price values.
        """
        cusips = set(self.cusips)
        dates = self.dates[1:] if synthetic else self.dates
        if start is not None:
            dates = dates[dates >= to_datetime(start)]
        if end is not None:
            if inclusive_end_date:
                dates = dates[dates <= to_datetime(end)]
            else:
                dates = dates[dates < to_datetime(end)]

        # Build dict of historical values for each CUSIP and
        # convert to DataFrame.
        hist_d = defaultdict(list)
        for d in dates:
            day_df = self.synthetic_day(d) if synthetic else self.day(d)
            # Add prices for CUSIPs with column data.
            for c, v in zip(day_df["CUSIP"].values, day_df[col].values):
                hist_d[c].append(v)
            # Add NaNs for CUSIPs that are missing.
            missing_cusips = cusips.symmetric_difference(set(day_df["CUSIP"]))
            for c in missing_cusips:
                hist_d[c].append(np.NaN)

        return pd.DataFrame(hist_d, index=dates)

    @lockutils.synchronized(
        "synthetic_difference_history",
        external=True,
        lock_path=root("data/US/synthetic_difference/file_maps"),
    )
    def _synthetic_difference_saved_history(self, col):
        """
        Get saved difference history if it exists.

        Parameters
        ----------
        col: str
            Column to perform synthetic difference history on.

        Returns
        -------
        fid: str
            Filename where calculated history is to be saved.
        history: dict
            Datetime key and respective synthetic
            difference values.
        """
        key = "|".join([f"{k}: {v}" for k, v in self.constraints.items()])
        json_dir = root("data/US/synthetic_difference/file_maps")
        history_dir = root("data/US/synthetic_difference/history")
        mkdir(json_dir)
        mkdir(history_dir)
        json_fid = f"US/synthetic_difference/file_maps/{col}"
        fid_map = load_json(json_fid, empty_on_error=True)
        try:
            # Find fid if key exists in file.
            filename = fid_map[key]
        except KeyError:
            # Get a unique fid, update the fid mappping file,
            # and return an empty history dict.
            filename = get_unique_fid(fid_map)
            fid_map[key] = filename
            dump_json(fid_map, json_fid)
            fid = history_dir / filename
            history = {}
            return fid, history
        else:
            fid = history_dir / filename
            try:
                history = pd.read_parquet(fid).squeeze().to_dict()
            except (FileNotFoundError, OSError, ArrowInvalid):
                history = {}
            return fid, history

    def get_synthetic_differenced_history(
        self,
        col,
        col2="PrevMarketValue",
        dropna=False,
        force_calculations=False,
    ):
        """
        Save the difference history of a column.

        col
        ----------
        Parametesr: str
            Column from :attr:`Index.df` to build synthetic
            differenced history for (e.g., 'OAS').
        dropna: bool, default=False
            If True drop columns with missing values for either
            `MarketValue` or specified column.

        Returns
        -------
        pd.Series
            Series of synthetically differenced market value
            weighting for specified column.
        """
        self._get_prev_market_value_history()

        fid, saved_history = self._synthetic_difference_saved_history(col)
        if force_calculations:
            saved_history = {}
        dates = self.dates[1:]
        a = np.zeros(len(dates))
        cols = ["MarketValue", col]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i, date in enumerate(dates):
                # Check if date exists in saved history.
                try:
                    a[i] = saved_history[date]
                except KeyError:
                    pass
                else:
                    if date == dates[-1]:
                        pass
                    else:
                        continue

                # Date does not exist in saved history.
                # Calculate the differenced history for
                # bonds which existed for both current and
                # previous days.
                if dropna:
                    df = self.synthetic_day(date).dropna(subset=cols)
                    prev_df = self.day(self.dates[i]).dropna(subset=cols)
                else:
                    df = self.synthetic_day(date)
                    prev_df = self.day(self.dates[i])
                prev_df = prev_df[prev_df.index.isin(df.index)]
                a[i] = (np.sum(df[col2] * df[col]) / np.sum(df[col2])) - (
                    np.sum(prev_df["MarketValue"] * prev_df[col])
                    / np.sum(prev_df["MarketValue"])
                )
                saved_history[date] = a[i]

        if not force_calculations:
            # Save synthetic difference history to file.
            new_history = pd.Series(saved_history, name="-").to_frame()
            new_history.to_parquet(fid)

        # Add the the cumulative synthetic differences to the current
        # value backwards in time to yield the synthetic history
        # of the specified column.
        current_val = np.sum(df["MarketValue"] * df[col]) / np.sum(
            df["MarketValue"]
        )
        offset = np.cumsum(a[::-1])
        synthetic_history = np.concatenate(
            [(current_val - offset)[::-1], [current_val]]
        )
        return pd.Series(synthetic_history, index=self.dates, name=col)

    def subset_current_universe(self):
        """ "
        Return an index consisting only of bonds which exist
        on the most current day in the index, (i.e., drop all
        bonds that have matured or fallen out of the index).

        Returns
        -------
        :class:`Index`:
            Index of most current constituents.
        """
        current_cusips = self.day(self.dates[-1], as_index=True).cusips
        return self.subset(cusip=current_cusips)

    def subset_current_universe_tickers(self):
        """ "
        Return an index consisting only of tickers which exist
        on the most current day in the index, (i.e., drop all
        tickers that have fully matured or fallen out of the index).

        Returns
        -------
        :class:`Index`:
            Index of most current constituents.
        """
        current_tickers = self.day(self.dates[-1], as_index=True).tickers
        return self.subset(ticker=current_tickers)

    def drop_ratings_migrations(self, allowable_ratings=None):
        """
        Drop any bonds who had rating migrations
        pushing them out of the current index.

        Parameters
        ----------
        allowable_ratings: tuple[str], optional
            Allowable rating range to keep bonds for.

        Returns
        -------
        :class:`Index`
            Index of bonds that spend entire series in
            current index rating boundaries.
        """
        # Load rating changes for bonds in index.
        cons = self.constraints

        if "isin" in cons:
            raise NotImplementedError(
                "Cannot drop ratings migrations in an Index based on Isins"
            )

        ig_indexes = [
            "in_stats_index",
            "in_returns_ix_index",
            "in_agg_returns_index",
            "in_agg_stats_index",
        ]
        index_constraints = {ix: cons.get(ix, False) for ix in ig_indexes}
        in_ig_index = sum(index_constraints.values())

        # Determine minimum and maximum allowable ratings.
        if allowable_ratings is not None:
            min_rating, max_rating = self.convert_input_ratings(
                allowable_ratings
            )
        elif "rating" in cons:
            # If ratings are specified, use those ratings.
            min_rating, max_rating = cons["rating"]
        elif in_ig_index:
            # If its an index member, use (AAA, BBB-).
            min_rating, max_rating = 0, 10
        else:
            # Allow all ratings.
            min_rating, max_rating = 0, 23

        # Return an index with all bonds outside allowable
        # range dropped. Use ISIN as ISINs dont change
        # over time.
        rating_change_df = self._ratings_changes_df
        downgraded_out_of_range = (
            rating_change_df["NumericRating_PREV"] <= max_rating
        ) & (rating_change_df["NumericRating_NEW"] > max_rating)
        upgraded_out_of_range = (
            rating_change_df["NumericRating_PREV"] >= min_rating
        ) & (rating_change_df["NumericRating_NEW"] < min_rating)

        df_to_drop = rating_change_df[
            upgraded_out_of_range | downgraded_out_of_range
        ]
        if len(df_to_drop):
            # There are bonds to drop. Ensure you keep any bonds that are
            # currently in the index, and drop the rest from the index.
            current_isins = self.day(self.dates[-1], as_index=True).isins
            isins_to_drop = set(df_to_drop["ISIN"]) - set(current_isins)
            if isins_to_drop:
                return self.subset(isin=isins_to_drop, special_rules="~ISIN")
            else:
                # The bonds to be dropped are still in the current index.
                return self
        else:
            # There are no bonds to drop.
            return self

    def drop_tail(self, col, pct=10):
        """
        Drop the widest tail portion of an index by given
        percent of cumulative market value.

        Parameters
        ----------
        col: str
            Column to find tail for.
        pct: int or float, default=10
            Size of tail by percent of cumulative market value.

        Returns
        -------
        :class:`Index`:
            Index the current :class:`Index` with the
            specified tail dropped.
        """
        df = self.df.sort_values(["Date", col])
        drop_threshold = 1 - (pct / 100)

        def daily_drop_tail(df):
            mv_percentile = np.cumsum(df["MarketValue"]) / np.sum(
                df["MarketValue"]
            )
            return df[mv_percentile < drop_threshold]

        return Index(df.groupby("Date").apply(daily_drop_tail))

    def get_tail(self, col, pct=10):
        """
        Get just the widest tail portion of an index by given
        percent of cumulative market value.

        Parameters
        ----------
        col: str
            Column to find tail for.
        pct: int or float, default=10
            Size of tail by percent of cumulative market value.

        Returns
        -------
        :class:`Index`:
            Index of the specified tail of the current :class:`Index`.
        """
        df = self.df.sort_values(["Date", col])
        drop_threshold = 1 - (pct / 100)

        def daily_keep_tail(df):
            mv_percentile = np.cumsum(df["MarketValue"]) / np.sum(
                df["MarketValue"]
            )
            return df[mv_percentile >= drop_threshold]

        return Index(df.groupby("Date").apply(daily_keep_tail))

    def get_cusip_history(self, cusip):
        """
        Get full history for specified cusip.

        Parameters
        ----------
        cusip: str
            Specified cusip.
        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and :attr:`Index.df`
            columns for specified cusip.
        """
        return pd.concat(
            [
                pd.DataFrame(self.day(d).loc[cusip, :]).T.set_index("Date")
                for d in self.dates
            ]
        )

    def market_value_weight_vol(self, col, window_size=20, annualized=False):
        """
        Market value weight the volatilities of the specified
        column using a rolling window approach. The variance
        of each bond in the index is computed daily and the
        average variance is then transformed back to volatility.

        Parameters
        ----------
        col: str
            Column name to weight, (e.g., ``'OAS'``).
        window_size: int, default=20
            Number of sample periods to use in rolling window.
        annualized: bool, default=True
            If True annualized the resulting volatilities by
            multiply by square root of 252.

        Returns
        -------
        vol: pd.Series
            Computed volatility with datetime index.
        """
        col_df = self.get_value_history(col)
        var_df = col_df.rolling(window_size, min_periods=window_size).var()
        mv_df = self.get_value_history("MarketValue").mask(var_df.isna())
        weights_df = mv_df.divide(np.sum(mv_df, axis=1).values, axis=0) ** 2
        vol = (np.sum(var_df * weights_df, axis=1) ** 0.5)[window_size - 1 :]
        if annualized:
            vol *= 252**0.5
        return vol

    def find_rating_changes(self, rating_agency):
        """
        Find rating changes of persistent index.

        Parameters
        ----------
        rating_agency: {'SP', 'Moody', 'Fitch'}.
            Rating agency to find changes for.

        Returns
        -------
        change_df: pd.DataFrame
            DataFrame of all rating changes, with no index,
            columns are: ``['cusip', 'date', 'change', 'new_val',
            'old_val', 'sector', 'subsector']``.
        """
        # Create numeric rating column for single rating agency.
        col = f"{rating_agency}Rating"
        new_col = col.replace("R", "NumericR")
        rating_vector = self.df[col].fillna("NR").values
        ratings_json = load_json("ratings")
        num_ratings = np.vectorize(ratings_json.__getitem__)(
            rating_vector
        ).astype(float)
        num_ratings[num_ratings == 0] = np.nan
        self.df[new_col] = num_ratings

        # Get rating history for all cusips.
        rating_df = self.get_value_history(new_col)
        change_cusips = []
        for cusip in rating_df.columns:
            if not check_all_equal(list(rating_df[cusip].dropna())):
                change_cusips.append(cusip)

        # Store direction and magnitude of rating changes and
        # the date of rating changes in separate dicts.
        change_dict = defaultdict(list)
        for cusip in change_cusips:
            ratings = rating_df[cusip].dropna().values
            diff = np.diff(ratings)
            nonzero_locs = np.nonzero(diff)[0]
            for loc in nonzero_locs:
                # Ensure at least 20 days with no change on either
                # side of rating change to include change.
                start = int(max(0, loc - 20))
                end = int(min(len(diff) - 1, loc + 20))
                local_diff = diff[start:end]
                changes = local_diff[local_diff != 0]
                if len(changes) != 1:
                    continue  # not true rating change
                change_val = diff[loc]
                if change_val == 0:
                    continue  # not true rating change

                rating_series = rating_df[cusip].dropna()
                date = rating_series.index[loc + 1]
                new_val = ratin

    @property
    @lru_cache(maxsize=None)
    def _treasury(self):
        """:class:`TreasuryCurve`:  Treasury curves."""
        return TreasuryCurve()

    def _get_prev_market_value_history(self, force=False):
        """
        Add previous market value as column as
        `PrevMarketValue` in :attr:`Index.df`.
        """
        # Stop if column is column is already present.
        if "PrevMarketValue" in list(self.df) and not force:
            return

        # Build dict of cusip/date pairs to prev market value.
        prev_mv_map = {}
        dates = self.dates
        for i, curr_date in enumerate(dates[1:]):
            prev_date = dates[i]
            prev_mv_i = pd.DataFrame(
                self.df.loc[self.df["Date"] == prev_date, "MarketValue"]
            ).squeeze()
            for cusip, val in prev_mv_i.items():
                prev_mv_map[(cusip, curr_date)] = val

        # Build column of previous market values and join to `self.df`.
        prev_mv = np.full(len(self.df), np.NaN)
        for i, key in enumerate(self.df["Date"].items()):
            prev_mv[i] = prev_mv_map.get(key, np.NaN)
        self.df["PrevMarketValue"] = prev_mv.astype("float32")

    def compute_total_returns(self):
        """
        Vectorized implementation for computing total returns
        for all cusips in index, adjusting for coupon delivery
        and reinvestment.

        Appends results as new column `TRet` in :attr:`Index.df`.
        """
        # Stop computation if already performed.
        if "TRet" in list(self.df):
            return

        # Compute total returns without adjusting for coupons.
        np.seterr(divide="ignore", invalid="ignore")
        price_df = self.get_value_history("DirtyPrice")
        cols = list(price_df)
        price = price_df.values
        tret = price[1:] / price[:-1] - 1

        # Find dates when coupons were paid by finding dates
        # with a decrease in accrued interest.
        accrued = self.get_value_history("AccruedInterest")[cols].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accrued_mask = np.zeros(accrued.shape)
            accrued_mask[1:] = np.where(np.diff(accrued, axis=0) < 0, 1, 0)

        # Find coupon rate, add coupon back into prices and assume
        # coupon is reinvested at same rate.
        # In Bloomberg this is done as two separate calculations.
        # For individual bonds, the coupon is not assumed to be
        # reinvested. However, when aggregating bonds for an
        # index, the `quantity` of bonds reflects the return
        # from the coupon and is included in excess returns.
        coupon_rate = self.get_value_history("CouponRate")[cols].values / 2
        coupon_adj_price = price + accrued_mask * coupon_rate
        reinvesting_multiplier = accrued_mask * (1 + coupon_rate / price)
        reinvesting_multiplier[reinvesting_multiplier == 0] = 1
        reinvesting_multiplier[np.isnan(reinvesting_multiplier)] = 1
        reinvesting_multiplier = np.cumprod(reinvesting_multiplier, axis=0)
        coupon_adj_tret = (
            coupon_adj_price[1:] / coupon_adj_price[:-1] - 1
        ) * reinvesting_multiplier[1:]

        # Combine both methods taking element-wise maximum to
        # account for day when coupon is paid.
        tret_df = pd.DataFrame(
            np.maximum(tret, coupon_adj_tret),
            index=price_df.index[1:],
            columns=cols,
        )

        # Append total returns to :attr:`Index.df`.
        self.df["TRet"] = np.NaN
        for date in tret_df.index:
            cusips = list(self.day(date).index)
            self.df.loc[self.df["Date"] == date, "TRet"] = tret_df.loc[
                date, cusips
            ].values

    def compute_excess_returns(self):
        """
        Compute excess returns for all cusips in index.

        Appends results as new column `XSRet` in :attr:`Index.df`.
        """
        # Stop computation if already performed.
        if "XSRet" in list(self.df):
            return

        # Compute total returns.
        self.compute_total_returns()

        # Fill missing 0.5 year KRD values with 0.
        krd_cols = [col for col in list(self.df) if "KRD" in col]
        self.df.loc[
            (self.df["KRD06mo"].isna())
            & (self.df[krd_cols].isna().sum(axis=1) == 1),
            "KRD06mo",
        ] = 0

        # Compute and append excess returns iteratively for each day.
        self.df["XSRet"] = np.NaN
        for date in self.dates[1:]:
            df = self.day(date)
            # Calculate hypothetical treasury weights for each cusip.
            weights = np.zeros([len(df), 7])
            weights[:, :-1] = df[krd_cols].values / self._treasury.KRDs(date)

            # Add cash component to make weights sum to 1.
            weights[:, -1] = 1 - np.sum(weights, axis=1)
            tsy_trets = np.sum(weights * self._treasury.trets(date), axis=1)
            ex_rets = df["TRet"].values - tsy_trets
            self.df.loc[self.df["Date"] == date, "XSRet"] = ex_rets

        # 0 out the excess return for on the run treasuries.
        tsy_loc = self.df["Sector"] == "TREASURIES"
        if tsy_loc.sum():
            cols = ["Date", "OriginalMaturity", "IssueDate"]
            for date, date_df in self.df.loc[tsy_loc, cols].groupby("Date"):
                on_the_run_treasuries = set(
                    date_df.groupby("OriginalMaturity")
                    .idxmax()["IssueDate"]
                    .values
                )
                self.df.loc[
                    (self.df["Date"] == date)
                    & (self.df.index.isin(on_the_run_treasuries)),
                    "XSRet",
                ] = 0

    def accumulate_individual_excess_returns(
        self, start_date=None, end_date=None
    ):
        """
        Accumulate excess returns for each cusip since start date.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.
        end_date: datetime
            Date to end aggregating returns.

        Returns
        -------
        pd.Series
            The accumlated excess return of each bond indexed
            by it cusip.
        """
        self._get_prev_market_value_history()
        self.compute_excess_returns()

        # Find excess and total returns in date range.
        xs_ret_df = self.get_value_history("XSRet")
        if start_date is not None:
            xs_ret_df = xs_ret_df[xs_ret_df.index > to_datetime(start_date)]
        if end_date is not None:
            xs_ret_df = xs_ret_df[xs_ret_df.index <= to_datetime(end_date)]

        # Calculate implied risk free returns.
        total_ret, t_ret_df = self.accumulate_individual_total_returns(
            start_date, end_date, return_total_ret_timeseries=True
        )
        rf_ret_df = t_ret_df - xs_ret_df
        # Calculate total returns over period and use treasury
        # returns to back out excess returns.
        rf_total_ret = np.prod(1 + rf_ret_df) - 1
        return (total_ret - rf_total_ret).rename("XSRet")

    def accumulate_individual_total_returns(
        self, start_date=None, end_date=None, return_total_ret_timeseries=False
    ):
        """
        Accumulate excess returns for each cusip since start date.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.
        end_date: datetime
            Date to end aggregating returns.
        return_total_ret_timeseries: bool, default=False.
            If ``True`` return both accumulated total returns as
            well as total return timeseries for each bond.

        Returns
        -------
        total_ret: pd.Series
            The accumlated total return of each bond indexed
            by it cusip.
        t_ret_df: pd.DataFrame
            Total return timeseries of each CUSIP, only returned if
            ``return_total_ret_timeseries`` is ``True``.
        """
        self._get_prev_market_value_history()
        self.compute_total_returns()

        # Find excess and total returns in date range.
        t_ret_df = self.get_value_history("TRet")
        if start_date is not None:
            t_ret_df = t_ret_df[t_ret_df.index > to_datetime(start_date)]
        if end_date is not None:
            t_ret_df = t_ret_df[t_ret_df.index <= to_datetime(end_date)]

        total_ret = (np.prod(1 + t_ret_df) - 1).rename("TRet")
        if return_total_ret_timeseries:
            return total_ret, t_ret_df
        else:
            return total_ret

    def aggregate_excess_returns(self, start_date=None, end_date=None):
        """
        Aggregate excess returns since start date for
        entire :class:`Index`.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.
        end_date: datetime
            Date to end aggregating returns.

        Returns
        -------
        float:
            Aggregated excess returns for entire :class:`Index`
            between specified dates.
        """
        self._get_prev_market_value_history()
        self.compute_excess_returns()

        # Find excess and total returns in date range.
        xs_rets = self.market_value_weight("XSRet", weight="PrevMarketValue")
        if start_date is not None:
            xs_rets = xs_rets[xs_rets.index > to_datetime(start_date)]
        if end_date is not None:
            xs_rets = xs_rets[xs_rets.index <= to_datetime(end_date)]

        # Calculate implied risk free returns.
        total_ret, t_rets = self.aggregate_total_returns(
            start_date, end_date, return_total_ret_timeseries=True
        )
        rf_rets = t_rets - xs_rets
        # Calculate total returns over period and use treasury
        # returns to back out excess returns.
        rf_total_ret = np.prod(1 + rf_rets) - 1
        return total_ret - rf_total_ret

    def aggregate_total_returns(
        self, start_date=None, end_date=None, return_total_ret_timeseries=False
    ):
        """
        Aggregate total returns since start date for
        entire :class:`Index`.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.
        end_date: datetime
            Date to end aggregating returns.
        return_total_ret_timeseries: bool, default=False
            If ``True`` return both aggregated total returns as
            well as total return timeseries.

        Returns
        -------
        total_ret: float
            Aggregated total returns for entire :class:`Index`
            between specified dates.
        t_rets: pd.Series
            Total return timeseries, only returned if
            ``return_total_ret_timeseries`` is ``True``.
        """
        self._get_prev_market_value_history()
        self.compute_total_returns()

        # Find excess and total returns in date range.
        t_rets = self.market_value_weight("TRet", weight="PrevMarketValue")
        if start_date is not None:
            t_rets = t_rets[t_rets.index > to_datetime(start_date)]
        if end_date is not None:
            t_rets = t_rets[t_rets.index <= to_datetime(end_date)]

        total_ret = np.prod(1 + t_rets) - 1
        if return_total_ret_timeseries:
            return total_ret, t_rets
        else:
            return total_ret

    def cumulative_excess_returns(self, start_date=None, end_date=None):
        """
        Get cumulative aggregated excess returns for the
        entire :class:`Index`.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.
        end_date: datetime
            Date to end aggregating returns.

        Returns
        -------
        pd.Series:
            Cumulative excess returns for full with datetime index
            for entire :class:`Index`.
        """
        self._get_prev_market_value_history()
        self.compute_excess_returns()

        # Find excess and total returns in date range.
        xs_rets = self.market_value_weight("XSRet", weight="PrevMarketValue")
        t_rets = self.market_value_weight("TRet", weight="PrevMarketValue")
        if start_date is not None:
            xs_rets = xs_rets[xs_rets.index > to_datetime(start_date)]
            t_rets = t_rets[t_rets.index > to_datetime(start_date)]
        if end_date is not None:
            xs_rets = xs_rets[xs_rets.index <= to_datetime(end_date)]
            t_rets = t_rets[t_rets.index <= to_datetime(end_date)]

        dates = xs_rets.index
        xsrets_a = np.zeros(len(dates))
        for i, date in enumerate(dates):
            xs_rets_i = xs_rets[xs_rets.index <= date]
            t_rets_i = t_rets[t_rets.index <= date]
            # Calculate implied risk free returns.
            rf_rets_i = t_rets_i - xs_rets_i
            # Calculate total returns over period and use treasury
            # returns to back out excess returns.
            total_ret_i = np.prod(1 + t_rets_i) - 1
            rf_total_ret_i = np.prod(1 + rf_rets_i) - 1
            xsrets_a[i] = total_ret_i - rf_total_ret_i
        return pd.Series(xsrets_a, index=dates)

    def _fill_missing_columns_with_bbg_data(self):
        """
        Fill columns with missing data (pre-Oct 2018)
        will data scraped form bloomberg.
        """
        cols = [
            # 'PaymentRank',
            "CollateralType",
            "Issuer",
            "CountryOfRisk",
            "CountryOfDomicile",
            "Currency",
            "MarketOfIssue",
        ]
        self.df["CollateralType"].replace("No Collateral", np.NaN, inplace=True)
        for col in cols:
            bbg_d = load_json(f"bbg_jsons/{col}")
            new_cats = set(bbg_d.values()) - set(self.df[col].cat.categories)
            # Add new values as categories then fill missing
            # data using bloomberg data.
            self.df[col] = self.df[col].cat.add_categories(new_cats)
            self.df[col] = self.df[col].fillna(self.df["CUSIP"].map(bbg_d))


# %%
def main():
    pass
    # %%
    from lgimapy.data import Database
    from lgimapy.utils import Time, load_json, dump_json, root, pprint
    import lgimapy.vis as vis
    from datetime import timedelta
    import matplotlib.pyplot as plt

    vis.style()
    db = Database()
    db.display_all_columns()
    # %%

    db.load_market_data(start="1/1/2020")
    self = db.build_market_index(in_stats_index=True, maturity=(5, 7))

    # %%
