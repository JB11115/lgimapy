import json
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from lgimapy.utils import check_all_equal, mkdir, root

# %%
class Index:
    """
    Class for indexes built by :class:`IndexBuilder`.

    Parameters
    ----------
    index_df: pd.DataFrame
        Index DataFrame from :meth:`IndexBuilder.build`.
    name: str, default=''
        Optional name of index.

    Attributes
    ----------
    df: pd.DataFrame
        Full index DataFrame.
    cuisps: List[str].
        List of all unique cusips in index.
    dates: List[datetime].
        Sorted list of all dates in index.
    """

    def __init__(self, index_df, name=""):
        self.df = index_df.set_index("CUSIP", drop=False)
        self.name = name
        self._column_cache = []
        self._day_cache = {}
        self._day_key_cache = defaultdict(str)

    def __repr__(self):
        start = self.dates[0].strftime("%m/%d/%Y")
        end = self.dates[-1].strftime("%m/%d/%Y")
        if start == end:
            return f"{self.name} Index {start}"
        else:
            return f"{self.name} Index {start} - {end}"

    @property
    @lru_cache(maxsize=None)
    def dates(self):
        """Memoized unique sorted dates in Index."""
        return sorted(list(set(self.df["Date"])))

    @property
    @lru_cache(maxsize=None)
    def cusips(self):
        """Memoized unique cusips in Index."""
        return list(set(self.df.index))

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
        date = pd.to_datetime(date)
        # Create cache key of all columns added to :attr:`Index.df`.
        cache_key = "_".join(self._column_cache)
        if cache_key == self._day_key_cache[date]:
            # No new column changes since last accessed,
            # treat like normal cache.
            try:
                df = self._day_cache[date]
            except KeyError:
                df = self.df[self.df["Date"] == date]
                self._day_cache[date] = df
        else:
            # New columns added since last accessed,
            # update cache and cache key.
            df = self.df[self.df["Date"] == date]
            self._day_cache[date] = df
            self._day_key_cache[date] = cache_key

        if as_index:
            return Index(df, self.name)
        else:
            return df

    def day_persistent_constituents(self, date, as_index=False):
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
        today_date = pd.to_datetime(date)
        prev_date = self.dates[list(self.dates).index(today_date) - 1]
        today_df = self.day(today_date)
        prev_df = self.day(prev_date)
        intersection_ix = set(today_df.index).intersection(prev_df.index)
        df = today_df[today_df.index.isin(intersection_ix)]
        if as_index:
            return Index(df, self.name)
        else:
            return df

    def clean_treasuries(self):
        """Clean treasury index for building model curve."""
        df = self.df[self.df["Ticker"] == "T"].copy()

        # Remove bonds that have current maturities less than the
        # original maturity of another tenor (e.g., remove 30 year bonds
        # once their matuirty is less than 10 years, 10 and 7,7 and 5, etc.).
        mat_map = {2: 0, 3: 2, 5: 3, 7: 5, 10: 7, 30: 10}
        mask = [
            my > mat_map[om]
            for my, om in zip(df["MaturityYears"], df["OriginalMaturity"])
        ]
        df = df[mask].copy()
        self.df = df.copy()

    def get_value_history(self, col):
        """
        Get history of any column for all cusips in Index.

        Parameters
        ----------
        col: str
            Column from :attr:`Index.df` to build history for (e.g., 'OAS').

        Returns
        -------
        hist_df: pd.DataFrame
            DataFrame with datetime index, CUSIP columns, and price values.
        """

        df = self.df.copy()
        temp_df = df[["CUSIP", "Date", col]].drop_duplicates()
        cusips = set(temp_df["CUSIP"])

        # Build dict of historical values for each CUSIP and
        # convert to DataFrame.
        hist_d = defaultdict(list)
        for d in self.dates:
            day_df = temp_df[temp_df["Date"] == d].copy()
            day_df.drop_duplicates("CUSIP", inplace=True)
            # Add prices for CUSIPs with column data.
            for c, v in zip(day_df["CUSIP"].values, day_df[col].values):
                hist_d[c].append(v)
            # Add NaNs for CUSIPs that are missing.
            missing_cusips = cusips.symmetric_difference(set(day_df["CUSIP"]))
            for c in missing_cusips:
                hist_d[c].append(np.NaN)

        hist_df = pd.DataFrame(hist_d, index=self.dates)
        return hist_df

    def get_cusip_history(self, cusip):
        """
        Get full history for specified cusip.

        Parameters
        ----------
        cusip: str
            Specified cusip.

        Returns
        -------
        hist_df: pd.DataFrame
            DataFrame with datetime index and :attr:`Index.df`
            columns for specified cusip.
        """
        return pd.concat(
            [
                pd.DataFrame(self.day(d).loc[cusip, :]).T.set_index("Date")
                for d in self.dates
            ]
        )

    def market_value_weight(self, col):
        """
        Market value weight a specified column vs entire
        index market value.

        Parameters
        ----------
        col: str
            Column name to weight, (e.g., 'OAS').

        Returns
        -------
        pd.Series:
            Series of market value weighting for specified column.
        """
        a = np.zeros(len(self.dates))
        for i, date in enumerate(self.dates):
            df = self.day(date)
            a[i] = np.sum(df["AmountOutstanding"] * df["DirtyPrice"] * df[col])
            a[i] /= np.sum(df["AmountOutstanding"] * df["DirtyPrice"])
        return pd.Series(a, index=self.dates, name=col)

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
        with open(root("data/ratings.json"), "r") as fid:
            ratings_json = json.load(fid)
        num_ratings = np.vectorize(ratings_json.__getitem__)(
            rating_vector
        ).astype(float)
        num_ratings[num_ratings == 0] = np.nan
        self.df[new_col] = num_ratings

        # Get rating history for all cusips.
        rating_df = self.get_value_history(new_col)
        print(rating_df.iloc[:10, :10])
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
                new_val = rating_series.values[loc + 1]
                old_val = rating_series.values[loc]
                last_date = rating_df[cusip].dropna().index[loc]
                change_dict["cusip"].append(cusip)
                change_dict["date"].append(date)
                change_dict["change"].append(change_val)
                change_dict["new_val"].append(new_val)
                change_dict["old_val"].append(old_val)
                change_dict["sector"].append(self.df.loc[date, "Sector"])
                change_dict["subsector"].append(self.df.loc[date, "Subsector"])

        change_df = pd.DataFrame(change_dict)
        return change_df

    def subset_value_by_rating(self, col, save=False, fid=None, path=None):
        """
        Find the market vaue weighted value of specified
        column by rating.

        Parameters
        ----------
        col: str
            Column from Index.df to market value weight.
        save: bool
            If true, store resulting DataFrame to fid.
        fid: str, default=None
            File name to store file, by default is input `col`.
        path: Path, default=None
            Path to store file, by default is `./data/subset_by_rating`.

        Returns
        -------
        subset_df: pd.DataFrame
            if save is false, return DataFrame of results
            with datetime index and numeric rating columns.
        """
        path = root("data/subset_by_rating") if path is None else path
        fid = col if fid is None else fid

        ratings = sorted(list(set(self.df["NumericRating"])))
        subset_dict = defaultdict(list)
        for date in self.dates:
            day_df = self.day(date)
            for rating in ratings:
                df = day_df[day_df["NumericRating"] == rating].copy()
                mvw = self.market_value_weight(col, df=df)
                subset_dict[rating].append(mvw)

        cols = [int(rating) for rating in ratings]
        subset_df = pd.DataFrame(subset_dict, index=self.dates, columns=cols)
        if save:
            self._save_subset_by_rating(subset_df, fid, path)
        else:
            return subset_df

    def _save_subset_by_rating(self, subset_df, fid, path):
        """
        Save subset_df to `../data/subset_by_rating/{col}.csv`
        creating the file if it doesn't exist, and overwriting
        data which already exists.

        Parameters
        ----------
        subset_df: pd.DataFrame
            subset_df from `Index.subset_value_by_rating` with
            datetime index and numeric rating columns.
        fid: str
            File name to store file.
        path: Path
            Path to store file.
        """
        fid = path.joinpath(f"{fid}.csv")
        mkdir(path)
        try:
            df = pd.read_csv(
                fid, index_col=0, parse_dates=True, infer_datetime_format=True
            )
        except FileNotFoundError:
            # File does not exist, create a new one.;
            subset_df.to_csv(fid)
        else:
            # File exists, append to it and sort.
            old_df = df[~df.index.isin(subset_df.index)]
            old_df.columns = [int(col) for col in old_df.columns]
            new_df = old_df.append(subset_df)
            new_df.sort_index(inplace=True)
            new_df.to_csv(fid)

    @property
    @lru_cache(maxsize=None)
    def _tsy_df(self):
        """pd.DataFrame: historical treasury curve DataFrame."""
        tsy_fid = root("data/treasury_curve_params.csv")
        tsy_df = pd.read_csv(
            tsy_fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )
        tsy_df = tsy_df[
            (tsy_df.index >= self.dates[0]) & (tsy_df.index <= self.dates[-1])
        ].copy()
        return tsy_df

    def cusip_excess_returns(self, cusips):
        """
        Calculate excess returns for specified cusips.

        Parameters
        ----------
        cusips: str, List[str], default=None
            Specified cusip(s) to compute excess returns for.

        Returns
        -------
        pd.DataFrame:
            DatFrame with datetime index, cusips columns,
            and excess return values.
        """
        if isinstance(cusips, str):
            cusips = [cusips]  # ensure cusips is list

        # Load treasury DataFrame and get KRDs and total returns.
        tsy_df = self._tsy_df
        tsy_krd_cols = [col for col in list(tsy_df) if "KRD" in col]
        tsy_tret_cols = [col for col in list(tsy_df) if "tret" in col]
        self._tsy_krds = tsy_df[tsy_krd_cols].values
        self._tsy_krd_trets = tsy_df[tsy_tret_cols].values

        # Store list of KRD columns for :attr:`Index.df`.
        self._krd_cols = [col for col in list(self.df) if "KRD" in col]

        ex_ret_dates = self.dates[1:]
        ex_rets = np.zeros([len(ex_ret_dates), len(cusips)])
        for i, cusip in enumerate(cusips):
            ex_rets[:, i] = self._single_cusip_excess_returns(cusip)

        return pd.DataFrame(ex_rets, index=ex_ret_dates, columns=cusips)

    def _single_cusip_excess_returns(self, cusip):
        """
        Calculate excess returns for a single cusip.

        Parameters
        ----------
        cusip: str
            Specified cusip to compute excess returns for.

        Returns
        -------
        """
        # Get KRDs and total returns from cusip.
        df = self.df[self.df.index == cusip]
        cusip_krds = df[self._krd_cols].values
        df[[*self._krd_cols, "Date"]]
        cusip_trets = (df["DirtyPrice"][1:] / df["DirtyPrice"][:-1]).values - 1
        if len(cusip_trets) < len(self.dates) - 1:
            # NaNs are present, find correct starting ix for trets
            # and KRDs and pad arrays with NaNs.
            start_ix = self.dates.index(df["Date"][0])
            temp_trets = np.nan * np.empty(len(self.dates) - 1)
            temp_trets[start_ix : start_ix + len(df) - 1] = cusip_trets
            cusip_trets = temp_trets
            temp_krds = np.nan * np.empty([len(self.dates), 6])
            temp_krds[start_ix : start_ix + len(df), :] = cusip_krds
            cusip_krds = temp_krds

        # Match hypothetical portfolios by KRDs, then add
        # cash component to make weight sum to 1.
        weights = np.zeros(self._tsy_krd_trets.shape)
        weights[:, :-1] = cusip_krds / self._tsy_krds
        weights[:, -1] = 1 - np.sum(weights, axis=1)
        tsy_trets = np.sum(weights * self._tsy_krd_trets, axis=1)[1:]
        return cusip_trets - tsy_trets

    def compute_total_returns(self):
        """
        Compute total returns for all cusips in index,
        appending result to :attr:`Index.df`.
        """
        # Stop computating if already performed.
        if "total_returns" in self._column_cache:
            return

        # Compute total returns without accounting for coupons.
        price_df = self.get_value_history("DirtyPrice")
        tret = price_df.values[1:] / price_df.values[:-1] - 1
        # Compute total returns accounting for coupons.
        cols = list(price_df)
        accrued = self.get_value_history("AccruedInterest")[cols].values
        accrued_mask = np.where(accrued == 0, 1, 0)
        coupon_rate = self.get_value_history("CouponRate")[cols].values / 2
        coupon_price = price_df.values + accrued_mask * coupon_rate
        coupon_tret = coupon_price[1:] / coupon_price[:-1] - 1
        # Combine both methods taking element-wise maximum to
        # account for day when coupon is paid.
        tret_df = pd.DataFrame(
            np.maximum(tret, coupon_tret),
            index=price_df.index[1:],
            columns=cols,
        )

        # Append total returns to :attr:`Index.df`.
        self.df["total_return"] = np.NaN
        for date in tret_df.index:
            cusips = list(self.day(date).index)
            self.df.loc[self.df["Date"] == date, "total_return"] = tret_df.loc[
                date, cusips
            ].values
        self._column_cache.append("total_returns")  # update cache

    def compute_excess_returns(self):
        """
        Compute excess returns for all cusips in index,
        appending result to :attr:`Index.df`.
        """
        # Stop computating if already performed.
        if "excess_returns" in self._column_cache:
            return

        # Compute total returns.
        self.compute_total_returns()

        # Load treasury DataFrame and save required column names.
        tsy_df = self._tsy_df
        tsy_krd_cols = [col for col in list(tsy_df) if "KRD" in col]
        tsy_tret_cols = [col for col in list(tsy_df) if "tret" in col]
        krd_cols = [col for col in list(self.df) if "KRD" in col]

        # Compute and append excess returns iteratively for each day.
        self.df["excess_return"] = np.NaN
        for date in self.dates[1:]:
            df = self.day(date)
            # Calculate hypothetical treasury weights for each cusip.
            weights = np.zeros([len(df), 7])
            weights[:, :-1] = (
                df[krd_cols].values
                / tsy_df.loc[date, tsy_krd_cols].values[None, :]
            )
            # Add cash component to make weights sum to 1.
            weights[:, -1] = 1 - np.sum(weights, axis=1)
            tsy_trets = np.sum(
                weights * tsy_df.loc[date, tsy_tret_cols].values, axis=1
            )
            ex_rets = df["total_return"].values - tsy_trets
            self.df.loc[self.df["Date"] == date, "excess_return"] = ex_rets
        self._column_cache.append("excess_returns")  # update cache

    def aggregate_excess_returns(self, start_date, index=True):
        """
        Aggregate excess returns since start date.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.
        index: bool, default=True
            If True, return aggregated excess returns
            for full, index, else return excess returns
            for individual cusips.

        Returns
        -------
        float or pd.Seires
            Aggregated excess returns for either full index
            or individual cusips.
        """
        self.compute_excess_returns()
        if index:
            # Find excess and total returns in date range.
            ex_rets = self.market_value_weight("excess_return")
            t_rets = self.market_value_weight("total_return")
            ex_rets = ex_rets[ex_rets.index > pd.to_datetime(start_date)]
            t_rets = t_rets[t_rets.index > pd.to_datetime(start_date)]
            # Calculate implied treasy returns.
            tsy_t_rets = t_rets - ex_rets
            # Calculate total returns over period.
            total_ret = np.prod(1 + t_rets) - 1
            tsy_total_ret = np.prod(1 + tsy_t_rets) - 1
            return total_ret - tsy_total_ret
        else:
            ## TODO: implement cusip level aggregate excess returns
            pass
