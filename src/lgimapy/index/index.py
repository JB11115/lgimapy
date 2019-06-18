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
        self._day_cache = {}

    def __repr__(self):
        start = self.dates[0].strftime("%m/%d/%Y")
        end = self.dates[-1].strftime("%m/%d/%Y")
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
        with date as key.

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
            DataFrame or Index for specified date.
        """
        date = pd.to_datetime(date)
        try:
            df = self._day_cache[date]
        except KeyError:
            self._day_cache[date] = self.df[self.df["Date"] == date].copy()
            df = self._day_cache[date]

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
        yesterday_date = self.dates[list(self.dates).index(today_date) - 1]
        today_df = self.day(today_date)
        yesterday_df = self.day(yesterday_date)
        intersection_ix = set(today_df.index).intersection(yesterday_df.index)
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
    def _hypothetical_treasuries(self):
        """pd.DataFrame: historical treasury curve DataFrame."""
        fid = root("data/treasury_curve_params.csv")
        return pd.read_csv(
            fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )

    def excess_returns(cusips=None):
        """
        Calculate excess returns for cusips.

        Parameters
        ----------
        cusips: str, List[str], default=None
            Specified cusip(s) to compute excess returns for,
            by default all cusips are selected.

        Returns
        -------
        pd.DataFrame:
            DatFrame with datetime index, cusips columns,
            and excess return values.
        """
        # Load treasury DataFrame and get KRDs and total returns.
        t_fid = root("data/treasury_curve_params.csv")
        t_df = pd.read_csv(
            t_fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )
        t_df = t_df[
            (t_df.index >= self.dates[0]) & (t_df.index <= self.dates[-1])
        ]
        t_krd_cols = [col for col in list(t_df) if "KRD" in col]
        t_tret_cols = [col for col in list(t_df) if "tret" in col]
        self._t_krds = t_df[t_krd_cols].values
        self._t_trets = t_df[t_tret_cols].values

        # Store list of KRD columns for :attr:`Index.df`.
        self._krd_cols = [col for col in list(self.df) if "KRD" in col]

    # def _excess_return(self, cusip):
    #     """
    #     Calculate excess returns for a single cusip.
    #
    #     Parameters
    #     ----------
    #     cusip: str
    #         Specified cusip to compute excess returns for.
    #
    #     Returns
    #     -------
    #     """
    #
    #     df = self.df[self.df.index == cusip]
    #     cusip_krds = df[self._krd_cols].values
    #
    #     weights = cusip_krds / self._t_krds
    #     weights


# %%
def main():
    from lgimapy.index import IndexBuilder
    from lgimapy.utils import Time
    import matplotlib.pyplot as plt

    ixb = IndexBuilder()
    ixb.load(dev=True, start="4/1/2019", end="5/1/2019")
    self = ixb.build(
        rating="IG", amount_outstanding=(300, None), OAS=(-10, 3000)
    )

    self._treasury_df = self._hypothetical_treasuries.copy()
    cusip = self.df.index[1]
    cusip

    self.df[self._krd_cols]
    list(df)
