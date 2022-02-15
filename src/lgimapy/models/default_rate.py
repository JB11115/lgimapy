import os
from functools import lru_cache

import numpy as np
import pandas as pd

from lgimapy.latex import Document
from lgimapy.utils import root, mkdir, load_pickle, dump_pickle

# %%


class DefaultRates:
    def __init__(self, ratings, lookbacks, db):
        self.ratings = ratings
        self.lookbacks = lookbacks
        self._max_lookback = 12 * max(lookbacks)
        self.db = db

    def load_month_start_index(self, date):
        self.db.load_market_data(date=date)
        return self.db.build_market_index(in_hy_returns_index=True)

    def fid(self, date):
        return f"{date.year}_{date.month:02.0f}"

    def _table_fid(self, rating, table_fid):
        return self.dir / rating / f"tables/{table_fid}.parquet"

    def _table_fids(self, rating, fid=None):
        table_fids = list((self.dir / rating / "tables").glob("*.parquet"))
        if fid is not None:
            current_fid = self._table_fid(rating, fid)
            if current_fid not in table_fids:
                table_fids.append(current_fid)
        return sorted(table_fids)

    def _og_ticker_fid(self, rating, fid):
        return self.dir / rating / f"tickers/{fid}.pickle"

    def _og_isin_fid(self, rating, fid):
        return self.dir / rating / f"isins/{fid}.parquet"

    def _distressed_ratio_fid(self, rating):
        return self.dir / rating / "distressed_ratio.parquet"

    def data_is_missing(self, fid):
        """Returns ``True`` if data is missing for date."""
        for rating in self.ratings.keys():
            # Check ticker, isin, and table files exist for current date.
            if not self._og_ticker_fid(rating, fid).exists():
                return True
            if not self._og_isin_fid(rating, fid).exists():
                return True
            if not self._table_fid(rating, fid).exists():
                return True

        return False

    @property
    @lru_cache(maxsize=None)
    def dates(self):
        """List of the start of every observed month."""
        return self.db.date("MONTH_STARTS")

    @property
    @lru_cache(maxsize=None)
    def dir(self):
        """Base directory for data."""
        return root("data/HY/default_rates")

    def build_dirs(self):
        """Build all directories if needed."""
        for rating in self.ratings.keys():
            mkdir(self.dir / rating / "tickers")
            mkdir(self.dir / rating / "isins")
            mkdir(self.dir / rating / "tables")

    @property
    @lru_cache(maxsize=None)
    def defaults(self):
        return self.db._defaults_df.copy()

    def update_distressed_ratio_files(self, fid, ix):
        """
        Update distressed ratio file for each rating at current date.
        """
        # Make original ticker file.
        for rating, rating_kws in self.ratings.items():
            dr_fid = self._distressed_ratio_fid(rating)
            try:
                old_df = pd.read_parquet(dr_fid)
            except (OSError, FileNotFoundError):
                old_df = pd.DataFrame()
            if fid in old_df.index:
                continue
            col = "AmountOutstanding"
            rating_ix = ix.subset(rating=rating_kws)
            ix_mv = rating_ix.total_value(col).iloc[0]
            distressed_ix = rating_ix.subset(OAS=(1000, None))
            if len(distressed_ix.df):
                distressed_mv = distressed_ix.total_value(col).iloc[0]
            else:
                distressed_mv = 0
            new_df = pd.DataFrame(
                {"distress_ratio": distressed_mv / ix_mv}, index=[fid]
            )
            updated_df = pd.concat((old_df, new_df), sort=True)
            updated_df.to_parquet(dr_fid)

    def make_og_index_files(self, fid, ix):
        """
        Make ticker and ISIN files for the starting universe
        for given subset rating index at each date.
        """
        # Make original ticker file.
        for rating, rating_kws in self.ratings.items():
            rating_ix = ix.subset(rating=rating_kws)
            tickers = set(rating_ix.tickers)
            dump_pickle(tickers, self._og_ticker_fid(rating, fid))

            # Make original isin file with inital amount outstanding.
            fields = ["AmountOutstanding", "Ticker"]
            isin_df = rating_ix.df.set_index("ISIN")[fields].rename_axis(None)
            isin_df.to_parquet(self._og_isin_fid(rating, fid))

    def _update_CDR_table(self, rating, fid, table_fid, ix):
        """
        Update cumulative default rate table for specific
        rating and forward date for the specified start date.
        """
        if fid < table_fid.stem:
            return  # Date not needed in file.

        # Load the data and check if update is required.
        try:
            table = pd.read_parquet(table_fid)
        except (OSError, FileNotFoundError):
            # Make new table file.
            og_tickers = load_pickle(self._og_ticker_fid(rating, fid))
            d = {}
            d["t"] = 0
            d["defaulted_tickers"] = ""
            d["n_defaults"] = 0
            d["withdrawn_tickers"] = ""
            d["n_withdrawls"] = 0
            d["n_tickers"] = len(og_tickers)
            d["issuer_MDR"] = 0
            d["issuer_CDR"] = 0
            d["mv_start"] = ix.total_value().iloc[0]
            d["mv_CDR"] = 0
            table = pd.Series(d, name=fid).to_frame().T
            table.to_parquet(table_fid)
            return  # No update necessary.

        # Check to see if file needs to be updated.
        prev = table.iloc[-1]  # previous row of table
        if prev["t"] == self._max_lookback:
            return  # File already completed.
        elif fid in table.index:
            return  # Date already in file.

        # Load tickers and ISINs from beginning of the period.
        og_tickers = load_pickle(self._og_ticker_fid(rating, table_fid.stem))
        og_isin_df = pd.read_parquet(self._og_isin_fid(rating, table_fid.stem))
        og_isins = set(og_isin_df.index.dropna())

        # Get issuer defaults, where either a bond from the original
        # index or ticker from original tickers has defaulted by
        # current fid's date.
        defaults_start = pd.to_datetime(table_fid.stem, format="%Y_%m")
        defaults_end = pd.to_datetime(fid, format="%Y_%m")
        issuer_def_df = self.defaults[
            (self.defaults["Date"] >= defaults_start)
            & (self.defaults["Date"] <= defaults_end)
            & (
                self.defaults["Ticker"].isin(og_tickers)
                | self.defaults["ISIN"].isin(og_isins)
            )
        ].copy()

        # If the original ISIN still exists in the dataset, revert the
        # ticker back to it's original value. This fixes the problem
        # of tickers changing over time, and counts them as only a
        # single default instead of 2 (M&A) or 0 (simple ticker change).
        og_isin_tickers = og_isin_df["Ticker"].to_dict()
        og_isin_locs = issuer_def_df["ISIN"].isin(og_isins)
        issuer_def_df.loc[og_isin_locs, "Ticker"] = issuer_def_df.loc[
            og_isin_locs, "ISIN"
        ].map(og_isin_tickers)

        # For market value (backwards looking) subset to only the
        # exact bonds that were in the index at the beggining of the
        # period, and use the original market value of each bond.
        mv_def_df = issuer_def_df[og_isin_locs].copy()
        og_isin_mv = og_isin_df["AmountOutstanding"].to_dict()
        mv_def_df["AmountOutstanding"] = mv_def_df["ISIN"].map(og_isin_mv)
        defaulted_mv = mv_def_df["AmountOutstanding"].sum()

        # Find new defaulted tickers. Look at all defaulted tickers
        # since the beggining of the period and subtract the tickers
        # that had defaulted prior to the current fid's date.
        all_defaulted_tickers = set(issuer_def_df["Ticker"].unique())
        prior_defaulted_tickers = set(prev["defaulted_tickers"].split())
        new_defaulted_tickers = all_defaulted_tickers - prior_defaulted_tickers
        n_defaults = len(new_defaulted_tickers)

        # Find new withdrawn tickers. Look at the original universe of
        # tickers, and remove any tickers that currently still exist,
        # the original tickers of bonds that still exist (fixes problem
        # of bonds changing tickers), tickers that have defaulted,
        # or tickers that have been withdrawn prior to current fid's date.
        current_tickers = set(ix.tickers)
        og_tickers_remaining = set(
            ix.subset(isin=og_isins).df["ISIN"].map(og_isin_tickers)
        )
        all_withdrawn_tickers = (
            og_tickers
            - current_tickers
            - og_tickers_remaining
            - all_defaulted_tickers
        )
        prior_withdrawn_tickers = set(prev["withdrawn_tickers"].split())
        new_withdrawn_tickers = all_withdrawn_tickers - prior_withdrawn_tickers
        n_withdraws = len(new_withdrawn_tickers)

        # Calulate issuer marginal default rate for current date.
        # Calcualted denominator according to Moody's methodology
        # where withdrawn tickers form current period are halved
        # to account for random timing of withdrawl within period.
        n_tickers = (
            table["n_tickers"].iloc[0]
            - len(prior_defaulted_tickers)
            - len(prior_withdrawn_tickers)
            - (n_withdraws / 2)
        )
        n_tickers = max(n_tickers, 0)  # can't be negative.
        if n_tickers == 0:
            issuer_mdr = 0  # divide by zero otherwise
        else:
            issuer_mdr = n_defaults / n_tickers

        # Calculate issuer cumulative default rate.
        mdr_history = np.array(list(table["issuer_MDR"]) + [issuer_mdr])
        issuer_cdr = 1 - np.product(1 - mdr_history)

        # Build next row in the table, append and save.
        d = {}
        d["t"] = prev["t"] + 1
        d["defaulted_tickers"] = " ".join(all_defaulted_tickers)
        d["n_defaults"] = n_defaults
        d["withdrawn_tickers"] = " ".join(all_withdrawn_tickers)
        d["n_withdrawls"] = n_withdraws
        d["n_tickers"] = n_tickers
        d["issuer_MDR"] = issuer_mdr
        d["issuer_CDR"] = issuer_cdr
        d["mv_start"] = prev["mv_start"]
        d["mv_CDR"] = defaulted_mv / prev["mv_start"]
        new_row = pd.Series(d, name=fid).to_frame().T
        new_table = table.append(new_row)
        new_table.to_parquet(table_fid)

    def update_CDR_tables(self, fid, ix):
        """Update all cumulative default rate tables for a start date."""
        for rating, rating_kws in self.ratings.items():
            rating_ix = ix.subset(rating=rating_kws)
            for table_fid in self._table_fids(rating, fid):
                self._update_CDR_table(rating, fid, table_fid, rating_ix)

    def check_default_data_is_up_to_date(self):
        last_date_to_update = self.dates[-1]
        last_observed_default = self.defaults["Date"].iloc[-1]
        defaults_fid = self.db.local("defaults.csv")
        default_fid_last_modification = pd.to_datetime(
            os.path.getmtime(defaults_fid), unit="s"
        )
        if last_date_to_update == max(
            last_date_to_update,
            last_observed_default,
            default_fid_last_modification,
        ):
            msg = (
                f"Final date to update is {last_date_to_update:%m/%d/%Y}, but "
                f"last observed default was {last_observed_default:%m/%d/%Y} "
                f"and default file was last modified on "
                f"{default_fid_last_modification:%m/%d/%Y}."
                f"Do you wish to continue with the update?\n"
                f"  [Y] Update\n  [N]  Exit\n"
            )
            update_data = ""
            while update_data not in {"Y", "N"}:
                update_data = input(msg).upper()
                if update_data == "N":
                    quit()

    def update_default_rate_data(self):
        """
        Update all data for default rates of specifed
        :attr:`lookbacks` and :attr:`ratings`.
        """
        self.build_dirs()
        self.check_default_data_is_up_to_date()
        for i, date in enumerate(self.dates):
            fid = self.fid(date)
            # if i == 120:
            if self.data_is_missing(fid):
                ix = self.load_month_start_index(date)
                self.make_og_index_files(fid, ix)
                self.update_CDR_tables(fid, ix)
                self.update_distressed_ratio_files(fid, ix)

    @lru_cache(maxsize=None)
    def default_rate(self, rating, lookback, metric="issuer"):
        dates, cdr = [], []
        t = 12 * lookback
        for table_fid in self._table_fids(rating):
            table = pd.read_parquet(table_fid)
            row = table[table["t"] == t].squeeze()
            if len(row):
                dates.append(pd.to_datetime(row.name, format="%Y_%m"))
                cdr.append(row[f"{metric}_CDR"])
        return pd.Series(cdr, index=dates, name="default_rate")

    @lru_cache(maxsize=None)
    def distressed_ratio(self, rating):
        dr = pd.read_parquet(self._distressed_ratio_fid(rating)).squeeze()
        dr.index = pd.to_datetime(dr.index, format="%Y_%m")
        return dr

    @property
    @lru_cache(maxsize=None)
    def sr_loan_survey(self):
        return self.db.load_bbg_data("US_SR_LOAN", "LEVEL").rename("sr_loan")
