import argparse
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sms

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root, mkdir, load_pickle, dump_pickle

vis.style()

# %%


class DefaultRates:
    def __init__(self, ratings, lookbacks):
        self.ratings = ratings
        self.lookbacks = lookbacks
        self._max_lookback = 12 * max(lookbacks)
        self.db = Database()

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
        df = pd.read_csv(self.dir / "defaults.csv", index_col=0)
        df.columns = ["Ticker", "Date", "ISIN"]
        bad_dates = {"#N/A Invalid Security"}
        bad_isins = {"#N/A Requesting Data...", "#N/A Field Not Applicable"}
        df = df[
            ~((df["Date"].isin(bad_dates)) | (df["ISIN"].isin(bad_isins)))
        ].copy()
        df["ISIN"].value_counts()
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True).rename_axis(None)

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
        if last_date_to_update > last_observed_default:
            msg = (
                f"Final date to update is {last_date_to_update:%m/%d/%Y}, but "
                f"last observed default was {last_observed_default:%m/%d/%Y}."
                f"Do you wish to continue with the update?\n"
                "  [Y] Update\n  [N]  Exit\n"
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


# ratings = {
#     "HY": "HY",
#     "BB": ("BB+", "BB-"),
#     "B": ("B+", "B-"),
#     "CCC": ("CCC+", "CCC-"),
# }
# lookbacks = [1, 3, 5]
# self = DefaultRates(ratings, lookbacks)

# %%

#
# # # %%
# cdr = self.default_rate("HY", 1, "issuer")
# vis.plot_timeseries(cdr, ytickfmt="{x:.0%}")
# vis.show()
#
# # %%
# db = Database()
# sr_loan = db.load_bbg_data("US_SR_LOAN", "LEVEL", start="1999")
# # %%
# x = 500
# fig, ax = vis.subplots()
# vis.plot_timeseries(sr_loan / x, label="large", color="navy", ax=ax)
# vis.plot_timeseries(cdr, label="Default Rate", color="k", ax=ax)
# ax.legend(fancybox=True, shadow=True)
# vis.show()
#
#
# # %%
# d = defaultdict(list)
# for raw_date in self.dates:
#     date = self.db.date("MONTH_START", raw_date)
#     self.db.load_market_data(date=date)
#     ix = self.db.build_market_index(
#         in_hy_stats_index=True,
#         in_hy_returns_index=True,
#         special_rules="USHYReturnsFlag | USHYStatisticsFlag",
#     )
#     distressed_ix = ix.subset(OAS=(1000, None))
#     ix_mv = ix.total_value("AmountOutstanding").iloc[0]
#     if len(distressed_ix.df):
#         distressed_mv = distressed_ix.total_value("AmountOutstanding").iloc[0]
#     else:
#         distressed_mv = 0
#
#     d["date"].append(date)
#     d["issuer_DR"].append(len(distressed_ix.df) / len(ix.df))
#     d["mv_DR"].append(distressed_mv / ix_mv)
#
# # len(d['date'])
# # d['issuer_DR'].append(d['issuer_DR'][-1])
# # d['mv_DR'].append(d['mv_DR'][-1])
#
# df = pd.DataFrame(d).set_index("date", drop=True).rename_axis(None)
# # %%
# x = 5
# fig, ax = vis.subplots()
# vis.plot_timeseries(
#     df["issuer_DR"].rank(pct=True), label="issuer", color="navy", ax=ax
# )
# vis.plot_timeseries(
#     df["mv_DR"].rank(pct=True), label="small", color="skyblue", ax=ax
# )
# vis.plot_timeseries(cdr.rank(pct=True), label="Default Rate", color="k", ax=ax)
# ax.legend(fancybox=True, shadow=True)
# vis.show()
#
# issuer_dr = df["issuer_DR"]
#
#
# # %%
# rating = "HY"


# %%
def update_default_rate_pdf(fid):
    # %%
    vis.style()
    ratings = {
        "HY": "HY",
        "BB": ("BB+", "BB-"),
        "B": ("B+", "B-"),
        "CCC": ("CCC+", "CCC-"),
    }
    lookbacks = [1, 3, 5]
    mod = DefaultRates(ratings, lookbacks)
    mod.update_default_rate_data()
    doc = Document(fid, path="reports/HY", fig_dir=True)
    doc.add_preamble(
        bookmarks=True,
        table_caption_justification="c",
        margin={"left": 0.5, "right": 0.5, "top": 1.5, "bottom": 1},
        header=doc.header(
            left="Default Rates",
            right=f"EOD {mod.dates[-1].strftime('%B %#d, %Y')}",
            height=0.5,
        ),
        footer=doc.footer(logo="LG_umbrella", height=-0.4, width=0.1),
    )
    doc.add_section("Default Rates")

    fig, ax = vis.subplots(figsize=(10, 5))
    colors = ["skyblue", "royalblue", "navy"]
    for lookback, color in zip(lookbacks[::-1], colors):
        dr = mod.default_rate("HY", lookback, "issuer")
        vis.plot_timeseries(
            dr,
            color=color,
            lw=3,
            alpha=0.8,
            label=f"{lookback} yr",
            ax=ax,
            start="2006",
            title="HY Issuer Default Rate",
        )
    ax.legend(fancybox=True, shadow=True)
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    doc.add_figure("HY_default_rates", width=0.9, dpi=200, savefig=True)

    fig, ax = vis.subplots(figsize=(10, 5))
    colors = ["k"] + vis.colors("ryb")
    for rating, color in zip(ratings.keys(), colors):
        dr = mod.default_rate(rating, 1, "issuer")
        vis.plot_timeseries(
            dr,
            color=color,
            lw=3,
            alpha=0.8,
            label=rating,
            ax=ax,
            start="2006",
            title="1 yr Issuer Default Rate",
        )
    ax.legend(fancybox=True, shadow=True)
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    doc.add_figure("1yr_default_rates", width=0.9, dpi=200, savefig=True)

    columns = doc.add_subfigures(n=2, valign="t")
    for title, column in zip(["Issuer", "MV"], columns):
        d = defaultdict(list)
        for rating in ratings.keys():
            for lookback in lookbacks:
                dr = mod.default_rate(rating, lookback, title.lower())
                d[rating].append(dr.iloc[-1])

        idx = [f"{lb} yr" for lb in lookbacks]
        table = pd.DataFrame(d, index=idx)
        with doc.start_edit(column):
            doc.add_table(
                table,
                caption=f"{title} Default Rates",
                col_fmt="lrrrr",
                font_size="Large",
                prec={col: "1%" for col in table.columns},
            )

    doc.save(save_tex=False)


# %%


def get_shift(df, col):
    if col == "sr_loan":
        return 6
    elif col == "distress_ratio":
        return 8
    shifts = {}
    for shift in range(1, 12):
        reg_df = df.copy()
        reg_df["shift"] = reg_df[col].shift(shift)
        reg_df.dropna(inplace=True)
        ols = sms.OLS(
            reg_df["default_rate"], sms.add_constant(reg_df["shift"])
        ).fit()
        shifts[shift] = ols.rsquared
    optimal_shift = max(shifts, key=shifts.get)
    return optimal_shift


# %%
def forecast_default_rate(self, rating):
    # %%
    rating = "HY"
    raw_df = pd.concat(
        (
            self.default_rate(rating, 1, "issuer"),
            self.distressed_ratio(rating),
            self.sr_loan_survey,
        ),
        axis=1,
    )
    raw_df["sr_loan"].fillna(method="ffill", inplace=True)
    raw_df.dropna(inplace=True)
    pct_df = raw_df.rank(pct=True)

    raw_reg_df = pct_df["default_rate"].copy().to_frame()

    x_cols = ["distress_ratio", "sr_loan"]
    shifts = {}
    for col in x_cols:
        shifts[col] = get_shift(pct_df, col)
        raw_reg_df[col] = pct_df[col].shift(shifts[col])

    reg_df = raw_reg_df.dropna()
    y = reg_df["default_rate"]
    x = sms.add_constant(reg_df[x_cols])
    ols = sms.OLS(y, x).fit()

    # %%
    y_pred = ols.predict(x).rename("pred")

    # %%
    pct_df.tail(10)
    reg_df.tail()
    pct_df["sr_loan"].iloc[-6]
    pct_df["sr_loan"].shift(6).iloc[-1]

    # %%
    curr_x = pd.DataFrame()
    curr_x["const"] = [1]
    curr_x["sr_loan"] = [pct_df["sr_loan"].iloc[-1]]
    n = shifts["distress_ratio"] - shifts["sr_loan"] + 1
    curr_x["distress_ratio"] = [pct_df["distress_ratio"].iloc[-n]]
    curr_pred = ols.predict(curr_x).iloc[0]
    def_rt = pct_df["default_rate"]
    idx = def_rt[def_rt <= curr_pred].sort_values().index[-1]
    raw_df.loc[idx, "default_rate"]
    pct_df
    # %%
    pred_x = pct_df["sr_loan"].to_frame()
    pred_x["distress_ratio"] = pct_df["distress_ratio"].shift(n - 1)
    y_pred = ols.predict(sms.add_constant(pred_x.dropna()))

    # %%

    plot_df = pd.concat((y, y_pred), axis=1)
    plot_df.plot()
    vis.show()


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print SRCH Columns"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.print:
        print(
            "\nBBG SRCH Result Columns:\n",
            "['Issuer Name', 'Ticker', 'Default Date', 'ISIN']",
        )
    else:
        fid = "Default_Rates"
        update_default_rate_pdf(fid)
