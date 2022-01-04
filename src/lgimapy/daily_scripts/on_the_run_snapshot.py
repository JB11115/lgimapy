from functools import lru_cache
from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document

# %%


class OnTheRunSnapshot:
    def __init__(self, accounts, lookback_dates, maturities):
        self.accounts = accounts
        self.maturities = maturities
        self.dates = ["OAS"] + lookback_dates
        self.db = Database()

    def load_data(self):
        self.db.load_market_data()
        ix = self.db.build_market_index()
        ix.expand_tickers()
        otr_ix = ix.subset_on_the_runs()
        current_otr_tickers = otr_ix.tickers
        self.ticker_positions = self.get_ticker_positions(
            current_otr_tickers, n=10
        )
        self.tickers = self.ticker_positions.index
        n = len(self.tickers)
        m = len(self.maturities)
        p = len(self.dates)
        self.oas = np.full([n, m, p], np.nan)
        self.bold_locs = np.zeros([n, m, p])
        self.beta_adj_oas = np.full([n, m, p], np.nan)
        self.bonds = pd.DataFrame(index=self.tickers)
        vmax_d = defaultdict(list)
        vmin_d = defaultdict(list)
        vcenter_d = defaultdict(list)
        current_isins = {}
        current_ix_oas = {}
        all_ticker_current_oas = {}
        for k, date in enumerate(self.dates):
            if k == 0:
                pass  # Data already loaded.
            else:
                self.db.load_market_data(date=self.db.date(date))

            full_ix = self.db.build_market_index()
            full_ix.expand_tickers()
            full_otr_ix = full_ix.subset_on_the_runs()
            otr_ix = full_otr_ix.subset(ticker=self.tickers)
            for j, maturity in enumerate(self.maturities):
                full_mat_ix = full_otr_ix.subset(
                    original_maturity=(maturity, maturity), in_stats_index=True
                )
                mat_ix = otr_ix.subset(original_maturity=(maturity, maturity))
                ticker_df = mat_ix.df.set_index("Ticker")
                if not len(ticker_df):
                    # No tickers for current maturity/lookback.
                    vmin_d[date].append(0)
                    vcenter_d[date].append(0)
                    vmax_d[date].append(0)
                    continue
                all_ticker_oas = full_mat_ix.df.set_index("Ticker")["OAS"]
                ticker_isins = self._as_array(ticker_df, "ISIN").astype(str)
                ticker_oas = self._as_array(ticker_df, "OAS")
                ix_oas = full_mat_ix.MEDIAN("OAS").iloc[0]
                if k == 0:
                    current_isins[maturity] = ticker_isins.copy()
                    current_ix_oas[maturity] = ix_oas
                    self.oas[:, j, k] = ticker_oas.values
                    self.bonds[maturity] = (
                        ticker_df["CouponRate"].apply(lambda x: f"{x:.2f}")
                        + " "
                        + ticker_df["MaturityDate"]
                        .apply(lambda x: f"{x.strftime('%b `%y')}")
                        .astype(str)
                    )
                    all_ticker_current_oas[maturity] = all_ticker_oas
                else:
                    self.bold_locs[:, j, k] = (
                        (
                            (ticker_isins != current_isins[maturity])
                            & ~ticker_isins.isna()
                            & ~current_isins[maturity].isna()
                        )
                        .astype(int)
                        .values
                    )

                    # Find change in OAS for prominent tickers.
                    ticker_oas_chg = self.oas[:, j, 0] - ticker_oas
                    self.oas[:, j, k] = ticker_oas_chg.values

                    # Find beta adjusted performance by forecasting
                    # spread change in terms of DTS.
                    ix_dts = full_mat_ix.MEDIAN("DTS").iloc[0]
                    ticker_dts = self._as_array(ticker_df, "DTS")
                    ix_chg_oas = current_ix_oas[maturity] - ix_oas
                    fcast_ticker_oas_chg = ticker_dts / ix_dts * ix_chg_oas
                    self.beta_adj_oas[:, j, k] = (
                        ticker_oas_chg - fcast_ticker_oas_chg
                    ).values

                    # Find change in oas for all tickers.
                    # This will be used for the color scheme.
                    all_ticker_dts = full_mat_ix.df.set_index("Ticker")["DTS"]
                    fcast_all_ticker = all_ticker_dts / ix_dts * ix_chg_oas
                    all_ticker_oas_chg = (
                        all_ticker_current_oas[maturity] - all_ticker_oas
                    )
                    all_ticker_beta_adj = (
                        (all_ticker_oas_chg - fcast_all_ticker)
                        .dropna()
                        .sort_values()
                    )
                    n = len(all_ticker_beta_adj)
                    vmin_d[date].append(all_ticker_beta_adj.iloc[int(0.1 * n)])
                    vcenter_d[date].append(np.median(all_ticker_beta_adj))
                    vmax_d[date].append(all_ticker_beta_adj.iloc[int(0.9 * n)])

        self.vmin = pd.DataFrame(vmin_d, index=self.maturities)
        self.vcenter = pd.DataFrame(vcenter_d, index=self.maturities)
        self.vmax = pd.DataFrame(vmax_d, index=self.maturities)

    def _as_array(self, df, col):
        return pd.Series(df[col].reindex(self.tickers))

    def get_ticker_positions(self, current_otr_tickers, n=10):
        """
        Get tickers that are in top `n` positions from specified
        portfolios in terms of OAD or DTS. Sort by DTS position size.
        """
        # Find tickers with on the run bonds and large positions in
        # the specified portfolios.
        ports = [
            self.db.load_portfolio(account=account) for account in self.accounts
        ]
        tickers_with_large_positions = set()
        for port in ports:
            for col in ["OAD", "DTS"]:
                ticker_ow = port.ticker_overweights(by=col)
                tickers_with_large_positions |= set(ticker_ow.index[:n])
                tickers_with_large_positions |= set(ticker_ow.index[-n:])
        tickers = tickers_with_large_positions & set(current_otr_tickers)

        # Store overweights in a DataFrame.
        df = pd.DataFrame(index=tickers)
        for port in ports:
            for col in ["OAD", "DTS"]:
                ticker_ow = port.ticker_overweights(by=col).reindex(tickers)
                df[f"{port.name}*{col}*OW"] = ticker_ow

        # Sort based on collective DTS overweight of all portfolios.
        sorted_order = pd.Series({ticker: 0 for ticker in tickers})
        for account in self.accounts:
            sorted_order += df[f"{account}*DTS*OW"]
        self.sorted_order = sorted_order.sort_values(ascending=False)
        return df.reindex(self.sorted_order.index)

    def _create_doc(self):
        today = self.db.date("today")
        fid = f"Prominent_OTR_Tickers_Snapshot_{today:%Y_%m_%d}"
        self.doc = Document(fid, path="reports/on_the_run_snapshots")
        self.doc.add_preamble(
            margin={
                "paperheight": 30,
                "paperwidth": 22,
                "left": 1,
                "right": 1,
                "top": 0.5,
                "bottom": 0.2,
            },
            bookmarks=True,
            header=self.doc.header(
                left="Prominent On the Run Tickers Snapshot",
                right=f"EOD {today:%B %#d, %Y}",
            ),
            footer=self.doc.footer(logo="LG_umbrella", width=0.08),
        )

    def build_pdf(self):
        self._create_doc()
        self._build_lookback_tables()
        self._build_maturity_tables()
        self.doc.save()
        self.doc.save_as(
            "Prominent_OTR_Tickers_Snapshot", path="reports/current_reports"
        )

    def _idx(self, key):
        """
        Return the array index for either maturity or date.
        """
        if isinstance(key, str):
            map = {v: k for k, v in dict(enumerate(self.dates)).items()}
        else:
            map = {v: k for k, v in dict(enumerate(self.maturities)).items()}
        return map[key]

    def _footnote(self):
        return """
            \\vspace{-2.5ex}
            \\scriptsize
            \\itemsep0.8em

            \\item
            *Bolded locations in spread change columns indicate that the
            on the run bond has changed since the referenced time period.

            \\item
            *Color indicates beta-adjusted performance compared to other
            on the run bonds over the specified lookback period.
            """

    def _prec(self, df):
        prec = {}
        for col in df.columns:
            if "DTS" in col:
                prec[col] = "1f"
            elif "OAD" in col:
                prec[col] = "2f"
            elif "Delta" in col:
                prec[col] = "0f"
            elif "OAS" in col:
                prec[col] = "0f"
            elif col.endswith("y"):
                prec[col] = "0f"
        return prec

    def _build_lookback_tables(self):
        self._create_doc()
        self.doc.add_bookmark("By Lookback")
        mat_cols = [f"{mat}y" for mat in self.maturities]
        for date in self.dates:
            idx = np.index_exp[:, :, self._idx(date)]
            positions = self.ticker_positions.copy()
            tables = {}
            table_arrays = {
                "OAS": self.oas,
                "bold_locs": self.bold_locs,
                "column_colors": self.beta_adj_oas,
            }
            for key, table_array in table_arrays.items():
                df = pd.DataFrame(
                    table_array[idx], index=self.tickers, columns=mat_cols
                )
                table = pd.concat((positions, df), axis=1)
                if key == "OAS":
                    table.dropna(subset=mat_cols, how="all", inplace=True)
                else:
                    table = table[table.index.isin(tables["OAS"].index)]
                tables[key] = table

            for col in tables["bold_locs"].columns:
                if col not in mat_cols:
                    tables["bold_locs"][col] = 0

            # Make midrules every 5 bonds to make table easy to read.
            sorted_index = self.sorted_order[
                self.sorted_order.index.isin(tables["OAS"].index)
            ]
            ow = sorted_index[sorted_index > 0]
            uw = sorted_index[sorted_index < 0]
            midrules = list(ow.index[::5][1:]) + list(uw.index[::5][1:])
            bold_locs = self.doc.df_to_locs(tables["bold_locs"])
            bold_changed_isins = {bold_locs: "\\textbf"} if bold_locs else None

            # Provide proper coloring to columns.
            if date == "OAS":
                color_cols = None
                color_kws = None
                pass  # No coloring required.
            else:
                color_cols = mat_cols
                color_kws = {}
                for col in mat_cols:
                    mat = int(col.strip("y"))
                    color_kws[col] = {
                        "vals": tables["column_colors"][col],
                        "vmin": self.vmin.loc[mat, date],
                        "center": self.vcenter.loc[mat, date],
                        "vmax": self.vmax.loc[mat, date],
                    }

            # Add table to document.
            self.doc.add_subsection(
                "Current" if date == "OAS" else f"$\Delta${date}"
            )
            self.doc.add_table(
                tables["OAS"],
                font_size="normalsize",
                col_fmt=f"l|rrrr|{'r' * len(self.maturities)}",
                multi_row_header=True,
                midrule_locs=midrules if midrules else None,
                specialrule_locs=str(uw.index[0]),
                prec=self._prec(tables["OAS"]),
                table_notes=self._footnote(),
                loc_style=bold_changed_isins,
                gradient_cell_col=color_cols,
                gradient_cell_kws=color_kws,
            )
            self.doc.add_pagebreak()
        self.doc.save()

    def _build_maturity_tables(self):
        self.doc.add_bookmark("By Maturity")
        oas_cols = [
            "OAS" if date == "OAS" else f"$\Delta${date}" for date in self.dates
        ]
        for maturity in self.maturities:
            idx = np.index_exp[:, self._idx(maturity), :]
            bonds = self.bonds[maturity].rename("Description")
            positions = self.ticker_positions.copy()
            tables = {}
            table_arrays = {
                "OAS": self.oas,
                "bold_locs": self.bold_locs,
                "column_colors": self.beta_adj_oas,
            }
            for key, table_array in table_arrays.items():
                df = pd.DataFrame(
                    table_array[idx], index=self.tickers, columns=oas_cols
                )
                table = pd.concat((bonds, positions, df), axis=1)
                if key == "OAS":
                    table.dropna(subset=oas_cols, how="all", inplace=True)
                else:
                    table = table[table.index.isin(tables["OAS"].index)]
                tables[key] = table

            for col in tables["bold_locs"].columns:
                if col not in oas_cols[1:]:
                    tables["bold_locs"][col] = 0

            # Make midrules every 5 bonds to make table easy to read.
            sorted_index = self.sorted_order[
                self.sorted_order.index.isin(tables["OAS"].index)
            ]
            ow = sorted_index[sorted_index > 0]
            uw = sorted_index[sorted_index < 0]
            midrules = list(ow.index[::5][1:]) + list(uw.index[::5][1:])
            bold_locs = self.doc.df_to_locs(tables["bold_locs"])
            bold_changed_isins = {bold_locs: "\\textbf"} if bold_locs else None

            # Provide proper coloring to columns.
            color_kws = {}
            for col in oas_cols[1:]:
                date = col.split("$")[-1]
                color_kws[col] = {
                    "vals": tables["column_colors"][col],
                    "vmin": self.vmin.loc[maturity, date],
                    "center": self.vcenter.loc[maturity, date],
                    "vmax": self.vmax.loc[maturity, date],
                }

            try:
                ow_uw_line = str(uw.index[0])
            except IndexError:
                ow_uw_line = None

            # Add table to document.
            self.doc.add_subsection(f"{maturity}yr")
            self.doc.add_table(
                tables["OAS"],
                col_fmt=f"ll|rrrr|r|{'r' * (len(self.dates) - 1)}",
                multi_row_header=True,
                adjust=True,
                midrule_locs=midrules if midrules else None,
                specialrule_locs=ow_uw_line,
                prec=self._prec(tables["OAS"]),
                table_notes=self._footnote(),
                loc_style=bold_changed_isins,
                gradient_cell_col=oas_cols[1:],
                gradient_cell_kws=color_kws,
            )
            self.doc.add_pagebreak()


def build_on_the_run_ticker_snapshot():
    lookback_dates = ["1D", "1W", "1M", "3M", "6M", "YTD"]
    maturities = [2, 3, 5, 7, 10, 20, 30]
    accounts = ["CITMC", "P-LD"]

    otr_snapshot = OnTheRunSnapshot(accounts, lookback_dates, maturities)
    self = OnTheRunSnapshot(accounts, lookback_dates, maturities)
    otr_snapshot.load_data()
    otr_snapshot.build_pdf()


# %%
if __name__ == "__main__":
    build_on_the_run_ticker_snapshot()
