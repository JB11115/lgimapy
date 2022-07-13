import multiprocessing as mp
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import BetaAdjustedPerformance
from lgimapy.utils import to_datetime, to_list

# %%


class SectorReport:
    def __init__(self, db, universe, date=None):
        self.universe = universe.upper()
        self.date = db.date("today") if date is None else to_datetime(date)
        self._db = db
        self._db.load_market_data(date=date)

    def _init_doc(self, input_fid):
        fid = f"{self.date:%Y-%m-%d}_{self.universe}_Sector_Report"
        if input_fid is not None:
            fid = f"{fid}_{input_fid}"

        if self.universe == "IG":
            path = "reports/sector_reports"
        elif self.universe == "HY":
            path = "reports/HY/sector_reports"
        else:
            raise ValueError(
                f"`universe` must be 'IG' or 'HY', got {self.universe}"
            )
        return Document(fid, path=path)

    @property
    def _portfolios(self):
        return {
            "IG": {
                "US LC": "US Long Credit",
                "US MC": "US Credit",
                "US LA": "Liability Aware Long Duration Credit",
                "US LGC": "US Long Government/Credit",
            },
            "HY": {"US HY": "US High Yield"},
        }[self.universe]

    def _get_sectors(self, sectors):
        if sectors is not None:
            return to_list(sectors, dtype=str)

        if self.universe == "IG":
            sectors = self._db.IG_sectors(with_tildes=True, drop_chevrons=True)
            sectors_to_add = {
                "LIFE": "~LIFE_FABN",
                "REITS": "FINANCE_COMPANIES",
            }
            for prior_sector, new_sector in sectors_to_add.items():
                prior_idx = sectors.index(prior_sector)
                sectors.insert(prior_idx + 1, new_sector)
            return sectors

        elif self.universe == "HY":
            return self._db.HY_sectors(with_tildes=True)

    def _load_portfolio_data(self, data=None):
        if data is not None:
            return data

        # No data provided.
        return {
            key: self._db.load_portfolio(
                portfolio_name, date=self.date, universe="stats"
            )
            for key, portfolio_name in self._portfolios.items()
        }

        return self._sector, page

    def _get_beta_adjusted_models(self, table_kws):
        d = {}
        if isinstance(table_kws, dict):
            pass  # Received table kwargs.
        else:
            # Only received table names, create emtpy kwargs dict.
            table_name_list = to_list(table_kws, dtype=str)
            table_kws = {table_name: {} for table_name in table_name_list}

        for table_name, kws in table_kws.items():
            mod = BetaAdjustedPerformance(self._db)
            mod.train(
                forecast="1m", universe=self.universe, date=self.date, **kws
            )
            d[table_name] = mod
        return d

    def _sector_kws(self, index_eligible_only=False, **kwargs):
        if self.universe == "IG":
            kws = {"source": "bloomberg"}
            if index_eligible_only:
                kws.update({"in_stats_index": True})
            else:
                kws.update({"unused_constraints": "in_stats_index"})
        elif self.universe == "HY":
            kws = {"source": "baml"}
            if index_eligible_only:
                kws.update({"in_H4UN_index": True})
            else:
                kws.update({"unused_constraints": "in_H4UN_index"})

        kws.update(**kwargs)
        return self._db.index_kwargs(self._sector, **kws)

    def _get_sector_page(self, sector):
        """Create single page for each sector."""
        page = self._doc.create_page()
        self._sector = sector.strip("~")
        page_name = self._sector_kws()["name"].replace("&", "\&")
        if sector.startswith("~"):
            page.add_subsection(page_name)
        else:
            page.add_section(page_name)

        self._add_overview_table(page)
        if self._beta_adjusted_model_d is not None:
            self._add_issuer_performance_tables(page)

        page.add_pagebreak()
        return self._sector, page

    def _add_overview_table(self, page):
        table, footnote = self._get_overview_table()
        prec = {}
        ow_cols = []
        for col in table.columns:
            if "OW" in col:
                ow_cols.append(col)

            if "BM %" in col:
                prec[col] = "2%"
            elif "DTS" in col:
                prec[col] = "1%"
            elif "OAD OW" in col:
                prec[col] = "+2f"
            elif "MV OW" in col:
                prec[col] = "+2%"
            elif "Analyst" in col:
                prec[col] = "+0f"

        issuer_table = table.iloc[
            len(self._summary_rows) : self._n_max_issuers + 2, :
        ].copy()
        ow_max = max(1e-5, issuer_table[ow_cols].max().max())
        ow_min = min(-1e-5, issuer_table[ow_cols].min().min())
        table_width = min(
            0.9, 0.3 + len(self._bm_weight_cols) / 10 + len(self._ports) / 10
        )
        edit = page.add_minipages(1, widths=[table_width])
        col_fmt = (
            f"llrc"
            f"|{'c' * len(self._bm_weight_cols)}"
            f"|{'c' * len(self._ports)}"
        )
        with page.start_edit(edit):
            page.add_table(
                table,
                table_notes=footnote,
                col_fmt=col_fmt,
                multi_row_header=True,
                midrule_locs=[table.index[len(self._summary_rows)], "Other"],
                prec=prec,
                adjust=True,
                gradient_cell_col=ow_cols,
                gradient_cell_kws={
                    "cmax": "army",
                    "cmin": "rose",
                    "vmax": ow_max,
                    "vmin": ow_min,
                },
            )

    @property
    def _overview_table_columns(self):
        cols = ["Rating", "Analyst*Score", "Sector*DTS"]
        for col in self._bm_weight_cols:
            cols.append(self._bm_weight_column(col))
        for port_key in self._ports.keys():
            ow_metric = "MV" if self._ow_metric == "Weight" else self._ow_metric
            cols.append(self._overweight_column(port_key))
        return cols

    def _bm_weight_column(self, col):
        return f"BM %*{col}"

    def _overweight_column(self, key):
        ow_metric = "MV" if self._ow_metric == "Weight" else self._ow_metric
        return f"{key}*{ow_metric} OW"

    @property
    def _summary_rows(self):
        if self.universe == "IG":
            return ["Total", "A-Rated", "BBB-Rated"]
        elif self.universe == "HY":
            return ["Total", "BB-Rated", "B-Rated", "CCC-Rated"]

    def _get_overview_table(self):
        """
        Get overview table for tickers in given sector including
        rating, % of major benchmarks, and LGIMA's overweights
        in major strategies.

        Returns
        -------
        table_df: pd.DataFrame
            Table of most important ``n`` tickers.
        """
        # Get DataFrame of all individual issuers.
        df_list = []
        ratings_list = []
        for port_key, port in self._ports.items():
            sector_port = port.subset(**self._sector_kws())
            ow_col = self._overweight_column(port_key)
            if sector_port.accounts:
                df_list.append(
                    sector_port.ticker_overweights(by=self._ow_metric).rename(
                        ow_col
                    )
                )
            else:
                df_list.append(pd.Series(name=ow_col, dtype=object))
            ticker_df = sector_port.ticker_df
            ratings_list.append(ticker_df["NumericRating"].dropna())
            if port_key in self._bm_weight_cols:
                # Get weight of each ticker in benchmark.
                df_list.append(
                    (ticker_df["BM_Weight"] / len(sector_port.accounts)).rename(
                        self._bm_weight_column(port_key)
                    )
                )

        ratings = (
            np.mean(pd.DataFrame(ratings_list), axis=0).round(0).astype(int)
        )
        df_list.append(
            self._db.convert_numeric_ratings(ratings).rename("Rating")
        )
        df = pd.DataFrame(df_list).T.rename_axis(None)
        if self.universe == "IG":
            ticker_df = self._ports["US MC"].ticker_df.copy()
        elif self.universe == "HY":
            ticker_df = self._ports["US HY"].ticker_df.copy()

        missing_tickers = set(df.index) - set(ticker_df.index)
        if len(missing_tickers) > 0:
            ticker_df.index = list(ticker_df.index)
            ix = self._db.build_market_index(ticker=missing_tickers)
            missing_ticker_df = ix.ticker_df
            for ticker in missing_tickers:
                try:
                    analyst_rating = missing_ticker_df.loc[
                        ticker, "AnalystRating"
                    ]
                except KeyError:
                    analyst_rating = np.nan
                s = pd.Series({"AnalystRating": analyst_rating}, name=ticker)
                ticker_df = pd.concat((ticker_df, s.to_frame().T))

        df["Analyst*Score"] = ticker_df["AnalystRating"][df.index].round(0)
        mv_weighted_dts = (ticker_df["DTS"] * ticker_df["MarketValue"])[
            df.index
        ]
        df["Sector*DTS"] = mv_weighted_dts / mv_weighted_dts.sum()

        # Find total overweight over all strategies.
        ow_cols = [col for col in df.columns if "OW" in col]
        bm_pct_cols = [col for col in df.columns if "BM %" in col]
        df["ow"] = np.sum(df[ow_cols], axis=1)

        # Get summary rows.
        summary_df = pd.DataFrame()
        str_cols = ["Rating", "Analyst*Score"]
        numeric_cols = [col for col in df.columns if col not in str_cols]
        self._overview_issuer_table_df = df.copy()  # for debugging
        for row_name in self._summary_rows:
            if row_name == "Total":
                df_row = df[numeric_cols].sum().rename(row_name)
            elif row_name == "A-Rated":
                df_row = (
                    df[df["Rating"].astype(str).str.startswith("A")][
                        numeric_cols
                    ]
                    .sum()
                    .rename(row_name)
                )
            else:
                rating_bucket = row_name.split("-")[0]
                df_row = (
                    df[
                        df["Rating"].astype(str).str.strip("+-")
                        == rating_bucket
                    ][numeric_cols]
                    .sum()
                    .rename(row_name)
                )
            for col in str_cols:
                df_row[col] = "-"
            summary_df = pd.concat((summary_df, df_row.to_frame().T))

        # Create `other` row if required.
        if len(df) > self._n_max_issuers:
            # Sort columns by combination of portfolio overweights
            # and market value. Lump together remaining tickers.
            df["bm"] = np.sum(df[bm_pct_cols], axis=1)
            df["importance"] = np.abs(df["ow"]) + 10 * df["bm"]
            df.sort_values("importance", ascending=False, inplace=True)
            df_top_tickers = df.iloc[: self._n_max_issuers - 1, :]
            df_other_tickers = df.iloc[self._n_max_issuers :, :]
            other_tickers = df_other_tickers[numeric_cols].sum().rename("Other")
            other_tickers["Rating"] = "-"
            other_tickers["Analyst*Score"] = "-"
            table_df = pd.concat(
                (
                    df_top_tickers.sort_values("ow", ascending=False),
                    other_tickers.to_frame().T,
                )
            )
            other_tickers = ", ".join(sorted(df_other_tickers.index))
            note = (
                f"\\scriptsize \\textit{{Other}} consists of {other_tickers}."
            )
        else:
            table_df = df.sort_values("ow", ascending=False)
            note = None

        table = pd.concat((summary_df, table_df))[self._overview_table_columns]
        return table, note

    def _add_issuer_performance_tables(self, page):
        n_tables = len(self._beta_adjusted_model_d)
        table_edits = to_list(
            page.add_minipages(
                n_tables,
                valign="t",
                widths=[0.7] if n_tables == 1 else None,
            ),
            dtype=int,
        )
        for edit, (title, mod) in zip(
            table_edits, self._beta_adjusted_model_d.items()
        ):
            try:
                table = mod.get_issuer_table(
                    **self._sector_kws(index_eligible_only=True),
                    return_type=self._return_type,
                ).reset_index(drop=True)
            except KeyError:
                continue
            table.drop("RatingBucket", axis=1, inplace=True)
            bold_rows = tuple(
                table[table["Issuer"].isin(self._summary_rows)].index
            )
            with page.start_edit(edit):
                page.add_table(
                    table,
                    prec=mod.table_prec(table),
                    col_fmt="llc|cc|ccc",
                    caption=f"{title} 1M Performance",
                    adjust=True,
                    hide_index=True,
                    font_size="scriptsize",
                    multi_row_header=True,
                    row_font={bold_rows: "\\bfseries"},
                    gradient_cell_col=["Out*Perform", "Impact*Factor"],
                    gradient_cell_kws={
                        "Out*Perform": {
                            "cmax": "steelblue",
                            "cmin": "firebrick",
                        },
                        "Impact*Factor": {"cmax": "oldmauve"},
                    },
                )
                page.add_vskip()

    def build_report(
        self,
        bm_weight_cols,
        ow_metric="OAD",
        return_type="XSRet",
        fid=None,
        sectors=None,
        portfolio_data=None,
        n_max_issuers=20,
        beta_adjusted_performance_tables=None,
    ):

        self._bm_weight_cols = to_list(bm_weight_cols, dtype=str)
        self._ow_metric = ow_metric
        self._return_type = return_type
        self._sectors = self._get_sectors(sectors)
        self._n_max_issuers = n_max_issuers
        self._ports = self._load_portfolio_data(portfolio_data)
        self._beta_adjusted_model_d = self._get_beta_adjusted_models(
            beta_adjusted_performance_tables
        )

        self._doc = self._init_doc(fid)
        self._doc.add_preamble(margin=1, bookmarks=True)
        self._pages = [
            self._get_sector_page(sector) for sector in tqdm(self._sectors)
        ]

        sector_names = [s.strip("~") for s in self._sectors]
        df = (
            pd.DataFrame(self._pages, columns=["sector", "page"])
            .set_index("sector")
            .reindex(sector_names)
        )
        for page in df["page"]:
            self._doc.add_page(page)

        self._doc.save()
        if fid is None:
            if self.universe == "IG":
                self._doc.save_as(
                    "IG_Sector_Report", path="reports/current_reports"
                )
            elif self.universe == "HY":
                self._doc.save_as("HY_Sector_Report", path="reports/HY")


# %%


def _test():
    sector_report = SectorReport(Database(), universe="IG")
    port_data = sector_report._load_portfolio_data()

    # %%
    self = SectorReport(Database(), universe="IG")
    self.build_report(
        bm_weight_cols=["US LC", "US MC"],
        ow_metric="OAD",
        return_type="XSRet",
        portfolio_data=port_data,
        beta_adjusted_performance_tables={
            "Long Credit": {"maturity": (10, None)},
            "5-10 yr Credit": {"maturity": (5, 10)},
        },
    )


def build_IG_sector_report():
    sector_report = SectorReport(Database(), universe="IG")
    sector_report = SectorReport(Database(), universe="IG")
    sector_report.build_report(
        bm_weight_cols=["US LC", "US MC"],
        ow_metric="OAD",
        return_type="XSRet",
        beta_adjusted_performance_tables={
            "Long Credit": {"maturity": (10, None)},
            "5-10 yr Credit": {"maturity": (5, 10)},
        },
    )
    # %%


def build_HY_sector_report():
    sector_report = SectorReport(Database(), universe="HY")
    sector_report.build_report(
        bm_weight_cols=["US HY"],
        ow_metric="Weight",
        return_type="TRet",
        beta_adjusted_performance_tables="Issuer Beta-Adjusted",
    )

    # %%


if __name__ == "__main__":
    build_IG_sector_report()
    build_HY_sector_report()
