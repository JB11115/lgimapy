from collections import defaultdict
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document, drop_consecutive_duplicates

# %%


def build_investable_universe_report(portfolio):
    db = Database()
    dates = {
        "YE21": db.date("YEAR_END", 2021),
        "YE20": db.date("YEAR_END", 2020),
    }
    report = InvestableUniverse(portfolio, dates, db)
    report.build_report()


class InvestableUniverse:
    def __init__(self, portfolio, dates, db):
        self.portfolio = portfolio
        self.date = db.date("today")
        self.dates = {"today": self.date, **dates}
        self._db = db
        self._small_ow_threshold = 0.001
        self._small_uw_threshold = -0.001

    @property
    def _rating_kws(self):
        return {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}

    @cached_property
    def _prev_date_keys(self):
        return sorted(self.dates, key=self.dates.get, reverse=True)[1:]

    @property
    def _sectors(self):
        return [
            "CHEMICALS",
            "CAPITAL_GOODS",
            "METALS_AND_MINING",
            "COMMUNICATIONS",
            "AUTOMOTIVE",
            "RETAILERS",
            "CONSUMER_CYCLICAL_EX_AUTOS_RETAILERS",
            "HEALTHCARE_PHARMA",
            "FOOD_AND_BEVERAGE",
            "CONSUMER_NON_CYCLICAL_OTHER",
            "ENERGY",
            "TECHNOLOGY",
            "TRANSPORTATION",
            "BANKS",
            "OTHER_FIN",
            "INSURANCE_EX_HEALTHCARE",
            "REITS",
            "UTILITY_HOLDCO",
            "UTILITY_OPCO",
            "SOVEREIGN",
            "NON_CORP_OTHER",
            "OTHER_INDUSTRIAL",
        ]

    @cached_property
    def _port(self):
        port = self._db.load_portfolio(self.portfolio, universe="stats")
        ix = self._db.build_market_index(isin=port.isins)
        missing_isins = set(port.isins) - set(ix.isins)
        port = port.subset(
            isin=missing_isins,
            special_rules="~ISIN",
        )
        port._constraints = {}
        port.expand_tickers()
        return port

    @lru_cache(maxsize=None)
    def _port_ticker_df(self, rating):
        df = self._port.subset(rating=self._rating_kws[rating]).ticker_df
        df["ReportSector"] = df.index.map(self._ticker_sector_map(rating))
        return df

    def _top_30_tickers(self, rating):
        return list(
            self._port.subset(rating=self._rating_kws[rating])
            .ticker_df["BM_DTS_PCT"]
            .sort_values(ascending=False)
            .index[:30]
        )

    @cached_property
    def _ix_d(self):
        d = {}
        keys = ["today", *self._prev_date_keys]
        for key in keys:
            self._db.load_market_data(date=self.dates[key])
            ix = self._db.build_market_index(isin=self._port.isins)
            ix.expand_tickers()
            d[key] = ix
        return d

    @lru_cache(maxsize=None)
    def _oas_df(self, rating):
        oas_level_df = pd.concat(
            (
                ix.subset(rating=self._rating_kws[rating])
                .ticker_df["OAS"]
                .rename(key)
                for key, ix in self._ix_d.items()
            ),
            axis=1,
        )
        df = oas_level_df["today"].rename("OAS").to_frame()
        for key in self._prev_date_keys:
            df[f"delta_{key}"] = oas_level_df["today"] - oas_level_df[key]

        return df

    def _ticker_sector_map(self, rating):
        ticker_sector_mv = []
        for sector in self._sectors:
            kws = self._db.index_kwargs(
                sector,
                unused_constraints="in_stats_index",
                rating=self._rating_kws[rating],
            )
            ticker_sector_df = self._port.subset(**kws).ticker_df[
                ["MarketValue"]
            ]
            ticker_sector_df["Sector"] = kws["name"]
            ticker_sector_mv.append(ticker_sector_df)

        return (
            pd.concat(ticker_sector_mv)
            .sort_values("MarketValue")
            .groupby("Ticker")
            .last()["Sector"]
            .to_dict()
        )

    def _large_cap_table(self, rating):
        df = pd.concat(
            [
                self._add_total(self._top_30_df(rating)),
                self._market_segment_row("HOSPITALS", rating),
                self._market_segment_row("NON_FIN_EX_TOP_30", rating),
                self._market_segment_row("FIN_EX_TOP_30", rating),
                self._market_segment_row("NON_CORP_EX_TOP_30", rating),
            ]
        )
        return self._format_table(df)

    def _small_ow_table(self, rating):
        port_df = self._port_ticker_df(rating)
        port_df = port_df[
            ~port_df.index.isin(self._used_tickers)
            & (port_df["DTS_PCT_Diff"] > 0)
            & (port_df["DTS_PCT_Diff"] < self._small_ow_threshold)
        ]
        tickers = port_df.index.unique()
        df = pd.concat(
            (
                self._port_ticker_df(rating).loc[tickers, self._port_cols],
                self._oas_df(rating).loc[tickers],
            ),
            axis=1,
        )
        table = self._sort_table(df.reset_index())
        return self._format_table(self._add_total(table))

    def _small_uw_table(self, rating):
        port_df = self._port_ticker_df(rating)
        port_df = port_df[
            ~port_df.index.isin(self._used_tickers)
            & (port_df["P_Weight"] > 0)
            & (port_df["DTS_PCT_Diff"] < 0)
            & (port_df["DTS_PCT_Diff"] > self._small_uw_threshold)
        ]
        tickers = port_df.index.unique()
        df = pd.concat(
            (
                self._port_ticker_df(rating).loc[tickers, self._port_cols],
                self._oas_df(rating).loc[tickers],
            ),
            axis=1,
        )
        table = self._sort_table(df.reset_index())
        return self._format_table(self._add_total(table))

    def _top_30_df(self, rating):
        t30 = self._top_30_tickers(rating)
        self._used_tickers |= set(t30)
        df = pd.concat(
            (
                self._port_ticker_df(rating).loc[t30, self._port_cols],
                self._oas_df(rating).loc[t30],
            ),
            axis=1,
        )
        return self._sort_table(df.reset_index())

    def _market_segment_row(self, segment, rating):
        d = {}
        kws = self._db.index_kwargs(
            segment,
            rating=self._rating_kws[rating],
            unused_constraints="in_stats_index",
        )
        d["ReportSector"] = kws["name"]
        d["Ticker"] = np.nan
        used_tickers = set()

        # Add change in OAS columns.
        for col, ix in zip(self._oas_cols, self._ix_d.values()):
            segment_ix = ix.subset(**kws)
            segment_ix._constraints = {}
            segment_ix = segment_ix.subset(
                ticker=self._used_tickers, special_rules="~Ticker"
            )
            used_tickers |= set(segment_ix.tickers)
            if col == "OAS":
                d[col] = segment_ix.OAS().iloc[0]
            else:
                d[col] = d["OAS"] - segment_ix.OAS().iloc[0]

        # Add portoflio metric columns.
        segment_port = self._port.subset(**kws)
        segment_port._constraints = {}
        segment_port = segment_port.subset(
            ticker=self._used_tickers, special_rules="~Ticker"
        )
        for col in self._port_cols[1:]:
            d[col] = segment_port.df[col].sum()

        return pd.Series(d).to_frame().T

    @property
    def _oas_cols(self):
        cols = [f"delta_{key}" for key in self._prev_date_keys]
        cols.insert(0, "OAS")
        return cols

    @property
    def _port_cols(self):
        return [
            "ReportSector",
            "P_Weight",
            "P_DTS_PCT",
            "BM_DTS_PCT",
            "DTS_PCT_Diff",
            "P_OAD",
            "BM_OAD",
            "OAD_Diff",
            "MarketValue",
        ]

    @property
    def _column_name_map(self):
        col_map = {
            "ReportSector": "Sector",
            "OAS": "OAS",
            "P_Weight": "Port*MV %",
            "P_DTS_PCT": "Port*DTS",
            "BM_DTS_PCT": "BM*DTS",
            "DTS_PCT_Diff": "DTS*OW",
            "P_OAD": "Port*OAD",
            "BM_OAD": "BM*OAD",
            "OAD_Diff": "OAD*OW",
        }
        for key in self._prev_date_keys:
            col_map[f"delta_{key}"] = f"$\\Delta$OAS*{key}"
        return col_map

    def _add_total(self, df):
        total_row = df[self._port_cols[1:]].sum().to_frame().T
        for col in self._oas_cols:
            total_row[col] = (df[col] * df["MarketValue"]).sum() / df[
                "MarketValue"
            ].sum()
        total_row["ReportSector"] = "Total"
        total_row["Ticker"] = np.nan
        return pd.concat((df, total_row))

    def _sort_table(self, df):
        sector_gdf = (
            df[["ReportSector", "BM_DTS_PCT"]]
            .groupby("ReportSector")
            .sum()
            .squeeze()
            .sort_values(ascending=False)
        )
        sorter_index = {sector: i for i, sector in enumerate(sector_gdf.index)}
        df["sort"] = df["ReportSector"].map(sorter_index)
        df.sort_values(
            ["sort", "BM_DTS_PCT"], ascending=[True, False], inplace=True
        )
        df["ReportSector"] = drop_consecutive_duplicates(df["ReportSector"])
        return df

    def _format_table(self, df):
        col_order = [
            "ReportSector",
            "Ticker",
            *self._oas_cols,
            *self._port_cols[1:-1],
        ]
        return (
            df[col_order]
            .rename(columns=self._column_name_map)
            .reset_index(drop=True)
        )

    @cached_property
    def _table_prec(self):
        d = {
            "OAS": "0f",
            "Port*MV %": "2%",
            "Port*DTS": "2%",
            "BM*DTS": "2%",
            "DTS*OW": "+2%",
            "Port*OAD": "2f",
            "BM*OAD": "2f",
            "OAD*OW": "+2f",
        }
        for col in self._column_name_map.values():
            if "Delta" in col:
                d[col] = "+0f"

        return d

    def _add_page(self, table, caption):
        total_loc = table[table["Sector"] == "Total"].index
        if total_loc == len(table) - 1:
            midrule_locs = table[table["Sector"] != " "].index[1:]
        else:
            midrule_locs = table[table["Sector"] != " "].index[1:-4]

        self._doc.add_table(
            table,
            caption=caption,
            col_fmt="lll|rrr|r|rrr|rrr",
            font_size="footnotesize",
            row_font={tuple(total_loc): "\\bfseries"},
            midrule_locs=midrule_locs,
            prec=self._table_prec,
            adjust=True,
            hide_index=True,
            multi_row_header=True,
            gradient_cell_col="DTS*OW",
            gradient_cell_kws={
                "cmax": "army",
                "cmin": "rose",
                "vmax": 0.02
                if "Top 30" in caption
                else 2 * self._small_ow_threshold,
                "vmin": -0.01
                if "Top 30" in caption
                else 2 * self._small_uw_threshold,
            },
        )
        self._doc.add_pagebreak()

    def _init_doc(self):
        doc = Document(
            f"{self._port.fid}_Investable_Universe_{self.date:%Y-%m-%d}",
            path=f"reports/investable_universe",
        )
        doc.add_preamble(
            margin={
                "paperheight": 26,
                "left": 1.5,
                "right": 1.5,
                "top": 0.5,
                "bottom": 0.2,
            },
            bookmarks=True,
            table_caption_justification="c",
            header=doc.header(
                left=f"{self._port.latex_name} Investable Universe",
                right=f"EOD {self.date:%B %#d, %Y}",
            ),
            footer=doc.footer(logo="LG_umbrella"),
        )
        return doc

    def build_report(self):
        self._doc = self._init_doc()
        for rating in self._rating_kws.keys():
            self._doc.add_bookmark(f"{rating}-Rated")
            self._doc.add_bookmark("Top 30", level=1)

            self._used_tickers = set()  # Don't use tickers more than once.
            self._add_page(
                self._large_cap_table(rating),
                caption=f"{rating}-Rated Top 30 Tickers by BM DTS \\%",
            )
            self._doc.add_bookmark("Small OW", level=1)
            self._add_page(
                self._small_ow_table(rating),
                caption=f"{rating}-Rated Small Overweights",
            )
            self._doc.add_bookmark("Small UW", level=1)
            self._add_page(
                self._small_uw_table(rating),
                caption=f"{rating}-Rated Small Underweights",
            )

        self._doc.save()
        self._doc.save_as("Investable_Universe", path="reports/current_reports")


if __name__ == "__main__":
    portfolio = "P-LD"
    build_investable_universe_report(portfolio)
