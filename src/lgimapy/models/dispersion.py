from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.latex import Document
from lgimapy.stats import IQR, QCV, MAD, RMAD
from lgimapy.utils import root, mkdir


from lgimapy.data import Database
from tqdm import tqdm

# %%


class Dispersion:
    def __init__(self, asset_class, db):
        self.asset_class = asset_class.upper()
        if self.asset_class not in {"IG", "HY"}:
            raise ValueError(f"`asset_class` must be either 'IG' or 'HY'")
        self._db = db

    @property
    @lru_cache(maxsize=None)
    def sectors(self):
        return {
            "IG": [
                "BASICS",
                "CAPITAL_GOODS",
                "COMMUNICATIONS",
                "AUTOMOTIVE",
                "RETAILERS",
                "FOOD_AND_BEVERAGE",
                "HEALTHCARE_PHARMA",
                "TOBACCO",
                "ENERGY",
                "TECHNOLOGY",
                "TRANSPORTATION",
                "US_BANKS",
                "YANKEE_BANKS",
                "FINANCIALS_EX_BANKS",
                "UTILITY",
                "HOSPITALS",
                "MUNIS",
                "SOVEREIGN",
            ],
        }[self.asset_class]

    @property
    def _rating_buckets(self):
        return {"IG": ["A", "BBB"], "HY": ["BB", "B"]}[self.asset_class]

    @property
    def _maturity_buckets(self):
        return {"IG": [10, 30], "HY": [None]}[self.asset_class]

    @property
    @lru_cache(maxsize=None)
    def _dir(self):
        dispersion_dir = {
            "IG": root("data/US/dispersion"),
            "HY": root("data/HY/dispersion"),
        }[self.asset_class]
        mkdir(dispersion_dir)
        return dispersion_dir

    def _intra_sector_fid(self, rating, maturity):
        return (
            self._dir / f"{rating}_{maturity}_intra_sector_dispersion.parquet"
        )

    def _inter_sector_fid(self, rating, maturity):
        return (
            self._dir / f"{rating}_{maturity}_inter_sector_dispersion.parquet"
        )

    @property
    def _trade_dates_fid(self):
        return self._dir / "dispersion_trade_dates.parquet"

    def _load(self, fid):
        try:
            return pd.read_parquet(fid)
        except FileNotFoundError:
            return None

    def _dates_to_analyze(self):
        analyzed_dates = self._load(self._trade_dates_fid)
        if analyzed_dates is None:
            return self._db.trade_dates(start="1/1/2000")
        else:
            last_analyzed_date = analyzed_dates["Date"].iloc[-1]
            return self._db.trade_dates(exclusive_start=last_analyzed_date)

    def update(self):
        """Update all data files for current asset class."""

        for date in tqdm(self._dates_to_analyze()):
            self._date = date
            self._db.load_market_data(date=date)
            ix = self._db.build_market_index(in_stats_index=True)
            for rating in self._rating_buckets:
                for maturity in self._maturity_buckets:
                    self._update_intra_sector_rating_files(rating, maturity, ix)
                    self._update_inter_sector_rating_files(rating, maturity, ix)
            self._update_trade_dates()

    def _rating_kws(self, rating):
        return {
            "A": (None, "A-"),
            "BBB": ("BBB+", "BBB-"),
            "BB": ("BB+", "BB-"),
            "B": ("B+", "B-"),
        }[rating]

    def _maturity_kws(self, maturity):
        return {
            None: (None, None),
            10: (8.25, 11),
            30: (25, 32),
        }[maturity]

    def _update_intra_sector_rating_files(self, rating, maturity, ix):
        d = {}
        d["OAS"] = ix.OAS().iloc[-1]
        for sector in self.sectors:
            sector_ix = ix.subset(
                **self._db.index_kwargs(
                    sector,
                    rating=self._rating_kws(rating),
                    maturity=self._maturity_kws(maturity),
                )
            )
            issuer_df = sector_ix.issuer_df.dropna(
                subset=["OAS", "MarketValue"]
            )
            n = len(issuer_df)
            if n <= 2:
                continue
            d[f"{sector}_n"] = n
            d[f"{sector}_IQR_plus_MAD"] = IQR(
                issuer_df["OAS"], weights=issuer_df["MarketValue"]
            ) + MAD(issuer_df["OAS"], weights=issuer_df["MarketValue"])
            d[f"{sector}_QCV_plus_RMAD"] = QCV(
                issuer_df["OAS"], weights=issuer_df["MarketValue"]
            ) + RMAD(issuer_df["OAS"], weights=issuer_df["MarketValue"])

        new_row = pd.Series(d, name=self._date).to_frame().T
        fid = self._intra_sector_fid(rating, maturity)
        old_df = self._load(fid)
        if old_df is None:
            new_row.to_parquet(fid)
        else:
            old_df = old_df[old_df.index != self._date]
            updated_df = pd.concat((old_df, new_row))
            updated_df.to_parquet(fid)

    def _update_inter_sector_rating_files(self, rating, maturity, ix):
        d = defaultdict(list)
        for sector in self.sectors:
            sector_ix = ix.subset(
                **self._db.index_kwargs(
                    sector,
                    rating=self._rating_kws(rating),
                    maturity=self._maturity_kws(maturity),
                )
            )
            if len(sector_ix.df):
                d["OAS"].append(sector_ix.OAS().iloc[-1])
                d["MarketValue"].append(sector_ix.total_value().iloc[-1])

        df = pd.DataFrame(d)
        new_row = (
            pd.Series(
                {
                    "OAS": ix.OAS().iloc[-1],
                    "IQR_plus_MAD": IQR(df["OAS"], weights=df["MarketValue"])
                    + MAD(df["OAS"], weights=df["MarketValue"]),
                    "QCV_plus_RMAD": QCV(df["OAS"], weights=df["MarketValue"])
                    + RMAD(df["OAS"], weights=df["MarketValue"]),
                },
                name=self._date,
            )
            .to_frame()
            .T
        )
        fid = self._inter_sector_fid(rating, maturity)
        old_df = self._load(fid)
        if old_df is None:
            new_row.to_parquet(fid)
        else:
            old_df = old_df[old_df.index != self._date]
            updated_df = pd.concat((old_df, new_row))
            updated_df.to_parquet(fid)

    def _update_trade_dates(self):
        fid = self._trade_dates_fid
        trade_dates = self._load(fid)
        new_row = pd.DataFrame({"Date": [self._date]})
        if trade_dates is None:
            new_row.to_parquet(fid)
        else:
            old_df = trade_dates[trade_dates["Date"] != self._date]
            updated_df = pd.concat((old_df, new_row)).sort_values("Date")
            updated_df.to_parquet(fid)

    def _clean_table_index(self, s):
        s.index = [idx.split("_IQR_plus_MAD")[0] for idx in s.index]
        s.index = [idx.split("_QCV_plus_RMAD")[0] for idx in s.index]
        s.index = [idx.split("_n")[0] for idx in s.index]
        return s

    def _date(self, raw_date):
        if raw_date is None:
            return self._db.date("today")
        else:
            return pd.to_datetime(raw_date)

    def _intra_sector_table(self, rating, maturity, date=None):
        date = self._date(date)
        table = pd.DataFrame(index=self.sectors)
        df = self._load(self._intra_sector_fid(rating, maturity))
        pctile_df = df.rank(pct=True)
        curr_oas_pctile = pctile_df["OAS"].loc[date]
        curr_regime_pctile_df = df[
            (pctile_df["OAS"] >= curr_oas_pctile - 0.1)
            & (pctile_df["OAS"] <= curr_oas_pctile + 0.1)
        ].rank(pct=True)

        n_issuer_cols = [col for col in df.columns if col.endswith("_n")]
        table[f"{rating}-Rated*# Issuers"] = self._clean_table_index(
            df[n_issuer_cols].loc[date]
        )
        abs_cols = [col for col in df.columns if col.endswith("_MAD")]
        table[f"{rating} Abs*Dispersion"] = self._clean_table_index(
            pctile_df[abs_cols].loc[date]
        )

        rel_cols = [col for col in df.columns if col.endswith("_RMAD")]
        table[f"{rating} Rel*Dispersion"] = self._clean_table_index(
            curr_regime_pctile_df[rel_cols].loc[date]
        )
        table.loc[
            table[f"{rating} Abs*Dispersion"].isna(), f"{rating} Rel*Dispersion"
        ] = np.nan

        table.index = [
            self._db.index_kwargs(sector)["name"] for sector in table.index
        ]
        return table

    def intra_sector_table(self, maturity, date=None):
        table = pd.concat(
            (
                self._intra_sector_table(rating, maturity, date)
                for rating in self._rating_buckets
            ),
            axis=1,
        ).dropna(how="all")
        for col in table.columns:
            if "Dispersion" in col:
                table[col] *= 100
        return table

    def _inter_sector_table(self, rating, maturity, date=None):
        date = self._date(date)
        df = self._load(self._inter_sector_fid(rating, maturity))
        pctile_df = df.rank(pct=True)
        curr_oas_pctile = pctile_df["OAS"].loc[date]
        curr_regime_pctile_df = df[
            (pctile_df["OAS"] >= curr_oas_pctile - 0.1)
            & (pctile_df["OAS"] <= curr_oas_pctile + 0.1)
        ].rank(pct=True)
        d = {
            "Inter-Sector Absolute": pctile_df.rank(pct=True)[
                "IQR_plus_MAD"
            ].loc[date],
            "Inter-Sector Relative": curr_regime_pctile_df.rank(pct=True)[
                "QCV_plus_RMAD"
            ].loc[date],
        }
        return pd.Series(d, name=f"{rating}-Rated")

    def overview_table(self, maturity, date=None):
        inter_df = 100 * pd.concat(
            (
                self._inter_sector_table(rating, maturity, date)
                for rating in self._rating_buckets
            ),
            axis=1,
        )

        intra_data_s = self.intra_sector_table(maturity, date).mean()
        intra_data_d = defaultdict(list)
        for rating in self._rating_buckets:
            col = f"{rating}-Rated"
            intra_data_d[col].append(intra_data_s[f"{rating} Abs*Dispersion"])
            intra_data_d[col].append(intra_data_s[f"{rating} Rel*Dispersion"])

        intra_df = pd.DataFrame(
            intra_data_d,
            index=[
                "Intra-Sector Absolute",
                "Intra-Sector Relative",
            ],
        )

        return pd.concat((inter_df, intra_df)).reindex(
            [
                "Inter-Sector Absolute",
                "Intra-Sector Absolute",
                "Inter-Sector Relative",
                "Intra-Sector Relative",
            ]
        )


# #
# db = Database()
# self = Dispersion("IG", db)
# self._date(None)
# self.update()
# maturity = 10
# rating = "A"
# self.overview_table(10)
# self.intra_sector_table(10)

# %%
# self._load(self._inter_sector_fid("A", 30))
# self._load(self._inter_sector_fid("A", 30)).isna().sum().sum()
# self._load(self._inter_sector_fid("BBB", 10)).isna().sum().sum()
#
# print(self._load(self._trade_dates_fid)["Date"].iloc[-1].strftime("%m/%d/%Y"))
#
# date = self._db.date("today")
# self.overview_table(maturity=30).round(2)
# # %%
# self.intra_sector_table(maturity=30)
#
# # %%
# maturity = 10
#
# # %%
# start = "11/23/16"
# oas = db.load_bbg_data("US_IG", "OAS", start=start)
# inter_a_10_list = []
# inter_b_10_list = []
# dates = []
# trade_dates = db.trade_dates(exclusive_start=start)
# for date in tqdm(trade_dates):
#     try:
#         table = self.overview_table(10, date=date)
#     except KeyError:
#         continue
#     dates.append(date)
#     inter_a_10_list.append(table.loc["Intra-Sector Absolute", "A-Rated"])
#     inter_b_10_list.append(table.loc["Intra-Sector Absolute", "BBB-Rated"])
#
# inter_a = pd.Series(inter_a_10_list, index=dates).to_frame()
# inter_b = pd.Series(inter_b_10_list, index=dates).to_frame()
#
# inter_a_smooth = inter_a.squeeze().rolling(5).mean().dropna()
# inter_b_smooth = inter_b.squeeze().rolling(5).mean().dropna()
#
# # %%
#
# inter_a_smooth = inter_a.squeeze().rolling(5).mean().dropna()
# inter_b_smooth = inter_b.squeeze().rolling(5).mean().dropna()
# vis.style()
#
# ax_left, ax_right = vis.plot_double_y_axis_timeseries(
#     inter_a_smooth.rename("Dispersion"),
#     oas.rename("US Credit OAS"),
#     color_left="navy",
#     color_right="k",
#     ret_axes=True,
#     ytickfmt_left="{x:.0%}",
#     plot_kws_right={"lw": 4},
# )
# vis.plot_timeseries(inter_b_smooth, color="darkorange", ax=ax_left)
#
# vis.show()
# vis.savefig("A-rated_10yr_inter-sector_dispersion")
#
# x.to_frame().to_csv("A-rated_10yr_inter-sector_dispersion.csv")
