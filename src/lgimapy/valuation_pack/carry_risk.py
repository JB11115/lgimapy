from collections import defaultdict
from datetime import datetime as dt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

from lgimapy import vis
from lgimapy.bloomberg import bdh, bdp, bds
from lgimapy.data import Database, Index
from lgimapy.models import find_drawdowns
from lgimapy.utils import load_json, root, Time

# %%


def main():
    # Load data since 2007 in memory for subsequent analysis.
    db = Database()

    # Store necessary metrics from US IG index for subsequent analysis.


# %%
class CarryRisk:
    """
    Estimate how much risk you need to take to generate
    1 bp of carry.

    Parameters
    ----------
    index: ``{'US_IG'}``
        Index to add risk to.
    long_term_start: datetime, default='2007-01-01'
        Start date for long term trend analysis.
    short_term_start: datetime, default= `2 years from current date`
        Start date for short term trend anlaysis.
    """

    def __init__(
        self,
        index,
        long_term_start="2007-01-01",
        short_term_start=Database().date("2y"),
    ):
        self.db = Database()
        self.lt_start = pd.to_datetime(long_term_start)
        self.st_start = pd.to_datetime(short_term_start)
        self.index = index
        self.compute_index_risk_metrics()
        self._d = defaultdict(list)
        self.risk_sources = {
            "us_hy": "US HY",
            "us_hy_1-3": "US HY 1-3 yr",
            "duration": "Duration",
            "wide": "Wide Sectors",
            "bbb": "BBB's",
            "cdx_ig": "CDX IG",
            "em": "EM",
        }

    def compute_index_risk_metrics(self):
        """
        Calculate DTS and get the total return series for
        specified index.
        """
        self.db.load_market_data(local=True)
        kwargs = {"US_IG": {"in_stats_index": True}}[self.index]
        ix = self.db.build_market_index(**kwargs)

        # Calculate current index spread level.
        self.ix_spread = ix.market_value_weight("OAS")[0]

        # Calculate current index DTS.
        ix.df.eval("DTS = OASD * OAS", inplace=True)
        self.ix_dts = ix.market_value_weight("DTS")[0]

        # Get total return series for index.
        if self.index == "US_IG":
            self.ix_tret_levels = self.db.load_bbg_data(
                "US_IG", "TRet", start=self.lt_start
            )
            self.ix_tret = np.log(self.ix_tret_levels).diff()[1:]

    def beta(self, df, start):
        df_beta = df[df.index >= start].copy()
        ix_tret = np.log(df_beta[self.index]).diff()[1:]
        rs_tret = np.log(df_beta["risk_source"]).diff()[1:]
        port_tret = weight * rs_tret + (1 - weight) * ix_tret
        beta, alpha, r2, p_value, std_err = linregress(ix_tret, port_tret)
        return beta - 1 if p_value < 0.05 else 0

    def compute_risk_metrics(self, rs_spread, rs_tret_levels):
        df_levels = pd.concat(
            [
                self.ix_tret_levels.rename(self.index),
                rs_tret_levels.rename("risk_source"),
            ],
            axis=1,
            sort=True,
        ).dropna()
        df_tret = np.log(df_levels).diff()[1:]
        weight = 1 / (rs_spread - self.ix_spread)
        df_tret["port"] = (
            weight * df_tret["risk_source"] + (1 - weight) * df_tret[self.index]
        )

        self._d["Weight"].append(weight)
        self._d["Spread"].append(rs_spread)
        self._d["Beta (2 years)"].append(self.beta(df, self.st_start))
        self._d["Beta (since 2007)"].append(self.beta(df, self.lt_start))

    def get_hy_risk_measures(self):
        """
        Get risk metrics for high yield.
        """

        hy_ix = self.db.build_market_index(in_hy_stats_index=True)
        # OAS.
        oas = hy_ix.market_value_weight("OAS")[0]
        # DTS.
        hy_ix.df.eval("DTS = OASD * OAS", inplace=True)
        dts = hy_ix.market_value_weight("DTS")[0]
        # Total Returns
        tret_levels = self.db.load_bbg_data("US_HY", "TRet", start=self.start)
        tret = np.log(tret_levels).diff()[1:]

        rs_spread = oas
        rs_dts = dts
        rs_tret_levels = tret_levels
        rs_tret = tret


self = CarryRisk("US_IG")

# %%


def find_missing_cds_spread(bb_number):
    """
    Find spread for CDS contracts with missing values
    using upfront price.
    """
    ykey = "MSG1 Curncy"
    quoted_price = 100 - bdp(bb_number, ykey, "UPFRONT_LAST").iloc[0, 0]
    maturity = bdp(bb_number, ykey, "MATURITY").iloc[0, 0]
    return bdp(
        "SP950JFJ",
        "Corp",
        fields="CDS REPL",
        ovrd={
            "CDS_QUOTED_PRICE": quoted_price,
            "MATURITY": maturity.strftime("%Y%m%d"),
        },
    ).iloc[0, 0]


def get_cdx_ig_dts():
    """float: Calculate current CDX IG DTS."""
    cdx_code = "CDX IG CDSI GEN 5Y"
    ykey = "MSG1 Curncy"
    spread = "5Y_MID_CDS_SPREAD"
    duration = "SW_CNV_RISK"

    # Scrape CDX IG constituents spreads and durations.
    cdx_members_df = bds(cdx_code, "Corp", "INDX_MEMBERS")
    cdx_members_df.columns = ["_", "_", "_", "_", "BB_numbers", "_"]
    cdx_members = cdx_members_df["BB_numbers"].values
    cdx_spreads = bdp(cdx_members, ykey, spread)
    cdx_durations = np.abs(bdp(cdx_members, ykey, duration))
    cdx_df = pd.concat([cdx_spreads, cdx_durations], axis=1, sort=False)

    # Impute missing spreads.
    if any(np.isnan(cdx_df[spread])):
        missing_bb_numbers = cdx_df[cdx_df[spread].isna()].index
        for bb_number in missing_bb_numbers:
            cdx_df.loc[bb_number, spread] = find_missing_cds_spread(bb_number)

    # Warn if there is missing data.
    if cdx_df.isna().sum().sum():
        print("WARNING: Missing Data in CDX IG DTS Calculation.")

    cdx_df["DTS"] = cdx_df[duration] * cdx_df[spread]
    dts = np.sum(cdx_df["DTS"]) / len(cdx_df)
    return dts


def get_hy_risk_measures(data):
    """
    Find change in risk measures for adding HY into
    the IG portfolio to increase carry by 1bp.
    """
    res = {}
    db = Database()

    # Get IG return history and current spread and DTS.
    ig_ret_levels = db.load_bbg_data("US_IG", "TRet", start="1/1/2007")
    db.load_market_data(local=True)
    ig_ix = db.build_market_index(in_stats_index=True)
    ig_ix.df.eval("DTS = OASD * OAS", inplace=True)
    ig_dts = ig_ix.market_value_weight("DTS").iloc[0, 0]

    hy_oas_hist = data.subset(in_hy_stats_index=True).market_value_weight("OAS")
    ig_spread = ig_oas_hist[-1]
    hy_spread = hy_oas_hist[-1]
    res["spread"] = hy_spread

    # Find current weighting for 1bp of carry.
    weight = 1 / (hy_spread - ig_spread)
    res["weight"] = weight

    # Find beta since 2007 and for past 2 years.
    port_oas_hist = weight * hy_oas_hist + (1 - weight) * ig_oas_hist
    beta, alpha, *stats = linregress(ig_oas_hist, port_oas_hist)
    res["beta_2007"] = beta - 1
    two_years_ago = db.date("2y")
    ig_oas_2y = ig_oas_hist[ig_oas_hist.index >= two_years_ago]
    port_oas_2y = port_oas_hist[port_oas_hist.index >= two_years_ago]
    beta, alpha, *stats = linregress(ig_oas_2y, port_oas_2y)
    res["beta_2y"] = beta - 1


# %%
import string

import numpy as np
import pandas as pd

from lgimapy.utils import Time

n = 26
letters = list(string.ascii_letters[:n])
letter_map = {i: letter for i, letter in enumerate(letters)}

randints = np.random.randint(0, n, size=100_000)
df = pd.DataFrame(randints)
df.columns = ["randints"]
df["rand2"] = np.random.randint(0, n, size=100_000)
df["rand3"] = np.random.randint(0, n, size=100_000)


temp = np.vectorize(letter_map.__getitem__)(df)
pd.DataFrame(temp).head()


temp = df[df.columns].map(letter_map)

help(df["randints"].map)

iters = 10
with Time() as t:
    name = "map"
    for __ in range(iters):
        temp = df["randints"].map(letter_map)
    t.split(name)
    name = "list comp"
    for __ in range(iters):
        temp = [letter_map[v] for v in df["randints"]]
    t.split(name)
    name = "numpy vect"
    for __ in range(iters):
        temp = np.vectorize(letter_map.__getitem__)(df)

    t.split(name)

df
