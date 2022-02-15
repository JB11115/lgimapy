from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import get_ordinal

vis.style()
# %%
db = Database()
start = pd.to_datetime("1/1/2017")

# %%
db.load_market_data(start=start)
tsy_30y = db.load_bbg_data("UST_30Y", "YTW", start=start)
ix = db.build_market_index(in_stats_index=True)
ix._create_day_cache()

# %%


def curve_10s_30s(ix):
    maturity_d = {10: (8.25, 11), 30: (25, 32)}
    maturities = sorted(list(maturity_d.keys()))
    weighting_methods_d = {
        "Issuer Weighted": "Count",
        "$ Weighted": "MarketValue",
        "Duration Weighted": "OAD_times_MV",
    }
    ratio_d = defaultdict(list)
    abs_d = defaultdict(list)
    dates = ix.dates
    for date in tqdm(dates):
        date_ix = ix.day(date, as_index=True)

        # Subset index by maturity, storing tickers in each maturity bucket.
        mat_ixs = {}
        mat_tickers = []
        for mat, mat_kws in maturity_d.items():
            mat_ix = date_ix.subset(maturity=mat_kws).issuer_index()
            mat_ix.df["Count"] = 1
            mat_ixs[mat] = mat_ix
            mat_tickers.append(set(mat_ix.tickers))

        # Keep only tickers in both maturity buckets.
        tickers = reduce(set.intersection, mat_tickers)
        ticker_matched_mat_ixs = {
            mat: mat_ix.subset(ticker=tickers)
            for mat, mat_ix in mat_ixs.items()
        }

        # First find raw values for each weighting method by maturity.
        mat_ticker_weight_raw_values_d = defaultdict(list)
        for mat, ticker_matched_mat_ix in ticker_matched_mat_ixs.items():
            ticker_matched_mat_ix.df["OAD_times_MV"] = (
                ticker_matched_mat_ix.df["OAD"]
                * ticker_matched_mat_ix.df["MarketValue"]
            )
            for weighting_method, weight_col in weighting_methods_d.items():
                mat_ticker_weight_raw_values_d[weighting_method].append(
                    ticker_matched_mat_ix.df[weight_col]
                )

        # Convert the raw values into maturity agnostic weights.
        ticker_weights_d = {}
        for (
            weighting_method,
            mat_ticker_weight_raw_values,
        ) in mat_ticker_weight_raw_values_d.items():
            raw_vals = pd.concat(mat_ticker_weight_raw_values, axis=1).sum(
                axis=1
            )
            ticker_weights_d[weighting_method] = raw_vals / raw_vals.sum()

        # Add the weights back into the ticker matched maturity indexes,
        # and compute the median OAS by weighting method.
        mat_OAS_by_weighting_method_d = defaultdict(dict)
        for mat, ticker_matched_mat_ix in ticker_matched_mat_ixs.items():
            for weighting_method, weight_col in weighting_methods_d.items():
                mat_OAS_by_weighting_method_d[mat][
                    weighting_method
                ] = ticker_matched_mat_ix.MEDIAN(
                    "OAS", weights=weight_col
                ).iloc[
                    0
                ]

        # Solve for the curve measure and added it to the data dict
        # for the current date.
        mat_OAS_by_weighting_method_d[maturities[1]]
        for weighting_method, weight_col in weighting_methods_d.items():
            val_30 = mat_OAS_by_weighting_method_d[maturities[1]][
                weighting_method
            ]
            val_10 = mat_OAS_by_weighting_method_d[maturities[0]][
                weighting_method
            ]
            ratio_d[weighting_method].append(val_30 / val_10)
            abs_d[weighting_method].append(val_30 - val_10)

    return pd.DataFrame(ratio_d, index=dates), pd.DataFrame(abs_d, index=dates)


ratio_df, abs_df = curve_10s_30s(ix)

# %%
def comp_plot(df):
    fig, ax = vis.subplots(figsize=(10, 8))
    colors = ["darkorchid", "navy", "skyblue"]
    for col, color in zip(df.columns, colors):
        curr = df[col].iloc[-1]
        pctile = df[col].rank(pct=True).iloc[-1] * 100
        vis.plot_timeseries(
            df[col],
            color=color,
            label=f"{col}: {curr:.2f} ({pctile:.0f}{get_ordinal(pctile)} %tile)",
            ax=ax,
            lw=1.5,
            median_line=True,
            median_line_kws={"color": color, "prec": 2},
        )

    vis.legend(ax)
    # vis.savefig('10s_30s_curves')


comp_plot(abs_df)
vis.show()


# %%
def comp_plot_with_treasury(s, tsy_30y):
    ax_left, ax_right = vis.plot_double_y_axis_timeseries(
        s.rename(f"10s/30s Curve\n({s.name})"),
        tsy_30y.rename("UST 30y Yield"),
        invert_right_axis=True,
        ytickfmt_right="{x:.1%}",
        figsize=(8, 5),
        color_left="navy",
        color_right="darkorchid",
        ret_axes=True,
    )


comp_plot_with_treasury(abs_df["$ Weighted"], tsy_30y)
vis.savefig("10s_30s_vs_UST_30y")


# %%
def plot_hy_ig_ratio(db, start):
    """Update plot for IG/HY ratio for cash bonds and cdx."""
    bbg_df = db.load_bbg_data(["US_HY", "US_IG"], "OAS", start=start)
    vis.plot_timeseries(
        bbg_df["US_HY"] / bbg_df["US_IG"],
        ylabel="HY/IG Ratio",
        color="navy",
        median_line=True,
        median_line_kws={"color": "firebrick", "prec": 1},
        figsize=(8, 6),
        pct_lines=(5, 95),
    )


plot_hy_ig_ratio(db, start)
vis.savefig("US_HY_IG_Ratio_sqare")

# %%
def plot_bbb_a_ratio(ix):
    ix_nonfin = ix.subset(financial_flag=0)
    ixs = {
        "10_A": ix_nonfin.subset(rating=("A+", "A-"), maturity=(8.25, 11)),
        "10_BBB": ix_nonfin.subset(
            rating=("BBB+", "BBB-"), maturity=(8.25, 11)
        ),
    }
    df = pd.concat(
        [ix.market_value_median("OAS").rename(key) for key, ix in ixs.items()],
        axis=1,
        sort=True,
    ).dropna(how="any")
    vis.plot_timeseries(
        df["10_BBB"] / df["10_A"],
        ylabel="Nonfin 10y BBB/A Ratio",
        color="navy",
        median_line=True,
        median_line_kws={"color": "firebrick", "prec": 1},
        pct_lines=(5, 95),
        legend={"loc": "upper right", "fancybox": True, "shadow": True},
    )


plot_bbb_a_ratio(ix)
vis.savefig("nonfin_10y_BBB_A_ratio")

# %%
def plot_median(s, color, ax):
    ax.axhline(np.median(s), lw=1, ls="--", color=color, label="_nolegend_")


def plot_ig_hy_spreads(db):
    df = db.load_bbg_data(["US_IG", "US_HY"], "OAS", start="2017")
    ax_left, ax_right = vis.plot_double_y_axis_timeseries(
        df["US_IG"].rename("US IG"),
        df["US_HY"].rename("US HY"),
        color_left="navy",
        color_right="darkorchid",
        alpha=0.8,
        lw=1.2,
        figsize=(8, 6),
        ret_axes=True,
    )
    plot_median(df["US_IG"], "navy", ax_left)
    plot_median(df["US_HY"], "darkorchid", ax_right)


plot_ig_hy_spreads(db)
vis.savefig("US_IG_HY_Spreads_square")
