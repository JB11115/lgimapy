import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sms

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import DefaultRates

vis.style()

# %%


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
    mod = DefaultRates(ratings, lookbacks, db=Database())
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
