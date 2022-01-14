from collections import defaultdict
from inspect import cleandoc

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database, Index
from lgimapy.latex import Document
from lgimapy.utils import root, get_ordinal


# %%
def update_cover_page(fid, db):
    """Create cover page for strategy meeting."""
    # %%
    vis.style()

    # %%
    fid = "HY_Cover_Page"
    doc = Document(fid, path="reports/HY", fig_dir=True, load_tex=fid)
    plot_corrected_H0A0_oas(doc, db)
    plot_index_oas("H4UN", doc)
    plot_hy_ig_ratios(doc, db)
    plot_negative_convexity(doc, db)
    doc.save(save_tex=True)

    # %%


# %%
def plot_hy_ig_ratios(doc, db):
    """Update plot for IG/HY ratio for cash bonds and cdx."""
    ix_ig = db.build_market_index(in_stats_index=True)
    ig_oas = ix_ig.market_value_median("OAS").rename("US_IG")
    bbg_df = db.load_bbg_data(
        ["US_HY", "CDX_IG", "CDX_HY"], "OAS", start=db.date("5y")
    )
    df = pd.concat([bbg_df, ig_oas], axis=1, sort=True).dropna(how="any")
    df["HY/IG Cash"] = df["US_HY"] / df["US_IG"]
    df["HY/IG CDX"] = df["CDX_HY"] / df["CDX_IG"]

    right_last = 100 * df["HY/IG CDX"].rank(pct=True).iloc[-1]
    right_label = f"CDX: {right_last:.0f}{get_ordinal(right_last)} %tile"
    left_last = 100 * df["HY/IG Cash"].rank(pct=True).iloc[-1]
    left_label = f"Cash: {left_last:.0f}{get_ordinal(left_last)} %tile"

    # Plot
    fig, ax_left = vis.subplots(figsize=(10, 5.8))
    ax_right = ax_left.twinx()
    ax_right.grid(False)

    ax_left.plot(df["HY/IG Cash"], c="navy", alpha=0.9, lw=2)
    ax_right.plot(
        df["HY/IG Cash"].iloc[:2], c="navy", alpha=0.9, lw=2, label=left_label
    )
    ax_left.set_ylabel("Cash", color="navy")
    ax_left.tick_params(axis="y", colors="navy")
    ax_left.axhline(np.median(df["HY/IG Cash"]), ls=":", lw=1.5, color="navy")

    ax_right.plot(
        df["HY/IG CDX"], c="goldenrod", alpha=0.9, lw=2, label=right_label
    )
    ax_right.axhline(
        np.median(df["HY/IG CDX"]),
        ls=":",
        lw=1.5,
        color="goldenrod",
        label="Median",
    )
    pct = {x: np.percentile(df["HY/IG CDX"], x) for x in [5, 95]}
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    ax_right.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    ax_right.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
    ax_right.set_title("HY/IG Ratios", fontweight="bold")
    ax_right.set_ylabel("CDX", color="goldenrod")
    ax_right.tick_params(axis="y", colors="goldenrod")
    vis.format_xaxis(ax_right, df["HY/IG CDX"], "auto")
    vis.set_percentile_limits(
        [df["HY/IG Cash"], df["HY/IG CDX"]], [ax_left, ax_right]
    )
    ax_right.legend(loc="upper left", shadow=True, fancybox=True)
    vis.savefig(doc.fig_dir / "HY_IG_ratio_cash_CDX")
    vis.close()


def plot_index_oas(index, doc):
    db = Database()
    fig, ax = vis.subplots(figsize=(10, 5.8))
    oas = db.load_baml_data(index, "OAS", start=db.date("5y")).dropna()
    med = np.median(oas)
    pct = {x: np.percentile(oas, x) for x in [5, 95]}

    pctile = int(np.round(100 * oas.rank(pct=True)[-1]))
    ordinal = get_ordinal(pctile)
    lbl = cleandoc(
        f"""
        {index} Index
        Last: {oas.iloc[-1]:.0f} bp ({pctile:.0f}{ordinal} %tile)
        Range: [{np.min(oas):.0f}, {np.max(oas):.0f}]
        """
    )
    oas_1y = oas[oas.index >= db.date("1y")]
    vis.plot_timeseries(
        oas_1y,
        color="navy",
        bollinger=True,
        title=index,
        ylabel="OAS",
        ax=ax,
        label=lbl,
    )
    ax.axhline(
        med, ls="--", lw=1.5, color="firebrick", label=f"Median: {med:.0f}"
    )
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    ax.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    ax.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)

    title = "$\\bf{5yr}$ $\\bf{Stats}$"
    ax.legend(loc="upper right", fancybox=True, title=title, shadow=True)
    vis.savefig(doc.fig_dir / f"cover_{index}")
    vis.close()


def get_label(s):
    last_val = s.iloc[-1]
    last_pct = 100 * s.rank(pct=True).iloc[-1]
    ord = get_ordinal(last_pct)
    return f"\n  {last_val:.0f} bp ({last_pct:.0f}{ord} %tile)"


def one_yr(s, db):
    return s[s.index >= db.date("1y")]


def plot_corrected_H0A0_oas(doc, db):
    rating_kwargs = {
        "BB": ("BB+", "BB-"),
        "B": ("B+", "B-"),
        "C": ("CCC+", "C"),
    }
    ix = db.build_market_index(in_H0A0_index=True)
    ix_oas_5y = ix.OAS()
    ix_oas_median = np.median(ix_oas_5y)
    ix_oas_1y = one_yr(ix_oas_5y, db)

    rating_oas_1y = {}
    rating_oas_5y = {}
    rating_oas_median = {}
    rating_mv_nominal = {}
    for rating, rating_kws in rating_kwargs.items():
        rating_ix = ix.subset(rating=rating_kws)
        rating_oas_5y[rating] = rating_ix.OAS().rename(rating)
        rating_oas_1y[rating] = one_yr(rating_oas_5y[rating], db)
        rating_oas_median[rating] = np.median(rating_oas_5y[rating])
        rating_mv_nominal[rating] = rating_ix.total_value().iloc[-1]

    ix_mv = sum(list(rating_mv_nominal.values()))
    rating_mv = {k: v / ix_mv for k, v in rating_mv_nominal.items()}

    hy_oas_df = pd.concat(
        (rating_oas_5y[rating] * rating_mv[rating] for rating in rating_kwargs),
        axis=1,
    )
    corrected_ix_oas_5y = hy_oas_df.sum(axis=1)
    corrected_ix_oas_5y -= corrected_ix_oas_5y.iloc[-1] - ix_oas_5y.iloc[-1]
    corrected_ix_oas_median = np.median(corrected_ix_oas_5y)
    corrected_ix_oas_1y = one_yr(corrected_ix_oas_5y, db)

    fig, (ax_top, ax_bottom) = vis.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 4]},
        figsize=(10, 5.8),
    )
    lw = 2
    vis.plot_timeseries(
        ix_oas_1y,
        lw=lw,
        color="grey",
        label=f"Historical H0A0: {get_label(ix_oas_5y)}",
        median_line=ix_oas_median,
        median_line_kws={"color": "grey", "alpha": 0.8, "label": "_nolegend_"},
        ax=ax_bottom,
    )
    vis.plot_timeseries(
        corrected_ix_oas_1y,
        color="k",
        lw=lw,
        label=f"Corrected H0A0: {get_label(corrected_ix_oas_5y)}",
        median_line=corrected_ix_oas_median,
        median_line_kws={"color": "k", "alpha": 0.8, "label": "_nolegend_"},
        ax=ax_bottom,
    )

    colors = dict(zip(rating_kwargs.keys(), vis.colors("ryb")))
    for rating, rating_kws in rating_kwargs.items():
        c = colors[rating]
        vis.plot_timeseries(
            rating_oas_1y[rating],
            color=c,
            lw=lw,
            median_line=rating_oas_median[rating],
            median_line_kws={
                "color": c,
                "alpha": 0.8,
                "label": "5yr median" if rating == "C" else "_nolegend_",
            },
            label=f"{rating}-Rated: {get_label(rating_oas_5y[rating])}",
            title="HY Spreads" if rating == "C" else None,
            ax=ax_top if rating == "C" else ax_bottom,
        )
    title = "$\\bf{5yr}$ $\\bf{Stats}$"
    for ax in [ax_top, ax_bottom]:
        ax.legend(
            title=title,
            fancybox=True,
            shadow=True,
            loc=2,
            bbox_to_anchor=(1, 1),
        )

    fid = "HY_spreads_cover_page"
    vis.savefig(doc.fig_dir / fid)
    vis.close()


def plot_negative_convexity(doc, db):
    ix = db.build_market_index(in_H0A0_index=True)
    rating_kwargs = {
        "BB": ("BB+", "BB-"),
        "B": ("B+", "B-"),
        "CCC": ("CCC+", "CCC-"),
    }
    colors = dict(zip(rating_kwargs.keys(), vis.colors("ryb")))
    fig, ax = vis.subplots(figsize=(10, 5.8))
    for rating, rating_kws in rating_kwargs.items():
        rating_ix = db.build_market_index(rating=rating_kws)
        mv = rating_ix.total_value()
        df = rating_ix.df.copy()
        df["CallPrice"] = 100 + df["CouponRate"] / 2
        ix_neg_conv = Index(df[df["DirtyPrice"] >= df["CallPrice"]])
        neg_conv = ix_neg_conv.total_value() / mv
        c = colors[rating]
        vis.plot_timeseries(
            neg_conv,
            color=c,
            lw=2,
            # alpha=0.5,
            median_line=True,
            median_line_kws={"color": c, "alpha": 0.8, "label": "_nolegend_"},
            label=f"{rating}: {neg_conv.iloc[-1]:.0%}",
            ax=ax,
            title="Bonds Trading Above First Call\n",
        )
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    ax.legend(
        fancybox=True,
        shadow=True,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.06),
    )
    fid = "bonds_trading_above_call_price"
    vis.savefig(doc.fig_dir / fid)
    vis.close()


if __name__ == "__main__":
    fid = "HY_Cover_Page"
    db = Database()
    db.load_market_data(start=db.date("5y"))
    update_cover_page(fid, db)
