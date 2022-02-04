from collections import defaultdict
from datetime import datetime as dt
from inspect import cleandoc
from shutil import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root, get_ordinal, mkdir


# %%
def update_cover_page(fid, db):
    """Create cover page for strategy meeting."""
    # Load market data and store indexes that will be re-used in memory.
    # %%
    vis.style()
    fig_dir = root("reports/valuation_pack/fig")

    ix_d = {}
    # db = Database()
    # db.load_market_data(start=db.date("5y"))
    ix_d["mc"] = db.build_market_index(in_stats_index=True)
    ix_d["lc"] = db.build_market_index(in_stats_index=True, maturity=(10, None))
    ix_d["10y"] = db.build_market_index(in_stats_index=True, maturity=(8, 12))
    ix_d["30y"] = db.build_market_index(in_stats_index=True, maturity=(25, 32))

    update_credit_overview(fig_dir, ix_d, save=True)
    # %%
    update_bbb_a_ratios(fig_dir, ix_d)
    update_hy_ig_ratios(fig_dir, ix_d)
    update_strategy_scores(fid, fig_dir)
    del ix_d


# %%


def save_to_global_pack(fid):
    src = root(f"reports/valuation_pack/fig/{fid}.png")
    dst_dir = root(f"reports/global_valuation_pack/fig")
    mkdir(dst_dir)
    dst = dst_dir / f"{fid}.png"
    copy(src, dst)


def update_credit_overview(fig_dir, ix_d, save=True):
    """
    Update overall view of credit plots for full market and long credit,
    including 5 year stats, last value, median, percentiles, and
    Bollinger bands. Overlay these plots with the a time series of the
    overall short term score from each strategy meeting.
    """
    db = Database()
    # Bollinger bands for market credit.
    fig, axes = vis.subplots(
        2,
        1,
        figsize=(9, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [6, 1]},
    )
    # Plot 5 year stats.
    oas = db.load_bbg_data("US_IG", "OAS", start=db.date("5y"))
    med = np.median(oas)
    pct = {x: np.percentile(oas, x) for x in [5, 95]}
    axes[0].axhline(
        med, ls="--", lw=1.5, color="firebrick", label=f"Median: {med:.0f}"
    )
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    axes[0].axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    axes[0].axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
    # Find corrected OAS data with proper volatility.
    ix_mc = ix_d["mc"].subset(start=db.date("1y"))
    ix_corrected = ix_mc.drop_ratings_migrations()
    oas_mc = ix_corrected.get_synthetic_differenced_history("OAS")

    # Get strategy meeting scoring data and combine with OAS.
    scores_df = (
        pd.read_excel(
            root("data/chicago_strategy_meeting_scores.xlsx"),
            index_col=0,
            sheet_name="Summary",
        )
        .loc["Short Term"]
        .astype(int)
    )
    df = pd.concat([oas_mc, scores_df], axis=1, sort=True)
    df["Short Term"].fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    pctile = int(np.round(100 * oas.rank(pct=True)[-1]))
    ordinal = get_ordinal(pctile)
    lbl = cleandoc(
        f"""
        Historical Stats Index
        Last: {oas[-1]:.0f} ({pctile:.0f}{ordinal} %tile)
        Range: [{np.min(oas):.0f}, {np.max(oas):.0f}]
        """
    )
    vis.plot_timeseries(
        oas, start=df.index[0], color="steelblue", label=lbl, ax=axes[0],
    )

    vis.plot_timeseries(
        df["OAS"],
        color="navy",
        bollinger=True,
        ylabel="OAS",
        ax=axes[0],
        legend=False,
        label="Corrected Index",
    )

    title = "$\\bf{5yr}$ $\\bf{Stats}$"
    axes[0].legend(loc="upper right", fancybox=True, title=title, shadow=True)
    axes[0].set_title("US Market Credit", fontweight="bold")
    # Plot short term scores below LC index.
    axes[1].plot(df["Short Term"], c="k", ls="--", lw=2)
    yticks = sorted(df["Short Term"].unique().astype(int))
    ytick_labels = [f"{v:.0f}" if v == 0 else f"{v:+.0f}" for v in yticks]
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(ytick_labels)
    axes[1].set_ylabel("Short Term\nStrategy Score", fontsize=12)
    cmap = sns.color_palette("coolwarm_r", 7).as_hex()
    plot_scores = scores_df.append(df["Short Term"].iloc[[0, -1]]).sort_index()
    fill = [np.min(plot_scores), np.max(plot_scores)]
    for i, (date, _) in enumerate(plot_scores.iloc[1:].items()):
        color_ix = int(plot_scores[i] + 3)
        axes[1].fill_betweenx(
            fill, plot_scores.index[i], date, color=cmap[color_ix], alpha=0.5
        )
    vis.format_xaxis(axes[1], df, "auto")
    plt.tight_layout()
    if save:
        vis.savefig(fig_dir / "US_MC_bollinger")
        save_to_global_pack("US_MC_bollinger")
        vis.close()
    else:
        vis.show()

    # Bollinger bands for long credit.
    fig, axes = vis.subplots(
        2,
        1,
        figsize=(9, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [7, 1]},
    )
    # Plot 5 year stats.
    oas = db.load_bbg_data("US_IG_10+", "OAS", start=db.date("5y"))
    med = np.median(oas)
    pct = {x: np.percentile(oas, x) for x in [5, 95]}
    axes[0].axhline(
        med, ls="--", lw=1.5, color="firebrick", label=f"Median: {med:.0f}"
    )
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    axes[0].axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    axes[0].axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
    # Find corrected OAS data with proper volatility.
    ix_lc = ix_d["lc"].subset(start=db.date("1y"))
    ix_corrected = ix_lc.drop_ratings_migrations()
    oas_lc = ix_corrected.get_synthetic_differenced_history("OAS")

    # Get strategy meeting scoring data and combine with OAS.
    df = pd.concat([oas_lc, scores_df], axis=1, sort=True)
    df["Short Term"].fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    pctile = int(np.round(100 * oas.rank(pct=True)[-1]))
    last_digit = pctile // 1 % 10
    ordinal = {1: "st", 2: "nd", 3: "rd"}.get(last_digit, "th")
    lbl = cleandoc(
        f"""
        Historical Stats Index
        Last: {oas[-1]:.0f} ({pctile:.0f}{ordinal} %tile)
        Range: [{np.min(oas):.0f}, {np.max(oas):.0f}]
        """
    )
    vis.plot_timeseries(
        oas, start=df.index[0], color="steelblue", label=lbl, ax=axes[0],
    )

    vis.plot_timeseries(
        df["OAS"],
        color="navy",
        bollinger=True,
        ylabel="OAS",
        ax=axes[0],
        legend=False,
        label="Corrected Index",
    )
    title = "$\\bf{5yr}$ $\\bf{Stats}$"
    axes[0].legend(loc="upper right", fancybox=True, title=title, shadow=True)
    axes[0].set_title("US Long Credit", fontweight="bold")

    # Plot short term scores below LC index.
    axes[1].plot(df["Short Term"], c="k", ls="--", lw=2)
    axes[1].set_ylabel("Short Term\nStrategy Score", fontsize=12)
    yticks = sorted(df["Short Term"].unique().astype(int))
    ytick_labels = [f"{v:.0f}" if v == 0 else f"{v:+.0f}" for v in yticks]
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(ytick_labels)
    cmap = sns.color_palette("coolwarm_r", 7).as_hex()
    plot_scores = scores_df.append(df["Short Term"].iloc[[0, -1]]).sort_index()
    fill = [np.min(plot_scores), np.max(plot_scores)]
    for i, (date, _) in enumerate(plot_scores.iloc[1:].items()):
        color_ix = int(plot_scores[i] + 3)
        axes[1].fill_betweenx(
            fill, plot_scores.index[i], date, color=cmap[color_ix], alpha=0.5
        )
    vis.format_xaxis(axes[1], df, "auto")
    plt.tight_layout()
    if save:
        vis.savefig(fig_dir / "US_LC_bollinger")
        vis.close()
    else:
        vis.show()


def update_hy_ig_ratios(fig_dir, ix_d):
    """Update plot for IG/HY ratio for cash bonds and cdx."""
    db = Database()
    bbg_df = db.load_bbg_data(
        ["US_HY", "CDX_IG", "CDX_HY"], "OAS", start=db.date("5y")
    )
    ix_mc = ix_d["mc"].subset(start=db.date("5y"))
    oas = ix_mc.market_value_median("OAS").rename("US_IG")
    df = pd.concat([bbg_df, oas], axis=1, sort=True).dropna(how="any")
    df["HY/IG Cash"] = df["US_HY"] / df["US_IG"]
    df["HY/IG CDX"] = df["CDX_HY"] / df["CDX_IG"]

    right_last = 100 * df["HY/IG CDX"].rank(pct=True).iloc[-1]
    right_label = f"CDX: {right_last:.0f}{get_ordinal(right_last)} %tile"
    left_last = 100 * df["HY/IG Cash"].rank(pct=True).iloc[-1]
    left_label = f"Cash: {left_last:.0f}{get_ordinal(left_last)} %tile"

    # Plot
    fig, ax_left = vis.subplots(figsize=(9, 6))
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
    vis.savefig(fig_dir / "HY_IG_ratio_cash_CDX")
    vis.close()


def update_bbb_a_ratios(fig_dir, ix_d):
    """Update plot for BBB/A nonfin ratio for 10y and 30y bonds."""
    # energy_sectors = [
    #     "INDEPENDENT",
    #     "REFINING",
    #     "OIL_FIELD_SERVICES",
    #     "INTEGRATED",
    #     "MIDSTREAM",
    # ]
    # ix_nonfin = ix_d["mc"].subset(
    #     financial_flag=0, sector=energy_sectors, special_rules="~Sector"
    # )
    ix_nonfin = ix_d["mc"].subset(financial_flag=0)
    ixs = {
        "30_A": ix_nonfin.subset(rating=("A+", "A-"), maturity=(25, 32)),
        "30_BBB": ix_nonfin.subset(rating=("BBB+", "BBB-"), maturity=(25, 32)),
        "10_A": ix_nonfin.subset(rating=("A+", "A-"), maturity=(8.25, 11)),
        "10_BBB": ix_nonfin.subset(
            rating=("BBB+", "BBB-"), maturity=(8.25, 11)
        ),
    }
    # ixs = {
    #     "30_A": ix_nonfin.subset(
    #         **db.index_kwargs(
    #             "A_NON_FIN_TOP_30_10+", rating=("A+", "A-"), maturity=(25, 32)
    #         )
    #     ),
    #     "30_BBB": ix_nonfin.subset(
    #         **db.index_kwargs(
    #             "BBB_NON_FIN_TOP_30_10+",
    #             rating=("BBB+", "BBB-"),
    #             maturity=(25, 32),
    #         )
    #     ),
    #     "10_A": ix_nonfin.subset(
    #         **db.index_kwargs(
    #             "A_NON_FIN_TOP_30", rating=("A+", "A-"), maturity=(8.25, 11)
    #         )
    #     ),
    #     "10_BBB": ix_nonfin.subset(
    #         **db.index_kwargs(
    #             "BBB_NON_FIN_TOP_30",
    #             rating=("BBB+", "BBB-"),
    #             maturity=(8.25, 11),
    #         )
    #     ),
    # }
    # %%
    df = pd.concat(
        [ix.market_value_median("OAS").rename(key) for key, ix in ixs.items()],
        axis=1,
        sort=True,
    ).dropna(how="any")
    df["10 yr"] = df["10_BBB"] / df["10_A"]
    df["30 yr"] = df["30_BBB"] / df["30_A"]

    # Plot
    fig, ax_left = vis.subplots(figsize=(9, 6))
    ax_right = ax_left.twinx()
    ax_right.grid(False)

    right_last = 100 * df["30 yr"].rank(pct=True).iloc[-1]
    right_label = f"30 yr: {right_last:.0f}{get_ordinal(right_last)} %tile"
    left_last = 100 * df["10 yr"].rank(pct=True).iloc[-1]
    left_label = f"10 yr: {left_last:.0f}{get_ordinal(left_last)} %tile"

    ax_left.plot(df["10 yr"], c="navy", alpha=0.9, lw=2)
    ax_right.plot(
        df["30 yr"].iloc[:2], c="navy", alpha=0.9, lw=2, label=left_label
    )
    ax_left.set_ylabel("10 yr", color="navy")
    ax_left.tick_params(axis="y", colors="navy")
    ax_left.axhline(np.median(df["10 yr"]), ls=":", lw=1.5, color="navy")

    ax_right.plot(
        df["30 yr"], c="goldenrod", alpha=0.9, lw=2, label=right_label
    )
    ax_right.axhline(
        np.median(df["30 yr"]),
        ls=":",
        lw=1.5,
        color="goldenrod",
        label="Median",
    )
    pct = {x: np.percentile(df["30 yr"], x) for x in [5, 95]}
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    ax_right.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    ax_right.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
    ax_right.set_title("Non-Fin BBB/A Ratio", fontweight="bold")
    ax_right.set_ylabel("30 yr", color="goldenrod")
    ax_right.tick_params(axis="y", colors="goldenrod")
    vis.format_xaxis(ax_right, df["30 yr"], "auto")
    vis.set_percentile_limits([df["10 yr"], df["30 yr"]], [ax_left, ax_right])
    ax_right.legend(loc="upper left", shadow=True, fancybox=True)
    vis.savefig(fig_dir / "BBB_A_nonfin_ratio")
    vis.close()
    save_to_global_pack("BBB_A_nonfin_ratio")
    # %%


# %%


def strategy_scores_plot(df, ax, cmap):
    """Build strategy scoring table from heatmap."""
    background = "lightgrey"
    sns.heatmap(
        df,
        cmap=cmap,
        alpha=1,
        linewidths=2,
        linecolor=background,
        cbar=False,
        ax=ax,
    )
    ax.xaxis.tick_top()
    ax.set_xticklabels("-3 -2 -1 0 +1 +2 +3".split(), fontsize=10)
    ax.set_yticklabels(df.index, fontsize=10)
    # b, t = ax.get_ylim()
    # ax.set_ylim(b + 0.5, t - 0.5)
    # y_loc = mpl.ticker.FixedLocator(np.arange(t, b + 1))
    # ax.yaxis.set_major_locator(y_loc)


def update_strategy_scores(fid, fig_dir):
    # %%
    vis.style(background="lightgrey")
    fig, axes = vis.subplots(
        2, 1, figsize=(4, 12), gridspec_kw={"height_ratios": [1, 3]}
    )
    cmap = mpl.colors.ListedColormap(["w", "skyblue", "navy"])

    # Overall Scores
    scores_df = (
        pd.read_excel(
            root("data/chicago_strategy_meeting_scores.xlsx"),
            index_col=0,
            sheet_name="Summary",
        )
        .iloc[:-3, -2:]
        .dropna(how="all")
        .fillna(0)
        .astype(int)
    )

    # Add short term score to page.
    score = scores_df.loc["Short Term"][-1]
    if score > 0:
        text = f"\\Huge \\textbf{{\\textcolor{{steelblue}}{{+{score}}}}}"
    elif score < 0:
        text = f"\\Huge \\textbf{{\\textcolor{{firebrick}}{{{score}}}}}"
    else:
        text = f"\\Huge \\textbf{{{score}}}"

    doc = Document(fid, path="reports/valuation_pack", load_tex=True)
    doc.start_edit("short_term_score")
    doc.add_text(text)
    doc.end_edit()
    doc.save_tex()

    # Bold Short and Long Term scores.
    scores_df.index = [
        f"$\\bf{{{ix}}}$" if "Term" in ix else ix for ix in scores_df.index
    ]
    # Create table for plotting.
    df = pd.DataFrame(np.zeros([len(scores_df), 7]), index=scores_df.index)
    df.columns = np.arange(-3, 4)
    for col in [0, 1]:
        for name, score in scores_df.iloc[:, col].items():
            df.loc[name, score] = col + 1

    strategy_scores_plot(df, axes[0], cmap)
    axes[0].set_title("Strategy Scoring\n", fontsize=15, fontweight="bold")

    # Load and clean individual scores, recording averages.
    scores_df = (
        pd.read_excel(
            root("data/chicago_strategy_meeting_scores.xlsx"),
            index_col=0,
            sheet_name="3 Months",
        )
        .iloc[:, -2:]
        .dropna(how="all")
    )

    # Drop non-credit scores.
    scores_df = scores_df[["(" not in name for name in scores_df.index]]
    avg_scores = np.mean(scores_df)
    # Drop people with no new scores.
    prev_date, curr_date = scores_df.columns
    scores_df = scores_df[~scores_df[curr_date].isna()]
    # If prev score is missing, fill it with current score for chart.
    scores_df.fillna(method="bfill", axis=1, inplace=True)

    # Create table for plotting.
    df = pd.DataFrame(np.zeros([len(scores_df), 7]), index=scores_df.index)
    df.columns = np.arange(-3, 4)
    for col in [0, 1]:
        for name, score in scores_df.iloc[:, col].items():
            df.loc[name, score] = col + 1

    strategy_scores_plot(df, axes[1], cmap)
    labels = [
        f"{col.strftime('%b')}\n{avg_scores[col]:.1f}"
        for col in scores_df.columns
    ]

    legend_elements = [
        mpl.patches.Patch(facecolor="lightgrey", label="Avg:"),
        mpl.patches.Patch(facecolor="navy", label=labels[1]),
        mpl.patches.Patch(facecolor="skyblue", label=labels[0]),
    ]
    axes[1].legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.2, -0.02),
        ncol=3,
        fontsize=14,
        frameon=False,
    )

    plt.tight_layout()
    vis.savefig(fig_dir / "strategy_scores")
    vis.close()
    vis.style()
    # %%


# %%

if __name__ == "__main__":
    fid = "first_draft"
    update_cover_page(fid)
