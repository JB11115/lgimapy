from pathlib import Path
from inspect import cleandoc

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database, Index
from lgimapy.latex import Document
from lgimapy.utils import root, Time, get_ordinal

vis.style()
# %%


def update_EUR_credit_overview():
    ix_d = load_data()
    doc = Document(
        "global_valuation_pack",
        path="reports/global_valuation_pack",
        fig_dir=True,
    )
    update_bbb_a_ratios(doc, ix_d, "darkgreen")
    update_EUR_spreads(doc, "darkgreen", save=True)


def load_data():
    db = Database(market="EUR")

    db.load_market_data(start=db.date("5y"))
    ix = db.build_market_index()

    ix_d = {}
    ix_d["mc"] = ix.copy()
    ix_d["lc"] = ix.subset(maturity=(10, None))
    ix_d["10y"] = ix.subset(maturity=(8, 12))
    ix_d["30y"] = ix.subset(maturity=(25, 32))
    return ix_d


# govt_bond_tickers = [
#     "BGB",
#     "BKO",
#     "BTPS",
#     "DBR",
#     "EFSF",
#     "EIB",
#     "ESM",
#     "EU",
#     "FRTR",
#     "ICTZ",
#     "IRISH",
#     "NETHER",
#     "OBL",
#     "PGB",
#     "RAGB",
#     "RFGB",
#     "SPGB",
# ]
#
#
# ix_temp = db.build_market_index(date="3/8/2021")
# ix_temp.df.sort_values("OAS").iloc[:50, :10]


def update_bbb_a_ratios(doc, ix_d, c_10):
    """Update plot for BBB/A nonfin ratio for 10y and 30y bonds."""
    ix_nonfin = ix_d["mc"].subset(financial_flag=0)
    ixs = {
        "10_A": ix_nonfin.subset(rating=("A+", "A-"), maturity=(8.25, 11)),
        "10_BBB": ix_nonfin.subset(
            rating=("BBB+", "BBB-"), maturity=(8.25, 11)
        ),
    }
    df = pd.concat(
        [ix.MEDIAN("OAS").rename(key) for key, ix in ixs.items()],
        axis=1,
        sort=True,
    )
    df["10 yr"] = df["10_BBB"] / df["10_A"]
    # Plot
    fig, ax = vis.subplots(figsize=(9, 6))

    ax.plot(df["10 yr"], c=c_10, alpha=0.9, lw=2)
    ax.set_ylabel("10 yr", color=c_10)
    ax.axhline(np.median(df["10 yr"]), ls=":", lw=1.5, color=c_10)
    pct = {x: np.percentile(df["10 yr"], x) for x in [5, 95]}
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    ax.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    ax.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
    ax.set_title("Non-Fin BBB/A Ratio", fontweight="bold")
    vis.format_xaxis(ax, df["10 yr"], "auto")
    ax.legend()
    vis.savefig(doc.fig_dir / "EUR_BBB_A_nonfin_ratio")
    vis.close()


# %%
def update_EUR_spreads(doc, color, save=True):
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
        gridspec_kw={"height_ratios": [7, 1]},
    )
    # Plot 5 year stats.
    oas = db.load_bbg_data("EU_IG", "OAS", start=db.date("5y"))
    oas_mc = oas[oas.index != pd.to_datetime("8/6/2020")].rename("OAS")
    oas_mc = oas_mc[oas_mc.index >= db.date("1y")]
    med = np.median(oas)
    pct = {x: np.percentile(oas, x) for x in [5, 95]}
    axes[0].axhline(
        med, ls="--", lw=1.5, color="firebrick", label=f"Median: {med:.0f}"
    )
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    axes[0].axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    axes[0].axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
    # Find corrected OAS data with proper volatility.
    # Get strategy meeting scoring data and combine with OAS.
    scores_df = (
        pd.read_excel(
            root("data/chicago_strategy_meeting_scores.xlsx"),
            index_col=0,
            sheet_name="Summary",
        )
        .loc["EUR"]
        .dropna()
        .astype(int)
        .rename("Short Term")
    )
    scores_df.index = pd.to_datetime(scores_df.index)
    # ix_mc = ix_d["mc"].subset(start=db.date("1y"))
    # oas_mc = ix_mc.get_synthetic_differenced_history("OAS").rename('OAS')
    df = pd.concat([oas_mc, scores_df], axis=1, sort=True)
    df["Short Term"].fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    pctile = int(np.round(100 * oas.rank(pct=True)[-1]))
    ordinal = get_ordinal(pctile)
    lbl = cleandoc(
        f"""
        Historical Returns Index
        Last: {oas[-1]:.0f} ({pctile:.0f}{ordinal} %tile)
        Range: [{np.min(oas):.0f}, {np.max(oas):.0f}]
        """
    )

    vis.plot_timeseries(
        df["OAS"],
        color=color,
        bollinger=True,
        ylabel="OAS",
        ax=axes[0],
        legend=False,
        label=lbl,
    )
    title = "$\\bf{5yr}$ $\\bf{Stats}$"
    axes[0].legend(loc="upper right", fancybox=True, title=title, shadow=True)
    axes[0].set_title(f"EUR Market Credit", fontweight="bold")

    # Plot short term scores below LC index.
    axes[1].plot(df["Short Term"], c="k", ls="--", lw=2)
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
    if save:
        vis.savefig(doc.fig_dir / f"EUR_MC_bollinger")
        vis.close()
    else:
        vis.show()


if __name__ == "__main__":
    update_EUR_credit_overview()
