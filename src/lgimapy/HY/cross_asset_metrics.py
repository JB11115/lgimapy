import itertools as it

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.bloomberg import bdh
from lgimapy.data import Database
from lgimapy.latex import Document

# %%


def update_sharpe_ratios_and_correlations():
    vis.style()
    doc = Document("Correlations_and_SRs", path="reports/HY", fig_dir=True)
    doc.add_preamble(
        bookmarks=True,
        table_caption_justification="c",
        margin={"left": 0.5, "right": 0.5, "top": 1.5, "bottom": 1},
    )
    doc.add_section("Cross Asset Measures")
    doc.add_subsection("Correlations")
    db = Database()

    assets = [
        "UST_5Y",
        "UST_10Y",
        "US_BBB",
        "US_BB",
        "US_B",
        "US_CCC",
        "SP500",
    ]
    colors = {
        "UST_5Y": "darkgreen",
        "UST_10Y": "limegreen",
        "US_BBB": "navy",
        "US_BB": "steelblue",
        "US_B": "skyblue",
        "US_CCC": "aqua",
        "SP500": "darkorchid",
    }
    names = dict(zip(assets, db.bbg_names(assets)))

    start = "1/1/2008"
    tret_raw = db.load_bbg_data(assets, "TRET", start=start).dropna()
    tret = np.log(tret_raw).diff().iloc[1:]
    fed_funds = bdh("FDTRMID", "Index", fields="PX_MID", start=start).squeeze()

    def get_corr(df, cols):
        return df[cols[0]].rolling(window=252).corr(df[cols[1]]).dropna()

    combo_cols = list(it.permutations(assets, 2))
    fig, axes = vis.subplots(len(assets), 1, sharex=True, figsize=(12, 14))
    for asset, ax in zip(assets, axes.flat):
        for cols in combo_cols:
            if cols[0] == asset:
                vis.plot_timeseries(
                    get_corr(tret, cols),
                    lw=1.6,
                    alpha=0.8,
                    color=colors[cols[1]],
                    label=names[cols[1]],
                    ax=ax,
                )
        ax.axhline(0, color="black", lw=1, label="_nolegend_")
        ax.set_title(names[asset], fontsize=10, fontweight="bold")
        vis.set_n_ticks(ax, 4)
        ax.legend(loc=2, bbox_to_anchor=(1, 1), fontsize=10)

    doc.add_figure("cross_asset_correlations", savefig=True)

    corr = tret.iloc[-252:].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    lt_corr = corr.where(~mask).dropna(how="all").dropna(how="all", axis=1)
    fig, ax = vis.subplots(figsize=(6, 6))
    sns.heatmap(
        lt_corr,
        cmap="coolwarm",
        center=0,
        vmax=1,
        vmin=-1,
        linewidths=0.3,
        cbar=False,
        annot=True,
        fmt=".2f",
        ax=ax,
    )

    ax.set_xticklabels(db.bbg_names(lt_corr.columns), ha="center", fontsize=12)
    ax.set_yticklabels(
        db.bbg_names(lt_corr.index),
        rotation=0,
        ha="right",
        va="center",
        fontsize=12,
    )
    ax.set_title("Current Correlations", fontweight="bold", fontsize=16)
    doc.add_figure("current_correlations", width=0.6, savefig=True)

    doc.add_subsection("Sharpe Ratios")
    periods = [1, 3, 5, 10]
    fig, axes = vis.subplots(len(periods), 1, sharex=True, figsize=(12, 14))
    df_list = []
    for period, ax in zip(periods, axes.flat):
        daily_fed_funds = (1 + fed_funds / 100) ** (1 / 252) - 1
        xsrets = (
            tret_raw.pct_change().subtract(daily_fed_funds, axis=0).dropna()
        )
        rolling_num = xsrets.rolling(window=period * 252).mean()
        rolling_denom = xsrets.rolling(window=period * 252).std()
        sharpe_ratio = np.sqrt(252) * (rolling_num / rolling_denom).dropna()
        df_list.append(sharpe_ratio.iloc[-1])
        for asset in assets:
            vis.plot_timeseries(
                sharpe_ratio[asset],
                lw=1.6,
                alpha=0.8,
                color=colors[asset],
                label=names[asset],
                ax=ax,
            )
        ax.axhline(1, color="black", lw=2, label="_nolegend_")
        ax.set_title(
            f"{period}yr Sharpe Ratios", fontsize=16, fontweight="bold"
        )
        ax.set_xlim((pd.to_datetime("1/1/2015")), None)

    axes[0].legend(
        fancybox=True, shadow=True, loc="upper center", ncol=len(assets)
    )
    for i in range(1, len(periods)):
        axes[i].get_legend().remove()

    sr_df = pd.concat(df_list, axis=1).T
    sr_df.columns = db.bbg_names(sr_df.columns)
    sr_df.index = [f"{p}yr" for p in periods]
    doc.add_table(sr_df, prec=2, caption="Annualized Sharpe Ratios")
    doc.add_figure("sharpe_ratios", savefig=True)
    doc.save()


if __name__ == "__main__":
    update_sharpe_ratios_and_correlations()
