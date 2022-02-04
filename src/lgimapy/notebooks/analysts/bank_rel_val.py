from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import get_ordinal, root, mkdir

vis.style()
path = root("latex/2021/Banks_Rel_Val")
mkdir(path)
# %%

db = Database()
db.load_market_data(start=db.date("5y"))

# %%

maturities = {
    "5": ((4, 6.2), (None, 10)),
    "10": ((7.5, 12.2), (None, 10)),
    "20+": ((19, 32), (None, None)),
}
bank_sectors = {
    "SIFI_BANKS_SR": ("AAA", "A-"),
    "SIFI_BANKS_SUB": ("BBB+", "BBB-"),
}
colors = {
    "SIFI_BANKS_SR": "navy",
    "SIFI_BANKS_SUB": "magenta",
}

date = "12/31/2019"

# %%
# US SIFI Banks maturity plots.
for mat, (mat_range, issue_years) in maturities.items():
    fig, ax = vis.subplots(figsize=(8, 5))
    for bank_sector, comp_ratings in bank_sectors.items():
        ix = db.build_market_index(
            **db.index_kwargs(
                bank_sector,
                maturity=mat_range,
                issue_years=issue_years,
                in_stats_index=True,
            )
        )
        oas = ix.OAS().rename(bank_sector)
        comp_ix = db.build_market_index(
            rating=comp_ratings,
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
        comp_oas = comp_ix.OAS()
        spread = (oas - comp_oas).fillna(method="ffill").dropna()
        current_spread = spread.iloc[-1]
        prev_spread = spread.loc[date]
        chg = current_spread - prev_spread
        pctile = 100 * spread.rank(pct=True).iloc[-1]
        ord = get_ordinal(pctile)
        median = np.median(spread)

        spread = spread[spread.index > "12/25/2018"]
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        if "SR" in bank_sector:
            title = "Sr (vs A-Rated)"
            n = 20
        else:
            title = "Sub (vs BBB-Rated)"
            n = 20
        lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
        tab = " " * int((n - len(title)) / 2)
        label = f"""
        {tab}{lbl}
        Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
        $\\Delta$ YE-2019: {chg:+.0f} bp
        5yr Median: {median:.0f} bp
        """
        color = colors[bank_sector]
        vis.plot_timeseries(
            spread,
            color=color,
            median_line=median,
            median_line_kws={
                "color": color,
                "ls": "--",
                "lw": 1,
                "alpha": 0.8,
                "label": "_nolegend_",
            },
            lw=1,
            alpha=0.8,
            label=label,
            ax=ax,
        )
        ax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)
        ax.legend(
            ncol=2,
            framealpha=0.0,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fontsize=12,
        )
        ax.set_title(
            f"{mat}yr Sifi Banks Rel Val", fontweight="bold", fontsize=12
        )
    vis.savefig(path / f"SIFI_{mat}yr")
    vis.close()

# %%
# US SIFI Banks sr vs sub plots.
for mat, (mat_range, issue_years) in maturities.items():
    ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[1],
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
    )
    oas = ix.OAS()
    comp_ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[0],
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
    )
    comp_oas = comp_ix.OAS()
    spread = (oas - comp_oas).fillna(method="ffill").dropna()
    current_spread = spread.iloc[-1]
    prev_spread = spread.loc[date]
    chg = current_spread - prev_spread
    pctile = 100 * spread.rank(pct=True).iloc[-1]
    ord = get_ordinal(pctile)
    median = np.median(spread)
    spread = spread[spread.index > "12/25/2018"]

    ratio = (oas / comp_oas).fillna(method="ffill").dropna()
    current_ratio = ratio.iloc[-1]
    current_ratio = ratio.iloc[-1]
    prev_ratio = ratio.loc[date]
    ratio_chg = current_ratio - prev_ratio
    ratio_pctile = 100 * ratio.rank(pct=True).iloc[-1]
    ratio_ord = get_ordinal(ratio_pctile)
    ratio_median = np.median(ratio)
    ratio = ratio[ratio.index > "12/25/2018"]

    title = "Sub vs Sr (abs)"
    n = 25
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    spread_label = f"""
    {tab}{lbl}
    Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {chg:+.0f} bp
    5yr Median: {median:.0f} bp
    """

    title = "Sub vs Sr (ratio)"
    n = 24
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    ratio_label = f"""
    {tab}{lbl}
    Current: {current_ratio:.2f} ({ratio_pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {ratio_chg:+.2f}
    5yr Median: {ratio_median:.2f}
    """

    lax, rax = vis.plot_double_y_axis_timeseries(
        spread,
        ratio,
        ylabel_left="OAS (abs)",
        ylabel_right="Ratio",
        plot_kws_left={"color": "darkorchid", "label": spread_label},
        plot_kws_right={"color": "firebrick", "label": ratio_label},
        ret_axes=True,
        figsize=(8, 5),
    )

    lax.axhline(0, color="k", lw=0.8, alpha=0.6)

    color = "darkorchid"
    vis.plot_timeseries(
        spread,
        color=color,
        median_line=median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=lax,
    )
    lax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)

    color = "firebrick"
    vis.plot_timeseries(
        ratio,
        color=color,
        median_line=ratio_median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=rax,
    )

    lax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.2, -0.1),
        fontsize=12,
    )
    rax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.8, -0.1),
        fontsize=12,
    )
    lax.set_title(
        f"{mat}yr Sifi Banks Sub vs Sr", fontweight="bold", fontsize=12
    )
    vis.show()
    vis.savefig(path / f"SIFI_{mat}yr_Sr_vs_Sub")
    vis.close()


# %%
# US banks maturity plots.
bank_sectors = {
    "US_BANKS_SR": ("AAA", "A-"),
    "US_BANKS_SUB": ("BBB+", "BBB-"),
}
colors = {
    "US_BANKS_SR": "navy",
    "US_BANKS_SUB": "magenta",
}

date = "12/31/2019"
for mat, (mat_range, issue_years) in maturities.items():
    fig, ax = vis.subplots(figsize=(8, 5))
    for bank_sector, comp_ratings in bank_sectors.items():
        ix = db.build_market_index(
            **db.index_kwargs(
                bank_sector,
                maturity=mat_range,
                issue_years=issue_years,
                in_stats_index=True,
            )
        )
        oas = ix.OAS().rename(bank_sector)
        comp_ix = db.build_market_index(
            rating=comp_ratings,
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
        comp_oas = comp_ix.OAS()
        spread = (oas - comp_oas).fillna(method="ffill").dropna()
        current_spread = spread.iloc[-1]
        prev_spread = spread.loc[date]
        chg = current_spread - prev_spread
        pctile = 100 * spread.rank(pct=True).iloc[-1]
        ord = get_ordinal(pctile)
        median = np.median(spread)

        spread = spread[spread.index > "12/25/2018"]
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        if "SR" in bank_sector:
            title = "Sr (vs A-Rated)"
            n = 20
        else:
            title = "Sub (vs BBB-Rated)"
            n = 20
        lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
        tab = " " * int((n - len(title)) / 2)
        label = f"""
        {tab}{lbl}
        Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
        $\\Delta$ YE-2019: {chg:+.0f} bp
        5yr Median: {median:.0f} bp
        """
        color = colors[bank_sector]
        vis.plot_timeseries(
            spread,
            color=color,
            median_line=median,
            median_line_kws={
                "color": color,
                "ls": "--",
                "lw": 1,
                "alpha": 0.8,
                "label": "_nolegend_",
            },
            lw=1,
            alpha=0.8,
            label=label,
            ax=ax,
        )
        ax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)
        ax.legend(
            ncol=2,
            framealpha=0.0,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fontsize=12,
        )
        ax.set_title(
            f"{mat}yr US Banks Rel Val", fontweight="bold", fontsize=12
        )
    vis.savefig(path / f"US_{mat}yr")
    vis.close()

# %%
# US Banks sr vs sub plots.
for mat, (mat_range, issue_years) in maturities.items():
    ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[1],
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
    )
    oas = ix.OAS()
    comp_ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[0],
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
    )
    comp_oas = comp_ix.OAS()
    spread = (oas - comp_oas).fillna(method="ffill").dropna()
    current_spread = spread.iloc[-1]
    prev_spread = spread.loc[date]
    chg = current_spread - prev_spread
    pctile = 100 * spread.rank(pct=True).iloc[-1]
    ord = get_ordinal(pctile)
    median = np.median(spread)
    spread = spread[spread.index > "12/25/2018"]

    ratio = (oas / comp_oas).fillna(method="ffill").dropna()
    current_ratio = ratio.iloc[-1]
    current_ratio = ratio.iloc[-1]
    prev_ratio = ratio.loc[date]
    ratio_chg = current_ratio - prev_ratio
    ratio_pctile = 100 * ratio.rank(pct=True).iloc[-1]
    ratio_ord = get_ordinal(ratio_pctile)
    ratio_median = np.median(ratio)
    ratio = ratio[ratio.index > "12/25/2018"]

    title = "Sub vs Sr (abs)"
    n = 25
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    spread_label = f"""
    {tab}{lbl}
    Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {chg:+.0f} bp
    5yr Median: {median:.0f} bp
    """

    title = "Sub vs Sr (ratio)"
    n = 24
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    ratio_label = f"""
    {tab}{lbl}
    Current: {current_ratio:.2f} ({ratio_pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {ratio_chg:+.2f}
    5yr Median: {ratio_median:.2f}
    """

    lax, rax = vis.plot_double_y_axis_timeseries(
        spread,
        ratio,
        ylabel_left="OAS (abs)",
        ylabel_right="Ratio",
        plot_kws_left={"color": "darkorchid", "label": spread_label},
        plot_kws_right={"color": "firebrick", "label": ratio_label},
        ret_axes=True,
        figsize=(8, 5),
    )

    lax.axhline(0, color="k", lw=0.8, alpha=0.6)

    color = "darkorchid"
    vis.plot_timeseries(
        spread,
        color=color,
        median_line=median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=lax,
    )
    lax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)

    color = "firebrick"
    vis.plot_timeseries(
        ratio,
        color=color,
        median_line=ratio_median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=rax,
    )

    lax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.2, -0.1),
        fontsize=12,
    )
    rax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.8, -0.1),
        fontsize=12,
    )
    lax.set_title(f"{mat}yr US Banks Sub vs Sr", fontweight="bold", fontsize=12)
    vis.savefig(path / f"US_{mat}yr_Sr_vs_Sub")
    vis.close()

# %%
# Aussie Banks maturity plots.
maturities = {
    "5": ((4, 6.2), (None, 10)),
    "10": ((7.5, 12.2), (None, 10)),
    "20+": ((19, 32), (None, None)),
}
bank_sectors = {
    "BANKS_SR": ("AAA", "A-"),
    "BANKS_SUB": ("BBB+", "BBB-"),
}
colors = {
    "BANKS_SR": "navy",
    "BANKS_SUB": "magenta",
}

date = "12/31/2019"
for mat, (mat_range, issue_years) in maturities.items():
    fig, ax = vis.subplots(figsize=(8, 5))
    for bank_sector, comp_ratings in bank_sectors.items():
        ix = db.build_market_index(
            **db.index_kwargs(
                bank_sector,
                country_of_risk="AU",
                maturity=mat_range,
                issue_years=issue_years,
                in_stats_index=None,
            )
        )
        oas = ix.OAS().rename(bank_sector)
        comp_ix = db.build_market_index(
            rating=comp_ratings,
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
        comp_oas = comp_ix.OAS()
        spread = (oas - comp_oas).fillna(method="ffill").dropna()
        current_spread = spread.iloc[-1]
        prev_spread = spread.loc[date]
        chg = current_spread - prev_spread
        pctile = 100 * spread.rank(pct=True).iloc[-1]
        ord = get_ordinal(pctile)
        median = np.median(spread)

        spread = spread[spread.index > "12/25/2018"]
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        if "SR" in bank_sector:
            title = "Sr (vs A-Rated)"
            n = 20
        else:
            title = "Sub (vs BBB-Rated)"
            n = 20
        lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
        tab = " " * int((n - len(title)) / 2)
        label = f"""
        {tab}{lbl}
        Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
        $\\Delta$ YE-2019: {chg:+.0f} bp
        5yr Median: {median:.0f} bp
        """
        color = colors[bank_sector]
        vis.plot_timeseries(
            spread,
            color=color,
            median_line=median,
            median_line_kws={
                "color": color,
                "ls": "--",
                "lw": 1,
                "alpha": 0.8,
                "label": "_nolegend_",
            },
            lw=1,
            alpha=0.8,
            label=label,
            ax=ax,
        )
        ax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)
        ax.legend(
            ncol=2,
            framealpha=0.0,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fontsize=12,
        )
        ax.set_title(
            f"{mat}yr Aussie Banks Rel Val", fontweight="bold", fontsize=12
        )
    vis.savefig(path / f"Aussie_{mat}yr")
    vis.close()

# %%
# Aussie Banks Sr vs Sub plots.
maturities = {
    "5": ((4, 6.2), (None, 10)),
    "10": ((7.5, 12.2), (None, 10)),
    "20+": ((19, 32), (None, None)),
}
bank_sectors = {
    "BANKS_SR": ("AAA", "A-"),
    "BANKS_SUB": ("BBB+", "BBB-"),
}
colors = {
    "BANKS_SR": "navy",
    "BANKS_SUB": "magenta",
}

date = "12/31/2019"
for mat, (mat_range, issue_years) in maturities.items():
    ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[1],
            country_of_risk="AU",
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=None,
        )
    )
    oas = ix.OAS()
    comp_ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[0],
            country_of_risk="AU",
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=None,
        )
    )
    comp_oas = comp_ix.OAS()
    spread = (oas - comp_oas).fillna(method="ffill").dropna()
    current_spread = spread.iloc[-1]
    prev_spread = spread.loc[date]
    chg = current_spread - prev_spread
    pctile = 100 * spread.rank(pct=True).iloc[-1]
    ord = get_ordinal(pctile)
    median = np.median(spread)
    spread = spread[spread.index > "12/25/2018"]

    ratio = (oas / comp_oas).fillna(method="ffill").dropna()
    current_ratio = ratio.iloc[-1]
    current_ratio = ratio.iloc[-1]
    prev_ratio = ratio.loc[date]
    ratio_chg = current_ratio - prev_ratio
    ratio_pctile = 100 * ratio.rank(pct=True).iloc[-1]
    ratio_ord = get_ordinal(ratio_pctile)
    ratio_median = np.median(ratio)
    ratio = ratio[ratio.index > "12/25/2018"]

    title = "Sub vs Sr (abs)"
    n = 25
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    spread_label = f"""
    {tab}{lbl}
    Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {chg:+.0f} bp
    5yr Median: {median:.0f} bp
    """

    title = "Sub vs Sr (ratio)"
    n = 24
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    ratio_label = f"""
    {tab}{lbl}
    Current: {current_ratio:.2f} ({ratio_pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {ratio_chg:+.2f}
    5yr Median: {ratio_median:.2f}
    """

    lax, rax = vis.plot_double_y_axis_timeseries(
        spread,
        ratio,
        ylabel_left="OAS (abs)",
        ylabel_right="Ratio",
        plot_kws_left={"color": "darkorchid", "label": spread_label},
        plot_kws_right={"color": "firebrick", "label": ratio_label},
        ret_axes=True,
        figsize=(8, 5),
    )

    lax.axhline(0, color="k", lw=0.8, alpha=0.6)

    color = "darkorchid"
    vis.plot_timeseries(
        spread,
        color=color,
        median_line=median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=lax,
    )
    lax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)

    color = "firebrick"
    vis.plot_timeseries(
        ratio,
        color=color,
        median_line=ratio_median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=rax,
    )

    lax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.2, -0.1),
        fontsize=12,
    )
    rax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.8, -0.1),
        fontsize=12,
    )
    lax.set_title(
        f"{mat}yr Aussie Banks Sub vs Sr", fontweight="bold", fontsize=12
    )
    vis.savefig(path / f"Aussie_{mat}yr_Sr_vs_Sub")
    vis.close()

# %%
# Japanese Banks maturity plots.
maturities = {
    "5": ((4, 6.2), (None, 10)),
    "10": ((7.5, 12.2), (None, 10)),
}
bank_sectors = {
    "BANKS_SR": ("AAA", "A-"),
    "BANKS_SUB": ("BBB+", "BBB-"),
}
colors = {
    "BANKS_SR": "navy",
    "BANKS_SUB": "magenta",
}

date = "12/31/2019"
for mat, (mat_range, issue_years) in maturities.items():
    fig, ax = vis.subplots(figsize=(8, 5))
    for bank_sector, comp_ratings in bank_sectors.items():
        ix = db.build_market_index(
            **db.index_kwargs(
                bank_sector,
                country_of_risk="JP",
                maturity=mat_range,
                issue_years=issue_years,
                in_stats_index=None,
            )
        )
        oas = ix.OAS().rename(bank_sector)
        comp_ix = db.build_market_index(
            rating=comp_ratings,
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=True,
        )
        comp_oas = comp_ix.OAS()
        spread = (oas - comp_oas).fillna(method="ffill").dropna()
        current_spread = spread.iloc[-1]
        prev_spread = spread.loc[date]
        chg = current_spread - prev_spread
        pctile = 100 * spread.rank(pct=True).iloc[-1]
        ord = get_ordinal(pctile)
        median = np.median(spread)

        spread = spread[spread.index > "12/25/2018"]
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        if "SR" in bank_sector:
            title = "Sr (vs A-Rated)"
            n = 20
        else:
            title = "Sub (vs BBB-Rated)"
            n = 20
        lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
        tab = " " * int((n - len(title)) / 2)
        label = f"""
        {tab}{lbl}
        Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
        $\\Delta$ YE-2019: {chg:+.0f} bp
        5yr Median: {median:.0f} bp
        """
        color = colors[bank_sector]
        vis.plot_timeseries(
            spread,
            color=color,
            median_line=median,
            median_line_kws={
                "color": color,
                "ls": "--",
                "lw": 1,
                "alpha": 0.8,
                "label": "_nolegend_",
            },
            lw=1,
            alpha=0.8,
            label=label,
            ax=ax,
        )
        ax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)
        ax.legend(
            ncol=2,
            framealpha=0.0,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fontsize=12,
        )
        ax.set_title(
            f"{mat}yr Japanese Banks Rel Val", fontweight="bold", fontsize=12
        )
    vis.savefig(path / f"Japanese_{mat}yr")
    vis.close()

# %%
# Japanese Banks Sr vs Sub plots.
maturities = {
    "5": ((4, 6.2), (None, 10)),
    "10": ((7.5, 12.2), (None, 10)),
}
bank_sectors = {
    "BANKS_SR": ("AAA", "A-"),
    "BANKS_SUB": ("BBB+", "BBB-"),
}
colors = {
    "BANKS_SR": "navy",
    "BANKS_SUB": "magenta",
}
date = "12/31/2019"

for mat, (mat_range, issue_years) in maturities.items():
    ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[1],
            country_of_risk="JP",
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=None,
        )
    )
    oas = ix.OAS()
    comp_ix = db.build_market_index(
        **db.index_kwargs(
            list(bank_sectors.keys())[0],
            country_of_risk="JP",
            maturity=mat_range,
            issue_years=issue_years,
            in_stats_index=None,
        )
    )
    comp_oas = comp_ix.OAS()
    spread = (oas - comp_oas).fillna(method="ffill").dropna()
    current_spread = spread.iloc[-1]
    prev_spread = spread.loc[date]
    chg = current_spread - prev_spread
    pctile = 100 * spread.rank(pct=True).iloc[-1]
    ord = get_ordinal(pctile)
    median = np.median(spread)
    spread = spread[spread.index > "12/25/2018"]

    ratio = (oas / comp_oas).fillna(method="ffill").dropna()
    current_ratio = ratio.iloc[-1]
    current_ratio = ratio.iloc[-1]
    prev_ratio = ratio.loc[date]
    ratio_chg = current_ratio - prev_ratio
    ratio_pctile = 100 * ratio.rank(pct=True).iloc[-1]
    ratio_ord = get_ordinal(ratio_pctile)
    ratio_median = np.median(ratio)
    ratio = ratio[ratio.index > "12/25/2018"]

    title = "Sub vs Sr (abs)"
    n = 25
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    spread_label = f"""
    {tab}{lbl}
    Current: {current_spread:.0f} bp ({pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {chg:+.0f} bp
    5yr Median: {median:.0f} bp
    """

    title = "Sub vs Sr (ratio)"
    n = 24
    lbl = "  ".join(f"$\\bf{t}$" for t in title.split())
    tab = " " * int((n - len(title)) / 2)
    ratio_label = f"""
    {tab}{lbl}
    Current: {current_ratio:.2f} ({ratio_pctile:.0f}{ord} %tile)
    $\\Delta$ YE-2019: {ratio_chg:+.2f}
    5yr Median: {ratio_median:.2f}
    """

    lax, rax = vis.plot_double_y_axis_timeseries(
        spread,
        ratio,
        ylabel_left="OAS (abs)",
        ylabel_right="Ratio",
        plot_kws_left={"color": "darkorchid", "label": spread_label},
        plot_kws_right={"color": "firebrick", "label": ratio_label},
        ret_axes=True,
        figsize=(8, 5),
    )

    lax.axhline(0, color="k", lw=0.8, alpha=0.6)

    color = "darkorchid"
    vis.plot_timeseries(
        spread,
        color=color,
        median_line=median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=lax,
    )
    lax.fill_between(spread.index, 0, spread.values, color=color, alpha=0.1)

    color = "firebrick"
    vis.plot_timeseries(
        ratio,
        color=color,
        median_line=ratio_median,
        median_line_kws={
            "color": color,
            "ls": "--",
            "lw": 1,
            "alpha": 0.8,
            "label": "_nolegend_",
        },
        lw=1,
        alpha=0.8,
        ax=rax,
    )

    lax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.2, -0.1),
        fontsize=12,
    )
    rax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.8, -0.1),
        fontsize=12,
    )
    lax.set_title(
        f"{mat}yr Japanese Banks Sub vs Sr", fontweight="bold", fontsize=12
    )
    vis.savefig(path / f"Japanese_{mat}yr_Sr_vs_Sub")
    vis.close()
