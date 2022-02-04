import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import root, get_ordinal

vis.style()


# %%
years = 15
sectors = [
    # "US_BANKS",
    # "YANKEE_BANKS",
    # "INSURANCE",
    # "REITS",
    # "BASICS",
    # "MIDSTREAM",
    # "ENERGY_EX_MIDSTREAM",
    # "TELECOM",
    # "MEDIA_ENTERTAINMENT",
    # "TECHNOLOGY",
    # "CAPITAL_GOODS",
    # "TRANSPORTATION",
    # "AUTOMOTIVE",
    # "RETAILERS",
    # "CONSUMER_PRODUCTS_FOOD_AND_BEVERAGE",
    # "TOBACCO",
    # "HEALTHCARE_PHARMA",
    # "UTILITY",
    "WIRELINES_WIRELESS",
]

db = Database()
db.load_market_data(start=db.date(f"{years}y"))


sector = sectors[0]
sector = "RAILROADS"
# %%
rating_kwargs = {"A-Rated": (None, "A-"), "BBB-Rated": ("BBB+", "BBB-")}
date = db.date("YTD")
ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
oas_d = {}
corrected_oas_d = {}
for key, ratings_kws in rating_kwargs.items():
    ix_rating = ix.subset(rating=ratings_kws)
    oas_d[key] = ix_rating.OAS()
    ix_1y = ix_rating.subset(start=date)
    ix_corrected = ix_1y.drop_ratings_migrations()
    corrected_oas_d[key] = ix_corrected.get_synthetic_differenced_history("OAS")

# %%
for sector in tqdm(sectors):
    ix_sector = db.build_market_index(
        **db.index_kwargs(sector, in_stats_index=True, maturity=(10, None))
    )
    sector_oas_d = {}
    corrected_sector_oas_d = {}
    for key, ratings_kws in rating_kwargs.items():
        if sector == "US_BANKS":
            sector_key = {
                "A-Rated": "SIFI_BANKS_SR",
                "BBB-Rated": "SIFI_BANKS_SUB",
            }[key]
            ix_sector_rating = ix_sector.subset(**db.index_kwargs(sector_key))
        elif sector == "UTILITY":
            sector_key = {
                "A-Rated": "UTILITY_1ST_MORTGAGE",
                "BBB-Rated": "UTILITY_1ST_MORTGAGE",
            }[key]
            ix_sector_rating = ix_sector.subset(**db.index_kwargs(sector_key))
        else:
            ix_sector_rating = ix_sector.subset(rating=ratings_kws)
        sector_oas_d[key] = ix_sector_rating.OAS()
        ix_sector_1y = ix_sector_rating.subset(start=date)
        ix_sector_corrected = ix_sector_1y.drop_ratings_migrations()
        corrected_sector_oas_d[
            key
        ] = ix_sector_corrected.get_synthetic_differenced_history("OAS")

    plot_lines = {}
    for key in rating_kwargs.keys():
        hist = sector_oas_d[key] - oas_d[key]
        corrected_hist = corrected_sector_oas_d[key] - corrected_oas_d[key]
        plot_lines[key] = hist
        plot_lines[f"{key} Corrected"] = corrected_hist
        plot_lines[f"{key} median"] = np.median(hist.dropna())

    fig, ax = vis.subplots(figsize=(8, 6))
    kwargs = {"lw": 1, "alpha": 0.8}
    colors = {"A-Rated": "navy", "BBB-Rated": "magenta"}

    for key in rating_kwargs.keys():
        spread = plot_lines[key]
        current_spread = spread.iloc[-1]
        pctile = spread.rank(pct=True).iloc[-1] * 100
        ord = get_ordinal(pctile)
        prev_spread = spread.loc[date]
        chg = current_spread - prev_spread
        prev_spread_corrected = plot_lines[f"{key} Corrected"].loc[date]
        corrected_chg = current_spread - prev_spread_corrected
        median = plot_lines[f"{key} median"]
        if sector == "US_BANKS":
            title = {"A-Rated": "Snr", "BBB-Rated": "Sub"}[key]
            n = 35
        elif sector == "UTILITY":
            title = {
                "A-Rated": "1st Mortgage/Opco",
                "BBB-Rated": "Unsecured/Holdco",
            }[key]
            n = 27
        else:
            title = key
            n = 30

        tab = " " * int((n - len(title)) / 2)
        label = f"""
        {tab}$\\bf{title}$
        Current: {current_spread:+.0f} bp ({pctile:.0f}{ord} %tile)
        $\\Delta$ YE-{date.year}: {chg:+.0f} bp
        Corrected $\\Delta$ YE-{date.year}: {corrected_chg:+.0f} bp
        {years}y Median: {median:+.0f} bp
        """

        # spread = spread[spread.index > "12/25/2018"]
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        vis.plot_timeseries(
            spread,
            color=colors[key],
            median_line=median,
            median_line_kws={
                "color": colors[key],
                "ls": "--",
                "lw": 1,
                "alpha": 0.8,
                "label": "_nolegend_",
            },
            label=label,
            ax=ax,
            **kwargs,
        )
        ax.fill_between(
            spread.index, 0, spread.values, color=colors[key], alpha=0.1
        )
    ax.legend(
        ncol=2,
        framealpha=0.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=12,
    )
    ax.set_title(
        f"{ix_sector.name} OAS vs Respective Rating Bucket",
        fontweight="bold",
        fontsize=12,
    )
    fid = root(f"latex/sector_summit/fig/{sector}")
    vis.savefig(fid)
    vis.savefig(sector)
