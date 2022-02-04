import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
import statsmodels.api as sm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.models import XSRETPerformance

vis.style()

# %%

db = Database()
start = db.date("YTD")
end = db.date("today")
date = start


def get_sector_vals(date, db):
    sectors = [
        "INDUSTRIALS",  # Industrials
        "BASICS",
        "~CHEMICALS",
        "~METALS_AND_MINING",
        "CAPITAL_GOODS",
        "COMMUNICATIONS",
        "~CABLE_SATELLITE",
        "~MEDIA_ENTERTAINMENT",
        "~WIRELINES_WIRELESS",
        "CONSUMER_CYCLICAL",
        "~AUTOMOTIVE",
        "~RETAILERS",
        "CONSUMER_NON_CYCLICAL",
        "~FOOD_AND_BEVERAGE",
        "~HEALTHCARE_EX_MANAGED_CARE",
        "~MANAGED_CARE",
        "~PHARMACEUTICALS",
        "~TOBACCO",
        "ENERGY",
        "~INDEPENDENT",
        "~INTEGRATED",
        "~OIL_FIELD_SERVICES",
        "~REFINING",
        "~MIDSTREAM",
        "ENVIRONMENTAL_IND_OTHER",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "~RAILROADS",
        "FINANCIALS",  # Financials
        "BANKS",
        "~SIFI_BANKS_SR",
        "~SIFI_BANKS_SUB",
        "~US_REGIONAL_BANKS",
        "~YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "LIFE",
        "P_AND_C",
        "REITS",
        "UTILITY",  # Utilities
        "NON_CORP",  # Non-Corp
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "UNIVERSITY",
    ]
    top_level_sector = "Industrials"
    db.load_market_data(date=date)
    ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
    d = defaultdict(list)
    for sector in sectors[1:]:
        kwargs = db.index_kwargs(sector.strip("~"))
        if sector == "FINANCIALS":
            top_level_sector = "Financial"
            continue
        elif sector == "UTILITY":
            top_level_sector = "Non Corp"
        elif sector == "NON_CORP":
            continue

        ix_sub = ix.subset(**kwargs)
        d["sector"].append(kwargs["name"])
        d[" "].append(top_level_sector)
        d["OAS"].append(ix_sub.OAS().iloc[0])
        d["\nMarket Value ($B)"].append(ix_sub.total_value().iloc[0] / 1e3)

    df = pd.DataFrame(d)
    return df


start_df = get_sector_vals(start, db)
df = get_sector_vals(end, db)
y = f"OAS {end: %m/%d/%Y}"
x = f"OAS {start: %m/%d/%Y}"
df[y] = df["OAS"]
df[x] = start_df["OAS"]
df = df[df["OAS"] > 50]
# %%
fig, ax = vis.subplots(figsize=(12, 11))
sns.scatterplot(
    x=x,
    y=y,
    hue=" ",
    size="\nMarket Value ($B)",
    sizes=(40, 200),
    alpha=0.7,
    palette="dark",
    data=df,
    ax=ax,
)
ax.legend(loc=2, bbox_to_anchor=(1.02, 1), prop={"size": 12})

ols = sm.OLS(df[y], sm.add_constant(df[x])).fit()
df["pred"] = ols.predict(sm.add_constant(df[x]))
df["resid"] = (df["pred"] - df[y]).abs()
x_lim = [df[x].min(), df[x].max()]
y_lim = ols.predict(sm.add_constant(x_lim))
ax.plot(x_lim, y_lim, lw=2, ls="--", color="grey")

df["raw_resid"] = df["pred"] - df[y]
df.set_index("sector")["raw_resid"].sort_values()


texts = []
threshold = 10

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _, row in df.iterrows():
        if np.abs(row["resid"]) < threshold:
            # Not an outlier -> don't label.
            continue
        texts.append(
            ax.annotate(
                row["sector"],
                xy=(row[x], row[y]),
                xytext=(1, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
            )
        )
    adjust_text(texts)

arrow = dict(arrowstyle="-|>, head_width=0.8, head_length=0.8", color="k", lw=5)
ax.annotate(
    " ",
    xy=(x_lim[0] + 20, y_lim[1] - 10),
    xytext=(x_lim[0] + 20, y_lim[1] - 50),
    arrowprops=arrow,
)
ax.annotate(
    "Under\nPerformers",
    ha="center",
    fontweight="bold",
    xy=(x_lim[0], y_lim[1]),
    xytext=(x_lim[0] + 20, y_lim[1] - 8),
)

ax.annotate(
    " ",
    xy=(x_lim[1] - 20, y_lim[0] + 10),
    xytext=(x_lim[1] - 20, y_lim[0] + 50),
    arrowprops=arrow,
)
ax.annotate(
    "Out\nPerformers",
    ha="center",
    fontweight="bold",
    xy=(x_lim[1], y_lim[0]),
    xytext=(x_lim[1] - 20, y_lim[0] + 2),
)

vis.savefig("ytd_sector_performance")


# %%

db = Database()
start = db.date("YTD")
end = db.date("today")
date = start


def get_sector_vals(date, db):
    sectors = [
        "CHEMICALS",
        "~INDEPENDENT",
        "~OIL_FIELD_SERVICES",
        "~REFINING",
        "~MIDSTREAM",
        "CAPITAL_GOODS",
        "METALS_AND_MINING",
        "COMMUNICATIONS",
        "AUTOMOTIVE",
        "RETAILERS",
        "CONSUMER_CYCLICAL_EX_AUTOS_RETAILERS",
        "HEALTHCARE_PHARMA",
        "FOOD_AND_BEVERAGE",
        "CONSUMER_NON_CYCLICAL_OTHER",
        "ENERGY",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "OTHER_INDUSTRIAL",
        "BANKS",
        "INSURANCE",
        "OTHER_FIN",
        "REITS",
        "UTILITY",
        "NON_CORP_OTHER",
    ]
    top_level_sector = "Industrials"
    db.load_market_data(date=date)
    ix = db.build_market_index(in_H0A0_index=True)
    d = defaultdict(list)
    for sector in sectors[1:]:
        kwargs = db.index_kwargs(
            sector.strip("~"), unused_constraints=["in_stats_index", "OAS"]
        )
        if sector == "BANKS":
            top_level_sector = "Financial"
            continue
        elif sector == "UTILITY":
            top_level_sector = "Non Corp"
        elif sector == "NON_CORP":
            continue

        ix_sub = ix.subset(**kwargs)
        d["sector"].append(kwargs["name"])
        d[" "].append(top_level_sector)
        d["OAS"].append(ix_sub.OAS().iloc[0])
        d["\nMarket Value ($B)"].append(ix_sub.total_value().iloc[0] / 1e3)

    df = pd.DataFrame(d)
    return df


start_df = get_sector_vals(start, db)
df = get_sector_vals(end, db)
y = f"OAS {end: %m/%d/%Y}"
x = f"OAS {start: %m/%d/%Y}"
df[y] = df["OAS"]
df[x] = start_df["OAS"]
df = df[df["OAS"] > 50]
# %%
fig, ax = vis.subplots(figsize=(12, 11))
sns.scatterplot(
    x=x,
    y=y,
    hue=" ",
    size="\nMarket Value ($B)",
    sizes=(40, 200),
    alpha=0.7,
    palette="dark",
    data=df,
    ax=ax,
)
ax.legend(loc=2, bbox_to_anchor=(1.02, 1), prop={"size": 12})

ols = sm.OLS(df[y], sm.add_constant(df[x])).fit()
df["pred"] = ols.predict(sm.add_constant(df[x]))
df["resid"] = (df["pred"] - df[y]).abs()
x_lim = [df[x].min(), df[x].max()]
y_lim = ols.predict(sm.add_constant(x_lim))
ax.plot(x_lim, y_lim, lw=2, ls="--", color="grey")

texts = []
threshold = 10

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _, row in df.iterrows():
        if np.abs(row["resid"]) < threshold:
            # Not an outlier -> don't label.
            continue
        texts.append(
            ax.annotate(
                row["sector"],
                xy=(row[x], row[y]),
                xytext=(1, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
            )
        )
    adjust_text(texts)

arrow = dict(arrowstyle="-|>, head_width=0.8, head_length=0.8", color="k", lw=5)
ax.annotate(
    " ",
    xy=(x_lim[0] + 20, y_lim[1] - 10),
    xytext=(x_lim[0] + 20, y_lim[1] - 50),
    arrowprops=arrow,
)
ax.annotate(
    "Under\nPerformers",
    ha="center",
    fontweight="bold",
    xy=(x_lim[0], y_lim[1]),
    xytext=(x_lim[0] + 20, y_lim[1] - 8),
)

ax.annotate(
    " ",
    xy=(x_lim[1] - 20, y_lim[0] + 10),
    xytext=(x_lim[1] - 20, y_lim[0] + 50),
    arrowprops=arrow,
)
ax.annotate(
    "Out\nPerformers",
    ha="center",
    fontweight="bold",
    xy=(x_lim[1], y_lim[0]),
    xytext=(x_lim[1] - 20, y_lim[0] + 2),
)

vis.savefig("ytd_HY_sector_performance")
