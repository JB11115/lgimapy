from collections import defaultdict
from pathlib import Path
from inspect import cleandoc

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database, Index
from lgimapy.latex import Document
from lgimapy.utils import root, load_json

vis.style()

# %%

fid_dir = Path(f"X:/Credit Strategy")
fid = fid_dir / "phoenix_analytics_dump_LGIMA_2020-10-05.csv"

df_raw = pd.read_csv(fid, engine="python")
# %%
def clean_data(df_raw):
    df = df_raw.iloc[4:].copy()
    d = defaultdict(list)
    for ugly_string in df["INSTRUMENT_LOOKUP"]:
        try:
            __, isin, cusip, __ = ugly_string.split(";")
        except (AttributeError, ValueError):
            d["ISIN"].append(np.nan)
            d["CUSIP"].append(np.nan)
            continue
        isin = isin[5:].upper()
        isin = np.nan if isin == "NOT SUPPLIED" else isin
        d["ISIN"].append(isin)
        cusip = cusip[6:].upper()
        cusip = np.nan if cusip == "NOT SUPPLIED" else cusip
        d["CUSIP"].append(cusip)

    df["CUSIP"] = d["CUSIP"]
    df["ISIN"] = d["ISIN"]
    return df


axioma_df = clean_data(df_raw)

# %%
db = Database()
db.display_all_columns()

db.load_market_data(date="10/5/2020")

# %%
ix = db.build_market_index(
    cusip=axioma_df["CUSIP"].dropna(),
    isin=axioma_df["ISIN"].dropna(),
    special_rules="CUSIP | ISIN",
)


def add_axioma_stats(ix, axioma_df):
    # df = ix.subset(OAD=(0.001, None), OAS=(0.001, 3000)).df
    ix.df.copy()

    axioma_cols = {
        "OAD": " EFFECTIVE_DURATION ",
        "OAS": " OAS_GOV ",
        "OASD": " SPREAD_DURATION ",
    }
    for col, axioma_col in axioma_cols.items():
        cusip_dict = axioma_df.set_index("CUSIP")[axioma_col].to_dict()
        cusip_dict = {
            k: v for k, v in cusip_dict.copy().items() if v != " -   "
        }
        df[f"axioma_{col}"] = pd.to_numeric(
            df["CUSIP"].map(cusip_dict).str.replace(",", "")
        )
    df.dropna(subset=axioma_cols.keys(), how="any", inplace=True)
    for col in axioma_cols.keys():
        df[f"{col}_abs_diff"] = df[col] - df[f"axioma_{col}"]
        df[f"{col}_abs_SE"] = df[f"{col}_abs_diff"] ** 2

        df[f"{col}_pct_diff"] = df[f"{col}_abs_diff"] / df[col]
        df[f"{col}_pct_SE"] = df[f"{col}_pct_diff"] ** 2

    maturity_bins = []
    for maturity in df["MaturityYears"]:
        if maturity < 5:
            maturity_bins.append("0-5 yrs")
        elif maturity < 10:
            maturity_bins.append("5-10 yrs")
        elif maturity < 20:
            maturity_bins.append("10-20 yrs")
        elif maturity < 30:
            maturity_bins.append("20-30 yrs")
        else:
            maturity_bins.append("30+ yrs")
    df["MaturityBin"] = maturity_bins

    lgima_sectors = load_json(f"lgima_sector_maps/2020-10-05")
    df["LGIMASector"] = df["CUSIP"].map(lgima_sectors)
    return Index(df)


# %%
ix_total = add_axioma_stats(ix, axioma_df)


# %%
doc = Document(
    'axioma_bloomberg_analytics_comparison',
    path='latex/axioma',
    fig_dir=True
)
doc.add_section('Sample Size')

# %%
counts = ix_total.df["MaturityBin"].value_counts()
fig, ax = vis.subplots(figsize=(8, 6))
counts.plot.bar(color='steelblue', alpha=0.8)
ax.grid(False, axis='x')
ax.set_title('Maturity Bucket Sample Size')
for label in ax.get_xticklabels():
    label.set_rotation(0)
# vis.show()
doc.add_figure('sample_size_maturity_bin', savefig=True)

# %%
counts = ix_total.df["LGIMASector"].value_counts()
fig, ax = vis.subplots(figsize=(8, 12))
counts.plot.barh(color='steelblue', alpha=0.8)
ax.grid(False, axis='y')
ax.set_title('Sector Sample Size')
doc.add_figure('sample_size_sector', savefig=True)
# vis.show()

# %%
stat = "OAS"

for stat in ["OAD", "OASD", "OAS"]:
    doc.add_pagebreak()
    doc.add_section(stat)
    # %%
    fig, axes = vis.subplots(3, 1, sharex=True, figsize=(12, 10))
    axes[0].plot(
        ix_total.df[stat],
        ix_total.df[f"axioma_{stat}"],
        "o",
        c="steelblue",
        alpha=0.7,
        ms=3,
    )
    axes[1].plot(
        ix_total.df[stat],
        ix_total.df[f"{stat}_abs_diff"],
        "o",
        c="steelblue",
        alpha=0.7,
        ms=3,
    )
    axes[2].plot(
        ix_total.df[stat],
        ix_total.df[f"{stat}_abs_SE"],
        "o",
        c="steelblue",
        alpha=0.7,
        ms=3,
    )
    limit = {"OAD": 1.5, "OASD": 1, "OAS": 3000}[stat]
    axes[2].set_ylim(0, limit)
    axes[2].set_xlabel(f"BBG {stat}")
    axes[0].set_ylabel(f"Axioma {stat}")
    axes[1].set_ylabel(f"Difference\n(BBG - Axioma)")
    axes[2].set_ylabel(f"Squared Error\n(Outliers Removed)")
    doc.add_figure(f'{stat}_overview', savefig=True)
    # vis.show()
    # %%
    maturity_bins = ["0-5 yrs", "5-10 yrs", "10-20 yrs", "20-30 yrs", "30+ yrs"]
    fig, ax = vis.subplots(figsize=(12, 6))
    col = f"{stat}_abs_diff"
    limits = {"OAD": (-1, 2.5), "OASD": (-1.5, 0.75), "OAS": (-100, 100)}[stat]
    df_temp = ix_total.df[
        (ix_total.df[col] >= limits[0]) & (ix_total.df[col] <= limits[1])
    ]
    sns.violinplot(
        ax=ax,
        data=df_temp,
        y=col,
        x="MaturityBin",
        order=maturity_bins,
        cut=0,
        inner=None,
        bw=0.5,
        linewidth=1,
        orient="v",
        palette="husl",
    )
    ax.axhline(0, c='k', ls='--', lw=1)
    ax.set_ylabel(f"{stat} Absolute Difference")
    doc.add_figure(f'{stat}_abs_diff', savefig=True)
    # vis.show()

    # %%
    fig, ax = vis.subplots(figsize=(12, 6))
    sns.boxplot(
        ax=ax,
        data=ix_total.df,
        y=f"{stat}_abs_SE",
        x="MaturityBin",
        order=maturity_bins,
        palette="husl",
        fliersize=3,
    )
    limit = {"OAD": 1.5, "OASD": 1.5, "OAS": 500}[stat]
    ax.set_ylim(-0.1, limit)
    ax.set_ylabel(f"{stat} Absolute Squared Error")
    doc.add_figure(f'{stat}_abs_SE', savefig=True)
    # vis.show()

    # %%
    fig, ax = vis.subplots(figsize=(12, 6))
    col = f"{stat}_pct_diff"
    limits = {"OAD": (-0.1, 0.1), "OASD": (-0.1, 0.1), "OAS": (-0.5, 0.5)}[stat]
    df_temp = ix_total.df[
        (ix_total.df[col] >= limits[0]) & (ix_total.df[col] <= limits[1])
    ]
    sns.violinplot(
        ax=ax,
        data=df_temp,
        y=col,
        x="MaturityBin",
        order=maturity_bins,
        cut=0,
        inner=None,
        bw=0.5,
        linewidth=1,
        orient="v",
        palette="husl",
    )
    ax.axhline(0, c='k', ls='--', lw=1)
    ax.set_ylabel(f"{stat} % Difference")
    vis.format_yaxis(ax, ytickfmt="{x:.1%}")
    doc.add_figure(f'{stat}_pct_diff', savefig=True)
    # vis.show()

    # %%
    fig, ax = vis.subplots(figsize=(12, 6))
    sns.boxplot(
        ax=ax,
        data=ix_total.df,
        y=f"{stat}_pct_SE",
        x="MaturityBin",
        order=maturity_bins,
        palette="husl",
        fliersize=3,
    )
    limit = {"OAD": 0.004, "OASD": 0.004, "OAS": 0.06}[stat]
    ax.set_ylim(-0.00005, limit)
    vis.format_yaxis(ax, ytickfmt="{x:.1%}")
    ax.set_ylabel(f"{stat} % Squared Error")
    doc.add_figure(f'{stat}_pct_SE', savefig=True)
    # vis.show()

    # %%
    fig, ax = vis.subplots(figsize=(12, 15))
    sns.boxplot(
        ax=ax,
        data=ix_total.df,
        x=f"{stat}_pct_SE",
        y="LGIMASector",
        palette="husl",
        fliersize=2,
        orient="h",
    )
    limit = {"OAD": 0.02, "OASD": 0.02, "OAS": 0.1}[stat]
    ax.set_xlim(-0.001, limit)
    vis.format_xaxis(ax, xtickfmt="{x:.1%}")
    ax.set_xlabel(f"{stat} % Squared Error")
    ax.set_ylabel("")
    doc.add_figure(f'{stat}_sector_pct_SE', savefig=True)
    # vis.show()

doc.save()
