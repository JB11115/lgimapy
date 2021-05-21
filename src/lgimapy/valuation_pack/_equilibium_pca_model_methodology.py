from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document

from equilibium_pca_model import do_pca, load_pca_bloom_data

vis.style()

# %%
doc = Document(
    "valuation_pack_methodology",
    path="reports/valuation_pack/methodology",
    fig_dir=True,
)

db = Database()
start_date = db.date("12y")
df = load_pca_bloom_data(start_date)
df_mean = df.rolling(window=250).mean()
df_std = df.rolling(window=250).std()
df_z = (df - df_mean) / df_std
df_pca, __ = do_pca(df_z.dropna(axis=0, how="any"))
df_reg = pd.concat([df_z, df_pca], join="outer", sort=True, axis=1).dropna(
    axis=0, how="any"
)


all_factors = {"PCA 1", "PCA 2", "PCA 3", "PCA 4"}
d = defaultdict(list)
for col in df_reg.columns:
    if col.startswith("PCA"):
        continue
    y = df_reg[col]
    ols = sm.OLS(y, df_reg[all_factors]).fit(cov_type="HC1")
    d["col"].append(col)
    total_r2 = ols.rsquared_adj
    for pc, beta in ols.params.sort_index().items():
        d[f"beta_{pc}"].append(beta)
    d["r2_total"].append(total_r2)
    for factor in sorted(all_factors):
        factors = all_factors - set([factor])
        ols = sm.OLS(y, df_reg[factors]).fit(cov_type="HC1")
        partial_r2 = max(0, total_r2 - ols.rsquared_adj)
        d[f"r2_{factor}"].append(partial_r2)

df_reg_result = pd.DataFrame(d).set_index("col").rename_axis(None)
df_reg_result.index = Database().bbg_names(df_reg_result.index)
columns = [
    "PC1: Economic Surprise",
    "PC2: Real Rates",
    "PC3: US Dollar",
    "PC4: Euro Area Risk",
]

# %%
df_partial_r2 = df_reg_result.sort_values("r2_total").filter(regex="r2_PCA")
df_partial_r2.columns = columns
fig, ax = vis.subplots(figsize=(8, 12))
ax.set_title(
    "Partial Variance Explained\nby Principal Components", fontweight="bold"
)
colors = ["#4D5359", "#8EB8E5", "#85BB65", "#DAA588"]
df_partial_r2.plot.barh(stacked=True, color=colors, ax=ax)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("$R^2$")
ax.grid(False, axis="y")
ax.legend(loc="lower right", fontsize=10, shadow=True, fancybox=True)
vis.savefig("contribution_to_r2", path=doc.fig_dir)

# %%
df_beta = -df_reg_result.sort_values("beta_PCA 2").filter(regex="beta_")
df_beta.columns = columns
index_agg = {
    "DM Rates": [
        "JGB 30Y",
        "UK 30Y",
        "Bund 30Y",
        "Bund 10Y",
        "UK 10Y",
        "JGB 10Y",
        "UST 10Y",
        "UST 30Y",
    ],
    "US IG": ["US Market Credit"],
    "US HY": ["US HY"],
    "EU HY": ["EU HY"],
    "EM Sov": ["EM Sov"],
    "S&P 500": ["S&P 500"],
    "Russell 2000": ["Russell 2000"],
    "MSCI EM": ["MSCI EM"],
    "VIX": ["Vix", "3M Vix"],
    "Gold": ["Gold"],
}
df_beta_agg = pd.concat(
    [df_beta.loc[vals].mean().rename(key) for key, vals in index_agg.items()],
    axis=1,
).T
fig, ax = vis.subplots(figsize=(12, 6))
ax.set_title("Drivers of Asset 1yr Z-Score", fontweight="bold")
df_beta_agg.plot.bar(color=colors, ax=ax)
# ax.set_xlim(-0.7, 0.7)
ax.set_ylabel("$\\beta$")
for label in ax.get_xticklabels():
    label.set_ha("left")
    label.set_rotation(-45)
ax.grid(False, axis="x")
ax.set_ylim(None, 0.4)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1),
    ncol=4,
    fontsize=10,
    shadow=True,
    fancybox=True,
)
vis.savefig("betas", path=doc.fig_dir)
vis.show()


# %%
start_date = Database().date("20y")
df = load_pca_bloom_data(start_date)
n = 250
df_z = ((df - df.rolling(window=n).mean()) / df.rolling(window=n).std())[n:]

n = int(len(df_z) / 2)
dates, var_ratio = [], []
for i in range(n):
    df_z_i = df_z.iloc[i : i + n + 1]
    dates.append(df_z_i.index[-1])
    __, var_ratio_i = do_pca(df_z_i)
    var_ratio.append(var_ratio_i)

var_ratio_df = pd.DataFrame(var_ratio, index=dates, columns=columns)
# %%
fig, ax = vis.subplots(figsize=(12, 6))
ax.set_title("Variance Explained over Model History", fontweight="bold")
var_ratio_df.plot.area(ax=ax, color=colors, alpha=0.8)
ax.set_ylabel("PCA Variance Explained")
ax.set_xlim(var_ratio_df.index[[0, -1]])
vis.format_xaxis(ax, var_ratio_df, "auto")
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
ax.legend(fancybox=True, shadow=True, loc="lower left", fontsize=10)
vis.savefig("explained_variance_ts", path=doc.fig_dir)
