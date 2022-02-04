from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sms


import lgimapy.vis as vis
from lgimapy.bloomberg import bdp
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import find_drawdowns

# %%
vis.style()
db = Database()
db.load_market_data(start="1/1/2010", local=True)

# %%
colors = {"BB": "#00203FFF", "B": "#51A389"}
ratings = {"B": ("B+", "B-"), "BB": ("BB+", "BB-")}
data = defaultdict(dict)
for rating, rating_kws in ratings.items():
    ix = db.build_market_index(
        end="12/31/2019",
        issue_years=(None, 0.2),
        rating=rating_kws,
        in_hy_stats_index=True,
    )
    cusips = ix.cusips
    ix = db.build_market_index(cusip=cusips, OAS=(0, 3000))
    spread_df = ix.get_value_history("OAS")
    prices_df = ix.get_value_history("DirtyPrice")
    d = defaultdict(list)
    defaults = bdp(cusips, "Corp", "DEFAULTED").squeeze()
    defaulted = defaults[defaults == "Y"].index

    data["price_df"][rating] = prices_df
    for cusip in prices_df.columns:
        prices = prices_df[cusip].dropna()
        spreads = spread_df[cusip].dropna()
        min_spread = np.min(prices)
        prices_below_80 = prices[prices <= 80]

        d["start"].append(prices.iloc[0])
        d["start_oas"].append(spreads.iloc[0])
        d["max"].append(np.max(prices))
        d["min"].append(min_spread)
        d["%_below_80"].append(len(prices_below_80) / len(prices))
        for p in range(60, 100, 5):
            if min_spread <= p:
                d[str(p)].append(1)
                # Find if bond recovered after dropping below p
                prices_below = prices[np.argmax(prices < p) :]
                if np.max(prices_below) >= 100:
                    d[f"{p}_recover"].append(1)
                else:
                    d[f"{p}_recover"].append(0)
            else:
                d[str(p)].append(0)
                d[f"{p}_recover"].append(0)

    df = pd.DataFrame(d, index=prices_df.columns)
    df["defaulted"] = 0
    df.loc[df.index.isin(defaulted), "defaulted"] = 1
    df["start_oas_bin"] = pd.qcut(df["start_oas"].values, q=5, labels=False)
    data["df"][rating] = df.copy()


# %%
doc = Document("HY_Distressed_Prices", path="latex/HY/2020", fig_dir=True)
doc.add_preamble(
    margin={"left": 0.75, "right": 0.75, "top": 0.75, "bottom": 0.75},
    table_caption_justification="c",
)
# %%
# Histograms of minimum price.
fig, axes = vis.subplots(1, 2, figsize=(16, 6))
for rating, df in data["df"].items():
    vis.plot_hist(
        df["min"],
        color=colors[rating],
        bins=40,
        alpha=0.6,
        label=rating,
        ax=axes[0],
        median_kws={"color": colors[rating]},
        median=True,
    )
vis.format_xaxis(axes[0], xtickfmt="${x:.0f}")
axes[0].set_title("Minimum Price in Lifetime")
axes[0].legend()
axes[0].set_xlim(*axes[0].get_xbound()[::-1])


for rating, df in data["df"].items():
    vis.plot_hist(
        df["min"],
        cumulative=True,
        color=colors[rating],
        bins=200,
        alpha=0.7,
        ax=axes[1],
    )
vis.format_xaxis(axes[1], xtickfmt="${x:.0f}")
axes[1].set_title("Minimum Price in Lifetime (Cumulative)")
axes[1].set_xlim(*ax.get_xbound()[::-1])
axes[1].set_xlim((120, None))
doc.add_figure("minimum_price_histograms", width=0.95, savefig=True)
# vis.show()

# %%
# Time at price level for all bonds
fig, axes = vis.subplots(1, 3, figsize=(16, 6))
for rating, price_df in data["price_df"].items():
    all_prices = price_df.values.ravel()
    prices = all_prices[~np.isnan(all_prices)]
    vis.plot_hist(
        prices,
        color=colors[rating],
        bins=200,
        alpha=0.6,
        label=rating,
        ax=axes[0],
    )
    vis.plot_hist(
        prices,
        color=colors[rating],
        cumulative=True,
        bins=200,
        alpha=0.6,
        label=rating,
        ax=axes[1],
    )
    vis.plot_hist(
        prices,
        color=colors[rating],
        cumulative=True,
        bins=200,
        alpha=0.6,
        label=rating,
        ax=axes[2],
    )

for ax in axes:
    ax.set_xlim(*ax.get_xbound()[::-1])
    vis.format_xaxis(ax, xtickfmt="${x:.0f}")
# axes[0].set_xlim((40, 120))
axes[0].set_xlim((120, 40))
# axes[2].set_xlim((50, 90))
axes[2].set_xlim((90, 30))
axes[1].set_xlim((120, 50))
axes[2].set_ylim((None, 0.12))
axes[0].legend()
axes[0].set_title("Time at each Price", y=1.05)
axes[1].set_title("Time below each Price", y=1.05)
axes[2].set_title("Zoomed-in Time below each Price", y=1.05)
doc.add_figure("time_at_each_price", width=0.95, savefig=True)
# vis.show()

# %%
# Time at price level for bonds conditional to reaching $80
cmap = sns.color_palette("husl", 5).as_hex()
fig, axes = vis.subplots(2, 1, figsize=(16, 12), sharex=True)
for ax, (rating, price_df) in zip(axes.flat, data["price_df"].items()):
    for i, c in enumerate(cmap):
        # bin = 4 - i
        bin = i
        df = data["df"][rating]
        bin_cusips = df[df["start_oas_bin"] == bin].index
        all_prices = price_df[bin_cusips].values.ravel()
        prices = all_prices[~np.isnan(all_prices)]
        vis.plot_hist(
            prices,
            color=c,
            bins=300,
            alpha=0.6,
            label=f"bin {bin}",
            ax=ax,
            prec=1,
        )

    ax.set_xlim((130, 60))
    vis.format_xaxis(ax, xtickfmt="${x:.0f}")

axes[0].legend()
axes[0].set_ylabel("B")
axes[1].set_ylabel("BB")
axes[0].set_title("Time at each Price by Starting OAS", y=1.05)
doc.add_figure(
    "time_at_each_price_by_starting_spread", width=0.95, savefig=True
)
# vis.show()
# %%
# for rating, df in data['df'].items():
#     print(rating)
#     for bin in range(5):
#         df_bin = df[df["start_oas_bin"] == bin]
#         oas = df_bin['start_oas']
#         print(f"{bin}: {oas.min()+1:.0f} - {oas.max():.0f}")


# %%


# Plot time at each price by starting oas bin.
# Time at price level for bonds conditional to reaching $80
fig, axes = vis.subplots(1, 3, figsize=(16, 6))
for rating, price_df in data["price_df"].items():
    below_80 = price_df[price_df < 80].sum() > 0
    cols = list(below_80[below_80].index)
    below_80_df = price_df[cols]
    all_prices = below_80_df.values.ravel()
    prices = all_prices[~np.isnan(all_prices)]
    vis.plot_hist(
        prices,
        color=colors[rating],
        bins=200,
        alpha=0.6,
        label=rating,
        ax=axes[0],
    )
    vis.plot_hist(
        prices,
        color=colors[rating],
        cumulative=True,
        bins=200,
        alpha=0.6,
        label=rating,
        ax=axes[1],
    )
    vis.plot_hist(
        prices,
        color=colors[rating],
        cumulative=True,
        bins=200,
        alpha=0.6,
        label=rating,
        ax=axes[2],
    )

for ax in axes:
    # ax.set_xlim(*ax.get_xbound()[::-1])
    vis.format_xaxis(ax, xtickfmt="${x:.0f}")

# axes[0].set_xlim((40, 120))
axes[0].set_xlim((120, 40))
# axes[1].set_xlim((40, 120))
axes[1].set_xlim((120, 40))
# axes[2].set_xlim((30, 90))
axes[2].set_xlim((90, 30))
axes[2].set_ylim((None, 0.25))
axes[0].legend()
axes[0].set_title("Time at each Price", y=1.05)
axes[1].set_title("Time below each Price", y=1.05)
axes[2].set_title("Zoomed-in Time below each Price", y=1.05)
doc.add_figure("time_at_each_price_below_80", width=0.95, savefig=True)

# %%
# Starting spread regression
fig, ax = vis.subplots(figsize=(8, 6))
for rating, df in data["df"].items():
    x = df["start_oas"]
    y = df["min"]
    ax.plot(
        x, y, "o", alpha=0.7, color=colors[rating], label=rating,
    )
    ols = sms.OLS(y, sms.add_constant(x)).fit()
    ax.plot(
        np.sort(x),
        np.sort(x) * ols.params[1] + ols.params[0],
        ls="--",
        lw=2,
        c=colors[rating],
        label=f"$R^2 = {ols.rsquared:.3f}$",
    )
ax.legend()
vis.format_yaxis(ax, ytickfmt="${x:.0f}")
ax.set(xlabel="OAS at Issuance (bp)", ylabel="Minimum Price in Lifetime")
doc.add_figure("starting_spread_regression", width=0.95, savefig=True)
# vis.show()

# %%
fig, ax = vis.subplots(figsize=(8, 6))
for rating, df in data["df"].items():
    x = df["start_oas"]
    y = df["%_below_80"]
    ax.plot(
        x, y, "o", alpha=0.7, color=colors[rating], label=rating,
    )
    ols = sms.OLS(y, sms.add_constant(x)).fit()
    ax.plot(
        np.sort(x),
        np.sort(x) * ols.params[1] + ols.params[0],
        ls="--",
        lw=2,
        c=colors[rating],
        label=f"$R^2 = {ols.rsquared:.3f}$",
    )
ax.legend()
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
ax.set(xlabel="OAS at Issuance (bp)", ylabel="Lifetime of Bond below $80")
doc.add_figure("time_below_80_vs_starting_spread", width=0.95, savefig=True)
# vis.show()

# %%
fig, ax = vis.subplots(figsize=(8, 6))
for rating, df in data["df"].items():
    df = df[df["%_below_80"] > 0].copy()
    x = df["start_oas"]
    y = df["%_below_80"]
    ax.plot(
        x, y, "o", alpha=0.7, color=colors[rating], label=rating,
    )
    ols = sms.OLS(y, sms.add_constant(x)).fit()
    # print(ols.summary())
    ax.plot(
        np.sort(x),
        np.sort(x) * ols.params[1] + ols.params[0],
        ls="--",
        lw=2,
        c=colors[rating],
        label=f"$R^2 = {ols.rsquared:.3f}$",
    )
ax.legend()
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
ax.set(xlabel="OAS at Issuance (bp)", ylabel="Lifetime of Bond below $80")
doc.add_figure(
    "time_below_80_vs_starting_spread_conditional", width=0.95, savefig=True
)
vis.show()


# %%
# Starting spread % chance of reaching different prices.
fig, axes = vis.subplots(1, 2, figsize=(10, 6), sharey=True)
for ax, (rating, df) in zip(axes.flat, data["df"].items()):
    prices = np.arange(60, 100, 5)
    d = defaultdict(list)
    for bin in range(5):
        df_bin = df[df["start_oas_bin"] == bin]
        for price in prices:
            d[str(bin)].append(df_bin[str(price)].sum() / len(df_bin))

    heatmap_df = pd.DataFrame(d, index=prices)
    cbar_ax = fig.add_axes([0.91, 0.2, 0.05, 0.6])
    sns.heatmap(
        heatmap_df,
        cmap="coolwarm",
        linewidths=0.2,
        # annot=True,
        # annot_kws={"fontsize": 7},
        # fmt=".2f",
        cbar=rating == "BB",
        cbar_ax=None if rating != "BB" else cbar_ax,
        cbar_kws={"label": "Probability of Reaching Given Price"},
        ax=ax,
    )
    ax.set_title(rating)
    ax.set_xlabel("Starting OAS Quintile")
axes[0].set_yticklabels(
    heatmap_df.index, ha="right", fontsize=12, va="center", rotation=0
)
axes[0].set_ylabel("Price ($)")
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("fig/heatmaps", dpi=300, bbox_inches="tight")
# plt.show()

# vis.show()

# %%
# Plot recovery and default rates for each rating
# and starting spread level.
def recovery_default(df, price):
    denom = np.sum(df[str(price)])
    recover = np.sum(df[f"{price}_recover"]) / denom
    default = np.sum(df["defaulted"]) / denom
    return recover, default


stat_df["start_oas_bin"].max()
fig, axes = vis.subplots(1, 2, figsize=(16, 6))
linestyles = ["-", "--", ":"]
prices = np.arange(60, 100, 5)
for rating, stat_df in data["df"].items():
    df_low_spread = stat_df[stat_df["start_oas_bin"] == 0].copy()
    df_high_spread = stat_df[stat_df["start_oas_bin"] == 4].copy()
    recover_d = defaultdict(list)
    default_d = defaultdict(list)
    for price in prices:
        recover, default = recovery_default(stat_df, price)
        recover_ls, default_ls = recovery_default(df_low_spread, price)
        recover_hs, default_hs = recovery_default(df_high_spread, price)
        recover_d[f"{rating}"].append(recover)
        recover_d[f"{rating} Tight Spread at Issue"].append(recover_ls)
        recover_d[f"{rating} Wide Spread at Issue"].append(recover_hs)
        default_d[f"{rating}"].append(default)
        default_d[f"{rating} Tight Spread at Issue"].append(default_ls)
        default_d[f"{rating} Wide Spread at Issue"].append(default_hs)

    recover_df = pd.DataFrame(recover_d, index=prices)
    default_df = pd.DataFrame(default_d, index=prices)

    for col, ls in zip(recover_df.columns, linestyles):
        axes[0].plot(recover_df[col], ls=ls, color=colors[rating], label=col)

    for col, ls in zip(default_df.columns, linestyles):
        axes[1].plot(default_df[col], ls=ls, color=colors[rating], label=col)

for ax in axes.flat:
    vis.format_xaxis(ax, xtickfmt="${x:.0f}")
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    ax.set_xlim(*ax.get_xbound()[::-1])
    ax.legend()

axes[0].set_title("Rate of Price Recovery to Par vs Price")
axes[1].set_title("Default Rate of Bonds vs Price")
doc.add_figure("recovery_default_rate_vs_price", width=0.95, savefig=True)
# vis.show()

# %%
# Find drawdowns.
fig, axes = vis.subplots(1, 2, figsize=(16, 6), sharey=True)
thresh = 20
for ax, (rating, price_df) in zip(axes.flat, data["price_df"].items()):
    drawdowns = np.zeros(len(price_df.columns))
    for i, cusip in enumerate(price_df.columns):
        prices = price_df[cusip].dropna()
        drawdowns[i] = len(
            find_drawdowns(prices, threshold=thresh, distance=40)
        )

    n_drawdowns = np.sum(drawdowns > 0) / len(drawdowns)
    print(f"{rating}: drawdowns: {n_drawdowns:.0%}")
    counts = pd.Series(Counter(drawdowns)) / len(drawdowns)
    ax.bar(
        counts.index,
        counts,
        width=0.8,
        alpha=0.7,
        color=colors[rating],
        label=rating,
    )
    ax.legend()
    ax.set_xticks(sorted(set(drawdowns)))
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    ax.set_xlabel("Number of Drawdowns")
    ax.grid(False, axis="x")


fig.suptitle(f"% of Bonds with ___ ${thresh} Drawdowns", y=1.1, fontsize=32)
doc.add_figure(f"drawdown_{thresh}", width=0.95, savefig=True)

# %%
# for rating, rating_kws in ratings.items():
#     ix = db.build_market_index(
#         start="6/1/2004",
#         end="10/1/2007",
#         issue_years=(None, 0.2),
#         rating=rating_kws,
#         in_hy_stats_index=True,
#     )
#     cusips = ix.cusips
#     ix = db.build_market_index(cusip=cusips, OAS=(0, 3000))
#     data["price_df_2004"][rating] = ix.get_value_history("DirtyPrice")


fig, axes = vis.subplots(1, 2, figsize=(16, 6))
for ax, (rating, price_df) in zip(axes.flat, data["price_df_2004"].items()):
    drawdowns = np.zeros(len(price_df.columns))
    for i, cusip in enumerate(price_df.columns):
        prices = price_df[cusip].dropna()
        drawdowns[i] = len(
            find_drawdowns(prices, threshold=thresh, distance=40)
        )

    n_drawdowns = np.sum(drawdowns > 0) / len(drawdowns)
    print(f"{rating}: drawdowns: {n_drawdowns:.0%}")
    counts = pd.Series(Counter(drawdowns)) / len(drawdowns)
    ax.bar(
        counts.index,
        counts,
        width=0.8,
        alpha=0.7,
        color=colors[rating],
        label=rating,
    )
    ax.legend()
    ax.set_xticks(sorted(set(drawdowns)))
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    ax.set_xlabel("Number of Drawdowns")
    ax.grid(False, axis="x")

fig.suptitle(f"% of Bonds with ___ ${thresh} Drawdowns", y=1.1, fontsize=32)
doc.add_figure(f"drawdown_{thresh}_2004", width=0.95, savefig=True)
# vis.show()


# %%
# for rating, rating_kws in ratings.items():
#     ix = db.build_market_index(
#         start="1/1/2010",
#         end="6/1/2019",
#         issue_years=(None, 0.2),
#         rating=rating_kws,
#         in_hy_stats_index=True,
#     )
#     cusips = ix.cusips
#     ix = db.build_market_index(cusip=cusips, OAS=(0, 3000), end="2/1/2020")
#     data["price_df_2010"][rating] = ix.get_value_history("DirtyPrice")


fig, axes = vis.subplots(1, 2, figsize=(16, 6), sharey=True)
# for thresh = in [10, 20, 30]:
for ax, (rating, price_df) in zip(axes.flat, data["price_df_2010"].items()):
    drawdowns = np.zeros(len(price_df.columns))
    for i, cusip in enumerate(price_df.columns):
        prices = price_df[cusip].dropna()
        drawdowns[i] = len(
            find_drawdowns(prices, threshold=thresh, distance=40)
        )

    n_drawdowns = np.sum(drawdowns > 0) / len(drawdowns)
    print(f"{rating}: drawdowns: {n_drawdowns:.0%}")
    counts = pd.Series(Counter(drawdowns)) / len(drawdowns)
    ax.bar(
        counts.index,
        counts,
        width=0.8,
        alpha=0.7,
        color=colors[rating],
        label=rating,
    )
    ax.legend()
    ax.set_xticks(sorted(set(drawdowns)))
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    ax.set_xlabel("Number of Drawdowns")
    ax.grid(False, axis="x")

fig.suptitle(f"% of Bonds with ___ ${thresh} Drawdowns", y=1.1, fontsize=32)
doc.add_figure(f"drawdown_{thresh}_2010", width=0.95, savefig=True)


# %%
# doc.save(save_tex=True)
