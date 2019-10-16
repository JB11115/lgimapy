import os
from datetime import datetime as dt
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sms
import squarify
from tqdm import tqdm

from lgimapy.bloomberg import bdh
from lgimapy.data import Database
from lgimapy.utils import savefig, root

pd.plotting.register_matplotlib_converters()
plt.style.use("fivethirtyeight")
# %matplotlib qt
os.chdir(root("src/lgimapy/vis"))


def colors(color):
    """
    LGIMA umbrella colors.

    Parameters
    ----------
    color: str
        Umbrella color or first letter of color to return.

    Returns
    -------
    str:
        Specified color hex code.
    """
    return {
        "red": "#DD2026",
        "r": "#DD2026",
        "blue": "#209CD8",
        "b": "#209CD8",
        "yellow": "#FDD302",
        "y": "#FDD302",
        "green": "#09933C",
        "g": "#09933C",
    }[color.lower()]


def coolwarm(x, center=0, symettric=False, quality=1000):
    """
    Create custom diverging color palette centered around
    specified value.

    Parameters
    ----------
    x: array-like
        Values to be converted to colors.
    center: float, default=0
        Central value for divergin colorbar.
    symmetric: # TODO bool, default=False
        If True, use symmetric gradient for each side of
        the color range, such that full color range
        would not be used for skewed data.
        If False, use full color gradient to both min and
        max of the value range, so that full color bar
        is used regardless of skew in data.
    quality: int, default=1000
        Total number of uniuqe colors to include in colorbar.
        Higher number gives better gradient between colors.

    Retuns
    ------
    List[str]:
        List of custom color pallete hex codes for each
        respective input value.
    """

    pal = sns.color_palette("coolwarm", quality).as_hex()

    # Convert x into array and split by center value.
    x = np.array(x)
    x_neg = np.append(x[x <= center], [center])
    x_pos = np.append([center], x[x > center])

    # Center both sides around given center.
    x_neg_center = (x_neg - np.min(x_neg)) / (np.max(x_neg) - np.min(x_neg))
    x_pos_center = (x_pos - np.min(x_pos)) / (np.max(x_pos) - np.min(x_pos))

    # Scale each side to color pallete index.
    x_neg_norm = (quality * 0.5 * x_neg_center[:-1]).astype(int)
    x_pos_norm = ((quality - 1) * (0.5 + 0.5 * x_pos_center[1:])).astype(int)

    return [pal[ix] for ix in np.concatenate([x_neg_norm, x_pos_norm])]


# %%


def main():
    fdir = "../fig"
    save = False


def spider_plots():
    # %%
    prev_df = pd.DataFrame(
        {
            "Economic Backdrop": [1, 0, 1, 0],
            "Liquidiyt/Monetary Policy": [1, 1, 1, 0],
            "Geopolitics": [-2, -1, -2, -2],
            "Corporate Strength (Fins)": [0, -1, -1, 0],
            "Corporate Strength (Non Fins)": [0, 0, 0, -1],
            "Supply/Demand (Fins)": [1, 1, 1, 1],
            "Supply/Demand (Non Fins)": [1, 1, 2, 1],
            "Short Term": [0, 0, 1, 0],
            "Long Term": [0, -1, 0, -2],
        },
        index=["US", "Europe", "UK", "Global"],
    ).T

    col = "US"

    df = pd.DataFrame(
        {
            "Economic Backdrop": [0, 0, 1, 0],
            "Liquidiyt/Monetary Policy": [1, 2, 1, 1],
            "Geopolitics": [-1, -1, -1, -1],
            "Corporate Strength (Fins)": [0, -1, -1, 0],
            "Corporate Strength (Non Fins)": [0, 0, 0, -1],
            "Supply/Demand (Fins)": [1, 1, 1, 0],
            "Supply/Demand (Non Fins)": [1, 1, 1, 1],
            "Short Term": [0, 1, 1, 1],
            "Long Term": [0, -1, 0, -2],
        },
        index=["US", "Europe", "UK", "Global"],
    ).T

    col = "US"

    # %%

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={"polar": True})

    colors = ["steelblue", "darkgreen", "firebrick", "darkorchid"]
    for ax, col, c in zip(axes.flat, prev_df.columns, colors):
        labels = list(prev_df.index)
        vals = prev_df[col].values
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        vals = np.concatenate((vals, [vals[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, vals, "o-", color="grey", linewidth=2, alpha=0.4)
        ax.fill(angles, vals, alpha=0.15, color="grey")
        ax.set_title(f"{col}\n", fontsize=14, fontweight="bold")
        ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=8)
        ax.set_yticks(np.arange(-3, 4))
        ax.set_yticklabels(np.arange(-3, 4), size=7)
        ax.grid(True)

    for ax, col, c in zip(axes.flat, df.columns, colors):
        labels = list(df.index)
        vals = df[col].values
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        vals = np.concatenate((vals, [vals[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, vals, "o-", color=c, linewidth=2)
        ax.fill(angles, vals, alpha=0.25, color=c)
        ax.set_title(f"{col}\n", fontsize=14, fontweight="bold")
        ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=8)
        ax.set_yticks(np.arange(-3, 4))
        ax.set_yticklabels(np.arange(-3, 4), size=7)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    # %%


def treemap():
    from lgimapy.data import Database

    db = Database()
    db.load_market_data()
    ix = db.build_market_index(
        rating=("BBB+", "BBB-"), financial_flag="non-financial"
    )

    # %%
    ix.df["market_value"] = ix.df["AmountOutstanding"] * ix.df["DirtyPrice"]
    df = ix.df[["Ticker", "market_value"]]
    df.set_index("Ticker", inplace=True)
    df = df.groupby(["Ticker"]).sum()
    df.sort_values("market_value", inplace=True, ascending=False)
    total = np.sum(df["market_value"])
    cumsum = np.cumsum(df.values) / total
    labels = [
        f"{i}\n${mv * 1e-6:.1f}B" if cs < 0.493 else ""
        for i, mv, cs in zip(df.index, df.values.ravel(), cumsum)
    ]

    df.sort_values("market_value", inplace=True)
    # %%
    plt.figure(figsize=[22, 14])
    colors = ["#DD2026", "#209CD8", "#FDD302", "#09933C"]
    squarify.plot(
        sizes=df["market_value"],
        pad=0.0001,
        # pad=False,
        label=labels[::-1],
        color=colors,
        alpha=0.7,
        text_kwargs={"fontsize": 7},
    )
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    savefig("BBB treemap")
    plt.show()


def linreg():
    # %%
    x = np.array([46, 12, 14, 18, 15, 19, 20, 30, 20, 20])
    bm = np.array([16.8, 10.7, 17.1, 12.7, -6.6, 16.4, -4.6, 10.2, 12.2, -6.8])
    lgim = np.array([19.8, 12.2, 18.3, 14.5, -4.7, 16.6, -4.1, 11, 12.3, -6.5])
    alpha = lgim - bm

    ols = sms.OLS(y, sms.add_constant(x)).fit()

    # Plot results.
    fig, ax = plt.subplots(1, 1, figsize=[8, 8])
    ax.plot(x, y, "o", ms=6, c="#209CD8")
    ax.plot(x, x * ols.params[1] + ols.params[0], lw=1.5, c="#FDD302")
    ax.set_xlabel("Downgrades in US Credit")
    ax.set_ylabel("LGIMA Outperformance")
    tick = mtick.StrMethodFormatter("{x:.1f}%")
    ax.yaxis.set_major_formatter(tick)
    savefig("dongrades vs outperformance")
    plt.show()
    # %%


def rolling_correlation():
    # %%
    from lgimapy.data import Database, Index
    from lgimapy.bloomberg import bdh

    # %%
    db = Database()
    start = "1/1/2007"
    db.load_market_data(start=start, local=True)

    # %%
    ix = db.build_market_index(
        start="1/1/2007", in_stats_index=True, maturity=(10, None)
    )

    # ix.df['mv'] = ix.df['AmountOutstanding'] * ix.df['DirtyPrice']
    #
    # gdf = ix.df.groupby('Issuer')
    #
    # mv_df = pd.DataFrame(gdf['mv'].agg(np.sum))
    # mv_df.sort_values('mv', ascending=False, inplace=True)
    # issuers = mv_df.index[:50]

    oas_df = ix.get_value_history("OAS")
    oas_a = oas_df.values
    window = 30
    corrs = []
    for i in range(window, len(oas_df)):
        a_i = oas_a[i : i + window, :].T
        a_i = a_i[~np.isnan(a_i).any(axis=1)]
        corrs.append(np.mean(np.corrcoef(a_i)))

    corr = (
        pd.Series(corrs, index=ix.dates[window:])
        .rolling(window=60, min_periods=60)
        .mean()
    )

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(corr, c="#209CD8")
    tick = mtick.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Correlation")
    fig.autofmt_xdate()
    savefig("rolling correlation")
    plt.show()

    # %%
    corr.index[0].year
    years = np.arange(corr.index[0].year, corr.index[-1].year + 1)

    corr_df = pd.DataFrame(corr, columns=["corr"])
    corr_df["year"] = [i.year for i in corr.index]
    gdf = pd.DataFrame(corr_df.groupby("year")["corr"].agg(np.mean))

    # %%
    x = np.array([46, 12, 14, 18, 15, 19, 20, 30, 20, 20])
    y = gdf.values[2:-1]
    ols = sms.OLS(y, sms.add_constant(x)).fit()

    # Plot results.
    fig, ax = plt.subplots(1, 1, figsize=[8, 8])
    ax.plot(x, y, "o", ms=6, c="#209CD8")
    ax.plot(x, x * ols.params[1] + ols.params[0], lw=1.5, c="#FDD302")
    ax.set_xlabel("Rolling Correlation")
    ax.set_ylabel("LGIMA Outperformance")
    tick = mtick.StrMethodFormatter("{x:.1f}%")
    ax.yaxis.set_major_formatter(tick)
    savefig("ro")
    plt.show()

    # %%
    luacoas = 100 * bdh("LULCOAS", "Index", start="1/1/2007", fields="PX_BID")

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax2 = ax.twinx()
    bar_dates = [pd.to_datetime(f"6/1/{y}") for y in gdf.index]
    ticks = [pd.to_datetime(f"1/1/{y}") for y in gdf.index]
    ax2.bar(bar_dates, gdf["corr"].values, width=200, color="grey", alpha=0.4)
    tick = mtick.StrMethodFormatter("{x:.0%}")
    ax2.yaxis.set_major_formatter(tick)
    ax2.grid(False)
    ax.set_ylim((None, 700))
    ax2.set_ylim((0.3, 0.65))
    ax.plot(luacoas.index, luacoas.values, lw=5, c="#209CD8")
    ax.grid(False, axis="x")
    ax.set_xticks(bar_dates)
    for t in ticks:
        ax.axvline(t, ymax=0.01, lw=0.5, c="k")
    ax.set_ylabel("Long Credit Index OAS", color="#209CD8", weight="bold")
    ax2_ylab = "Average Annual Rolling Correlation"
    ax2.set_ylabel(ax2_ylab, color="grey", weight="bold")
    ax.set_xlabel("Date", weight="bold")
    savefig("yearly correlation with index OAS")
    plt.show()

    luacoas.to_csv("long_credit_index.csv")
    index_corr_df = pd.DataFrame(
        {"date": bar_dates, "annual correlation": gdf["corr"].values}
    )
    index_corr_df.to_csv("annual_index_correlation.csv")

    # %%

    disps = []
    for i in range(len(oas_df)):
        a_i = oas_a[i]
        a_i = np.sort(a_i[~np.isnan(a_i)])
        n = len(a_i)
        disps.append(a_i[int(0.75 * n)] - a_i[int(0.25 * n)])

    disp = (
        pd.Series(disps, index=ix.dates).rolling(window=1, min_periods=1).mean()
    )

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(corr, c="#209CD8")
    tick = mtick.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Correlation")
    fig.autofmt_xdate()
    # savefig("rolling correlation")
    plt.show()
    # %%
    years = np.arange(disp.index[0].year, disp.index[-1].year + 1)

    disp_df = pd.DataFrame(disp, columns=["disp"])
    disp_df["year"] = [i.year for i in disp.index]
    disp_gdf = pd.DataFrame(disp_df.groupby("year")["disp"].agg(np.mean))
    disp_gdf

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax2 = ax.twinx()
    bar_dates = [pd.to_datetime(f"6/1/{y}") for y in disp_gdf.index]
    ticks = [pd.to_datetime(f"1/1/{y}") for y in disp_gdf.index]
    ax2.bar(
        bar_dates, disp_gdf["disp"].values, width=200, color="grey", alpha=0.4
    )
    fs = 24
    tick = mtick.StrMethodFormatter("{x:.0f}")
    ax2.yaxis.set_major_formatter(tick)
    ax2.grid(False)
    ax.set_ylim((None, 500))
    # ax2.set_ylim((0.3, 0.65))
    ax.plot(luacoas.index, luacoas.values, lw=5, c="#209CD8")
    ax.grid(False, axis="x")
    ax.set_xticks(bar_dates)
    for t in ticks:
        ax.axvline(t, ymax=0.01, lw=0.5, c="k")
    ax.set_ylabel(
        "Long Credit Index OAS", color="#209CD8", weight="bold", fontsize=fs
    )
    ax2_ylab = "Annual Dispersion (bp)"
    ax2.set_ylabel(ax2_ylab, color="grey", weight="bold", fontsize=fs)
    # savefig("yearly dispersion with index OAS")
    plt.show()

    # %%
    index_disp_df = pd.DataFrame(
        {"date": bar_dates, "annual dispersion": disp_gdf["disp"].values}
    )
    index_disp_df.to_csv("annual_index_dispersion.csv")

    gdf[2:-1]
    lr_df = pd.DataFrame(
        {
            "downgrades": x,
            "correlation": gdf["corr"].values[2:-1],
            "dispersion": disp_gdf["disp"].values[2:-1],
        }
    )

    ols = sms.OLS(alpha, sms.add_constant(lr_df)).fit()
    ols.summary()

    # %%
    disp_gdf
    plt.figure()
    plt.plot(disp_gdf["disp"].values[2:-1], alpha, "o")
    plt.show()

    # %%

    alpha_df = pd.DataFrame(
        {
            "date": disp_gdf["disp"].index[2:-1],
            "dispersion": disp_gdf["disp"].values[2:-1],
            "alpha": alpha,
        }
    )
    alpha_df.to_csv("dispersion_vs_alpha.csv")


def libor_vs_long_credit_yield():
    two_yrs_ago = dt.today() - timedelta(365 * 2)
    lc_yield = bdh("LULCYW", "Index", fields="PX_LAST", start=two_yrs_ago) / 100
    libor = bdh("US0003M", "Index", fields="PX_LAST", start=two_yrs_ago) / 100
    diff = (lc_yield - libor).dropna()

    # %%
    lw = 3
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(lc_yield, c="steelblue", lw=lw, label="Long Credit Yield")
    axes[0].plot(libor, c="firebrick", lw=lw, label="3 Month Libor")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim((None, 0.058))
    tick = mtick.StrMethodFormatter("{x:.1%}")
    axes[0].yaxis.set_major_formatter(tick)
    axes[1].plot(diff, c="k", lw=lw, label="Long Credit Yield - Libor")
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(tick)
    axes[1].set_xlabel("Date")
    savefig("libor_vs_long_credit_yield")
    plt.show()


def highlighted_sector_downgrades():
    n_highlight = 5  # number to highlight
    start_year = 2006

    # Load and clean data.
    df = pd.read_csv("Moody_sector_downgrades.csv", index_col=0)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    cols = [c for c in df.columns if int(c) >= start_year]
    df = df[cols].fillna(0)
    sorted(list(df.index))

    # Combine sectors that are similar.
    def combine_sectors(df, sectors, new_sector):
        all_sectors = list(df.index)
        other_sectors = [s for s in all_sectors if s not in sectors]
        df_sectors = pd.DataFrame(
            df.loc[sectors, :].sum(), columns=[new_sector]
        ).T
        return df.loc[other_sectors, :].append(df_sectors)

    ute_sectors = ["ELECTRIC", "NATURAL_GAS", "UTILITY_OTHER"]
    energy_sectors = [
        "INDEPENDENT",
        "INTEGRATED",
        "OIL_FIELD_SERVICES",
        "REFINING",
        "MIDSTREAM",
    ]
    df = combine_sectors(df, ute_sectors, "Utilities")
    df = combine_sectors(df, energy_sectors, "Energy")

    # Find most downgraded sectors and aggregate others together.
    least_downgraded = df.sum(axis=1).sort_values().index[:-n_highlight]
    df = combine_sectors(df, least_downgraded, "Other")
    df.index = [ix.replace("_", " ").title() for ix in df.index]

    # %%
    clrs = [colors(c) for c in "bgry"] + ["#002D72", "grey"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    df.T.plot.bar(stacked=True, ax=ax, color=clrs, rot=0, alpha=0.8)
    ax.set_ylabel("Number of Downgrades", fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(False, axis="x")
    savefig("Moody_sector_downgrades")
    plt.show()


def number_of_issuers_timeseries():
    db = Database()
    db.load_market_data(start="1/1/2014", local=True)
    ix_a = db.build_market_index(rating=("AAA", "A-"), in_stats_index=True)
    ix_b = db.build_market_index(rating=("BBB+", "BBB-"), in_stats_index=True)
    n_issuers = np.zeros([len(ix_a.dates), 2])
    for i, day in enumerate(ix_a.dates):
        df_a = ix_a.day(day)
        df_b = ix_b.day(day)
        n_issuers[i, 0] = len(df_a["Issuer"].dropna().unique())
        n_issuers[i, 1] = len(df_b["Issuer"].dropna().unique())

    # %%
    df = pd.DataFrame(n_issuers, columns=["A", "BBB"], index=ix_a.dates)
    for col in df.columns:
        df[col] = df[col] / df[col][0]
    df *= 100

    df = df[df.index < pd.to_datetime("7/1/2019")]

    df.plot(alpha=0.5, color=["steelblue", "firebrick"])


def market_value_timeseries():
    df = pd.read_csv(
        "data.csv", index_col=0, parse_dates=True, infer_datetime_format=True
    )
    df.sort_index(inplace=True)
    for col in df.columns:
        df[col] = [float(cell.replace(",", "")) for cell in df[col]]
    df = df.divide(df.sum(axis=1), axis=0)
    df = df[df.index > pd.to_datetime("6/1/2008")]

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.stackplot(
        df.index,
        df["BBB"],
        df["A"],
        df["AA"],
        df["AAA"],
        # labels=,
        colors=[colors(c) for c in "rbyg"],
        alpha=0.8,
    )
    tick = mtick.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)
    plt.margins(0, 0)
    ax.grid(False)
    ax.set_title("Market Value Contribution to Index")
    savefig("market_value_contribution_to_index")
    plt.show()


def ebitda_by_sector():
    # %%
    df = pd.read_clipboard(
        names=["sector", "change_mn", "change", "level"], sep="\s{2,}"
    )
    df["sector"] = df["sector"].str.replace(" e", "e")
    df["change"] = [float(val.replace("%", "")) for val in df["change"]]

    # %%
    pal = sns.set_palette("coolwarm")
    fig, ax = plt.subplots(1, 1, figsize=[8, 8])
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["change"], color=coolwarm(df["change"]), alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["sector"])
    ax.grid(False, axis="y")
    tick = mtick.StrMethodFormatter("{x:.0f}%")
    ax.xaxis.set_major_formatter(tick)
    ax.set_title("EBITDA Growth by Sector (y/y)")
    savefig("EBITDA_by_sector")
    plt.show()
    # %%
