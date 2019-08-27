import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import statsmodels.api as sms
import squarify

pd.plotting.register_matplotlib_converters()
plt.style.use("fivethirtyeight")

from lgimapy.utils import savefig

# %matplotlib qt
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
    db.load_market_data(dev=True)
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
        f"{i}\n${mv * 1e-6:.1f}B" if cs < 0.47 else ""
        for i, mv, cs in zip(df.index, df.values.ravel(), cumsum)
    ]

    df.sort_values("market_value", inplace=True)
    # %%
    plt.figure(figsize=[18, 12])
    colors = ["#DD2026", "#209CD8", "#FDD302", "#09933C"]
    squarify.plot(
        sizes=df["market_value"],
        pad=0.0001,
        # pad=False,
        label=labels[::-1],
        color=colors,
        alpha=0.7,
        # fontsize=8,
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
    from lgimapy.data import Index
    from lgimapy.bloomberg import bdh

    # %%
    db = Database()
    db.load_market_data(local=True)

    # %%
    ix = db.build_market_index(
        start="1/1/2007",
        rating="IG",
        is_144A=False,
        OAS=(1e-6, 3000),
        maturity=(10, None),
        issuer=issuers,
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
    luacoas = 100 * bdh("LUACOAS", "Index", start="1/1/2007", field="PX_BID")
    import matplotlib.ticker as ticker
    import matplotlib.dates as mdates

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
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax2 = ax.twinx()
    bar_dates = [pd.to_datetime(f"6/1/{y}") for y in disp_gdf.index]
    ticks = [pd.to_datetime(f"1/1/{y}") for y in disp_gdf.index]
    ax2.bar(
        bar_dates, disp_gdf["disp"].values, width=200, color="grey", alpha=0.4
    )
    tick = mtick.StrMethodFormatter("{x:.0f}")
    ax2.yaxis.set_major_formatter(tick)
    ax2.grid(False)
    ax.set_ylim((None, 700))
    # ax2.set_ylim((0.3, 0.65))
    ax.plot(luacoas.index, luacoas.values, lw=5, c="#209CD8")
    ax.grid(False, axis="x")
    ax.set_xticks(bar_dates)
    for t in ticks:
        ax.axvline(t, ymax=0.01, lw=0.5, c="k")
    ax.set_ylabel("Long Credit Index OAS", color="#209CD8", weight="bold")
    ax2_ylab = "Annual Dispersion (bp)"
    ax2.set_ylabel(ax2_ylab, color="grey", weight="bold")
    ax.set_xlabel("Date", weight="bold")
    savefig("yearly dispersion with index OAS")
    plt.show()

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
