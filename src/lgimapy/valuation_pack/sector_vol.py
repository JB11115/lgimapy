import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import linregress
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import load_json, root


# %%


def update_volatility_indicators():
    """Update volatility indicator figures."""
    vis.style()
    path = root("latex/valuation_pack/fig")
    sector_df, history_dict = get_sector_vol_data()
    sector_df
    plot_sector_spread_vs_vol(sector_df, path=path)
    plot_regression_timeseries_results(history_dict, path=path)
    plot_maturity_bucket_spread_vs_vol()


def diagonal_deviation(x, y, threshold):
    """
    Find diagonal deviation for data points from
    a linear regression.
    """
    reg_res = sm.RLM(y, sm.add_constant(x)).fit()
    alpha, beta = reg_res.params
    y_resid = y - (x * beta + alpha)
    x_resid = x - (y - alpha) / beta
    deviation = (
        np.sign(y_resid)
        * np.abs(x_resid * y_resid)
        / (x_resid ** 2 + y_resid ** 2) ** 0.5
    )
    return (deviation - np.mean(deviation)) / np.std(deviation)


def get_sector_vol_data():
    """
    Get historical sector volatility and spread data, and a
    current snapshot of data including market value and rating.
    """
    sectors = [
        "CHEMICALS",
        "METALS_AND_MINING",
        "CAPITAL_GOODS",
        "CABLE_SATELLITE",
        "MEDIA_ENTERTAINMENT",
        "WIRELINES_WIRELESS",
        "AUTOMOTIVE",
        "RETAILERS",
        "FOOD_AND_BEVERAGE",
        "HEALTHCARE_EX_MANAGED_CARE",
        "MANAGED_CARE",
        "PHARMACEUTICALS",
        "TOBACCO",
        "INDEPENDENT",
        "INTEGRATED",
        "OIL_FIELD_SERVICES",
        "REFINING",
        "MIDSTREAM",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "SIFI_BANKS_SR",
        "SIFI_BANKS_SUB",
        "YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "FINANCE_COMPANIES",
        "LIFE",
        "P&C",
        "REITS",
        "UTILITY",
        "OWNED_NO_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SOVEREIGN",
        "UNIVERSITY",
    ]
    db = Database()
    db.load_market_data(start=db.date("1.2y"), local=True)

    ratings = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
    kwargs = load_json("indexes")

    warnings.simplefilter(action="ignore", category=FutureWarning)
    ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
    d = defaultdict(list)
    history_d = defaultdict(list)
    top_level_sector = "Industrials"
    for sector in sectors:
        for rating, rating_kws in ratings.items():
            ix_sector = ix.subset(**kwargs[sector], rating=rating_kws)
            # Get top level sector.
            if sector == "SIFI_BANKS_SR":
                top_level_sector = "Financials"
            elif sector == "UTILITY":
                top_level_sector = "Non-Corp"

            # Calculate spread and volatility.
            oas = ix_sector.market_value_weight("OAS")
            vol = ix_sector.market_value_weight_vol("OAS")
            if not len(vol):
                # Not enough days to create volatility.
                continue

            # Calculate current values of each sector.
            curr_ix_sector = ix_sector.day(ix.dates[-1], as_index=True)

            # Store historical results.
            history_d["vol"].append(vol.rename(f"{ix_sector.name}|{rating}"))
            history_d["oas"].append(oas.rename(f"{ix_sector.name}|{rating}"))
            # Store current snapshot.
            d["name"].append(ix_sector.name)
            d["top_level_sector"].append(top_level_sector)
            d["sector"].append(ix_sector.name)
            d["rating"].append(rating)
            d["vol"].append(vol[-1])
            d["oas"].append(oas[-1])
            d["oad"].append(curr_ix_sector.market_value_weight("OAD").iloc[-1])
            d["XSRet"].append(
                1e4
                * ix_sector.aggregate_excess_returns(start_date=db.date("YTD"))
            )
            d["market_val"].append(ix_sector.total_value()[-1])
    warnings.simplefilter(action="default", category=FutureWarning)

    return pd.DataFrame(d), history_d


def plot_sector_spread_vs_vol(df, path):
    """Plot sectors vs spread with robust regression line."""
    # Create labeled columns to use with seaborn's scatter plot.
    df["\nMarket Value ($B)"] = (df["market_val"] / 1e3).astype(int)
    df["\nRating"] = df["rating"]
    df[" "] = df["top_level_sector"]  # no label in legend

    fig, ax = vis.subplots(figsize=(10, 6))

    # Plot robust regression results.
    sns.regplot(
        x=df["vol"],
        y=df["oas"],
        robust=True,
        color="k",
        line_kws={"lw": 2, "alpha": 0.5},
        scatter_kws={"alpha": 0},
        n_boot=1000,
        ci=95,
        ax=ax,
    )

    # Plot scatter points.
    sns.scatterplot(
        x="vol",
        y="oas",
        hue=" ",
        style="\nRating",
        size="\nMarket Value ($B)",
        sizes=(40, 200),
        alpha=0.7,
        palette="dark",
        data=df,
        markers=["D", "o"],
        ax=ax,
    )

    # Annotate labels with diagonal deviation from regression line
    # greater than set threshold of 1 z-score.
    threshold = 1
    df["deviation"] = diagonal_deviation(df["vol"], df["oas"], threshold)
    for _, row in df.iterrows():
        if np.abs(row["deviation"]) < threshold:
            # Not an outlier -> don't label.
            continue
        ax.annotate(
            row["name"],
            xy=(row["vol"], row["oas"]),
            xytext=(1, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Sector Volatility Snapshot", fontweight="bold")
    ax.set_xlabel("Daily Spread Volatility (bp)")
    ax.set_ylabel("OAS (bp)")
    ax.legend(loc=2, bbox_to_anchor=(1.02, 1), prop={"size": 10})
    vis.savefig("sector_spread_vs_vol", path=path)
    vis.close()


def plot_regression_timeseries_results(history_dict, path):
    """
    Run robust regression for every day in history.
    plot alpha and beta for regression through time,
    and plot timeseries of Z-scores for sectors'
    deviations from the regression line.
    """
    # %%
    vol_df = pd.concat(history_dict["vol"], axis=1, sort=True)
    oas_df = pd.concat(history_dict["oas"], axis=1, sort=True)
    cols, dates = vol_df.columns, vol_df.index
    n, m = vol_df.shape

    # Run robust regression for each day.
    alpha_a, beta_a, scale_a = np.zeros(n), np.zeros(n), np.zeros(n)
    deviation_a = np.zeros((n, m))
    vol = vol_df.values
    oas = oas_df.values
    i = 0
    n_lookback_days = 20
    warnings.simplefilter(action="ignore", category=RuntimeWarning)

    for i in range(n):
        # Peform regression for current date.
        i_start = max(0, i - n_lookback_days)
        x = vol[i_start : i + 1, :].ravel()
        y = oas[i_start : i + 1, :].ravel()
        # x, y = vol[i, :], oas[i, :]
        mask = ~(np.isnan(y) | np.isnan(x))  # remove nans
        reg_res = sm.RLM(
            y[mask], sm.add_constant(x[mask]), M=sm.robust.norms.Hampel()
        ).fit()
        # reg_res.summary()
        alpha_a[i], beta_a[i] = reg_res.params
        scale_a[i] = reg_res.scale

        # Compute deviation from regression line for each sector.
        y_resid = (y - (x * beta_a[i] + alpha_a[i]))[-m:]
        x_resid = (x - (y - alpha_a[i]) / beta_a[i])[-m:]

        deviation_a[i, :] = (
            np.sign(y_resid)
            * np.abs(x_resid * y_resid)
            / (x_resid ** 2 + y_resid ** 2) ** 0.5
        )
    warnings.simplefilter(action="default", category=RuntimeWarning)

    alpha = pd.Series(alpha_a, index=dates, name="$\\alpha$")
    beta = pd.Series(beta_a, index=dates, name="$\\beta$")
    scale = pd.Series(scale_a, index=dates, name="$\\beta$")

    dev_df = pd.DataFrame(deviation_a, index=dates, columns=cols)

    def zscore(df, window, min_periods=0):
        roll = df.rolling(window=window, min_periods=min_periods)
        mean = roll.mean().shift(1)
        std = roll.std(ddof=0).shift(1)
        z_score_df = (df - mean) / std
        # Remove days with no values.
        start = max(2, min_periods)
        return z_score_df.iloc[start:, :]  # remove days with no values

    z_score_df = zscore(dev_df, window=300, min_periods=60)

    # %%
    # Plot alpha and beta timeseries.
    # vis.plot_double_y_axis_timeseries(
    #     alpha,
    #     beta,
    #     ylabel_left="$\\alpha$",
    #     ylabel_right="$\\beta$",
    #     plot_kws_left={"color": "darkgoldenrod"},
    #     plot_kws_right={"color": "navy"},
    #     alpha=0.7,
    #     lw=4,
    #     figsize=(8, 6),
    #     xtickfmt="auto",
    # )
    vis.plot_timeseries(
        beta,
        title="Compensation for Volatility",
        ylabel="bp increase in OAS per\nbp increase in Daily Spread Vol",
        xtickfmt="auto",
        color="navy",
        lw=4,
    )
    # vis.show()
    vis.savefig(f"sector_vol_beta", path=path)
    vis.close()
    # %%

    # Plot sectors timeseries through time.
    n_out = 4
    current_dev = z_score_df.iloc[-1, :].sort_values()
    ix = current_dev.index
    outliers = list(ix[:n_out]) + list(ix[-n_out:])

    fig, ax = vis.subplots(figsize=(15, 8))
    for col in z_score_df.columns:
        ax.plot(
            z_score_df[col], c="lightgrey", lw=1, alpha=0.7, label="_nolegend_"
        )
    colors = sns.color_palette("husl", n_out * 2).as_hex()
    for sector, color in zip(reversed(outliers), colors):
        vis.plot_timeseries(
            z_score_df[sector],
            lw=2.5,
            color=color,
            alpha=0.9,
            label=f"{sector.replace('|', ' (')})",
            ax=ax,
        )
    vis.format_xaxis(ax, s=z_score_df, xtickfmt="auto")
    ax.set_title("Sector Volatility Compensation Timeseries", fontweight="bold")
    ax.set_ylabel("Deviation Z-score")
    ax.legend(loc="upper left", fancybox=True, shadow=True)
    # vis.show()
    vis.savefig(f"sector_vol_timeseries", path=path)
    vis.close()


def plot_maturity_bucket_spread_vs_vol():
    """
    Plot current spread vs volatility subset by maturity bucket,
    rating, and fin/non-fin/non-corp.
    """
    # %%
    db = Database()
    db.load_market_data(start=db.date("2m"), local=True)

    maturity_buckets = {
        "20+": (20, None),
        "10-20": (10, 20),
        "7-10": (7, 10),
        "2-7": (2, 7),
    }
    rating_buckets = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
    sector_buckets = {"Industrials": 0, "Financials": 1, "Non-Corp": 2}

    warnings.simplefilter(action="ignore", category=FutureWarning)
    ix = db.build_market_index(in_stats_index=True)
    d = defaultdict(list)
    for maturity_cat, maturities in maturity_buckets.items():
        for rating_cat, ratings in rating_buckets.items():
            for sector, fin_flag in sector_buckets.items():
                ix_cat = ix.subset(
                    maturity=maturities, rating=ratings, financial_flag=fin_flag
                )
                # Calculate spread and volatility.
                oas = ix_cat.market_value_weight("OAS")
                vol = ix_cat.market_value_weight_vol("OAS")
                if not len(vol):
                    # Not enough days to create volatility.
                    continue
                # Store current snapshot.
                d["\nMaturity Bucket"].append(maturity_cat)
                d[" "].append(sector)
                d["\nRating"].append(rating_cat)
                d["vol"].append(vol[-1])
                d["oas"].append(oas[-1])
    warnings.simplefilter(action="default", category=FutureWarning)
    df = pd.DataFrame(d)

    # %%

    fig, ax = vis.subplots(figsize=(8, 7.5))
    sns.scatterplot(
        x="vol",
        y="oas",
        hue=" ",
        style="\nRating",
        size="\nMaturity Bucket",
        sizes=(40, 200),
        alpha=0.7,
        palette="dark",
        markers=["D", "o"],
        data=df,
        ax=ax,
    )
    ax.set_title("Maturity Bucket Volatility", fontweight="bold")
    ax.set_xlabel("Daily Spread Volatility (bp)")
    ax.set_ylabel("OAS (bp)")
    ax.legend(loc=2, bbox_to_anchor=(1.02, 1), prop={"size": 10})
    vis.savefig("maturity_bucket_spread_vs_vol", path=path)
    vis.close()


def plot_sector_xsret_vs_oad(sector_df):
    """
    Plot YTD excess returns vs duration for each sector.
    """
    # %%

    df = sector_df.copy()

    # %%

    df["\nMarket Value ($B)"] = (df["market_val"] / 1e3).astype(int)
    df["\nRating"] = df["rating"]
    df[" "] = df["top_level_sector"]  # no label in legend

    fig, ax = vis.subplots(figsize=(10, 6))

    # Plot robust regression results.
    sns.regplot(
        x=df["vol"],
        y=df["oad"],
        robust=True,
        color="k",
        line_kws={"lw": 2, "alpha": 0.5},
        scatter_kws={"alpha": 0},
        n_boot=1000,
        ci=95,
        ax=ax,
    )

    # Plot scatter points.
    sns.scatterplot(
        x="vol",
        y="oas",
        hue=" ",
        style="\nRating",
        size="\nMarket Value ($B)",
        sizes=(40, 200),
        alpha=0.7,
        palette="dark",
        data=df,
        markers=["D", "o"],
        ax=ax,
    )
    ax.set_title("Sector Excess Returns", fontweight="bold")
    ax.set_xlabel("YTD Excess Returns (bp)")
    ax.set_ylabel("OAD (yrs)")
    ax.legend(loc=2, bbox_to_anchor=(1.02, 1), prop={"size": 10})
    vis.show()
