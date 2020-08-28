import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from adjustText import adjust_text
from scipy.stats import linregress
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database, convert_sectors_to_fin_flags
from lgimapy.latex import Document
from lgimapy.models import rolling_zscore
from lgimapy.utils import load_json, root

from volatility import update_voltility_model

# %%


def update_volatility_indicators(fid, db):
    """Update volatility indicator figures."""
    # db = Database()
    # db.load_market_data(start=db.date("5y"), local=True)
    # fid = 'temp'
    vis.style()
    update_voltility_model()
    path = root("reports/valuation_pack/fig")
    page_maturities = {
        "Long Credit": (10, None),
        "5-10 yr Credit": (5, 10),
    }
    doc = Document(
        fid, path="reports/valuation_pack", fig_dir=True, load_tex=True
    )
    for page, maturity in page_maturities.items():
        table_df = get_sector_table(maturity, path=path, db=db)
        plot_sector_spread_vs_vol(table_df, maturity, path=path)
        make_sector_table(table_df, page, doc)
        make_1m_sector_table(table_df, maturity, page, doc, db=db)
    plot_maturity_bucket_spread_vs_vol(path=path, db=db)
    doc.save_tex()


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


def get_sector_table(maturity, path, db):
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
        "US_REGIONAL_BANKS",
        "YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "FINANCE_COMPANIES",
        "LIFE",
        "P&C",
        "REITS",
        "UTILITY",
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SUPRANATIONAL",
        "SOVEREIGN",
        "UNIVERSITY",
    ]

    ratings = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
    kwargs = load_json("indexes")

    warnings.simplefilter(action="ignore", category=FutureWarning)
    ix = db.build_market_index(
        in_stats_index=True, maturity=maturity, start=db.date("1.2y")
    )
    model_xsret_df = get_forecasted_xsrets(db)
    d = defaultdict(list)
    history_d = defaultdict(list)
    top_level_sector = "Industrials"
    for sector in sectors:
        for rating, rating_kws in ratings.items():
            # break
            # rating = "BBB"
            # rating_kws = ("BBB+", "BBB-")
            # sector = "AUTOMOTIVE"
            # rating = "A"
            # rating_kws = ("AAA", "A-")
            ix_sector = ix.subset(**kwargs[sector], rating=rating_kws)
            # Get top level sector.
            if sector == "SIFI_BANKS_SR":
                top_level_sector = "Financials"
            elif sector == "UTILITY":
                top_level_sector = "Non-Corp"

            # Calculate volatility.
            vol = ix_sector.market_value_weight_vol("OAS")
            if not len(vol):
                # Not enough days to create volatility.
                continue

            # Calculate current values of each sector.
            try:
                curr_ix_sector = ix_sector.day(ix.dates[-1], as_index=True)
            except KeyError:
                # Sector has some history but no longer exists.
                continue

            # Calculate spread.
            ix_sector_corrected = ix_sector.drop_ratings_migrations()
            try:
                oas = ix_sector_corrected.get_synthetic_differenced_history(
                    "OAS"
                )
            except UnboundLocalError:
                # Not enough history.
                continue

            oas_ytd = oas[oas.index >= db.date("YTD")]
            oas_1m = oas[oas.index >= db.date("1m")]
            oas_min = np.min(oas_ytd)

            # Calculate forecasted vs realized excess returns.
            # %%
            ix_sector_1m = ix_sector.subset(start=db.date("1m"))
            sector_model_xsret_df = model_xsret_df[
                (model_xsret_df["RatingBucket"] == rating)
                & model_xsret_df.index.isin(ix_sector_1m.cusips)
            ]
            # temp_df = pd.concat(
            #     (
            #         sector_model_xsret_df,
            #         df[["Issuer", "MaturityDate", "IssueDate", "OAS", "OAD"]],
            #     ),
            #     join="inner",
            #     axis=1,
            # )
            # temp_df["ModelXSRet"] = temp_df["ModelXSRet"].values * 1e4
            # temp_df["RealXSRet"] = temp_df["RealXSRet"].values * 1e4
            # temp_df["OutPerform"] = temp_df["RealXSRet"] - temp_df["ModelXSRet"]
            # temp_df["weight"] = (
            #     temp_df["weight"].values / temp_df["weight"].sum()
            # )
            # temp_df.to_csv(f"{rating}_Autos_XSRet.csv")
            # %%
            for col in ["ModelXSRet", "RealXSRet"]:
                d[col].append(
                    1e4
                    * np.sum(
                        sector_model_xsret_df[col]
                        * sector_model_xsret_df["weight"]
                    )
                    / sector_model_xsret_df["weight"].sum()
                )
            d["ModelResid"].append(d["ModelXSRet"][-1] - d["RealXSRet"][-1])

            # Store historical results.
            history_d["vol"].append(vol.rename(f"{ix_sector.name}|{rating}"))
            history_d["oas"].append(oas.rename(f"{ix_sector.name}|{rating}"))
            # Store current snapshot.
            d["name"].append(ix_sector.name)
            d["top_level_sector"].append(top_level_sector)
            d["raw_sector"].append(sector)
            d["sector"].append(ix_sector.name)
            d["rating"].append(rating)
            d["vol"].append(vol[-1])
            d["oas"].append(oas[-1])
            d["oas_1m_chg"].append(oas[-1] - oas_1m[0])
            d["oas_ytd_chg"].append(oas[-1] - oas_ytd[0])
            d["oas_min"].append(np.min(oas_ytd))
            d["oas_max"].append(np.max(oas_ytd))
            d["oas_%tile"].append(oas_ytd.rank(pct=True)[-1])
            d["oas_%_widening"].append((oas[-1] / oas_min) - 1)
            d["oad"].append(curr_ix_sector.market_value_weight("OAD").iloc[-1])
            d["XSRet"].append(
                1e4
                * ix_sector.aggregate_excess_returns(start_date=db.date("YTD"))
            )
            d["market_val"].append(ix_sector.total_value()[-1] / 1e3)
    warnings.simplefilter(action="default", category=FutureWarning)
    table_df = pd.DataFrame(d)

    # Add volatility model results to table.
    vol_z_scores = vol_model_zscores(history_d, maturity, path)
    vol_model = np.full(len(table_df), np.NaN)
    for i, key in enumerate(table_df.set_index("name")["rating"].items()):
        vol_model[i] = vol_z_scores.get(key, np.NaN)
    # Make negative to have same sign as xsret model.
    table_df["vol_model"] = -vol_model

    # Add xsret model results to table.
    xs_ret_z_scores_df = xsret_model_zscores(sectors, maturity, db)
    xs_ret_d, xs_ret_change_d = {}, {}
    for idx, vals in xs_ret_z_scores_df.iterrows():
        key = tuple(idx.split("|"))
        xs_ret_d[key] = vals["current"]
        xs_ret_change_d[key] = vals["prev"]

    curr_xs_rets = np.full(len(table_df), np.NaN)
    prev_xs_rets = np.full(len(table_df), np.NaN)
    for i, key in enumerate(table_df.set_index("raw_sector")["rating"].items()):
        curr_xs_rets[i] = xs_ret_d.get(key, np.NaN)
        prev_xs_rets[i] = xs_ret_change_d.get(key, np.NaN)
    table_df["xsret_model"] = curr_xs_rets
    table_df["xsret_prev"] = prev_xs_rets
    # temp_fid = root("src/lgimapy/valuation_pack/temp.csv")
    # table_df.dropna(subset=["xsret_model", "vol_model"]).to_csv(temp_fid)
    return table_df.dropna(subset=["xsret_model", "vol_model"])


def plot_sector_spread_vs_vol(df, maturity, path):
    """Plot sectors vs spread with robust regression line."""
    # Create labeled columns to use with seaborn's scatter plot.
    df["\nMarket Value ($B)"] = (df["market_val"]).astype(int)
    df["\nRating"] = df["rating"]
    df[" "] = df["top_level_sector"]  # no label in legend

    fig, ax = vis.subplots(figsize=(9, 5))

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
    texts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in df.iterrows():
            if np.abs(row["deviation"]) < threshold:
                # Not an outlier -> don't label.
                continue
            texts.append(
                ax.annotate(
                    row["name"],
                    xy=(row["vol"], row["oas"]),
                    xytext=(1, 3),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=8,
                )
            )
        adjust_text(texts)
    ax.set_title("Long Credit Sector Volatility", fontweight="bold")
    ax.set_xlabel("Daily Spread Volatility (bp)")
    ax.set_ylabel("OAS (bp)")
    ax.legend(loc=2, bbox_to_anchor=(1.02, 1), prop={"size": 10})
    mat_fid = f"{maturity[0]}_{maturity[1]}"
    plot_fid = f"sector_spread_vs_vol_{mat_fid}"
    vis.savefig(plot_fid, path=path)
    vis.close()


def vol_model_zscores(history_dict, maturity, path):
    """
    Run robust regression for every day in history.
    plot alpha and beta for regression through time,
    and plot timeseries of Z-scores for sectors'
    deviations from the regression line.
    """
    maturity_fid = f"{maturity[0]}_{maturity[1]}"
    # history_dict = history_d.copy()
    vol_df = pd.concat(history_dict["vol"], axis=1, sort=True)
    oas_df = pd.concat(history_dict["oas"], axis=1, sort=True)
    vol_df.shape
    oas_df.shape
    oas_df
    cols, dates = vol_df.columns, vol_df.index
    n, m = vol_df.shape

    # Run robust regression for each day.
    alpha_a, beta_a, scale_a = np.zeros(n), np.zeros(n), np.zeros(n)
    deviation_a = np.zeros((n, m))
    vol = vol_df.values
    oas = oas_df.values
    i = 0
    n_lookback_days = 20
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        for i in range(n):
            # Peform regression for current date.
            i_start = max(0, i - n_lookback_days)
            x = vol[i_start : i + 1, :].ravel()
            y = oas[i_start : i + 1, :].ravel()
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

    alpha = pd.Series(alpha_a, index=dates, name="$\\alpha$")
    beta = pd.Series(beta_a, index=dates, name="$\\beta$")
    scale = pd.Series(scale_a, index=dates, name="$\\beta$")

    dev_df = pd.DataFrame(deviation_a, index=dates, columns=cols)
    z_score_df = rolling_zscore(dev_df, window=300, min_periods=60)

    vis.plot_timeseries(
        beta,
        start=Database().date("5m"),
        title="Compensation for Volatility",
        ylabel="bp increase in OAS per\nbp increase in Daily Spread Vol",
        xtickfmt="auto",
        color="navy",
        lw=4,
    )
    vis.savefig(f"sector_vol_beta_{maturity_fid}", path=path)
    vis.close()

    # Plot sectors timeseries through time.
    # n_out = 4
    # current_dev = z_score_df.iloc[-1, :].sort_values()
    # ix = current_dev.index
    # outliers = list(ix[:n_out]) + list(ix[-n_out:])
    #
    # fig, ax = vis.subplots(figsize=(15, 8))
    # for col in z_score_df.columns:
    #     ax.plot(
    #         z_score_df[col],
    #         c="lightgrey",
    #         lw=1,
    #         alpha=0.7,
    #         label="_nolegend_",
    #     )
    # colors = sns.color_palette("husl", n_out * 2).as_hex()
    # for sector, color in zip(reversed(outliers), colors):
    #     vis.plot_timeseries(
    #         z_score_df[sector],
    #         lw=2.5,
    #         color=color,
    #         alpha=0.9,
    #         label=f"{sector.replace('|', ' (')})",
    #         ax=ax,
    #     )
    # vis.format_xaxis(ax, s=z_score_df, xtickfmt="auto")
    # ax.set_title(
    #     "Long Credit Sector Volatility Compensation Timeseries",
    #     fontweight="bold",
    # )
    # ax.set_ylabel("Deviation Z-score")
    # ax.legend(loc="upper left", fancybox=True, shadow=True)
    # vis.savefig(f"sector_vol_timeseries_{maturity_fid)}", path=path)
    # vis.close()

    # Return current values for each sector/rating combination.
    current_zscores_s = z_score_df.iloc[-1, :]
    current_zscores = {
        tuple(sector.split("|")): z for sector, z in current_zscores_s.items()
    }
    return current_zscores


def plot_maturity_bucket_spread_vs_vol(path, db):
    """
    Plot current spread vs volatility subset by maturity bucket,
    rating, and fin/non-fin/non-corp.
    """
    maturity_buckets = {
        "20+": (20, None),
        "10-20": (10, 20),
        "7-10": (7, 10),
        "2-7": (2, 7),
    }
    rating_buckets = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
    sector_buckets = {"Industrials": 0, "Financials": 1, "Non-Corp": 2}

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        ix = db.build_market_index(in_stats_index=True)
        d = defaultdict(list)
        for maturity_cat, maturities in maturity_buckets.items():
            for rating_cat, ratings in rating_buckets.items():
                for sector, fin_flag in sector_buckets.items():
                    ix_cat = ix.subset(
                        maturity=maturities,
                        rating=ratings,
                        financial_flag=fin_flag,
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
    df = pd.DataFrame(d)

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
    ax.set_title("Volatility by Maturity", fontweight="bold")
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

    df["\nMarket Value ($B)"] = (df["market_val"]).astype(int)
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


def make_sector_table(sector_df, name, doc):
    col_map = {
        "raw_sector": "raw_sector",
        "name": "Sector",
        "rating": "Rating",
        "oas": "OAS*(bp)",
        "oas_ytd_chg": "YTD*$\\Delta$OAS",
        "oas_min": "YTD*Min",
        "oas_max": "YTD*Max",
        "oas_%_widening": "% Wide*of Tights",
        "oas_%tile": "YTD*%tile",
        "oad": "OAD*(yr)",
        "market_val": "Mkt Val*(\\$B)",
        "XSRet": "YTD*XSRet",
        "vol": "$\\sigma$*(bp)",
        "vol_model": "Vol*Model",
        "xsret_model": "XSRet*Model",
        "xsret_prev": "xsret_prev",
    }
    prec = {
        "OAS*(bp)": "0f",
        "YTD*$\\Delta$OAS": "0f",
        "YTD*Min": "0f",
        "YTD*Max": "0f",
        "YTD*%tile": "0f",
        "% Wide*of Tights": "0%",
        "OAD*(yr)": "1f",
        "Mkt Val*(\\$B)": "0f",
        "YTD*XSRet": "0f",
        "$\\sigma$*(bp)": "1f",
        "Vol*Model": "1f",
        "XSRet*Model": "1f",
        "Model*Change": "0f",
    }
    # df_ = table_df.copy()
    # df = df_.copy()
    kwargs = load_json("indexes")

    df = sector_df.copy()
    df["fin"] = convert_sectors_to_fin_flags(df["raw_sector"])
    df["oas_%tile"] = 100 * df["oas_%tile"].values

    fin_flags = {
        f"{name} Non-Fin Sector Breakdown": 0,
        f"{name} Fin Sector Breakdown": 1,
        f"{name} Non-Corp Sector Breakdown": 2,
    }

    for caption, fin_flag in fin_flags.items():
        table = (
            df[df["fin"] == fin_flag][col_map.keys()]
            .sort_values("xsret_model")
            .rename(columns=col_map)
            .reset_index(drop=True)
        )
        # sorted(list(table_df['raw_sector'].unique()))
        # Get row colors for level 3 sectors.
        sector_row_colors = {
            "ENERGY": "magicmint",
            "COMMUNICATIONS": "opal",
            "CONSUMER_NON_CYCLICAL": "sage",
            "BANKS": "powderblue",
            "UTILITY": "magicmint",
            "SOVS": "sage",
            "OTHER": "opal",
        }
        sector_locs = {}
        for sector, color in sector_row_colors.items():
            bottom_sector_names = {
                "BANKS": {
                    "SIFI_BANKS_SR",
                    "SIFI_BANKS_SUB",
                    "US_REGIONAL_BANKS",
                    "YANKEE_BANKS",
                },
                "COMMUNICATIONS": {
                    "WIRELINES_WIRELESS",
                    "CABLE_SATELLITE",
                    "MEDIA_ENTERTAINMENT",
                },
                "ENERGY": {
                    "INDEPENDENT",
                    "MIDSTREAM",
                    "INTEGRATED",
                    "OIL_FIELD_SERVICES",
                    "REFINING",
                },
                "CONSUMER_NON_CYCLICAL": {
                    "FOOD_AND_BEVERAGE",
                    "HEALTHCARE_EX_MANAGED_CARE",
                    "MANAGED_CARE",
                    "PHARMACEUTICALS",
                    "TOBACCO",
                },
                "UTILITY": {"UTILITY"},
                "SOVS": {"OWNED_NO_GUARANTEE", "SOVEREIGN"},
                "OTHER": {"HOSPITALS", "MUNIS", "UNIVERSITY"},
            }[sector]

            locs = tuple(
                table.loc[table["raw_sector"].isin(bottom_sector_names)].index
            )
            if locs:
                sector_locs[locs] = color
        table["Model*Change"] = -rank_change(table, "XSRet*Model", "xsret_prev")
        table.drop(["raw_sector", "xsret_prev"], axis=1, inplace=True)

        doc.start_edit(f"{name.replace(' ', '_')}_{fin_flag}_sector_table")
        doc.add_table(
            table,
            # hide_index=True,
            prec=prec,
            col_fmt="llc|cccccl|cccc|ccl",
            caption=caption,
            adjust=True,
            font_size="scriptsize",
            multi_row_header=True,
            row_color=sector_locs,
            col_style={"YTD*%tile": "\pctbar{6}"},
            gradient_cell_col=["Vol*Model", "XSRet*Model"],
            gradient_cell_kws={"cmax": "orchid", "cmin": "orange",},
            arrow_col="Model*Change",
            arrow_kws={
                "cmax": "orange",
                "cmin": "orchid",
                "vmax": len(table) // 3,
                "vmin": len(table) // 3,
            },
        )
        doc.end_edit()


def make_1m_sector_table(sector_df, maturity, name, doc, db):
    col_map = {
        "raw_sector": "raw_sector",
        "name": "Sector",
        "rating": "Rating",
        "oas": "OAS*(bp)",
        "oas_1m_chg": "1M*$\\Delta$OAS",
        "RealXSRet": "Real*XSRet",
        "ModelXSRet": "FCast*XSRet",
        "ModelResid": "Out*Perform",
    }
    prec = {
        "OAS*(bp)": "0f",
        "1M*$\\Delta$OAS": "0f",
        "Real*XSRet": "0f",
        "FCast*XSRet": "0f",
        "Out*Perform": "0f",
    }
    kwargs = load_json("indexes")

    df = sector_df.copy()
    df["fin"] = convert_sectors_to_fin_flags(df["raw_sector"])
    df["ModelResid"] = -df["ModelResid"].values

    # Add index to df.
    # df = table_df.copy()
    ix = db.build_market_index(
        start=db.date("1m"), maturity=maturity, in_stats_index=True
    )
    ix_xsret = 1e4 * ix.aggregate_excess_returns()
    ix_corrected = ix.drop_ratings_migrations()
    ix_oas = ix_corrected.get_synthetic_differenced_history("OAS")
    df = df.append(pd.Series(dtype="object"), ignore_index=True)
    n = len(df) - 1
    df.loc[n, "raw_sector"] = "index"
    df.loc[n, "name"] = f"{name} Stats Index"
    df.loc[n, "rating"] = "-"
    df.loc[n, "oas"] = ix_oas[-1]
    df.loc[n, "oas_1m_chg"] = ix_oas[-1] - ix_oas[0]
    df.loc[n, "RealXSRet"] = 1e4 * ix.aggregate_excess_returns()
    df.loc[n, "ModelXSRet"] = 0
    df.loc[n, "ModelResid"] = 0
    fin_flags = {
        f"{name} Non-Fin Sector 1M Performance": 0,
        f"{name} Fin Sector 1M Performance": 1,
        f"{name} Non-Corp Sector 1M Performance": 2,
    }
    date = db.date("1m").strftime("%m/%d/%Y")

    for caption, fin_flag in fin_flags.items():
        table = (
            df[(df["fin"] == fin_flag) | (df["raw_sector"] == "index")][
                col_map.keys()
            ]
            .dropna()
            .sort_values("ModelResid", ascending=False)
            .rename(columns=col_map)
            .reset_index(drop=True)
        )
        # Get row colors for level 3 sectors.
        sector_row_colors = {
            "ENERGY": "magicmint",
            "COMMUNICATIONS": "opal",
            "CONSUMER_NON_CYCLICAL": "sage",
            "BANKS": "powderblue",
            "UTILITY": "magicmint",
            "SOVS": "sage",
            "OTHER": "opal",
        }
        sector_locs = {}
        for sector, color in sector_row_colors.items():
            bottom_sector_names = {
                "BANKS": {
                    "SIFI_BANKS_SR",
                    "SIFI_BANKS_SUB",
                    "US_REGIONAL_BANKS",
                    "YANKEE_BANKS",
                },
                "COMMUNICATIONS": {
                    "WIRELINES_WIRELESS",
                    "CABLE_SATELLITE",
                    "MEDIA_ENTERTAINMENT",
                },
                "ENERGY": {
                    "INDEPENDENT",
                    "MIDSTREAM",
                    "INTEGRATED",
                    "OIL_FIELD_SERVICES",
                    "REFINING",
                },
                "CONSUMER_NON_CYCLICAL": {
                    "FOOD_AND_BEVERAGE",
                    "HEALTHCARE_EX_MANAGED_CARE",
                    "MANAGED_CARE",
                    "PHARMACEUTICALS",
                    "TOBACCO",
                },
                "UTILITY": {"UTILITY"},
                "SOVS": {"OWNED_NO_GUARANTEE", "SOVEREIGN"},
                "OTHER": {"HOSPITALS", "MUNIS", "UNIVERSITY"},
            }[sector]

            locs = tuple(
                table.loc[table["raw_sector"].isin(bottom_sector_names)].index
            )
            if locs:
                sector_locs[locs] = color
        table.drop(["raw_sector"], axis=1, inplace=True)
        index_loc = tuple(table[table["Rating"] == "-"].index)
        if fin_flag == 0:
            fnote = f"Performance since {date}."
        elif fin_flag == 2:
            fnote = "."
        else:
            fnote = None

        doc.start_edit(f"{name.replace(' ', '_')}_1m_{fin_flag}_sector_table")
        doc.add_table(
            table,
            # hide_index=True,
            prec=prec,
            col_fmt="llc|cc|ccc",
            caption=caption,
            table_notes=fnote,
            table_notes_justification="l",
            adjust=True,
            font_size="scriptsize",
            multi_row_header=True,
            row_font={index_loc: "\\bfseries"},
            row_color=sector_locs,
            gradient_cell_col="Out*Perform",
            gradient_cell_kws={"cmax": "steelblue", "cmin": "firebrick"},
        )
        doc.end_edit()


# %%


def plot_sector_ytd_exsret_heat_scatter(df):
    db = Database()
    ix_oas = db.load_bbg_data("US_IG_10+", "OAS", start="1/1/2020")
    ix_oas_start = ix_oas.iloc[0]
    start_date = ix_oas.index[0].strftime("%m/%d/%Y")

    # %%

    df = long_credit_df.copy()
    fig, ax = vis.subplots()
    df["starting_spread"] = (df["oas"] - df["oas_ytd_chg"]) / 1
    points = ax.scatter(
        df["oad"], df["starting_spread"], c=df["XSRet"], cmap="coolwarm"
    )
    cbar = fig.colorbar(points)
    cbar.ax.set_ylabel("YTD Excess Returns (bp)")
    ax.set_xlabel("OAD (yrs)")
    ax.set_ylabel(f"OAS on {start_date} (bp)")
    vis.savefig("sector_xsret")

    os.getcwd()
    df.sort_values("XSRet")

    # %%
    fig, ax = vis.subplots()
    sns.regplot(
        x=df["oas_ytd_chg"],
        y=df["XSRet"],
        color="k",
        line_kws={"lw": 2, "alpha": 0.5},
        ci=95,
        ax=ax,
    )
    vis.savefig("oas_chg_vs_xsret")


def calc_cum_xs_ret(ix, name=None):
    ex_rets = ix.market_value_weight("XSRet", weight="PrevMarketValue")
    t_rets = ix.market_value_weight("TRet", weight="PrevMarketValue")
    rf_rets = t_rets - ex_rets

    cum_xs_ret_a = np.zeros(len(t_rets))
    for i, __ in enumerate(t_rets):
        total_ret = np.prod(1 + t_rets[: i + 1]) - 1
        rf_total_ret = np.prod(1 + rf_rets[: i + 1]) - 1
        cum_xs_ret_a[i] = total_ret - rf_total_ret
    return 1e4 * pd.Series(cum_xs_ret_a, index=t_rets.index, name=name)


def _xsret_zscore(ix_sector, cum_xs_ret_ix, n):
    cum_xs_ret_sector = calc_cum_xs_ret(ix_sector)
    cum_xs_ret = cum_xs_ret_sector - cum_xs_ret_ix
    rolling_xs_ret = cum_xs_ret[n:] - cum_xs_ret[:-n].values
    z_rolling_xs_ret = (
        rolling_xs_ret - rolling_xs_ret.mean()
    ) / rolling_xs_ret.std()
    return z_rolling_xs_ret[-1]


def xsret_model_zscores(sectors, maturity, db, n_months=6, lookback="1m"):
    """
    Get DataFrame of current and previous z scores and change in
    rank for each sector/rating combination.
    """
    n = int(n_months * 21)
    ratings = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
    kwargs = load_json("indexes")
    ix = db.build_market_index(in_stats_index=True, maturity=maturity)
    prev_date = db.date(lookback)

    # Store excess returns for A and BBB indexes.
    cum_xs_ret_ix_d = {}
    for rating, rating_kws in ratings.items():
        xsret = calc_cum_xs_ret(ix.subset(rating=rating_kws))
        cum_xs_ret_ix_d[("current", rating)] = xsret
        cum_xs_ret_ix_d[("prev", rating)] = xsret[xsret.index <= prev_date]

    # Find excess return of each sector less the index
    current_zscores = {}
    for sector in sectors:
        for rating, rating_kws in ratings.items():
            # Get current z-score
            try:
                ix_sector = ix.subset(**kwargs[sector], rating=rating_kws)
                ix_sector_current = ix_sector.subset_current_universe()
            except IndexError:
                continue

            # Succesfully calculated excess returns for current sector.
            curr_z = _xsret_zscore(
                ix_sector_current, cum_xs_ret_ix_d[("current", rating)], n
            )
            if np.isnan(curr_z):
                continue
            try:
                ix_sector_old = ix_sector.subset(end=prev_date)
                ix_sector_prev = ix_sector_old.subset_current_universe()
            except IndexError:
                prev_z = np.NaN
            else:
                prev_z = _xsret_zscore(
                    ix_sector_prev, cum_xs_ret_ix_d[("prev", rating)], n
                )
            current_zscores[f"{sector}|{rating}"] = (curr_z, prev_z)

    df = pd.DataFrame(current_zscores, index=["current", "prev"]).T
    df["change"] = -rank_change(df)
    return df


def rank_change(df, current="current", prev="prev"):
    curr_ranks = df.sort_values(current).reset_index()["index"]
    curr_ranks = pd.Series(curr_ranks.index, curr_ranks.values, name="curr")
    prev_ranks = df.sort_values(prev).reset_index()["index"]
    prev_ranks = pd.Series(prev_ranks.index, prev_ranks.values, name="prev")
    comb_df = pd.concat((curr_ranks, prev_ranks), axis=1)
    comb_df["change"] = comb_df["curr"] - comb_df["prev"]
    return comb_df["change"]


def get_forecasted_xsrets(db, lookback="1m"):
    """
    Get forecasted excess returns from specified lookback.
    """
    # Use stas index to find an index eligible bonds over
    # entire period. Then build an index of these bonds
    # where they do not drop out for any reason.
    lookback_date = db.date(lookback)
    stats_ix = db.build_market_index(in_stats_index=True, start=lookback_date)
    ix = db.build_market_index(cusip=stats_ix.cusips, start=lookback_date)
    ix.df["DTS"] = ix.df["OAS"] * ix.df["OASD"]
    ix.df["RatingBucket"] = np.NaN
    ix.df.loc[ix.df["NumericRating"] <= 7, "RatingBucket"] = "A"
    # ix.df.loc[ix.df["NumericRating"].isin((4, 5, 6)), "RatingBucket"] = "A"
    ix.df.loc[ix.df["NumericRating"].isin((8, 9, 10)), "RatingBucket"] = "BBB"
    d = defaultdict(dict)
    for date in ix.dates:
        # Get the days data.
        date_ix = ix.day(date, as_index=True)
        date_cusips = date_ix.cusips
        date_df = date_ix.df.copy()

        # Subset the bonds which have not been forecasted already.
        already_been_forecasted = (
            pd.Series(date_df["RatingBucket"].items())
            .isin(d["ModelXSRet"])
            .values
        )
        in_rating_bucket = ~date_df["RatingBucket"].isna()
        pred_df = date_df[~already_been_forecasted & in_rating_bucket]
        # d["n_pred"][date] = len(pred_df)
        if not len(pred_df):
            # No new bonds to forecast.
            continue

        # Calculate excess returns and weights from the date to current date.
        ix_from_date = ix.subset(start=date)
        xsrets = ix_from_date.accumulate_individual_excess_returns()
        weights = ix_from_date.get_value_history("MarketValue").sum()

        # Perform regression to find expected excess returns.
        x_cols = ["OAS", "OAD", "DTS"]
        reg_df = pd.concat((date_df[x_cols], xsrets), axis=1).dropna()
        X = sm.add_constant(reg_df[x_cols])
        ols = sm.OLS(reg_df["XSRet"], X).fit()
        # d["r2"][date] = ols.rsquared
        pred_df = pd.concat(
            (
                pred_df[["RatingBucket", *x_cols]],
                xsrets,
                weights.rename("weight"),
            ),
            axis=1,
            join="inner",
            sort=False,
        )

        X_pred = sm.add_constant(pred_df[x_cols], has_constant="add")
        pred_df["pred"] = ols.predict(X_pred)
        pred_df
        # Store forecasted values.
        for cusip, row in pred_df.iterrows():
            key = (cusip, row["RatingBucket"])
            d["ModelXSRet"][key] = row["pred"]
            d["RealXSRet"][key] = row["XSRet"]
            d["weight"][key] = row["weight"]

    # Create DataFrame of all forecasted cusip/rating bucket combinations.
    df = pd.Series(d["ModelXSRet"]).to_frame()
    df.columns = ["ModelXSRet"]
    list_d = defaultdict(list)
    for i, (key, model_xsret) in enumerate(df["ModelXSRet"].items()):
        cusip, rating_bucket = key
        list_d["CUSIP"].append(cusip)
        list_d["RatingBucket"].append(rating_bucket)
        for col in ["RealXSRet", "weight"]:
            list_d[col].append(d[col][key])
    for col, vals in list_d.items():
        df[col] = vals
    df.set_index("CUSIP", drop=True, inplace=True)
    return df


# %%


def _xs_ret_z_score_plots():
    # ix_sector = ix.subset(
    # **kwargs[sector], rating=rating_kws).subset_current_universe()
    # cum_xs_ret_sector = calc_cum_xs_ret(ix_sector, sector)
    # cum_xs_ret = cum_xs_ret_sector - cum_xs_ret_ix[rating]
    # pd.Series((ix_sector.name))

    # %%
    # vis.plot_multiple_timeseries(
    #     [
    #         cum_xs_ret.rename(f"{rating} {ix_sector.name}  - {rating} Index"),
    #         cum_xs_ret_sector.rename(f"{rating} {ix_sector.name}"),
    #         cum_xs_ret_ix[rating].rename(f"{rating} Index"),
    #     ],
    #     title="Cumulative Excess Returns",
    #     xtickfmt="auto",
    # )
    # vis.savefig(f"{sector}_cum_xsret_comp_to_index")
    # vis.show()

    # %%
    n_months_list = [3, 6, 9, 12]

    # vis.plot_timeseries(
    #     rolling_xs_ret,
    #     xtickfmt="auto",
    #     title=(
    #     f"{n_months} Month Rolling Window Cumulative "
    #     f"Excess Returns {ix_sector.name} vs Index"
    # )
    #     figsize=(16, 8),
    # )
    # vis.savefig(f"{sector}_raw_{n_months}_rolling_window_xsret_")
    # vis.show()

    fig, axes = vis.subplots(4, 1, figsize=(16, 12))
    for ax, n_months in zip(axes.flat, n_months_list):
        n = n_months * 21
        rolling_xs_ret = cum_xs_ret[n:] - cum_xs_ret[:-n].values
        z_rolling_xs_ret = (
            rolling_xs_ret - rolling_xs_ret.mean()
        ) / rolling_xs_ret.std()

        vis.plot_timeseries(
            z_rolling_xs_ret,
            xtickfmt="auto",
            title=f"{n_months} Month Rolling Z-Score {ix_sector.name}",
            ax=ax,
        )
        ax.fill_between(
            z_rolling_xs_ret.index,
            z_rolling_xs_ret,
            0,
            color="steelblue",
            alpha=0.4,
        )
    vis.savefig(f"{sector}_z_score_rolling_window_xsret")
    vis.show()


# %%
def _forecast_xsrets_validation_plots():
    # %%
    temp = pd.concat(
        (
            pd.Series(d["RealXSRet"]).rename("XSRet"),
            pd.Series(d["ModelXSRet"]).rename("pred"),
        ),
        axis=1,
    )
    temp *= 1e4
    fig, ax = vis.subplots()
    ax.plot(temp["pred"], temp["XSRet"], "o", ms=2, alpha=0.5, c="steelblue")
    ax.set(
        xlabel="Predicted Excess Return (bp)",
        ylabel="Realized Excess Return (bp)",
        ylim=(None, 2500),
    )
    xlim = (0, 2500)
    ax.plot(xlim, xlim, "--", c="firebrick", lw=1.5)
    vis.savefig("pred_vs_real_xsret")
    vis.show()

    # %%
    rsq = pd.Series(d["r2"])
    rsq.index = [(rsq.index[-1] - idx).days for idx in rsq.index]
    fig, ax = vis.subplots()
    ax.plot(rsq, "-o", ms=5, lw=2, color="steelblue")
    ax.set(xlabel="Days Prior to Estimation", ylabel="$R^2$", xlim=(32, 0))
    vis.savefig("pred_xsret_r2_vs_days_to_estimation")
    vis.show()

    # %%
    n = pd.Series(d["n_pred"])
    n.index = [(n.index[-1] - idx).days for idx in n.index]
    n = n[:-1]

    fig, ax = vis.subplots()
    ax.bar(n.index, n, width=0.8, color="steelblue", alpha=0.9)
    ax.grid(False, axis="x")
    ax.set(
        xlabel="Days Prior to Estimation",
        ylabel="Number of Bonds Forecasted",
        yscale="log",
        xlim=(32, 0),
    )
    for rect in ax.patches:
        height = rect.get_height()
        label = "" if height == 0 else f"{height:,.0f}"
        if height < 0:
            height = 0

        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            label,
            fontsize=10,
            ha="center",
            va="bottom",
        )

    # vis.show()
    vis.savefig("pred_xsret_n_bonds")
