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
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import rolling_zscore, XSRETPerformance
from lgimapy.utils import root, get_ordinal


# %%


def update_sector_models(fid, db):
    """Update volatility indicator figures."""
    vis.style()
    doc = Document("Sector_Valuations", path="reports/HY", fig_dir=True)

    db = Database()
    today = db.date("today").strftime("%B %#d, %Y")
    db.load_market_data(start=db.date("HY_START"))
    ix = db.build_market_index(in_H4UN_index=True, start=db.date("1.2y"))

    table_df = get_sector_table(ix, db)
    # %%
    doc = Document("Sector_Valuations", path="reports/HY", fig_dir=True)
    doc.add_preamble(
        margin={
            "paperheight": 20,
            "paperwidth": 20,
            "left": 0.5,
            "right": 0.5,
            "top": 0.5,
            "bottom": 0.2,
        },
        bookmarks=True,
        header=doc.header(left="Sector Breakdown", right=f"EOD {today}"),
        footer=doc.footer(logo="LG_umbrella"),
    )
    make_forward_looking_sector_table(table_df, doc)
    doc.save()
    # %%


# %%
def get_sector_table(ix, db):
    """
    Get historical sector volatility and spread data, and a
    current snapshot of data including market value and rating.
    """
    sectors = [
        "AUTOMAKERS",
        "AUTOPARTS",
        "HOME_BUILDERS",
        "BUILDING_MATERIALS",
        "CHEMICALS",
        "METALS_AND_MINING",
        "AEROSPACE_DEFENSE",
        "DIVERSIFIED_CAPITAL_GOODS",
        "PACKAGING",
        "BEVERAGE",
        "FOOD",
        "PERSONAL_AND_HOUSEHOLD_PRODUCTS",
        "ENERGY_EXPLORATION_AND_PRODUCTION",
        "GAS_DISTRIBUTION",
        "OIL_REFINING_AND_MARKETING",
        "HEALTH_FACILITIES",
        "MANAGED_CARE",
        "PHARMA",
        "GAMING",
        "HOTELS",
        "RECREATION_AND_TRAVEL",
        "REAL_ESTATE",
        "REITS",
        "ENVIRONMENTAL",
        "SUPPORT_SERVICES",
        "CABLE_SATELLITE",
        "MEDIA_CONTENT",
        "TELECOM_SATELLITE",
        "TELECOM_WIRELESS",
        "SOFTWARE",
        "HARDWARE",
        "AIRLINES",
        "UTILITY",
    ]
    ratings = {"BB": ("BB+", "BB-"), "B": ("B+", "B-")}

    warnings.simplefilter(action="ignore", category=FutureWarning)
    d = defaultdict(list)
    history_d = defaultdict(list)
    for sector in sectors:
        for rating, rating_kws in ratings.items():
            ix_sector = ix.subset(
                **db.index_kwargs(sector, rating=rating_kws, source="BAML")
            )
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

            # Store historical results.
            history_d["vol"].append(vol.rename(f"{ix_sector.name}|{rating}"))
            history_d["oas"].append(oas.rename(f"{ix_sector.name}|{rating}"))
            # Store current snapshot.
            d["name"].append(ix_sector.name)
            d["raw_sector"].append(sector)
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
    vol_z_scores = vol_model_zscores(history_d, ratings)
    vol_model = np.full(len(table_df), np.NaN)
    for i, key in enumerate(table_df.set_index("name")["rating"].items()):
        vol_model[i] = vol_z_scores.get(key, np.NaN)
    table_df["vol_model"] = vol_model

    # Add sector vs median results to table.
    median_z_scores = historical_sector_median_zscore(sectors, ratings, db)
    median_model = np.full(len(table_df), np.NaN)
    for i, key in enumerate(table_df.set_index("name")["rating"].items()):
        median_model[i] = median_z_scores.get(key, np.NaN)
    table_df["carry_model"] = median_model

    # Add xsret model results to table.
    xs_ret_z_scores_df = xsret_model_zscores(
        ix, sectors, ratings, db, n_months=6, lookback="1m"
    )
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
    # Make negative so positive is cheaper, negative richer.
    table_df["xsret_model"] = -curr_xs_rets
    table_df["xsret_prev"] = -prev_xs_rets
    table_df.dropna(subset=["xsret_model", "vol_model"], inplace=True)
    # temp_fid = root("src/lgimapy/valuation_pack/temp.csv")
    # table_df.dropna(subset=["xsret_model", "vol_model"]).to_csv(temp_fid)
    return table_df


def vol_model_zscores(history_dict, ratings):
    """
    Run robust regression for every day in history.
    plot alpha and beta for regression through time,
    and plot timeseries of Z-scores for sectors'
    deviations from the regression line.
    """
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

    # Return current values for each sector/rating combination.
    current_zscores_s = z_score_df.iloc[-1, :]
    current_zscores = {
        tuple(sector.split("|")): z for sector, z in current_zscores_s.items()
    }
    return current_zscores


def make_forward_looking_sector_table(table_df, doc):
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
        "carry_model": "Carry*Model",
        "xsret_model": "Momentum*Model",
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
        "Carry*Model": "1f",
        "Momentum*Model": "1f",
        "Model*Change": "0f",
    }

    df = table_df.copy()
    df["oas_%tile"] = 100 * df["oas_%tile"].values
    table = (
        df[col_map.keys()]
        .sort_values("xsret_model", ascending=False)
        .rename(columns=col_map)
        .reset_index(drop=True)
    )
    cuttoff = int(len(table) / 2)
    title = "HY Sector Breakdown"
    # if title == 'HY Cheapest Screening Sectors':
    #     table = table.iloc[:cuttoff].copy()
    # elif title == 'HY Richest Screening Sectors':
    #     table = table.iloc[cuttoff:].copy()

    # Get row colors for level 3 sectors.
    sector_row_colors = {
        "ENERGY": "magicmint",
        "COMMUNICATIONS": "opal",
        "CONSUMER_NON_CYCLICAL": "sage",
    }
    sector_locs = {}
    for sector, color in sector_row_colors.items():
        bottom_sector_names = {
            "COMMUNICATIONS": {
                "CABLE_SATELLITE",
                "MEDIA_CONTENT",
                "TELECOM_SATELLITE",
                "TELECOM_WIRELESS",
            },
            "ENERGY": {
                "ENERGY_EXPLORATION_AND_PRODUCTION",
                "GAS_DISTRIBUTION",
                "OIL_REFINING_AND_MARKETING",
            },
            "CONSUMER_NON_CYCLICAL": {
                "BEVERAGE",
                "FOOD",
                "PERSONAL_AND_HOUSEHOLD_PRODUCTS",
                "HEALTH_FACILITIES",
                "MANAGED_CARE",
                "PHARMA",
            },
        }[sector]

        locs = tuple(
            table.loc[table["raw_sector"].isin(bottom_sector_names)].index + 1
        )
        if locs:
            sector_locs[locs] = color

    # Add table to document.
    table["Model*Change"] = rank_change(table, "Momentum*Model", "xsret_prev")
    table.drop(["raw_sector", "xsret_prev"], axis=1, inplace=True)
    table.index += 1
    footnote = """
        \\tiny
        Sectors which screen \\color{orange} \\textbf{cheap}
        \\color{black} are at the top of the table while sectors
        which screen \\color{orchid} \\textbf{rich}
        \\color{black} are at the bottom.
        """

    doc.add_table(
        table,
        # hide_index=True,
        prec=prec,
        col_fmt="llc|cccccl|cccc|cccl",
        caption=title,
        table_notes=footnote,
        adjust=True,
        font_size="scriptsize",
        multi_row_header=True,
        row_color=sector_locs,
        col_style={"YTD*%tile": "\pctbar{6}"},
        gradient_cell_col=["Vol*Model", "Carry*Model", "Momentum*Model"],
        gradient_cell_kws={
            "cmax": "orange",
            "cmin": "orchid",
        },
        arrow_col="Model*Change",
        arrow_kws={
            "cmax": "orange",
            "cmin": "orchid",
            "vmax": len(table) // 3,
            "vmin": len(table) // 3,
        },
    )


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


def xsret_model_zscores(ix, sectors, ratings, db, n_months=6, lookback="1m"):
    """
    Get DataFrame of current and previous z scores and change in
    rank for each sector/rating combination.
    """
    n = int(n_months * 21)
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
                ix_sector = ix.subset(
                    **db.index_kwargs(sector, rating=rating_kws, source="BAML")
                )
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
    return df


def rank_change(df, current="current", prev="prev"):
    curr_ranks = df.sort_values(current).reset_index()["index"]
    curr_ranks = pd.Series(curr_ranks.index, curr_ranks.values, name="curr")
    prev_ranks = df.sort_values(prev).reset_index()["index"]
    prev_ranks = pd.Series(prev_ranks.index, prev_ranks.values, name="prev")
    comb_df = pd.concat((curr_ranks, prev_ranks), axis=1)
    comb_df["change"] = comb_df["curr"] - comb_df["prev"]
    return comb_df["change"]


def historical_sector_median_zscore(sectors, ratings, db):
    ix = db.build_market_index(in_H4UN_index=True)
    ix_rating = {
        rating: ix.subset(rating=rating_kws)
        for rating, rating_kws in ratings.items()
    }
    ix_oas_per_oad = {
        rating: ix.OAS() / ix.market_value_weight("OAD")
        for rating, ix in ix_rating.items()
    }

    historical_median_z_scores = {}
    for sector in sectors:
        for rating in ratings.keys():
            ix_sector = ix_rating[rating].subset(
                **db.index_kwargs(sector, source="BAML")
            )
            sector_oas_per_oad = (
                ix_sector.OAS() / ix_sector.market_value_weight("OAD")
            )
            abs_diff = (sector_oas_per_oad - ix_oas_per_oad[rating]).dropna()
            ratio = (sector_oas_per_oad / ix_oas_per_oad[rating]).dropna()
            abs_z = (abs_diff - abs_diff.mean()) / abs_diff.std()
            ratio_z = (ratio - ratio.mean()) / ratio.std()
            try:
                z = (abs_z.iloc[-1] + ratio_z.iloc[-1]) / 2
            except IndexError:
                continue
            historical_median_z_scores[(ix_sector.name, rating)] = z
    return historical_median_z_scores


# %%


def _xsret_z_score_plots():
    # %%
    n_months = 6
    lookback = "1m"
    n = int(n_months * 21)
    prev_date = db.date(lookback)

    # Store excess returns for A and BBB indexes.
    cum_xs_ret_ix = {}
    for rating, rating_kws in ratings.items():
        xsret = calc_cum_xs_ret(ix.subset(rating=rating_kws))
        cum_xs_ret_ix[rating] = xsret
    # %%
    sector = "HEALTHCARE"
    ix_sector = ix.subset(
        **db.index_kwargs(sector, rating=rating_kws, source="BAML")
    ).subset_current_universe()
    cum_xs_ret_sector = calc_cum_xs_ret(ix_sector, sector)
    cum_xs_ret = (cum_xs_ret_sector - cum_xs_ret_ix[rating]).dropna()
    ix_sector.name

    # %%
    vis.plot_multiple_timeseries(
        [
            cum_xs_ret.rename(f"{rating} {ix_sector.name}  - {rating} Index"),
            cum_xs_ret_sector.rename(f"{rating} {ix_sector.name}"),
            cum_xs_ret_ix[rating].rename(f"{rating} Index"),
        ],
        title="Cumulative Excess Returns",
        xtickfmt="auto",
    )
    # vis.savefig(f"{sector}_cum_xsret_comp_to_index")
    vis.show()
    # %%
    n_months_list = [3, 6, 9]

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

    fig, axes = vis.subplots(len(n_months_list), 1, figsize=(16, 12))
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
    # vis.savefig(f"{sector}_z_score_rolling_window_xsret")
    vis.show()
    # %%


# %%
