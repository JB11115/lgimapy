import warnings
from collections import defaultdict
from datetime import timedelta, date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
from statsmodels.regression.quantile_regression import QuantReg

import lgimapy.vis as vis
from lgimapy.bloomberg import bdh
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root

# from dateutil.relativedelta import relativedelta
# from lgimapy.utils import load_json, dump_json
# from statsmodels.tsa.stattools import adfuller

# %%


def update_equilibrium_model(fid):
    """
    Program to replicate GS risk appetite indicator and PCA analysis on asset
    prices. Source papers saved in X:/Credit Strategy/Quant Research

    PCA model illustrates degree to which various markets are inconsistently
    priced with each other.

    df: a dataframe of Bloomberg levels for different assets
    df_pca: a dataframe of the PCA factors derived from z-scoring df
    tab_results: a dataframe of PCA model predictions for asset prices
    df_risk: a dataframe of risk sentiment indicators
    """
    vis.style()
    # fid = "temp"
    # pca on asset prices
    df, df_pca, equilibrium_table = run_pca_analysis()

    # calculate gs-style risk indicators
    df_risk = run_risk_indicators(df, df_pca)
    # %%
    doc = Document(fid, path="valuation_pack", fig_dir=True, load_tex=True)
    doc.start_edit("equilibrium_model_table")
    doc.add_table(
        format_table(equilibrium_table),
        caption="Equilibrium Cross Asset Model Results",
        font_size="scriptsize",
        adjust=True,
        col_fmt="lrrrrr",
        prec=2,
        gradient_cell_col="Current less Model * 1yr Z-score",
        gradient_cell_kws={
            "vals": equilibrium_table["cur_resid_sign"],
            "cmax": "orange",
            "cmin": "orchid",
        },
        multi_row_header=True,
    )
    doc.end_edit()
    doc.save_tex()
    # %%
    # save the risk indicators and pca factors
    # save_data(df_risk, df_pca)


def format_table(df):

    no_dec = [
        "GOLD",
        "CDX_HY",
        "CDX_IG",
        "EM_CORP",
        "EM_IG",
        "EU_IG",
        "EU_HY",
        "ITRAXX_XOVER",
        "US_HY",
        "ITRAXX_MAIN",
        "EM_HY",
        "EM_SOV",
        "US_IG",
        "US_IG_10+",
    ]

    yields = [
        "JGB_10Y",
        "JGB_30Y",
        "UST_10Y",
        "UST_30Y",
        "ITALY_10Y",
        "ITALY_30Y",
        "UK_10Y",
        "UK_30Y",
        "SPAIN_10Y",
        "SPAIN_30Y",
        "BUND_10Y",
        "BUND_30Y",
    ]

    d = defaultdict(list)
    for ix, row in df.iterrows():
        if ix in no_dec:
            d["Current * Level"].append(f"{row['cur_level']:.0f}")
            d["Model * Level"].append(f"{row['Norm_mod_val']:.0f}")
        elif ix in yields:
            d["Current * Level"].append(f"{row['cur_level']:.2%}")
            d["Model * Level"].append(f"{row['Norm_mod_val']:.2%}")
        else:
            d["Current * Level"].append(f"{row['cur_level']:.2f}")
            d["Model * Level"].append(f"{row['Norm_mod_val']:.2f}")

        d["Current * 1yr Z-score"].append(row["cur_1y_z"])
        d["Current less Model * 1yr Z-score"].append(row["cur_resid"])
        d["$R^2$"].append(row["R2"])

    # Format index names.
    new_index = Database().bbg_names(df.index)
    bold_names = {"US Market Credit", "US Long Credit", "US HY"}
    new_index = [
        f"\\textbf{{{ix}}}" if ix in bold_names else ix for ix in new_index
    ]
    return pd.DataFrame(d, index=new_index)


def run_risk_indicators(df, df_pca):

    start_date = Database().date("12y")

    df_add = load_additional_bloom_data(start_date)
    df_bloom = pd.concat([df, df_add], join="outer", sort=True, axis=1)

    df_ratios = calc_risk_ratios(df_bloom)
    df_ratios.dropna(axis=0, how="any", inplace=True)

    df_risk = calc_risk_appetite_indexes(df_ratios)

    bbg_st_dt = "20000101"
    gs_global_MAP = bdh("GSERMWD", "INDEX", "PX_LAST", bbg_st_dt)

    # plot risk ratios
    # create plots for valuation pack
    path = root("latex/valuation_pack/fig")
    df_reg = pd.concat(
        [df_risk, df_pca, gs_global_MAP], join="outer", sort=True, axis=1
    )

    df_reg.dropna(axis=0, how="any", inplace=True)
    df_reg.rename(columns={"PX_LAST": "GS_MAP"}, inplace=True)

    # %%

    # Plot Credit risk appetite.
    credit_app = df_risk["credit_appetite"]
    oas = Database().load_bbg_data("US_IG", "OAS", start=credit_app.index[0])

    fig, ax = vis.subplots(figsize=(8, 5))
    color = "steelblue"
    vis.plot_timeseries(
        credit_app,
        title=f"Credit Risk Appetite: {credit_app[-1]:.1f}",
        ylabel="Z-Score",
        color=color,
        ax=ax,
        lw=1.5,
    )
    ax.fill_between(
        credit_app.index, credit_app.values, 0, color=color, alpha=0.5
    )
    vis.savefig("credit_risk_appetite", path=path)

    # vis.plot_timeseries(oas, color="darkorange", ax=ax)
    # vis.show()
    # %%

    # fig, ax = vis.subplots(figsize=(8, 5))
    #
    # _, right_ax = vis.plot_double_y_axis_timeseries(
    #     oas, credit_app, invert_right_axis=True, ax=ax, ret_axes=True
    # )
    # right_ax.fill_between(
    #     credit_app.index, credit_app.values, 0, color="firebrick", alpha=0.5
    # )
    # vis.set_percentile_limits([oas, credit_app], [ax, right_ax])
    # vis.show()

    # %%

    # Plot 1: PCA 1 vs risk appetite indicator
    # plot_table = {
    #     "risk_appetite": ("PCA 1", -0.12),
    #     "credit_appetite": ("PCA 1", -0.12),
    #     "govie_appetite": ("PCA 1", -0.12),
    #     "vol_appetite": ("PCA 1", -0.12),
    #     "curncy_appetite": ("PCA 1", -0.12),
    #     "equity_appetite": ("PCA 1", -0.12),
    #     "GS_MAP": ("risk_appetite", 1),
    # }
    #
    # for key, (text, scalar) in plot_table.items():
    #     title = f"{key} vs {text}"
    #     figsize = (8, 5.5)
    #
    #     vis.plot_multiple_timeseries(
    #         [
    #             df_reg[key].rename(f"{key}"),
    #             scalar * df_reg[text].rename(f"{text}"),
    #         ],
    #         ylabel="1yr z-score",
    #         title=title,
    #         figsize=figsize,
    #         xtickfmt="auto",
    #     )
    #
    #     vis.savefig(f"{key}_vs_{text}", path=path)
    #     vis.close()

    return df_risk


def run_pca_analysis():

    # get bloomberg data on asset classes, z-score it, run pca routine
    start_date = Database().date("12y")
    df = load_pca_bloom_data(start_date)
    df_z = (df - df.rolling(window=250).mean()) / df.rolling(window=250).std()
    # df_pca = run_pca(df_z.drop(['US_IG'], axis=1))  #use to drop an asset
    df_pca, _ = do_pca(df_z.dropna(axis=0, how="any"))
    # sys_risk = do_rolling_pca(df_z.dropna(axis=0, how='any'))
    df_reg = pd.concat([df_z, df_pca], join="outer", sort=True, axis=1).dropna(
        axis=0, how="any"
    )

    # create plots for valuation pack
    path = root("latex/valuation_pack/fig")

    # Plot 1: all the R2 for each asset class for PCA 1,2,3,4 model
    X = df_reg[["PCA 1", "PCA 2", "PCA 3", "PCA 4"]]

    col_names = []
    d = defaultdict(list)
    t = defaultdict(list)
    # loop through each asset in df_reg frame and do a regression
    for col in df_reg.columns:
        if not col.startswith("PCA"):
            col_names.append(col)
            y = df_reg[col]
            OLS_result = sm.OLS(y, X).fit(cov_type="HC1")
            LAD_result = QuantReg(y, X).fit(q=0.5, cov_type="HC1")
            resid = y - OLS_result.predict(X)
            d["OLS_res"].append(OLS_result)
            d["LAD_res"].append(LAD_result)
            d["OLS_r2"].append(OLS_result.rsquared)
            d["OLS_resid"].append(resid)
            d["LAD_r2"].append(LAD_result.rsquared)  # doesn't work
            t["cur_level"].append(df[col].iloc[-1])
            t["cur_1y_z"].append(df_z[col].iloc[-1])
            t["pred_1y_z"].append(OLS_result.predict(X).iloc[-1])
            t["cur_resid"].append(resid[-1])
            t["mean_resid"].append(resid.mean())
            t["std"].append(df[col].iloc[-250:].std())
            t["R2"].append(OLS_result.rsquared)

    reg_results = pd.DataFrame(d, index=col_names)
    table = pd.DataFrame(t, index=col_names)
    mod_values = (
        table["cur_level"]
        - (table["cur_resid"] - table["mean_resid"]) * table["std"]
    )
    table = pd.concat([table, mod_values], join="outer", sort=True, axis=1)
    table.rename(columns={0: "Norm_mod_val"}, inplace=True)

    asset_signs_reversed = [
        "SP500",
        "SP500_GROW",
        "SP500_VALU",
        "RUSSELL_2000",
        "MSCI_EU",
        "MSCI_ACWI",
        "MSCI_EM",
        "GOLD",
        "OIL",
        "COPPER",
    ]

    table["cur_resid_sign"] = [
        -1 * table.loc[ix, "cur_resid"]
        if ix in asset_signs_reversed
        else table.loc[ix, "cur_resid"]
        for ix in table.index
    ]
    table = table.sort_values("cur_resid_sign")

    figsize = (8, 11)
    fig, ax = vis.subplots(figsize=figsize)
    r2 = reg_results["OLS_r2"].sort_values()
    ax.barh(np.arange(len(r2)), r2)
    ax.set_title("Adj R2 for PCA factor model")
    ax.set_yticks(np.arange(len(r2)))
    ax.set_yticklabels(r2.index, fontsize=8)

    vis.savefig("PCA_R2_values", path=path)
    vis.close()

    # Plots showing deviation of asset class from PCA model implied level
    plot_table = {
        "US_IG": "US IG",
        "US_HY": "US HY",
        "US_IG_10+": "US IG Long",
        "EU_IG": "EU IG",
        "EU_HY": "EU HY",
    }

    for key, text in plot_table.items():
        title = f"{text} (1yr z-score) vs PCA model deviation"
        figsize = (8, 5.5)
        PCA_resid = reg_results.loc[key, "OLS_resid"].rename("PCA residual")
        vis.plot_multiple_timeseries(
            [df_reg[key].rename(f"{text}"), PCA_resid],
            ylabel="1yr z-score",
            title=title,
            figsize=figsize,
            xtickfmt="auto",
        )

        vis.savefig(f"{key}_deviation", path=path)
        vis.close()

    # Plots showing PCA factor identification
    bbg_st_dt = "20000101"

    CESI_global = bdh("CESIG10", "INDEX", "PX_LAST", bbg_st_dt)
    real_10y = bdh("USGGT10Y", "INDEX", "PX_LAST", bbg_st_dt)
    usd_twi = bdh("USTWBGD", "INDEX", "PX_LAST", bbg_st_dt)
    real_z = (
        real_10y - real_10y.rolling(window=250).mean()
    ) / real_10y.rolling(window=250).std()
    usd_z = (usd_twi - usd_twi.rolling(window=250).mean()) / usd_twi.rolling(
        window=250
    ).std()

    sov_risk = (df["ITALY_10Y"] + df["SPAIN_10Y"]) / 2 - df["BUND_10Y"]
    sov_z = (sov_risk - sov_risk.rolling(window=250).mean()) / sov_risk.rolling(
        window=250
    ).std()

    pca_list = [df_pca, CESI_global, real_10y, real_z, usd_z, sov_z]
    pca_frame = pd.concat(pca_list, join="outer", sort=True, axis=1)
    new_cols = list(pca_frame.columns)
    new_cols[-5:] = ["CESIG10", "REAL_10Y", "REAL_Z", "USD_Z", "SOV_Z"]
    pca_frame.columns = new_cols

    pca_ID = {
        "CESIG10": (-5, "PCA 1"),
        "REAL_Z": (-0.5, "PCA 2"),
        "USD_Z": (0.66, "PCA 3"),
        "SOV_Z": (-1, "PCA 4"),
    }

    for factor, (scalar, pca) in pca_ID.items():
        title = f"{pca} (rescaled) vs {factor}"
        figsize = (8, 5.5)

        vis.plot_multiple_timeseries(
            [
                pca_frame[pca].rename(pca) * scalar,
                pca_frame[factor].rename(f"{factor}"),
            ],
            ylabel=f"{factor}",
            start="1-1-2009",
            title=title,
            figsize=figsize,
            xtickfmt="auto",
        )

        vis.savefig(f"{pca.replace(' ', '_')}_v_{factor}", path=path)
        vis.close()

    # %%
    fig, axes = vis.subplots(2, 2, figsize=(12, 8), sharex=True)
    pca_ID = {
        "CESIG10": (-5, "PCA 1", "Economic Surprise"),
        "REAL_Z": (-0.5, "PCA 2", "Real Rates"),
        "USD_Z": (0.66, "PCA 3", "US Dollar"),
        "SOV_Z": (-1, "PCA 4", "Euro Area Risk"),
    }

    for ax, (factor, (scalar, pca, name)) in zip(axes.flat, pca_ID.items()):
        title = f"{pca} vs {name}"
        df_plot = pd.concat(
            [
                pca_frame[pca].rename(f"{pca} (rescaled)") * scalar,
                pca_frame[factor].rename(name),
            ],
            axis=1,
            sort=True,
        ).fillna(method="ffill")

        vis.plot_multiple_timeseries(
            df_plot,
            c_list=["black", "darkorchid"],
            ylabel=name,
            start="1-1-2009",
            title=title,
            xtickfmt="auto",
            alpha=0.9,
            lw=1.2,
            ax=ax,
            legend={"loc": "lower left", "fancybox": True, "shadow": True},
        )

    vis.savefig("pca_vs_factors", path=path)
    vis.close()

    # %%

    return df, df_pca, table


def do_pca(df):
    # as currently used, function assumes df comes in already z-scored

    # pca = PCA(n_components = 'mle', svd_solver = 'full')
    pca = PCA(n_components=8)
    df3 = pca.fit_transform(df)

    df_pca = pd.DataFrame(
        data=df3,
        columns=[f"PCA {i+1}" for i in range(df3.shape[1])],
        index=df.index,
    )

    # print(pca.explained_variance_ratio_)
    # df_pca = pca.singular_values_

    return df_pca, pca.explained_variance_ratio_


def do_rolling_pca(df):

    window = 500  # create a 2y window

    np_data = df.to_numpy()
    num_rows = np_data.shape[0]

    d = defaultdict(list)
    for row in range(num_rows - window):
        np_data_windowed = np_data[row : row + window]
        pca = PCA(n_components=8)
        pca.fit_transform(np_data_windowed)
        d["explained_var"].append(sum(pca.explained_variance_ratio_))

    sys_risk = pd.DataFrame(d, index=df.index[-(num_rows - window) :])

    return sys_risk


def save_data(df_risk, df_pca):

    fids = [
        root("data/risk_appetite/risk_app_data.csv"),
        root("data/risk_appetite/pca_data.csv"),
    ]

    for fid, df in zip(fids, [df_risk, df_pca]):
        try:
            old_data = pd.read_csv(
                fid, index_col=0, parse_dates=True, infer_datetime_format=True
            )

        except FileNotFoundError:

            df.to_csv(fid)

        else:
            old_date = old_data.index.max()
            cur_date = df.index.max()

            # check if old data needs updating
            if old_date < cur_date:

                # remove overlapping dates in the old_data
                old_data = old_data[:-2]
                df_list = [old_data, df]
                all_data = pd.concat(df_list, join="outer", sort=False, axis=0)

                all_data.to_csv(fid)

    return


def calc_risk_appetite_indexes(df):

    df2 = (df - df.rolling(window=250).mean()) / df.rolling(window=250).std()
    df2.dropna(axis=0, how="any", inplace=True)

    indic_dict = {
        "credit_appetite": "(US_HY_vs_IG + EU_HY_vs_IG + EM_HY_vs_IG + US_IG"
        "+ US_BBB_vs_A + US_B_vs_BB + EU_IG + EM_Corp)/8",
        "govie_appetite": "(US_10y + US_30y + BUND_10y + BUND_30y"
        "+ Spain_spread + Italy_spread)/6",
        "vol_appetite": "(VIX + VVIX + Rates_vol + Oil_vol"
        " + US_PutCall_ratio + EU_PutCall_ratio + US_Skew)/7",
        "curncy_appetite": "(USD_trade_wgt + EURUSD + CHFGBP + JPYAUD"
        "+ GOLD)/5",
        "equity_appetite": "(EM_vs_DM_equity + Small_vs_Large"
        "+ Low_vol_vs_SP500 + Growth_vs_Value"
        "+ Fin_vs_Staples + Cyclicals_vs_Defensive)/6",
    }

    df_list = []
    for indic_name, indic_list in indic_dict.items():
        df_list.append(df2.eval(indic_list).rename(indic_name))

    df3 = pd.concat(df_list, join="outer", sort=True, axis=1)
    eval_instruct = "(" + " + ".join(df3.columns) + ")/5"
    df_list.append(df3.eval(eval_instruct).rename("risk_appetite"))
    df3 = pd.concat(df_list, join="outer", sort=True, axis=1)

    return df3


def calc_risk_ratios(df):

    # these are designed so that higher = higher risk appetite (hra)
    risk_ratios = {
        "US_HY_vs_IG": "-1 * CDX_HY / CDX_IG",  # higher=hra
        "EU_HY_vs_IG": "-1 * ITRAXX_XOVER / ITRAXX_MAIN",  # higher=hra
        "EM_HY_vs_IG": "-1 * EM_HY / EM_IG",  # higher=hra
        "US_IG": "-1 * US_IG",  # higher=hra
        "US_BBB_vs_A": "-1 * US_BBB / US_A",  # higher=hra
        "US_B_vs_BB": "-1 * US_B / US_BB",  # higher=hra
        "EU_IG": "-1 * EU_IG",  # higher=hra
        "EM_Corp": "-1 * EM_CORP",  # higher=hra
        "US_10y": "UST_10Y",  # higher=higher r.a.
        "US_30y": "UST_30Y",  # higher=higher r.a.
        "BUND_10y": "BUND_10Y",  # higher=higher r.a.
        "BUND_30y": "BUND_30Y",  # higher=higher r.a.
        "Spain_spread": "-1 * (SPAIN_10Y - BUND_10Y)",  # higher=higher r.a.
        "Italy_spread": "-1 * (ITALY_10Y - BUND_10Y)",  # higher=higher r.a.
        "VIX": "-1 * (VIX_3M + VIX_6M)/2",  # higher=higher r.a.
        "VVIX": "-1 * VVIX",  # higher=higher r.a.
        "Rates_vol": "-1 * MOVE",  # higher=higher r.a.
        "Oil_vol": "-1 * OIL_VOL",  # higher=higher r.a.
        "USD_trade_wgt": "-1 * USD_TW",  # higher=higher r.a.
        "EURUSD": "EURUSD",  # higher=higher r.a.
        "CHFGBP": "-1 * CHFGBP",  # higher=higher r.a.
        "JPYAUD": "-1 * JPYAUD",  # higher=higher r.a.
        "GOLD": "-1 * GOLD",  # higher=higher r.a.
        "US_PutCall_ratio": "-1 * US_PC_RATIO",  # higher=higher r.a.
        "EU_PutCall_ratio": "-1 * EUR_PC_RATIO",  # higher=higher r.a.
        "US_Skew": "-1 * US_SKEW",  # higher=higher r.a.
        "EM_vs_DM_equity": "MSCI_EM / SP500",  # higher=higher r.a.
        "Small_vs_Large": "RUSSELL_2000 / RUSSELL_1000",  # higher=higher r.a.
        "Low_vol_vs_SP500": "-1 * SP500_LOWV / SP500",  # higher=higher r.a.
        "Growth_vs_Value": "SP500_GROW / SP500_VALU",  # higher=higher r.a.
        "Fin_vs_Staples": "SP500_FINS / SP500_STAP",  # higher=higher r.a.
        "Cyclicals_vs_Defensive": "(SP500_MATS + SP500_DISC + SP500_INDU"
        "+ SP500_INFO) / (SP500_HEAL + SP500_STAP + SP500_TELE"
        "+ SP500_UTIL)",  # higher=higher r.a.
    }

    df_list = []
    for risk_name, risk_equation in risk_ratios.items():
        df_list.append(df.eval(risk_equation).rename(risk_name))

    return pd.concat(df_list, join="outer", sort=True, axis=1)


def load_additional_bloom_data(start_date):
    """
    Load a number of bloomberg time series that will be used to construct
    indicators that will make up the risk appetite indicator
    """
    db = Database()

    sec_dict = {
        "PRICE": [
            "VIX_6M",
            "VVIX",
            "CHFGBP",
            "US_PC_RATIO",
            "US_SKEW",
            "EUR_PC_RATIO",
        ],
        "OAS": ["US_A", "US_BBB", "US_HY", "US_BB", "US_B"],
        "PB_RATIO": [
            "SP500_FINS",
            "SP500_STAP",
            "SP500_INFO",
            "SP500_ENER",
            "SP500_HEAL",
            "SP500_UTIL",
            "SP500_INDU",
            "SP500_DISC",
            "SP500_TELE",
            "SP500_MATS",
            "SP500_LOWV",
            "RUSSELL_1000",
        ],
    }

    prev = db.date("today")
    df = pd.concat(
        [
            db.load_bbg_data(securities, field, start=start_date, end=prev)
            for field, securities in sec_dict.items()
        ],
        axis=1,
        sort=True,
    )
    return df.fillna(method="ffill").dropna()


def load_pca_bloom_data(start_date):
    """
    Load a number of bloomberg time series that will be used to do the pca
    analysis
    """
    db = Database()

    sec_dict = {
        "PRICE": [
            "VIX",
            "VIX_3M",
            "MOVE",
            "OIL",
            "OIL_VOL",
            "COPPER",
            "JPYAUD",
            "USD_TW",
            "EURUSD",
            "GOLD",
        ],
        "OAS": [
            "ITRAXX_MAIN",
            "ITRAXX_XOVER",
            "CDX_IG",
            "CDX_HY",
            "US_IG",
            "US_IG_10+",
            "US_HY",
            "EU_IG",
            "EU_HY",
            "EM_SOV",
            "EM_IG",
            "EM_HY",
            "EM_CORP",
        ],
        "YTW": [
            "UST_10Y",
            "UST_30Y",
            "BUND_10Y",
            "BUND_30Y",
            "ITALY_10Y",
            "ITALY_30Y",
            "SPAIN_10Y",
            "SPAIN_30Y",
            "JGB_10Y",
            "JGB_30Y",
            "UK_10Y",
            "UK_30Y",
        ],
        "PB_RATIO": [
            "SP500",
            "SP500_GROW",
            "SP500_VALU",
            "RUSSELL_2000",
            "MSCI_EU",
            "MSCI_ACWI",
            "MSCI_EM",
        ],
    }

    prev = db.date("today")
    df = pd.concat(
        [
            db.load_bbg_data(securities, field, start=start_date, end=prev)
            for field, securities in sec_dict.items()
        ],
        axis=1,
        sort=True,
    )
    jans = [pd.to_datetime(f"1-1-{y}") for y in range(2008, 2021)]
    holidays = set(db.holiday_dates) | set(jans)
    df = df[~df.index.isin(holidays)]

    # impute missing credit data with iterative regression approach
    imp = IterativeImputer(max_iter=10, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed_df = (
            pd.DataFrame(
                imp.fit_transform(df), index=df.index, columns=list(df)
            )
            .fillna(method="ffill")
            .dropna()
        )
    return imputed_df


if __name__ == "__main__":
    update_risk_appetite_indicators("temp")
