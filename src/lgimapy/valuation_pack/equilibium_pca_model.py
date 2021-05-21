import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root


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
    doc = Document(
        fid, path="reports/valuation_pack", fig_dir=True, load_tex=True
    )

    df_levels, df_pca, equilibrium_table = run_pca_analysis()

    plot_pc_vs_factors(df_levels, df_pca, doc)
    plot_credit_risk_appetite(df_levels, df_pca, doc)

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
    doc.save()
    # doc.save_tex()


def format_table(df):
    no_dec = [
        "SP500",
        "SP500_GROW",
        "SP500_VALU",
        "RUSSELL_2000",
        "MSCI_EU",
        "MSCI_ACWI",
        "MSCI_EM",
        "GOLD",
        "COPPER",
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
            d["Model * Level"].append(f"{row['model_level']:.0f}")
        elif ix in yields:
            d["Current * Level"].append(f"{row['cur_level']:.2%}")
            d["Model * Level"].append(f"{row['model_level']:.2%}")
        else:
            d["Current * Level"].append(f"{row['cur_level']:.2f}")
            d["Model * Level"].append(f"{row['model_level']:.2f}")

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


def plot_credit_risk_appetite(df_levels, df_pca, doc):
    """Plot the credit risk appetite."""
    start_date = Database().date("12y")
    df_add = load_additional_bloom_data(start_date)
    df_bloom = pd.concat([df_levels, df_add], join="outer", sort=True, axis=1)
    df_ratios = calc_risk_ratios(df_bloom)
    df_ratios.dropna(axis=0, how="any", inplace=True)
    df_risk = calc_risk_appetite_indexes(df_ratios)

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
    vis.savefig("credit_risk_appetite", path=doc.fig_dir)
    vis.close()


def run_pca_analysis():
    """
    Perform PCA analysis using 1yr Z-scores for each asset class.
    """
    # Get bloomberg data on asset classes, z-score it, run pca routine.
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

    # For assets which use PB ratio Z-scores, find the book value
    # So PB ratio can be converted back to index price level.
    pb_ratio_assets = {
        "SP500",
        "SP500_GROW",
        "SP500_VALU",
        "RUSSELL_2000",
        "MSCI_EU",
        "MSCI_ACWI",
        "MSCI_EM",
    }
    pb_ratio = db.load_bbg_data(pb_ratio_assets, "PB_RATIO")
    price = db.load_bbg_data(pb_ratio_assets, "price")
    scalar_multiple = {}
    for asset in pb_ratio_assets:
        pbr = pb_ratio[asset].dropna().iloc[-1]
        p = price[asset].dropna().iloc[-1]
        scalar_multiple[asset] = p / pbr

    # For each asset, perform a regression using PC's to find
    # what the model suggests current level should be, and
    # record residual for the model in terms of Z-score.
    X = df_reg[["PCA 1", "PCA 2", "PCA 3", "PCA 4"]]
    col_names = []
    d, t = defaultdict(list), defaultdict(list)
    for col in df_reg.columns:
        if not col.startswith("PCA"):
            col_names.append(col)
            y = df_reg[col]
            OLS_result = sm.OLS(y, X).fit(cov_type="HC1")
            pred_z = OLS_result.predict(X)
            resid = y - pred_z
            scalar = scalar_multiple.get(col, 1)
            model_level = scalar * ((pred_z * df_std[col]) + df_mean[col])
            d["OLS_res"].append(OLS_result)
            d["OLS_r2"].append(OLS_result.rsquared_adj)
            d["OLS_resid"].append(resid)
            t["cur_level"].append(scalar * df[col].iloc[-1])
            t["cur_1y_z"].append(df_z[col].iloc[-1])
            t["pred_1y_z"].append(pred_z.iloc[-1])
            t["cur_resid"].append(resid[-1])
            t["model_level"].append(model_level.iloc[-1])
            t["R2"].append(OLS_result.rsquared_adj)

    # reg_results = pd.DataFrame(d, index=col_names)
    table = pd.DataFrame(t, index=col_names)

    # Flip residual signs for assets that are rich when current
    # level is greater than model level, such as equities.
    asset_signs_reversed = [
        "SP500",
        "SP500_GROW",
        "SP500_VALU",
        "RUSSELL_2000",
        "MSCI_EU",
        "MSCI_ACWI",
        "MSCI_EM",
        "GOLD",
        "COPPER",
        "OIL",
        "VIX",
        "VIX_3M",
        "OIL_VOL",
        "MOVE",
        "USD_TW",
    ]

    table["cur_resid_sign"] = [
        -table.loc[ix, "cur_resid"]
        if ix in asset_signs_reversed
        else table.loc[ix, "cur_resid"]
        for ix in table.index
    ]
    table = table.sort_values("cur_resid_sign")
    return df, df_pca, table


def do_pca(df):
    """Perform PCA on given DataFrame."""
    pca = PCA(n_components=4)
    pca_data = pca.fit_transform(df)
    df_pca = pd.DataFrame(
        data=pca_data,
        columns=[f"PCA {i+1}" for i in range(pca_data.shape[1])],
        index=df.index,
    )
    return df_pca, pca.explained_variance_ratio_


def plot_pc_vs_factors(df_levels, df_pca, doc):
    """Plot all 4 identified PC's vs their respective factors/"""
    # Load factor data.
    db = Database()
    sov_risk = (
        df_levels["ITALY_10Y"] + df_levels["SPAIN_10Y"]
    ) / 2 - df_levels["BUND_10Y"]
    start = "1/1/2000"
    df = pd.concat(
        [
            db.load_bbg_data("UST_10Y", "YTW", start=start),
            db.load_bbg_data("UST_10Y_RY", "YTW", start=start),
            db.load_bbg_data("USD_TW", "price", start=start),
            sov_risk.rename("SOV_risk"),
        ],
        axis=1,
    ).fillna(method="ffill")
    df.columns = ["UST_10Y", "REAL", "USD", "SOV"]
    df["UST_10Y"] = df["UST_10Y"].diff(12 * 5)  # 12 week change
    n = 250
    for col in df.columns:
        s = df[col]
        df[f"{col}_Z"] = (s - s.rolling(n).mean()) / s.rolling(n).std()

    # Combine PCA and factor data together.
    df_pca_and_factors = pd.concat(
        [
            df,
            df_pca.rename(columns={f"PCA {i}": f"PC {i}" for i in range(5)}),
        ],
        axis=1,
    ).fillna(method="ffill")
    fig, axes = vis.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Build plots, using a scalar to properly scale PC's to factors.
    pca_ID = {
        "UST_10Y": (
            "PC 1",
            None,
            "$\\Delta$10y Yield",
            "12wk Change in US 10y Yield",
        ),
        "REAL_Z": (
            "PC 2",
            None,
            "Real Rates",
            "1y Rolling Z-Score\nUS Generic 10y Real Yield",
        ),
        "USD_Z": (
            "PC 3",
            0.66,
            "US Dollar",
            "1y Rolling Z-Score\nUS Fed Trade Weighted $ Index",
        ),
        "SOV_Z": (
            "PC 4",
            -1,
            "Euro Area Risk",
            (
                "1y Rolling Z-Score\n$\\dfrac{Spain\ 10y + Italy\ 10y}{2} "
                "- Bund\ 10y$"
            ),
        ),
    }
    for ax, (factor, (pc, scalar, name, des)) in zip(axes.flat, pca_ID.items()):
        title = f"{pc} vs {name}\n\n"
        pc_col = f"{pc} (rescaled)"
        df_plot = pd.concat(
            [
                df_pca_and_factors[pc].rename(pc_col),
                df_pca_and_factors[factor].rename(des),
            ],
            axis=1,
            sort=True,
        ).dropna()

        # Scale PC's to underlying series.
        if scalar is None:
            x = sm.add_constant(df_plot[pc_col])
            ols = sm.OLS(df_plot[des], x).fit()
            df_plot[pc_col] = ols.predict(x)
        else:
            df_plot[pc_col] *= scalar

        vis.set_n_ticks(ax, 5)
        vis.plot_multiple_timeseries(
            df_plot,
            c_list=["black", "darkorchid"],
            ylabel=name,
            title=title,
            xtickfmt="auto",
            alpha=0.8,
            lw=1.2,
            ax=ax,
            legend={
                "loc": "upper center",
                "bbox_to_anchor": (0.5, 1.25),
                "ncol": 2,
                "fancybox": True,
                "shadow": True,
                "fontsize": 10,
            },
        )
        if factor == "UST_10Y":
            vis.format_yaxis(ax, ytickfmt=("{x:.1%}"))
    vis.savefig("pca_vs_factors", path=doc.fig_dir)
    vis.close()


def calc_risk_appetite_indexes(df):

    df_z = (
        (df - df.rolling(window=250).mean()) / df.rolling(window=250).std()
    ).dropna(axis=0, how="any")

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
        df_list.append(df_z.eval(indic_list).rename(indic_name))

    df_risk_app = pd.concat(df_list, join="outer", sort=True, axis=1)
    eval_instruct = "(" + " + ".join(df_risk_app.columns) + ")/5"
    df_list.append(df_risk_app.eval(eval_instruct).rename("risk_appetite"))
    return pd.concat(df_list, sort=True, axis=1)


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

    return pd.concat(df_list, sort=True, axis=1)


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
    holidays = set(db.holiday_dates()) | set(jans)
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


# %%
if __name__ == "__main__":
    update_equilibrium_model("val_pack_temp")
