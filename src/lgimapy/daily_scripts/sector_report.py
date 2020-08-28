import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root

vis.style()
# %%
# db.load_market_data(start=db.date('1y'), local=True)
# ix = db.build_market_index()

# %%
def create_sector_report():
    db = Database()
    date = db.date("today")
    pdf_path = root("reports/sector_reports")
    fid = f"{date.strftime('%Y-%m-%d')}_Sector_Report"

    strategy_names = {
        "US Long Credit": "US LC",
        "US Credit": "US C",
        "Liability Aware Long Duration Credit": "US LA",
        "US Long Government/Credit": "US LGC",
    }

    strategies = {
        name: db.load_portfolio(strategy=strat)
        for strat, name in strategy_names.items()
    }

    sectors = [
        "BASICS",
        "CHEMICALS",
        "METALS_AND_MINING",
        "CAPITAL_GOODS",
        "COMMUNICATIONS",
        "CABLE_SATELLITE",
        "MEDIA_ENTERTAINMENT",
        "WIRELINES_WIRELESS",
        "CONSUMER_CYCLICAL",
        "AUTOMOTIVE",
        "RETAILERS",
        "CONSUMER_NON_CYCLICAL",
        "FOOD_AND_BEVERAGE",
        "HEALTHCARE_EX_MANAGED_CARE",
        "MANAGED_CARE",
        "PHARMACEUTICALS",
        "TOBACCO",
        "ENERGY",
        "INDEPENDENT",
        "INTEGRATED",
        "OIL_FIELD_SERVICES",
        "REFINING",
        "MIDSTREAM",
        "ENVIRONMENTAL_IND_OTHER",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "RAILROADS",
        "BANKS",
        "SIFI_BANKS_SR",
        "SIFI_BANKS_SUB",
        "US_REGIONAL_BANKS",
        "YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "LIFE",
        "P&C",
        "REITS",
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "UNIVERSITY",
    ]

    # %%
    doc = Document(fid, path=pdf_path)
    doc.add_preamble(margin=1, bookmarks=True)
    for sector in tqdm(sectors):
        sector_kwargs = db.index_kwargs(sector, in_stats_index=None)
        sector_kws = sector_kwargs
        doc.add_section(sector_kwargs["name"].replace("&", "\&"))
        table, footnote = get_overview_table(sector_kwargs, strategies)
        # sector_kws = sector_kwargs
        prec = {}
        ow_cols = []
        for col in table.columns:
            if "BM %" in col:
                prec[col] = "2%"
            elif "OAD OW" in col:
                ow_cols.append(col)
                prec[col] = "2f"
        ow_max = max(0, table[ow_cols].max().max())
        ow_min = min(0, table[ow_cols].min().min())
        sector
        doc.add_table(
            table,
            table_notes=footnote,
            col_fmt="llc|cc|cccc",
            prec=prec,
            multi_row_header=True,
            adjust=True,
            gradient_cell_col=ow_cols,
            gradient_cell_kws={
                "cmax": "orchid",
                "cmin": "orange",
                "vmax": ow_max,
                "vmin": ow_min,
            },
        )
        doc.add_pagebreak()
    doc.save(save_tex=True)

    # %%


# %%


def get_overview_table(sector_kws, strategies, n=20):
    """
    Get overview table for tickers in given sector including
    rating, % of major benchmarks, and LGIMA's overweights
    in major strategies.

    Returns
    -------
    table_df: pd.DataFrame
        Table of most important ``n`` tickers.
    """
    df_list = []
    ratings_list = []
    for name, strat in strategies.items():
        sector_strat = strat.subset(**sector_kws)
        ow_col = f"{name}*OAD OW"
        if sector_strat.accounts:
            df_list.append(sector_strat.ticker_overweights().rename(ow_col))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                df_list.append(pd.Series(name=ow_col))
        ticker_df = sector_strat.ticker_df
        ratings_list.append(ticker_df["NumericRating"].dropna())
        if name in {"US LC", "US C"}:
            # Get weight of each ticker in benchmark.
            df_list.append(
                (ticker_df["BM_Weight"] / len(sector_strat.accounts)).rename(
                    f"BM %*{name}"
                )
            )

    ratings = np.mean(pd.DataFrame(ratings_list)).round(0).astype(int)
    df_list.append(db.convert_numeric_ratings(ratings).rename("Rating"))
    df = pd.DataFrame(df_list).T.rename_axis(None)
    df["Analyst*Score"] = 0  # TODO: Use analyst score.
    table_cols = [
        "Rating",
        "Analyst*Score",
        "BM %*US LC",
        "BM %*US C",
        "US LC*OAD OW",
        "US C*OAD OW",
        "US LA*OAD OW",
        "US LGC*OAD OW",
    ]
    # Find total overweight over all strategies.
    ow_cols = [col for col in df.columns if "OAD OW" in col]
    bm_pct_cols = [col for col in df.columns if "BM %" in col]
    df["ow"] = np.sum(df[ow_cols], axis=1)
    if len(df) > n:
        # Sort columns by combination of portfolio overweights
        # and market value. Lump together remaining tickers.
        df["bm"] = np.sum(df[bm_pct_cols], axis=1)
        df["sort"] = np.abs(df["ow"]) + 10 * df["bm"]
        df.sort_values("sort", ascending=False, inplace=True)
        df_top_tickers = df.iloc[: n - 1, :]
        df_other_tickers = df.iloc[n:, :]
        other_tickers = df_other_tickers.sum().rename("Other")
        other_tickers["Rating"] = "-"
        table_df = df_top_tickers.append(other_tickers)
        other_tickers = ", ".join(sorted(df_other_tickers.index))
        note = f"\\scriptsize \\textit{{Other}} consists of {other_tickers}."
    else:
        table_df = df.sort_values("ow", ascending=False)
        note = None

    return table_df[table_cols], note
