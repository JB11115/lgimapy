import multiprocessing as mp
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
from lgimapy.models import XSRETPerformance
from lgimapy.utils import root

vis.style()
# %%


def create_sector_report():
    db = Database()
    date = db.date("today")
    pdf_path = root("reports/sector_reports")
    fid = f"{date.strftime('%Y-%m-%d')}_Sector_Report"

    # Load strategy data and store in strategy dict.
    strategy_names = {
        "US Long Credit": "US LC",
        "US Credit": "US MC",
        "Liability Aware Long Duration Credit": "US LA",
        "US Long Government/Credit": "US LGC",
    }
    strategy_d = {
        name: db.load_portfolio(strategy=strat, universe="stats")
        for strat, name in strategy_names.items()
    }

    # Train excess return models and store in dict.
    db_xsret = Database()
    db_xsret.load_market_data(start=db.date("2m"))
    db_xsret.make_thread_safe()
    xsret_model_d = {}
    model_maturities = {
        "Long Credit": (10, None),
        "5-10 yr Credit": (5, 10),
    }
    for name, maturity in model_maturities.items():
        mod = XSRETPerformance(db_xsret)
        mod.train(forecast="1m", maturity=maturity)
        xsret_model_d[name] = mod

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
    # Create document, append each page, and save.
    doc = Document(fid, path=pdf_path)
    doc.add_preamble(margin=1, bookmarks=True)
    # pool = mp.Pool(processes=mp.cpu_count() - 2)
    # res = [
    #     pool.apply_async(
    #         get_sector_page, args=(sector, doc, strategy_d, xsret_model_d)
    #     )
    #     for sector in sectors
    # ]
    # pages = [r.get() for r in res]
    pages = [
        get_sector_page(sector, doc, strategy_d, xsret_model_d)
        for sector in tqdm(sectors)
    ]
    df = (
        pd.DataFrame(pages, columns=["sector", "page"])
        .set_index("sector")
        .reindex(sectors)
    )
    for page in df["page"]:
        doc.add_page(page)
    doc.save(save_tex=False)

    # %%


# %%
def get_sector_page(sector, doc, strategy_d, xsret_model_d):
    """Create single page for each sector."""

    page = doc.create_page()
    sector_kwargs = Database().index_kwargs(sector)
    page_name = sector_kwargs["name"].replace("&", "\&")
    page.add_section(page_name)

    page = add_overview_table(page, sector, strategy_d, n=20)
    page = add_issuer_performance_tables(page, sector_kwargs, xsret_model_d)

    page.add_pagebreak()
    return sector, page


def add_overview_table(page, sector, strategy_d, n):
    sector_kwargs = Database().index_kwargs(
        sector, unused_constraints="in_stats_index"
    )
    table, footnote = _get_overview_table(sector_kwargs, strategy_d, n)
    prec = {}
    ow_cols = []
    for col in table.columns:
        if "BM %" in col:
            prec[col] = "2%"
        elif "OAD OW" in col:
            ow_cols.append(col)
            prec[col] = "2f"

    issuer_table = table.iloc[3 : n + 2, :].copy()
    ow_max = max(1e-5, issuer_table[ow_cols].max().max())
    ow_min = min(-1e-5, issuer_table[ow_cols].min().min())
    edit = page.add_subfigures(1, widths=[0.75])
    with page.start_edit(edit):
        page.add_table(
            table,
            table_notes=footnote,
            col_fmt="llc|cc|cccc",
            multi_row_header=True,
            midrule_locs=[table.index[3], "Other"],
            prec=prec,
            adjust=True,
            gradient_cell_col=ow_cols,
            gradient_cell_kws={
                "cmax": "orchid",
                "cmin": "orange",
                "vmax": ow_max,
                "vmin": ow_min,
            },
        )
    return page


def _get_overview_table(sector_kwargs, strategy_d, n):
    """
    Get overview table for tickers in given sector including
    rating, % of major benchmarks, and LGIMA's overweights
    in major strategies.

    Returns
    -------
    table_df: pd.DataFrame
        Table of most important ``n`` tickers.
    """
    table_cols = [
        "Rating",
        "Analyst*Score",
        "BM %*US LC",
        "BM %*US MC",
        "US LC*OAD OW",
        "US MC*OAD OW",
        "US LA*OAD OW",
        "US LGC*OAD OW",
    ]

    # Get DataFrame of all individual issuers.
    df_list = []
    ratings_list = []
    for name, strat in strategy_d.items():
        sector_strat = strat.subset(**sector_kwargs)
        ow_col = f"{name}*OAD OW"
        if sector_strat.accounts:
            df_list.append(sector_strat.ticker_overweights().rename(ow_col))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                df_list.append(pd.Series(name=ow_col))
        ticker_df = sector_strat.ticker_df
        ratings_list.append(ticker_df["NumericRating"].dropna())
        if name in {"US LC", "US MC"}:
            # Get weight of each ticker in benchmark.
            df_list.append(
                (ticker_df["BM_Weight"] / len(sector_strat.accounts)).rename(
                    f"BM %*{name}"
                )
            )

    ratings = np.mean(pd.DataFrame(ratings_list)).round(0).astype(int)
    df_list.append(Database().convert_numeric_ratings(ratings).rename("Rating"))
    df = pd.DataFrame(df_list).T.rename_axis(None)
    df["Analyst*Score"] = strat.ticker_df["AnalystRating"][df.index].round(0)

    # Find total overweight over all strategies.
    ow_cols = [col for col in df.columns if "OAD OW" in col]
    bm_pct_cols = [col for col in df.columns if "BM %" in col]
    df["ow"] = np.sum(df[ow_cols], axis=1)

    # Get summary rows.
    summary_rows = ["Total", "A Rated", "BBB Rated"]
    summary_df = pd.DataFrame()
    for name in summary_rows:
        if name == "Total":
            df_row = df.sum().rename(name)
        else:
            df_row = (
                df[df["Rating"].astype(str).str.startswith(name[0])]
                .sum()
                .rename(name)
            )
        df_row["Rating"] = "-"
        df_row["Analyst*Score"] = "-"
        summary_df = summary_df.append(df_row)

    # Create `other` row if required.
    if len(df) > n:
        # Sort columns by combination of portfolio overweights
        # and market value. Lump together remaining tickers.
        df["bm"] = np.sum(df[bm_pct_cols], axis=1)
        df["importance"] = np.abs(df["ow"]) + 10 * df["bm"]
        df.sort_values("importance", ascending=False, inplace=True)
        df_top_tickers = df.iloc[: n - 1, :]
        df_other_tickers = df.iloc[n:, :]
        other_tickers = df_other_tickers.sum().rename("Other")
        other_tickers["Rating"] = "-"
        other_tickers["Analyst*Score"] = "-"
        table_df = df_top_tickers.sort_values("ow", ascending=False).append(
            other_tickers
        )
        other_tickers = ", ".join(sorted(df_other_tickers.index))
        note = f"\\scriptsize \\textit{{Other}} consists of {other_tickers}."
    else:
        table_df = df.sort_values("ow", ascending=False)
        note = None

    return summary_df.append(table_df)[table_cols], note


def add_issuer_performance_tables(page, sector_kwargs, xsret_model_d):
    table_edits = page.add_subfigures(2)
    for edit, (title, mod) in zip(table_edits, xsret_model_d.items()):
        table = mod.get_issuer_table(**sector_kwargs).reset_index(drop=True)
        table.drop("RatingBucket", axis=1, inplace=True)
        bold_rows = tuple(
            table[table["Issuer"].isin({"A-Rated", "BBB-Rated"})].index
        )
        with page.start_edit(edit):
            page.add_table(
                table,
                prec=mod.table_prec(table),
                col_fmt="llc|cc|ccc",
                caption=f"{title} 1M Performance",
                adjust=True,
                hide_index=True,
                font_size="scriptsize",
                multi_row_header=True,
                row_font={bold_rows: "\\bfseries"},
                gradient_cell_col=["Out*Perform", "Impact*Factor"],
                gradient_cell_kws={
                    "Out*Perform": {"cmax": "steelblue", "cmin": "firebrick"},
                    "Impact*Factor": {"cmax": "oldmauve"},
                },
            )
            page.add_vskip()

    return page


# %%


if __name__ == "__main__":
    create_sector_report()

# mod = XSRETPerformance(db)
# mod.train(maturity=(10, None))
#
#
