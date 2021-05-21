import multiprocessing as mp
import warnings
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
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

    sectors = db.IG_sectors(with_tildes=True, with_chevrons=True)
    sectors = [s for s in sectors if not s.startswith(">>")]
    sector_names = [s.strip("~") for s in sectors]
    # %%
    # Create document, append each page, and save.
    doc = Document(fid, path=pdf_path)
    doc.add_preamble(margin=1, bookmarks=True)

    pages = [
        get_sector_page(sector, doc, strategy_d, xsret_model_d)
        for sector in tqdm(sectors)
    ]
    # pages = joblib.Parallel(n_jobs=6)(
    #     joblib.delayed(get_sector_page)(sector, doc, strategy_d, xsret_model_d)
    #     for sector in sectors
    # )
    df = (
        pd.DataFrame(pages, columns=["sector", "page"])
        .set_index("sector")
        .reindex(sector_names)
    )
    for page in df["page"]:
        doc.add_page(page)
    doc.save(save_tex=False)

    # %%


# %%
def get_sector_page(sector, doc, strategy_d, xsret_model_d):
    """Create single page for each sector."""

    page = doc.create_page()
    sector_name = sector.strip("~")
    sector_kwargs = Database().index_kwargs(sector_name)
    page_name = sector_kwargs["name"].replace("&", "\&")
    if sector.startswith("~"):
        page.add_subsection(page_name)
    else:
        page.add_section(page_name)

    page = add_overview_table(page, sector_name, strategy_d, n=20)
    page = add_issuer_performance_tables(page, sector_kwargs, xsret_model_d)

    page.add_pagebreak()
    return sector_name, page


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
        elif "DTS" in col:
            prec[col] = "1%"
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
            col_fmt="llcc|cc|cccc",
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
        "Sector*DTS",
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
    ticker_df = strategy_d["US MC"].ticker_df.copy()
    df["Analyst*Score"] = ticker_df["AnalystRating"][df.index].round(0)
    mv_weighted_dts = (ticker_df["DTS"] * ticker_df["MarketValue"])[df.index]
    df["Sector*DTS"] = mv_weighted_dts / mv_weighted_dts.sum()

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
import pdb
