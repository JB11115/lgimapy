import multiprocessing as mp
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root

vis.style()

# %%


def create_sector_summit_report():
    db = Database()

    date = db.date("today")
    pdf_path = root("latex/sector_summit")
    fid = f"{date.strftime('%Y-%m-%d')}_Sector_Summit"

    # Load strategy data and store in strategy dict.
    strat = "US Long Credit"
    strat = db.load_portfolio(strategy=strat, universe="stats")
    banks = strat.df["Sector"] == "BANKING"
    rules = {
        " Snr": banks & (strat.df["CollateralType"] == "UNSECURED"),
        " Sub": banks & (strat.df["CollateralType"] == "SUBORDINATED"),
    }
    strat.df["Ticker"] = strat.df["Ticker"].astype(str)
    for name, rule in rules.items():
        strat.df.loc[rule, "Ticker"] = strat.df.loc[rule, "Ticker"] + name

    issuer_oas_df = get_issuer_oas_data()
    sectors = [
        # "US_BANKS",
        # "YANKEE_BANKS",
        # "INSURANCE",
        # "REITS",
        # "BASICS",
        # "MIDSTREAM",
        # "ENERGY_EX_MIDSTREAM",
        # "TELECOM",
        # "MEDIA_ENTERTAINMENT",
        "TECHNOLOGY",
        # "CAPITAL_GOODS",
        # "TRANSPORTATION",
        # "AUTOMOTIVE",
        # "RETAILERS",
        # "CONSUMER_PRODUCTS_FOOD_AND_BEVERAGE",
        # "TOBACCO",
        # "HEALTHCARE_PHARMA",
        # "UTILITY",
    ]
    sector_kwargs = db.index_kwargs(
        sectors[0], unused_constraints="in_stats_index"
    )
    n = 15

    # Create document, append each page, and save.
    doc = Document(fid, path=pdf_path)
    doc.add_preamble(margin=1, bookmarks=True)

    pages = [
        get_sector_page(sector, doc, strat, issuer_oas_df)
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
def get_issuer_oas_data(market="US"):
    db = Database(market=market)

    # Make dict of dates to analyze and then choose nearest traded dates.
    today = db.date("today")
    dates = {
        "today": db.date("today"),
        "YE-2020": db.nearest_date("12/31/2020"),
        "YE-2019": db.nearest_date("12/31/2019"),
    }
    currency = {"US": "USD"}[market]
    maturity_d = {"LC": ((10, None), 30)}
    # Load data for all dates.
    d = defaultdict(list)
    for key, date in dates.items():
        db.load_market_data(date=date)
        for maturity, (mat_kws, max_issue) in maturity_d.items():
            ix = db.build_market_index(
                rating="IG",
                maturity=mat_kws,
                currency=currency,
            )
            # Differentiate Senior and Sub banks in the ticker.
            banks = ix.df["Sector"] == "BANKING"
            rules = {
                " Snr": banks & (ix.df["CollateralType"] == "UNSECURED"),
                " Sub": banks & (ix.df["CollateralType"] == "SUBORDINATED"),
            }
            ix.df["Ticker"] = ix.df["Ticker"].astype(str)
            for name, rule in rules.items():
                ix.df.loc[rule, "Ticker"] = ix.df.loc[rule, "Ticker"] + name
            if key == "today":
                d[maturity].append(ix.ticker_df)
            else:
                d[maturity].append(ix.ticker_df["OAS"].rename(key))

    df_d = {}
    for maturity in maturity_d.keys():
        df = pd.concat(d[maturity], axis=1).rename_axis(None)
        for date in dates.keys():
            if date == "today":
                continue
            df[f"$\\Delta$OAS*{date}"] = df["OAS"] - df[date]
        df_d[maturity] = df.copy()

    return df_d["LC"]


def get_sector_page(sector, doc, strat, issuer_oas_df):
    """Create single page for each sector."""

    page = doc.create_page()
    sector_kwargs = Database().index_kwargs(sector)
    page_name = sector_kwargs["name"].replace("&", "\&")
    page.add_section(page_name)
    page = add_overview_table(page, sector, strat, issuer_oas_df, n=20)
    page.add_pagebreak()
    return sector, page


def add_overview_table(page, sector, strat, issuer_oas_df, n):
    sector_kwargs = Database().index_kwargs(
        sector, unused_constraints="in_stats_index"
    )
    table, footnote = _get_overview_table(
        sector_kwargs, strat, issuer_oas_df, n
    )
    prec = {}
    for col in table.columns:
        if "BM %" in col:
            prec[col] = "2%"
        elif "OAD OW" in col:
            prec[col] = "2f"
        elif "OAS" in col:
            prec[col] = "0f"
        elif "DTS" in col:
            prec[col] = "1%"

    issuer_table = table.iloc[3 : n + 2, :].copy()
    ow_max = max(1e-5, issuer_table["US LC*OAD OW"].max())
    ow_min = min(-1e-5, issuer_table["US LC*OAD OW"].min())
    edit = page.add_subfigures(1, widths=[0.75])
    with page.start_edit(edit):
        page.add_table(
            table,
            table_notes=footnote,
            col_fmt="llc|cc|ccc|c",
            multi_row_header=True,
            midrule_locs=[table.index[3], "Other"],
            prec=prec,
            adjust=True,
            gradient_cell_col=["$\\Delta$OAS*YE-2019", "US LC*OAD OW"],
            gradient_cell_kws={
                "US LC*OAD OW": {
                    "cmax": "orchid",
                    "cmin": "orange",
                },
                "$\\Delta$OAS*YE-2019": {
                    "cmax": "firebrick",
                    "cmin": "steelblue",
                },
            },
        )
    return page


def _get_overview_table(sector_kwargs, strat, issuer_oas_df, n):
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
        "DTS",
        "OAS",
        "$\\Delta$OAS*YE-2020",
        "$\\Delta$OAS*YE-2019",
        "US LC*OAD OW",
    ]

    # Get DataFrame of all individual issuers.
    df_list = []
    ratings_list = []
    sector_strat = strat.subset(**sector_kwargs)
    ow_col = "US LC*OAD OW"
    if sector_strat.accounts:
        df_list.append(sector_strat.ticker_overweights().rename(ow_col))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            df_list.append(pd.Series(name=ow_col))
    ticker_df = sector_strat.ticker_df
    ratings_list.append(ticker_df["NumericRating"].dropna())
    df_list.append(
        (ticker_df["BM_Weight"] / len(sector_strat.accounts)).rename(
            "BM %*US LC"
        )
    )

    ratings = np.mean(pd.DataFrame(ratings_list)).round(0).astype(int)
    df_list.append(Database().convert_numeric_ratings(ratings).rename("Rating"))
    df = pd.DataFrame(df_list).T.rename_axis(None)
    df["Analyst*Score"] = ticker_df["AnalystRating"][df.index].round(0)
    mv_weighted_dts = (ticker_df["DTS"] * ticker_df["MarketValue"])[df.index]
    df["DTS"] = mv_weighted_dts / mv_weighted_dts.sum()
    df["MarketValue"] = ticker_df["MarketValue"]

    # Find total overweight over all strategies.
    ow_cols = [col for col in df.columns if "OAD OW" in col]
    bm_pct_cols = [col for col in df.columns if "BM %" in col]
    df["ow"] = np.sum(df[ow_cols], axis=1)

    # Add OAS columns.
    oas_df = issuer_oas_df[issuer_oas_df.index.isin(df.index)].copy()
    oas_cols = [col for col in table_cols if "OAS" in col]
    oas_df = oas_df[oas_cols]
    df = pd.concat((df, oas_df), axis=1)
    # Get summary rows.
    summary_rows = ["Total", "A Rated", "BBB Rated"]
    summary_df = pd.DataFrame()
    for name in summary_rows:
        if name == "Total":
            df_sub = df.copy()
            df_row = df.sum().rename(name)
        else:
            df_sub = df[df["Rating"].astype(str).str.startswith(name[0])]
            df_row = df_sub.sum().rename(name)

        df_row["Rating"] = "-"
        df_row["Analyst*Score"] = "-"
        df_row["OAS"] = (df_sub["OAS"] * df_sub["MarketValue"]).sum() / df_sub[
            "MarketValue"
        ].sum()
        df_row["$\\Delta$OAS*YE-2020"] = np.nan
        df_row["$\\Delta$OAS*YE-2019"] = np.nan
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
        other_tickers["OAS"] = np.nan
        other_tickers["$\\Delta$OAS*YE-2020"] = np.nan
        other_tickers["$\\Delta$OAS*YE-2019"] = np.nan
        table_df = df_top_tickers.sort_values("ow", ascending=False).append(
            other_tickers
        )
        other_tickers = ", ".join(sorted(df_other_tickers.index))
        note = f"\\scriptsize \\textit{{Other}} consists of {other_tickers}."
    else:
        table_df = df.sort_values("ow", ascending=False)
        note = None

    final_df = summary_df.append(table_df)[table_cols]
    return final_df, note


# %%

if __name__ == "__main__":
    create_sector_summit_report()
