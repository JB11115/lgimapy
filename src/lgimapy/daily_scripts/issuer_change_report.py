import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database, BondBasket
from lgimapy.latex import Document, latex_table
from lgimapy.stats import percentile
from lgimapy.utils import root, mkdir, cp

# %%


def build_issuer_change_report():
    # ---------------------------------- #
    sort_column = "month"

    db = Database(market="US")
    today = db.date("today")
    if today.month >= 3:
        last_col_date = db.date("LAST_YEAR_END")
    else:
        last_col_date = db.date("LAST_YEAR_END", db.date("LAST_YEAR_END"))
    last_col_label = f"$\\Delta$YE'{last_col_date:%y}"

    # ---------------------------------- #

    n_cols = 4
    col_length_thresh = 100

    # markets = ["US", "EUR", "GBP"]
    markets = ["US"]
    maturities = ["10y", "30y"]
    units = ["abs", "pct"]
    page_types = ["nonfin", "fin"]

    market = "US"
    maturity = "30y"
    unit = "abs"
    page = "nonfin"

    # Build date formatting func for column labels.
    if last_col_label is None:
        date_fmt_d = {}
    else:
        date_fmt_d = {last_col_date: last_col_label}
    date_formatter = lambda x: date_fmt_d.get(
        x, f"$\\Delta${x.strftime('%#m/%d')}"
    )

    fid = f"{today.strftime('%Y_%m_%d')}_Issuer_Change_Report"
    doc = Document(fid, path="reports/issuer_change")
    doc.add_preamble(
        margin={
            "paperheight": 31,
            "left": 0.5,
            "right": 0.5,
            "top": 0.5,
            "bottom": 0.2,
        },
        bookmarks=True,
        ignore_bottom_margin=True,
        header=doc.header(
            left="Issuer Change Report",
            right=f"EOD {today.strftime('%B %#d, %Y')}",
        ),
        footer=doc.footer(logo="LG_umbrella"),
    )

    page_names = {
        "nonfin": "Non-Financials",
        "fin": "Financials, Utilities, Non Corp",
    }
    excel_fid = root("reports/current_reports/Issuer_Change_Data.xlsx")
    excel = pd.ExcelWriter(excel_fid)

    # %%
    for market in markets:
        doc.add_section(market)
        data_d, dates_d = get_raw_data(market, last_col_date)
        for maturity in maturities:
            for unit in units:
                doc.add_subsection(f"{maturity} {unit} change")
                for page in page_types:
                    # Get table layout for current page.
                    doc.add_text(f"\\textbf{{{page_names[page]}}}")
                    table_layout, sector_dfs = get_table_layout(
                        data_d,
                        market,
                        maturity,
                        unit,
                        page,
                        n_cols,
                        col_length_thresh,
                    )
                    # Save data to excel.
                    clean_data = pd.concat(sector_dfs.values())
                    clean_data.to_excel(
                        excel, sheet_name=f"{market}_{maturity}"
                    )
                    # Format PDF page.
                    midrule = calculate_midrule(
                        clean_data, sort_column, market, unit, page
                    )
                    columns = doc.add_subfigures(n=4, valign="t")
                    for i, (col, column_edit) in enumerate(
                        zip(table_layout.columns, columns)
                    ):
                        with doc.start_edit(column_edit):
                            for table_name in table_layout[col].dropna():
                                sector_df = sector_dfs[table_name]
                                table = format_table(
                                    sector_df,
                                    table_name,
                                    unit,
                                    sort_column,
                                    dates_d,
                                    midrule,
                                    date_formatter,
                                )
                                doc.add_table(table)
                    doc.add_pagebreak()

    add_methodology_section(doc)
    doc.save()
    doc.save_as("Issuer_Change_Report", path="reports/current_reports")
    excel.save()


def get_raw_data(market, last_col_date):
    db = Database(market=market)

    # Make dict of dates to analyze and then choose nearest traded dates.
    today = db.date("today")
    mtd_date = db.date("MTD")
    if today.day < 10:
        mtd_date = db.date("MTD", mtd_date)
    dates = {
        "today": db.date("today"),
        "week": db.date("6d"),
        "month": mtd_date,
        "year_end": db.nearest_date(last_col_date),
    }
    currency = {"US": "USD"}[market]
    maturity_d = {"10y": ((8.25, 11), 2), "30y": ((25, 32), 5)}
    # Load data for all dates.
    d = defaultdict(list)
    for key, date in dates.items():
        db.load_market_data(date=date)
        for maturity, (mat_kws, max_issue) in maturity_d.items():
            ix = db.build_market_index(
                rating="IG",
                maturity=mat_kws,
                issue_years=(None, max_issue),
                currency=currency,
            )
            ix.expand_tickers()
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
            df[f"{date}_abs_change"] = df["OAS"] - df[date]
            df[f"{date}_pct_change"] = df["OAS"] / df[date] - 1
        df_d[maturity] = df.copy()

    return df_d, dates


def clean_sector_table(df, name):
    df["LGIMA_Sector"] = name
    dropped_cols = [
        "PrevMarketValue",
        "DirtyPrice",
        "CleanPrice",
        "AnalystRating",
        "TRet",
        "XSRet",
        "week",
        "month",
        "year_end",
        "Ticker",
    ]
    return df.drop(columns=dropped_cols).dropna(subset=["OAS", "DTS"])


def load_table_layout(fid):
    try:
        return pd.read_csv(fid, index_col=0)
    except FileNotFoundError:
        return None


def get_market_sectors(market, page):
    return {
        "US": {
            "nonfin": [
                "BASICS",
                "CAPITAL_GOODS",
                "COMMUNICATIONS",
                "CONSUMER_CYCLICAL",
                "CONSUMER_NON_CYCLICAL_EX_HEALTHCARE",
                "HEALTHCARE_PHARMA",
                "ENERGY",
                "TECHNOLOGY",
                "TRANSPORTATION",
            ],
            "fin": [
                "BANKS",
                "BROKERAGE_ASSETMANAGERS_EXCHANGES",
                "LIFE",
                "P_AND_C",
                "REITS",
                "UTILITY",  # Utilities
                "OWNED_NO_GUARANTEE",
                "GOVERNMENT_GUARANTEE",
                "HOSPITALS",
                "MUNIS",
                "SOVEREIGN",
                "SUPRANATIONAL",
                "UNIVERSITY",
            ],
        }
    }[market][page]


def get_table_layout(
    df_d, market, maturity, unit, page, n_cols, col_length_thresh
):
    """
    Get table layouts for current page and cleaned DataFrames for
    each sector. Save current data to current excel sheet.
    """
    sectors = get_market_sectors(market, page)

    # Separate DataFrame into sector components.
    db = Database(market=market)
    df = df_d[maturity].copy()
    df["Ticker"] = df.index
    bb = BondBasket(df, index=None)

    table_const = 2
    sector_dfs, sector_lengths = {}, {}
    for sector in sectors:
        kwargs = db.index_kwargs(sector, unused_constraints=["in_stats_index"])
        name = kwargs["name"]
        sector_df = clean_sector_table(bb.subset(**kwargs).df, name)
        if len(sector_df) < 4:
            continue
        sector_dfs[name] = sector_df
        sector_lengths[name] = len(sector_df) + table_const

    # Load table layout, solving for a new one if none exists.
    data_dir = root("data/issuer_change/")
    mkdir(data_dir)
    fid = data_dir / f"{market}_{maturity}_{page}.csv"
    table_layout = load_table_layout(fid)
    if table_layout is None:
        table_layout = optimize_table_layout(
            sector_lengths, n_cols, col_length_thresh
        )

    # Check that table layout is within set tolerance for number of rows.
    max_col_len = 0
    for col in table_layout.columns:
        current_col_len = 0
        for table in table_layout[col].dropna():
            current_col_len += sector_lengths[table]
        max_col_len = max(max_col_len, current_col_len)

    # Re-optimize table layout if required
    if max_col_len > col_length_thresh:
        table_layout = optimize_table_layout(
            sector_lengths, n_cols, col_length_thresh
        )

    # Save current layout.
    table_layout.to_csv(fid)
    return table_layout, sector_dfs


def optimize_table_layout(sector_length_d, n_cols, col_length_thresh):
    # Get all possible combinations as list of lists. One for keys
    # and one for values (lengths of each table).
    def get_partitions(seq, k):
        n = len(seq)
        groups = []

        def generate_partitions(i):
            if i >= n:
                yield list(map(tuple, groups))
            else:
                if n - i > k - len(groups):
                    for group in groups:
                        group.append(seq[i])
                        yield from generate_partitions(i + 1)
                        group.pop()

                if len(groups) < k:
                    groups.append([seq[i]])
                    yield from generate_partitions(i + 1)
                    groups.pop()

        return generate_partitions(0)

    keys = get_partitions(list(sector_length_d.keys()), n_cols)
    values = get_partitions(list(sector_length_d.values()), n_cols)

    # Find optimal layout index.
    df_rows = []
    for val in values:
        df_rows.append([sum(layout) for layout in val])
    df = pd.DataFrame(df_rows)
    std = df.std(axis=1)
    index = int(std.argmin())

    # Format table such that columns are in order from longest (left)
    # to shortest (right) and format into a DataFrame.
    column_table_names = next(it.islice(keys, index, None))
    column_lengths = list(df.loc[index])
    max_column_length = np.max(column_lengths)
    if max_column_length > col_length_thresh:
        print(column_lengths)
        raise ValueError(
            "Max column length exceeded during layout optimization."
        )
    col_order = np.argsort(column_lengths)[::-1]
    column_tables_df = pd.DataFrame(column_table_names).T[col_order]
    column_tables_df.columns = np.arange(len(column_tables_df.columns))
    return column_tables_df


def format_table(df, sector, unit, sort_col, dates, midrule, date_formatter):
    periods = ["week", "month", "year_end"]
    cols = ["OAS", "DTS", *[f"{period}_{unit}_change" for period in periods]]
    new_cols = [
        "OAS",
        "DTS",
        *[date_formatter(dates[period]) for period in periods],
    ]
    new_sort_col = new_cols[periods.index(sort_col) + 2]
    table = df.sort_values(f"{sort_col}_{unit}_change", ascending=False).copy()
    # Get DTS as a MV weighted % of the sector.
    table["DTS"] = (table["DTS"] * table["MarketValue"]) / (
        table["DTS"] * table["MarketValue"]
    ).sum()
    # Find locations of A-rated issuers.
    a_rated = tuple(table[table["NumericRating"] <= 7].index)

    # Update column names with dates.
    table = table[cols]
    table.columns = new_cols

    # Get precision formatting for each column.
    col_unit = {"pct": "0%", "abs": "0f"}[unit]
    col_prec = {col: col_unit for col in table.columns}
    col_prec["OAS"] = "0f"
    col_prec["DTS"] = "1%"

    # Find index for midrule.
    try:
        midrule_loc = [table[table[new_sort_col] < midrule].index[0]]
    except IndexError:
        midrule_loc = None
    latex_table(
        table,
        caption=sector.replace("&", "\\&"),
        col_fmt="lrrrrr",
        prec=col_prec,
        row_font={a_rated: "\\bfseries"},
        midrule_locs=midrule_loc,
        gradient_cell_col=new_sort_col,
        adjust=True,
    )

    return latex_table(
        table,
        caption=sector.replace("&", "\\&"),
        col_fmt="lrrrrr",
        prec=col_prec,
        row_font={a_rated: "\\bfseries"},
        midrule_locs=midrule_loc,
        gradient_cell_col=new_sort_col,
        adjust=True,
    )


def calculate_midrule(df, sort_col, market, unit, page):
    db = Database(market=market)
    col = f"{sort_col}_{unit}_change"
    sectors = [
        db.index_kwargs(sector)["name"]
        for sector in get_market_sectors(market, page)
    ]
    df_page = df[df["LGIMA_Sector"].isin(sectors)]
    return percentile(df_page[col], weights=df_page["MarketValue"])


def add_methodology_section(doc):
    doc.add_section("Methodology")
    doc.add_text(
        """
        \\begin{itemize}

        \\item
        The OAS and DTS for issuers were found for each date
        by taking the market value weighted average of the respective
        metric for all bonds of each issuer that both had
        8.25 - 11 years until
        maturity and were issued in the last 2 years (10 year
        page), or 25 - 32 years until maturity and issued within the
        last 5 years (30 year page).

        \\item
        The DTS column provides the DTS of each issuer as a
        (market value weighted) percentage of the overall sector.
        As such, a large issuer with high DTS will have a large
        sector DTS \\% in the table.
        Due to the market value weighting, a large issuer with
        low DTS will tend to have a higher sector DTS \\% than
        a small issuer with high DTS.

        \\item
        Bold tickers represented A-rated issuers

        \\item
        The midrule lines within each table represent the
        market value weighted median move of all the issuers
        on the given page (i.e., all those with the same currency,
        maturity, and non-fin/fin categorization) over the time
        horizon of the sorted (colored) column.

        \\end{itemize}
        """
    )


# %%
if __name__ == "__main__":
    build_issuer_change_report()
