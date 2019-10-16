import multiprocessing as mp
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from datetime import datetime as dt
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document, latex_table
from lgimapy.utils import load_json, mkdir, root, Time

# %%


def build_long_credit_snapshot():
    # Define filename and directories to save data.
    fid = f"{dt.today().strftime('%Y-%m-%d')}_Long_Credit_Snapshot"
    csv_path = root("data/long_credit_snapshot")
    pdf_path = root("reports/long_credit_snapshot")
    mkdir(csv_path)

    # Define sectors for each table.
    market_sectors = [
        "STATS_all",
        "~AAA",
        "~AA",
        "~A",
        "~BBB",
        "FINANCIALS",
        "INDUSTRIALS",
        "~INDUSTRIALS_EX_ENERGY_MM",
        "UTILITY",
        "~ELECTRIC",
        "~NATURAL_GAS",
        "~UTILITY_OTHER",
    ]
    corp_secotrs = [
        "BASICS",
        "~METALS_AND_MINING",
        "CAPITAL_GOODS",
        "COMMUNICATIONS",
        "~CABLE_SATELLITE",
        "~MEDIA_ENTERTAINMENT",
        "~WIRELINES_WIRELESS",
        "CONSUMER_CYCLICAL",
        "~AUTOMOTIVE",
        "~RETAILERS",
        "CONSUMER_NON_CYCLICAL",
        "~FOOD_AND_BEVERAGE",
        "~HEALTHCARE",
        "~MANAGED_CARE",
        "~PHARMACEUTICALS",
        "~TOBACCO",
        "ENERGY",
        "~INDEPENDENT",
        "~INTEGRATED",
        "~OIL_FIELD_SERVICES",
        "~REFINING",
        "~MIDSTREAM",
        "INDUSTRIAL_OTHER",
        "~UNIVERSITY",
        "TECHNOLOGY",
        "TRANSPORTATION",
    ]
    non_corp_sectors = [
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "MUNIS",
        "SOVEREIGN",
        "SUPRANATIONAL",
    ]
    fin_sectors = [
        "BANKS",
        "~SIFI_BANKS_SR",
        "~SIFI_BANKS_SUB",
        "~US_REGIONAL_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "FINANCE_COMPANIES",
        "LIFE",
        "P&C",
        "REITS",
    ]
    all_sectors = market_sectors + corp_secotrs + non_corp_sectors + fin_sectors

    # Find dates for daily, week, month, and year to date calculations.
    # Store dates not to be used in table with a `~` before them.
    db = Database()
    tdates = db.trade_dates

    today = tdates[-1]
    last_trade = partial(bisect_left, tdates)
    dates_dict = {
        "~today": tdates[-1],
        "~6m": db.nearest_date(today - timedelta(183)),
        "Daily": tdates[-2],
        "WTD": tdates[last_trade(today - timedelta(today.weekday() + 1)) - 1],
        "MTD": tdates[last_trade(today.replace(day=1)) - 1],
        "YTD": tdates[last_trade(today.replace(month=1, day=1)) - 1],
    }

    # Create indexes for all sectors.
    dates_dict["~start"] = min(dates_dict.values())
    kwargs = load_json("indexes")
    db.load_market_data(start=dates_dict["~start"], local=True)
    ix_dict = {
        sector: db.build_market_index(
            **kwargs[sector.strip("~")], maturity=(10, None)
        )
        for sector in all_sectors
    }

    # Get current market value of the entrie long credit index.
    full_ix_today = ix_dict["STATS_all"].subset(date=today)
    ix_mv = full_ix_today.total_value()[0]

    # Create DataFrame of snapshot values with each sector as a row.
    # Use multiprocessing to speed up computation.
    # pool = mp.Pool(processes=mp.cpu_count())
    pool = mp.Pool(processes=7)
    results = [
        pool.apply_async(
            get_snapshot_values, args=(ix_dict[sector], dates_dict, ix_mv)
        )
        for sector in all_sectors
    ]
    rows = [p.get() for p in results]
    df = pd.concat(rows)
    df.to_csv(csv_path / f"{fid}.csv")
    df = pd.read_csv(csv_path / f"{fid}.csv", index_col=0)

    # Make table captions.
    close_key = f"Close: {dates_dict['~today'].strftime('%m/%d/%Y')}"
    dates_key = ", \hspace{0.2cm} ".join(
        f"{k}: $\Delta${dates_dict[k].strftime('%m/%d')}"
        for k in "Daily WTD MTD".split()
    )
    colors_key = " \hspace{0.3cm} ".join(
        f"\color{{{c}}}{{2-Z move {tw}}}"
        for c, tw in zip(["blue", "red"], ["tighter", "wider"])
    )
    captions = [
        make_caption("Market Sectors", 11.9, close_key),
        make_caption("IG Corp Sectors", 6.7, dates_key),
        "IG Non-Corp Sectors",
        make_caption("IG Fin Sectors", 9.3, colors_key),
    ]

    # Create daily snapshot tables and save to pdf.
    doc = Document(fid, path=pdf_path)
    doc.add_preamble(
        margin={"left": 0.75, "right": 0.75, "top": 0.2, "bottom": 0.2},
        page_numbers=False,
    )
    sector_list = [market_sectors, corp_secotrs, non_corp_sectors, fin_sectors]
    for sector_subset, cap in zip(sector_list, captions):
        doc.add_table(make_table(df, sector_subset, cap, kwargs))
    doc.save()


def make_caption(title, distance_cm, key):
    """Make table caption with key spaced from title."""
    return f"{title} \hspace{{{distance_cm}cm}} \\normalfont{{{key}}}"


def make_table(full_df, sectors, caption, ix_kwargs):
    """
    Subset full DataFrame to specific sectors and build
    LaTeX formatted table.
    """
    # Sort Index to desired order.
    sorted_ix = [ix_kwargs[s.strip("~")]["name"] for s in sectors]
    df = full_df[full_df.index.isin(sorted_ix)].copy()
    df = df.reindex(sorted_ix)

    # Find colors locations for daily changes with greater than 2-Z move.
    col = list(df.columns).index("Daily*OAS")
    df_colors = df[~df["color"].isna()]
    color_locs = {
        (list(df.index).index(sector), col): f"\color{{{color}}}"
        for sector, color in zip(df_colors.index, df_colors["color"])
    }
    df.drop("color", axis=1, inplace=True)

    # Add indent to index column for subsectors.
    final_ix = []
    for i, (ix, s) in enumerate(zip(df.index, sectors)):
        if s == "STATS_all":
            final_ix.append("US Long Credit Index")
            continue
        if s.startswith("~"):
            try:
                if sectors[i + 1].startswith("~"):
                    final_ix.append("\hspace{1mm} $\\vdash$ " + ix)
                else:
                    final_ix.append("\hspace{1mm} $\lefthalfcup$ " + ix)
            except IndexError:
                # Last item in table, must be halfcup.
                final_ix.append("\hspace{1mm} $\lefthalfcup$ " + ix)
        else:
            final_ix.append(ix)
    df.index = final_ix

    # Find location for midrule bars.
    midrule_locs = [ix for ix in df.index if not ix.startswith("\\hspace{")]
    if caption == "IG Non-Corp Sectors":
        midrule_locs = None

    # Format market cap by precision by value for 2 sig figs.
    df["Mkt*Cap"] = [
        f"{v:.1%}" if v < 0.1 else f"{v:.0%}" for v in df["Mkt*Cap"]
    ]
    # Format remaining columns.
    prec = {
        "Close*OAS": "0f",
        "Close*Price": "2f",
        "Daily*OAS": "1f",
        "Daily*Price": "2f",
        "WTD*OAS": "1f",
        "WTD*Price": "1f",
        "MTD*OAS": "0f",
        "MTD*Price": "1f",
        "YTD*OAS": "0f",
        "YTD*Price": "1f",
        "OAD": "1f",
        "YTD*XSRet": "0f",
        "6m*Min": "0f",
        "6m*Max": "0f",
        "6m*%tile": "0f",
    }

    table = latex_table(
        df,
        caption=caption,
        font_size="footnotesize",
        col_fmt="l|cc|cc|cc|cc|cc|c|c|c|ccl",
        prec=prec,
        midrule_locs=midrule_locs,
        col_style={"6m*%tile": "\mybar"},
        loc_style=color_locs,
        multi_row_header=True,
        adjust=True,
        greeks=False,
    )
    return table


def get_snapshot_values(ix, dates, index_mv):
    """
    Get single row from IG snapshot table for
    specified sector.

    Parameters
    ----------
    ix: :class:`Index`
        Index of the sector for current row.
    dates: Dict[str: datetime].
        Dict of pertinent dates for constructing row.
    index_mv: float
        Current total IG Long Credit Index market value.

    Returns
    -------
    pd.DataFrame:
        DataFrame row of IG snapshot table for specified sector.
    """
    # Get synthetic OAS and price histories.
    oas = ix.get_synthetic_differenced_history("OAS")
    price = ix.get_synthetic_differenced_history("DirtyPrice")

    # Get current state of the index and 6 month OAS history.
    ix_today = ix.subset(date=dates["~today"])
    oas_6m = oas[oas.index > dates["~6m"]]
    oas_change = np.diff(oas_6m)
    oas_z_scores = (oas_change - np.mean(oas_change)) / np.std(oas_change)
    z = oas_z_scores[-1]

    # Builid row of the table.
    d = OrderedDict([("Close*OAS", oas[-1]), ("Close*Price", price[-1])])
    for col_name, date in dates.items():
        if col_name.startswith("~"):
            continue
        d[f"{col_name}*OAS"] = oas[-1] - oas[date]
        d[f"{col_name}*Price"] = price[-1] - price[date]

    d["OAD"] = ix_today.market_value_weight("OAD")[0]
    d["YTD*XSRet"] = 1e4 * ix.aggregate_excess_returns(start_date=dates["YTD"])
    d["Mkt*Cap"] = ix_today.total_value()[0] / index_mv
    d["6m*Min"] = np.min(oas_6m)
    d["6m*Max"] = np.max(oas_6m)
    d["6m*%tile"] = 100 * oas_6m.rank(pct=True)[-1]
    d["color"] = "blue" if z < -2 else "red" if z > 2 else None
    return pd.DataFrame(d, index=[ix.name])


if __name__ == "__main__":
    with Time():
        build_long_credit_snapshot()
