import multiprocessing as mp
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from datetime import datetime as dt
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd

from lgimapy.bloomberg import get_bloomberg_subsector
from lgimapy.data import Database, Index
from lgimapy.latex import Document, latex_table
from lgimapy.utils import load_json, mkdir, root, Time

# %%


def make_credit_snapshots(date=None):
    """Build credit snapshots and sitch them together."""
    indexes = ["US_IG", "US_IG_10+"]
    for index in indexes:
        build_credit_snapshot(
            index, date=date, include_portfolio_positions=True
        )


def update_credit_snapshots():
    """
    Create all .csv files for past month if they do
    not exist.
    """
    db = Database()
    trade_dates = db.trade_dates(start=db.date("1m"))
    indexes = ["US_IG", "US_IG_10+"]
    for index in indexes:
        # Find dates with missing .csv files for index.
        fid = SnapshotConfig(index).fid
        saved_fids = root("data/credit_snapshots").glob(f"*_{fid}*.csv")
        saved_dates = [fid.stem.split("_")[0] for fid in saved_fids]
        missing_dates = [
            date
            for date in trade_dates
            if date.strftime("%Y-%m-%d") not in saved_dates
        ]
        # Build .csv file for each missing date.
        for date in missing_dates:
            build_credit_snapshot(
                index, date=date, include_portfolio_positions=True, pdf=False
            )


class SnapshotConfig:
    """
    Snapshot configs for each benchmark index.
    """

    def __init__(self, index):
        self.index = index

    @property
    def fid(self):
        return {
            "US_IG": "Market_Credit",
            "US_IG_10+": "Long_Credit",
            "US_HY": "High_Yield",
        }[self.index]

    @property
    def rep_account(self):
        return {"US_IG": "CITMC", "US_IG_10+": "P-LD", "US_HY": None}[
            self.index
        ]

    @property
    def market_sectors(self):
        return {
            "US_IG": [
                "STATS_US_IG",
                "AAA",
                "AA",
                "A",
                # "~A_FIN",
                # "~A_NON_FIN_TOP_30",
                # "~A_NON_FIN_EX_TOP_30",
                # "~A_NON_CORP",
                "BBB",
                "~BBB_FIN",
                "~BBB_NON_FIN_TOP_30",
                "~BBB_NON_FIN_EX_TOP_30",
                "~BBB_NON_CORP",
            ],
            "US_IG_10+": [
                "STATS_US_IG_10+",
                "AAA",
                "AA",
                "A",
                # "~A_FIN",
                # "~A_NON_FIN_TOP_30_10+",
                # "~A_NON_FIN_EX_TOP_30_10+",
                # "~A_NON_CORP",
                "BBB",
                "~BBB_FIN",
                "~BBB_NON_FIN_TOP_30_10+",
                "~BBB_NON_FIN_EX_TOP_30_10+",
                "~BBB_NON_CORP",
            ],
            "US_HY": ["STATS_HY", "~BB", "~B", "~CCC"],
        }[self.index]

    @property
    def kwarg_updates(self):
        return {
            "US_IG": {"in_stats_index": True},
            "US_IG_10+": {"in_stats_index": True, "maturity": (10, None)},
            "US_HY": {"in_stats_index": False, "in_hy_stats_index": True},
        }[self.index]


def build_credit_snapshot(
    index, date=None, pdf=True, include_portfolio_positions=True
):
    """
    Create snapshot for respective index.

    Parameters
    ----------
    index: ``{'US_IG_10+', 'US_IG', 'US_HY'}``
        Index to build snapshot for.
    date: datetime, optional
        Date of close to build snapshot for, by default the
        most recent trade date.
    pdf: bool, default=True
        Whether to save a pdf.
    include_portfolio_positions: bool
        If ``True`` include a column for current portfolio
        positions in resepective index.
    """
    # Define filename and directories to save data.
    db = Database()
    today = db.date("today") if date is None else pd.to_datetime(date)
    config = SnapshotConfig(index)

    fid = f"{today.strftime('%Y-%m-%d')}_{config.fid}_Snapshot"
    csv_path = root("data/credit_snapshots")
    pdf_path = root("reports/credit_snapshots")
    mkdir(csv_path)
    mkdir(pdf_path)

    # Define sectors for each table.
    # Subsectors are indicated with a leading `~`.
    # Major sectors are indicated with a leading `^`
    sectors = [
        "INDUSTRIALS",  # Industrials
        "BASICS",
        "~CHEMICALS",
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
        "~HEALTHCARE_EX_MANAGED_CARE",
        "~MANAGED_CARE",
        "~PHARMACEUTICALS",
        "~TOBACCO",
        "ENERGY",
        "~INDEPENDENT",
        "~INTEGRATED",
        "~OIL_FIELD_SERVICES",
        "~REFINING",
        "~MIDSTREAM",
        "ENVIRONMENTAL_IND_OTHER",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "~RAILROADS",
        "FINANCIALS",  # Financials
        "BANKS",
        "~SIFI_BANKS_SR",
        "~SIFI_BANKS_SUB",
        "~US_REGIONAL_BANKS",
        "~YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "LIFE",
        "P&C",
        "REITS",
        "UTILITY",  # Utilities
        "NON_CORP",  # Non-Corp
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "UNIVERSITY",
    ]

    all_sectors = config.market_sectors + sectors

    # Find dates for daily, week, month, and year to date calculations.
    # Store dates not to be used in table with a `~` before them.
    dates_dict = {
        "~today": today,
        "~6m": db.date("6m", today),
        "Daily": db.date("yesterday", today),
        "WTD": db.date("WTD", today),
        "MTD": db.date("MTD", today),
        # "QTD": db.nearest_date(pd.to_datetime("6/29/2019")),
        "YTD": db.date("YTD", today),
    }

    # Create indexes for all sectors.
    dates_dict["~start"] = min(dates_dict.values())
    kwargs = load_json("indexes")
    # for key in kwargs.keys():
    #     kwargs[key].update({"rating": ("AAA", "AA-")})
    db.load_market_data(start=dates_dict["~start"], end=today, local=True)
    ix_dict = {
        sector: db.build_market_index(
            **{**kwargs[sector.strip("^~")], **config.kwarg_updates}
        )
        for sector in all_sectors
    }

    # Get current market value of the entire long credit index.
    bm_name = config.market_sectors[0]
    full_ix_today = ix_dict[bm_name].subset(date=today)
    ix_mv = full_ix_today.total_value()[0]
    ix_xsrets = ix_dict[bm_name].market_value_weight("XSRet")
    ix_xsrets_6m = ix_xsrets[ix_xsrets.index > dates_dict["~6m"]]

    # Get current state of portfolio.
    if include_portfolio_positions:
        rep_account = db.load_portfolio(account=config.rep_account, date=today)
    else:
        rep_account = None

    pool = mp.Pool(processes=mp.cpu_count() - 2)
    results = [
        pool.apply_async(
            get_snapshot_values,
            args=(ix_dict[sector], dates_dict, ix_mv, ix_xsrets, rep_account),
        )
        for sector in all_sectors
    ]
    rows = [p.get() for p in results]
    table_df = pd.concat(rows, sort=False)
    table_df.to_csv(csv_path / f"{fid}.csv")
    table_df = pd.read_csv(csv_path / f"{fid}.csv", index_col=0)

    if not pdf:
        return

    # Make table captions.
    dates_fmt = [f"Close: {dates_dict['~today'].strftime('%m/%d/%Y')}"]
    dates_fmt += [
        f"{k}: $\Delta${dates_dict[k].strftime('%m/%d')}"
        for k in "Daily WTD MTD".split()
        # for k in "Daily MTD QTD".split()
    ]

    dates_key = ", \hspace{0.2cm} ".join(dates_fmt)
    colors_key = (
        "Color Key: 2-Z move \color{blue}tighter \color{black} ("
        "\color{red}wider\color{black}) relative to the Index"
    )

    captions = [
        f"Market Sectors \hspace{{6.2cm}} \\normalfont{{{dates_key}}}",
        f"IG Sectors  \hspace{{9.2cm}} \\normalfont{{{colors_key}}}",
    ]

    # Create daily snapshot tables and save to pdf.
    doc = Document(fid, path=pdf_path)
    doc.add_preamble(
        margin={"left": 0.5, "right": 0.5, "top": 0.5, "bottom": 0.2},
        page_numbers=False,
        ignore_bottom_margin=True,
    )
    doc.add_background_image("umbrella", scale=1.1, vshift=1, alpha=0.04)
    # doc.add_background_image("xmas_tree", scale=1.1, vshift=2, alpha=0.08)
    sector_list = [config.market_sectors, sectors]
    for sector_subset, cap in zip(sector_list, captions):
        doc.add_table(
            make_table(
                table_df,
                sector_subset,
                cap,
                kwargs,
                include_portfolio_positions,
            )
        )
    doc.save()


def make_table(full_df, sectors, caption, ix_kwargs, include_overweights):
    """
    Subset full DataFrame to specific sectors and build
    LaTeX formatted table.
    """
    # Sort Index to desired order.
    sorted_ix = [ix_kwargs[s.strip("^~")]["name"] for s in sectors]
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
    df.drop("z", axis=1, inplace=True)
    if not include_overweights:
        df.drop("Over*Weight", axis=1, inplace=True)

    # Add indent to index column for subsectors.
    final_ix = []
    for i, (ix, s) in enumerate(zip(df.index, sectors)):
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

    # Find location for major sector formatting and sector bars.
    major_sectors = [
        df.index[0],
        "Industrials",
        "Financials",
        "Utilities",
        "Non-Corp",
    ]

    midrule_locs = [
        "AAA",
        "BBB",
        "Basics",
        "Capital Goods",
        "Communications",
        "Consumer Cyclical",
        "Consumer Non-Cyclical",
        "Energy",
        "Env/Ind. Other",
        "Technology",
        "Transportation",
        "Banks",
        "Brokers/Asset Mngr",
        "Life",
        "REITs",
        "Gov Owned, No Guar",
        "Hospitals",
        "Munis",
        "Sovereigns",
        "Universities",
    ]

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
        # "QTD*OAS": "0f",
        # "QTD*Price": "1f",
        "YTD*OAS": "0f",
        "YTD*Price": "1f",
        "OAD": "1f",
        "YTD*XSRet": "0f",
        "6m*Min": "0f",
        "6m*Max": "0f",
        "6m*%tile": "0f",
    }
    col_fmt = "l|cc|cc|cc|cc|cc|c|c|c|ccl"
    if include_overweights:
        col_fmt = f"{col_fmt}|c"
        prec["Over*Weight"] = "2f"

    table = latex_table(
        df,
        caption=caption,
        font_size="footnotesize",
        col_fmt=col_fmt,
        prec=prec,
        midrule_locs=midrule_locs,
        specialrule_locs=major_sectors,
        row_color={"header": "darkgray", tuple(major_sectors): "lightgray"},
        row_font={"header": "\color{white}\\bfseries"},
        col_style={"6m*%tile": "\pctbar"},
        loc_style=color_locs,
        multi_row_header=True,
        adjust=True,
        greeks=False,
    )
    return table


def get_snapshot_values(ix, dates, index_mv, index_xsrets, portfolio):
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
    index_xsrets: pd.Series
        Excess return series for index.
    portfolio: :class:`Index`
        Current state of representative account portfolio
        as an :class:`Index`.

    Returns
    -------
    pd.DataFrame:
        DataFrame row of IG snapshot table for specified sector.
    """
    # Get synthetic OAS and price histories.
    try:
        oas = ix.get_synthetic_differenced_history("OAS")
        price = ix.get_synthetic_differenced_history("DirtyPrice")
    except UnboundLocalError:
        # DataFrame is emtpy.
        d = OrderedDict([("Close*OAS", np.nan), ("Close*Price", np.nan)])
        for col_name, date in dates.items():
            if col_name.startswith("~"):
                continue
            d[f"{col_name}*OAS"] = np.nan
            d[f"{col_name}*Price"] = np.nan
        d["OAD"] = np.nan
        d["YTD*XSRet"] = np.nan
        d["Mkt*Cap"] = 0
        d["6m*Min"] = np.nan
        d["6m*Max"] = np.nan
        d["6m*%tile"] = 0
        d["z"] = None
        d["Over*Weight"] = np.nan
        return pd.DataFrame(d, index=[ix.name])

    # Get current state of the index and 6 month OAS/XSRet history.
    ix_today = ix.subset(date=dates["~today"])
    oas_6m = oas[oas.index > dates["~6m"]]
    sector_xsrets = ix.market_value_weight("XSRet")
    sector_xsrets_6m = sector_xsrets[sector_xsrets.index > dates["~6m"]]
    xsrets = sector_xsrets_6m - index_xsrets
    xsret_z_scores = (xsrets - np.mean(xsrets)) / np.std(xsrets)
    z = xsret_z_scores[-1]
    z = 0 if np.isnan(z) else z

    # Builid row of the table.
    d = OrderedDict([("Close*OAS", oas[-1]), ("Close*Price", price[-1])])
    for col_name, date in dates.items():
        if col_name.startswith("~"):
            continue
        try:
            d[f"{col_name}*OAS"] = oas[-1] - oas[date]
            d[f"{col_name}*Price"] = price[-1] - price[date]
        except KeyError:
            d[f"{col_name}*OAS"] = np.nan
            d[f"{col_name}*Price"] = np.nan

    d["OAD"] = ix_today.market_value_weight("OAD")[0]
    d["YTD*XSRet"] = 1e4 * ix.aggregate_excess_returns(start_date=dates["YTD"])
    d["Mkt*Cap"] = ix_today.total_value()[0] / index_mv
    d["6m*Min"] = np.min(oas_6m)
    d["6m*Max"] = np.max(oas_6m)
    d["6m*%tile"] = 100 * oas_6m.rank(pct=True)[-1]
    d["z"] = z
    d["color"] = "blue" if z > 2 else "red" if z < -2 else None

    # Determine current portfolio overweight.
    if portfolio is not None:
        # Find all CUSIPs that fit current sector, whether they
        # are index eligible or not.
        unused_constraints = {"in_stats_index", "maturity", "name"}
        constraints = {
            k: v
            for k, v in ix.constraints.items()
            if k not in unused_constraints
        }
        sector_portfolio = portfolio.subset(**constraints)
        d["Over*Weight"] = np.sum(sector_portfolio.df["OAD_Diff"])

    else:
        d["Over*Weight"] = np.nan

    return pd.DataFrame(d, index=[ix.name])


def convert_sectors_to_fin_flags(sectors):
    """
    Convert sectors to flag indicating if they
    are non-financial (0), financial (1), or other (2).

    Parameters
    ----------
    sectors: pd.Series
        Sectors

    Returns
    -------
    fin_flags: nd.array
        Array of values indicating if sectors are non-financial (0),
        financial (1), or other (Treasuries, Sovs, Govt owned).
    """

    financials = {
        "P&C",
        "LIFE",
        "APARTMENT_REITS",
        "BANKING",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "RETAIL_REITS",
        "HEALTHCARE_REITS",
        "OTHER_REITS",
        "FINANCIAL_OTHER",
        "FINANCE_COMPANIES",
        "OFFICE_REITS",
    }
    other = {
        "TREASURIES",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "INDUSTRIAL_OTHER",
        "GOVERNMENT_GUARANTEE",
        "OWNED_NO_GUARANTEE",
    }
    fin_flags = np.zeros(len(sectors))
    fin_flags[sectors.isin(financials)] = 1
    fin_flags[sectors.isin(other)] = 2
    return fin_flags


if __name__ == "__main__":
    date = "6/29/2020"
    with Time():
        make_credit_snapshots(date)
        # update_credit_snapshots()
