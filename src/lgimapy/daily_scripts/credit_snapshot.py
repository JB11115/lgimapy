import multiprocessing as mp
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from datetime import datetime as dt
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd

from lgimapy.bloomberg import get_bloomberg_subsector
from lgimapy.data import Database, Index, IG_sectors, HY_sectors
from lgimapy.latex import Document, latex_table, merge_pdfs
from lgimapy.utils import load_json, mkdir, root, Time, restart_program

# %%


def make_credit_snapshots(date=None, include_portfolio=True):
    """Build credit snapshots and sitch them together."""
    indexes = ["US_IG_10+", "US_IG", "US_HY"]
    indexes = ["US_IG_10+", "US_IG"]
    fids = []
    for index in indexes:
        fid = build_credit_snapshot(
            index, date=date, include_portfolio_positions=include_portfolio
        )
        fids.append(fid)
    pdf_path = "reports/credit_snapshots"
    merge_pdfs("Credit_Snapshot", fids, path=pdf_path)


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
            # restart_program(RAM_threshold=75)
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
        return {"US_IG": "CITMC", "US_IG_10+": "P-LD", "US_HY": "PMCHY"}[
            self.index
        ]

    @property
    def horizon(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return 6
        elif self.index == "US_HY":
            return 7

    @property
    def overview_sectors(self):
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
                "AJP_WINNERS",
                "AJP_LOSERS",
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
                "AJP_WINNERS",
                "AJP_LOSERS",
            ],
            "US_HY": [],
        }[self.index]

    @property
    def sectors(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return IG_sectors(with_tildes=True)
        elif self.index == "US_HY":
            return HY_sectors(with_tildes=True, sectors_only=False)

    @property
    def midrule_locs(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return [
                "AAA",
                "BBB",
                "AJP Winners",
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
        elif self.index == "US_HY":
            return [
                "Basics",
                "Capital Goods",
                "Beverage",
                "Energy",
                "Healthcare",
                "Gaming",
                "Real Estate",
                "Environmental",
                "TMT",
                "Technology",
                "Transportation",
                "Utilities",
            ]

    @property
    def title(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return "IG"
        elif self.index == "US_HY":
            return "HY"

    @property
    def table_types(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return ["sector", "sector"]
        elif self.index == "US_HY":
            return [
                "cross-asset",
                "sector",
                "gainers_losers",
                "sector_not_corrected",
            ]

    @property
    def footnote(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return [None, None]
        elif self.index == "US_HY":
            return [
                (
                    "\\tiny Summary statistics and \\%tiles use a 5yr "
                    "history for credit, rates, and LIBOR with a "
                    "1yr history for equity indexes and the VIX."
                ),
                (
                    "\\tiny Historical spreads and yields corrected "
                    "to reflect current investable universe."
                ),
                None,
                "\\tiny Raw historical spreads and yields.",
            ]

    def captions(self, dates_key, colors_key):
        if self.index == "US_IG":
            return (
                (
                    f"Market Credit IG Overview \hspace{{4.2cm}} "
                    f"\\normalfont{{{dates_key}}}"
                ),
                (
                    f"Market Credit IG Sectors  \hspace{{6.8cm}} "
                    f"\\normalfont{{{colors_key}}}"
                ),
            )
        elif self.index == "US_IG_10+":
            return (
                (
                    f"Long Credit IG Overview \hspace{{4.6cm}} "
                    f"\\normalfont{{{dates_key}}}"
                ),
                (
                    f"Long Credit IG Sectors  \hspace{{7.2cm}} "
                    f"\\normalfont{{{colors_key}}}"
                ),
            )
        elif self.index == "US_HY":
            return (
                (
                    f"Cross-Asset Overview \hspace{{5.2cm}} "
                    f"\\normalfont{{{dates_key}}}"
                ),
                (
                    f"{self.title} Sectors  \hspace{{9.2cm}} "
                    f"\\normalfont{{{colors_key}}}"
                ),
                f"H4UN Gainers/Losers",
                (
                    f"{self.title} Sectors  \hspace{{9.2cm}} "
                    f"\\normalfont{{{colors_key}}}"
                ),
            )

    @property
    def table_sectors(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return [self.overview_sectors, self.sectors]
        elif self.index == "US_HY":
            return [None, self.sectors, None, self.sectors]

    @property
    def kwarg_updates(self):
        return {
            "US_IG": {
                "source": "bloomberg",
                "in_stats_index": True,
                "OAS": (0, 3000),
            },
            "US_IG_10+": {
                "source": "bloomberg",
                "in_stats_index": True,
                "maturity": (10, None),
                "OAS": (0, 3000),
            },
            "US_HY": {
                "source": "baml",
                "in_H4UN_index": True,
                "OAS": (0, None),
            },
        }[self.index]

    @property
    def return_type(self):
        return {"US_IG": "XSRet", "US_IG_10+": "XSRet", "US_HY": "OAS"}[
            self.index
        ]

    @property
    def umbrella_vshift(self):
        return {"US_IG": 1, "US_IG_10+": 1, "US_HY": -1}[self.index]

    @property
    def margin(self):
        if self.index in {"US_IG", "US_IG_10+"}:
            return {
                "left": 0.5,
                "right": 0.5,
                "top": 0.5,
                "bottom": 0.2,
                "paperheight": 29,
            }
        elif self.index == "US_HY":
            return {
                "left": 0.5,
                "right": 0.5,
                "top": 0.5,
                "bottom": 0.2,
                "paperheight": 32,
            }


def build_credit_snapshot(
    index, date=None, pdf=True, include_portfolio_positions=True
):
    """
    Create snapshot for respective indexg.

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
    config = SnapshotConfig(index)

    # Define filename and directories to save data.
    db = Database()
    today = db.date("today") if date is None else pd.to_datetime(date)

    fid = f"{today.strftime('%Y-%m-%d')}_{config.fid}_Snapshot"
    csv_path = root("data/credit_snapshots")
    pdf_path = "reports/credit_snapshots"
    mkdir(csv_path)

    # Find dates for daily, week, month, and year to date calculations.
    # Store dates not to be used in table with a `~` before them.
    dates_dict = {
        "~today": today,
        "Daily": db.date("yesterday", today),
        "WTD": db.date("WTD", today),
        "MTD": db.date("MTD", today),
        # "QTD": db.nearest_date(pd.to_datetime("6/29/2019")),
        "YTD": db.date("YTD", today),
        f"~{config.horizon}m": db.date(f"{config.horizon}m", today),
    }

    # Create indexes for all sectors.
    dates_dict["~start"] = min(dates_dict.values())
    db.load_market_data(start=dates_dict["~start"], end=today)
    all_sectors = config.overview_sectors + config.sectors
    ix_dict = {}
    for sector in all_sectors:
        kwargs = db.index_kwargs(sector.strip("^~"), **config.kwarg_updates)
        if sector in {"LDB1_BBB", "HUC3_CCC"}:
            kwargs.pop("in_H4UN_index")
        ix_dict[sector] = db.build_market_index(**kwargs)

    # Get current market value of the entire long credit index.
    bm_name = all_sectors[0]
    bm_ix_today = ix_dict[bm_name].subset(date=today)
    bm_mv = bm_ix_today.total_value()[0]
    bm_rets = ix_dict[bm_name].market_value_weight(config.return_type)

    # Get current state of portfolio.
    if include_portfolio_positions:
        rep_account = db.load_portfolio(account=config.rep_account, date=today)
    else:
        rep_account = None

    pool = mp.Pool(processes=mp.cpu_count() - 2)
    results = [
        pool.apply_async(
            get_snapshot_values,
            args=(
                ix_dict[sector],
                dates_dict,
                bm_mv,
                bm_rets,
                rep_account,
                True,  # perform historical index corrections
                config,
            ),
        )
        for sector in all_sectors
    ]
    rows = [p.get() for p in results]
    table_df = pd.concat(rows, sort=False)
    table_df.to_csv(csv_path / f"{fid}.csv")
    table_df = pd.read_csv(csv_path / f"{fid}.csv", index_col=0)

    if not pdf:
        return

    if config.index == "US_HY":
        results = [
            pool.apply_async(
                get_snapshot_values,
                args=(
                    ix_dict[sector],
                    dates_dict,
                    bm_mv,
                    bm_rets,
                    rep_account,
                    False,  # Don't perform historical index corrections
                    config,
                ),
            )
            for sector in all_sectors
        ]
        rows = [p.get() for p in results]
        no_historical_correction_table_df = pd.concat(rows, sort=False)
    # Make table captions.
    dates_fmt = [f"Close: {dates_dict['~today'].strftime('%m/%d/%Y')}"]
    dates_fmt += [
        f"{k}: $\Delta${dates_dict[k].strftime('%m/%d')}"
        for k in "Daily WTD MTD".split()
        # for k in "Daily MTD QTD".split()
    ]
    dates_key = ", \hspace{0.2cm} ".join(dates_fmt)
    colors_key = (
        "Color Key: 2-Z move \color{blue}tighter \color{black} "
        "(\color{red}wider\color{black}) relative to the Index"
    )

    # Create daily snapshot tables and save to pdf.
    doc = Document(fid, path=pdf_path)
    doc.add_preamble(
        margin=config.margin,
        ignore_bottom_margin=True,
    )
    doc.add_background_image(
        "umbrella", scale=1.1, vshift=config.umbrella_vshift, alpha=0.04
    )

    if config.index in {"US_IG", "US_IG_10+"}:
        table_dfs = [table_df, table_df]
    elif config.index == "US_HY":
        table_dfs = [
            get_cross_asset_table(db, dates_dict),
            table_df,
            get_h4un_gainers_losers_table(db, dates_dict, n=10),
            no_historical_correction_table_df,
        ]

    # doc.add_background_image("xmas_tree", scale=1.1, vshift=2, alpha=0.08)
    for table_type, df, sector_subset, cap, footnote in zip(
        config.table_types,
        table_dfs,
        config.table_sectors,
        config.captions(dates_key, colors_key),
        config.footnote,
    ):
        if table_type == "sector_not_corrected":
            doc.add_pagebreak()
        doc.add_table(
            make_latex_table(
                table_type,
                df,
                sector_subset,
                cap,
                footnote,
                db._index_kwargs_dict(config.kwarg_updates["source"]),
                include_portfolio_positions,
                config,
            )
        )
    doc.save(save_tex=False)
    return doc.fid


def make_latex_table(
    table_type,
    table_df,
    sectors,
    caption,
    footnote,
    ix_kwargs,
    include_overweights,
    config,
):
    if config.index in {"US_IG", "US_IG_10+"}:
        return make_ig_table(
            table_df, sectors, caption, ix_kwargs, include_overweights, config
        )
    elif config.index == "US_HY":
        if table_type == "cross-asset":
            return make_cross_asset_table(table_df, caption, footnote)
        elif table_type.startswith("sector"):
            return make_hy_sector_table(
                table_df,
                sectors,
                caption,
                footnote,
                ix_kwargs,
                include_overweights,
                config,
            )
        elif table_type == "gainers_losers":
            return make_hy_gainers_loser_table(table_df, caption)


def make_ig_table(
    full_df, sectors, caption, ix_kwargs, include_overweights, config
):
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
    ow_col = [col for col in df.columns if "*OW" in col][0]
    if not include_overweights:
        df.drop(ow_col, axis=1, inplace=True)

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
        f"{config.horizon}m*Min": "0f",
        f"{config.horizon}m*Max": "0f",
        f"{config.horizon}m*%tile": "0f",
    }
    col_fmt = "l|cc|cc|cc|cc|cc|c|c|c|ccl"
    if include_overweights:
        col_fmt = f"{col_fmt}|c"
        prec[ow_col] = "2f"

    table = latex_table(
        df,
        caption=caption,
        font_size="footnotesize",
        col_fmt=col_fmt,
        prec=prec,
        midrule_locs=config.midrule_locs,
        specialrule_locs=major_sectors,
        row_color={"header": "darkgray", tuple(major_sectors): "lightgray"},
        row_font={"header": "\color{white}\\bfseries"},
        col_style={f"{config.horizon}m*%tile": "\pctbar"},
        loc_style=color_locs,
        multi_row_header=True,
        adjust=True,
        greeks=False,
    )

    return table


def make_hy_sector_table(
    full_df, sectors, caption, footnote, ix_kwargs, include_overweights, config
):
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
    ow_col = [col for col in df.columns if "*OW" in col][0]
    if not include_overweights:
        df.drop(ow_col, axis=1, inplace=True)

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
        "Autos",
    ]

    # Format market cap by precision by value for 2 sig figs.
    df["Mkt*Cap"] = [
        f"{v:.1%}" if v < 0.1 else f"{v:.0%}" for v in df["Mkt*Cap"]
    ]
    # Format remaining columns.
    prec = {
        "Close*OAS": "0f",
        "Close*YTW": "2%",
        "Daily*OAS": "1f",
        "Daily*YTW": "2%",
        "WTD*OAS": "1f",
        "WTD*YTW": "2%",
        "MTD*OAS": "0f",
        "MTD*YTW": "2%",
        # "QTD*OAS": "0f",
        # "QTD*YTW": "2%",
        "YTD*OAS": "0f",
        "YTD*YTW": "2%",
        "Mod Dur*to Worst": "1f",
        "MTD*TRet": "2%",
        "YTD*TRet": "2%",
        f"{config.horizon}m*Min": "0f",
        f"{config.horizon}m*Med": "0f",
        f"{config.horizon}m*Max": "0f",
        f"{config.horizon}m*%tile": "0f",
    }
    col_fmt = "l|cc|cc|cc|cc|cc|c|cc|c|ccc|l"
    if include_overweights:
        col_fmt = f"{col_fmt}|c"
        prec[ow_col] = "1%"

    table = latex_table(
        df,
        caption=caption,
        table_notes=footnote,
        font_size="footnotesize",
        col_fmt=col_fmt,
        prec=prec,
        midrule_locs=config.midrule_locs,
        specialrule_locs=major_sectors,
        row_font={"header": "\\bfseries"},
        col_style={f"{config.horizon}m*%tile": "\pctbar"},
        loc_style=color_locs,
        multi_row_header=True,
        adjust=True,
        greeks=False,
    )

    return table


def make_hy_gainers_loser_table(table_df, cap):
    bar_cols = ["MTD Return", " MTD Return", " MTD Return ", "MTD  Return"]
    table = latex_table(
        table_df,
        caption=cap,
        col_fmt="r|lr|lr|lr|lr",
        adjust=True,
        row_font={"header": "\\bfseries"},
        prec={col: "1%" for col in bar_cols},
        div_bar_col=bar_cols,
        div_bar_kws={"cmin": "firebrick", "cmax": "steelblue"},
        multi_row_header=True,
    )
    return table


def make_cross_asset_table(table_df, cap, footnote):
    prec = {
        "Daily*%tile": "1%",
        "WTD*%tile": "1%",
        "MTD*%tile": "0%",
        "YTD*%tile": "0%",
        f"%tile": "0f",
    }
    # Custom set precision for columns with varrying precision by row.
    formatted_table_df = table_df.copy()
    cols = ["Close", "Daily", "WTD", "MTD", "YTD", "Min", "Median", "Max"]
    change_cols = ["Daily", "WTD", "MTD", "YTD"]
    d = defaultdict(list)
    for asset, row in table_df.iterrows():
        for col in cols:
            if row["asset_class"] == "credit":
                d[col].append(f"{row[col]:.0f}")
            elif row["asset_class"] == "rates":
                d[col].append(f"{row[col]:.2%}")
            elif row["asset_class"] == "equities":
                if col in change_cols:
                    d[col].append(f"{row[col]:.1%}")
                else:
                    d[col].append(f"{row[col]:,.0f}")
            elif row["asset_class"] == "vix":
                d[col].append(f"{row[col]:,.1f}")

    for col in cols:
        formatted_table_df[col] = d[col]

    col_fmt = "l|c|cc|cc|cc|cc|ccc|l"
    midrule_locs = ["UST 10Y", "S\\&P 500", "Vix"]
    table = latex_table(
        formatted_table_df.drop("asset_class", axis=1),
        caption=cap,
        table_notes=footnote,
        col_fmt=col_fmt,
        midrule_locs=midrule_locs,
        prec=prec,
        adjust=True,
        col_style={f"%tile": "\pctbar"},
        row_font={"header": "\\bfseries"},
        multi_row_header=True,
    )
    return table


def get_snapshot_values(
    ix, dates, bm_mv, bm_rets, portfolio, corrected, config
):
    if config.index in {"US_IG", "US_IG_10+"}:
        return get_ig_snapshot_values(
            ix, dates, bm_mv, bm_rets, portfolio, config
        )
    elif config.index == "US_HY":
        return get_hy_snapshot_values(
            ix, dates, bm_mv, bm_rets, portfolio, corrected, config
        )


def get_ig_snapshot_values(ix, dates, bm_mv, bm_rets, portfolio, config):
    """
    Get single row from IG snapshot table for
    specified sector.

    Parameters
    ----------
    ix: :class:`Index`
        Index of the sector for current row.
    dates: Dict[str: datetime].
        Dict of pertinent dates for constructing row.
    bm_mv: float
        Current total IG Long Credit Index market value.
    bm_rets: pd.Series
        Excess or total return series for the benchmark index.
    portfolio: :class:`Index`
        Current state of representative account portfolio
        as an :class:`Index`.

    Returns
    -------
    pd.DataFrame:
        DataFrame row of IG snapshot table for specified sector.
    """
    # Get synthetic OAS and price histories.
    ow_col = "-*OW" if portfolio is None else f"{portfolio.name}*OW"
    try:
        ix_corrected = ix.drop_ratings_migrations()
        oas = ix_corrected.get_synthetic_differenced_history("OAS")
        price = ix_corrected.get_synthetic_differenced_history("DirtyPrice")
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
        d[f"{config.horizon}m*Min"] = np.nan
        d[f"{config.horizon}m*Max"] = np.nan
        d[f"{config.horizon}m*%tile"] = 0
        d["z"] = None
        d[ow_col] = np.nan
        return pd.DataFrame(d, index=[ix.name])

    # Get current state of the index and month OAS/XSRet history.
    ix_today = ix.subset(date=dates["~today"])
    oas_horizon = oas[oas.index > dates[f"~{config.horizon}m"]]
    sector_rets = ix.market_value_weight(config.return_type)
    xsrets = sector_rets - bm_rets
    xsrets = xsrets[xsrets.index >= dates[f"~{config.horizon}m"]]
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
    d["Mkt*Cap"] = ix_today.total_value()[0] / bm_mv
    d[f"{config.horizon}m*Min"] = np.min(oas_horizon)
    d[f"{config.horizon}m*Max"] = np.max(oas_horizon)
    d[f"{config.horizon}m*%tile"] = 100 * oas_horizon.rank(pct=True)[-1]
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
        d[ow_col] = np.sum(sector_portfolio.df["OAD_Diff"])

    else:
        d[ow_col] = np.nan

    return pd.DataFrame(d, index=[ix.name])


def get_hy_snapshot_values(
    ix, dates, bm_mv, bm_rets, portfolio, corrected, config
):
    """
    Get single row for HY snapshot table for specified sector.

    Parameters
    ----------
    ix: :class:`Index`
        Index of the sector for current row.
    dates: Dict[str: datetime].
        Dict of pertinent dates for constructing row.
    bm_mv: float
        Current total IG Long Credit Index market value.
    bm_rets: pd.Series
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
    ow_col = "-*OW" if portfolio is None else f"{portfolio.name}*OW"
    if corrected:
        try:
            allowable_ratings = ix.constraints["rating"]
        except KeyError:
            allowable_ratings = ("BB+", "B-")
        try:
            ix_corrected = ix.drop_ratings_migrations(allowable_ratings)
            oas = ix_corrected.get_synthetic_differenced_history("OAS")
            ytw = (
                ix_corrected.get_synthetic_differenced_history("YieldToWorst")
                / 100
            )
        except UnboundLocalError:
            # DataFrame is emtpy.
            d = OrderedDict([("Close*OAS", np.nan), ("Close*Price", np.nan)])
            for col_name, date in dates.items():
                if col_name.startswith("~"):
                    continue
                d[f"{col_name}*OAS"] = np.nan
                d[f"{col_name}*Price"] = np.nan
            d["Mod Dur*to Worst"] = np.nan
            d["MTD*TRet"] = np.nan
            d["YTD*TRet"] = np.nan
            d["Mkt*Cap"] = 0
            d[f"{config.horizon}m*Min"] = np.nan
            d[f"{config.horizon}m*Med"] = np.nan
            d[f"{config.horizon}m*Max"] = np.nan
            d[f"{config.horizon}m*%tile"] = 0
            d["z"] = None
            d[ow_col] = np.nan
            return pd.DataFrame(d, index=[ix.name])
    else:
        oas = ix.OAS()
        ytw = ix.market_value_weight("YieldToWorst") / 100

    # Get current state of the index and 6 month OAS/XSRet history.
    ix_today = ix.subset(date=dates["~today"])
    oas_horizon = oas[oas.index > dates[f"~{config.horizon}m"]]
    rel_oas = -(oas.diff() - bm_rets.diff())
    rel_oas = rel_oas[rel_oas.index > dates[f"~{config.horizon}m"]]
    rel_z = (rel_oas - np.mean(rel_oas)) / np.std(rel_oas)
    z = rel_z.iloc[-1]
    z = 0 if np.isnan(z) else z

    # Builid row of the table.
    d = OrderedDict([("Close*OAS", oas[-1]), ("Close*YTW", ytw[-1])])
    for col_name, date in dates.items():
        if col_name.startswith("~"):
            continue
        try:
            d[f"{col_name}*OAS"] = oas.iloc[-1] - oas[date]
            d[f"{col_name}*YTW"] = ytw.iloc[-1] - ytw[date]
        except KeyError:
            d[f"{col_name}*OAS"] = np.nan
            d[f"{col_name}*YTW"] = np.nan

    d["Mod Dur*to Worst"] = ix_today.market_value_weight("ModDurtoWorst").iloc[
        0
    ]
    d["MTD*TRet"] = ix.aggregate_total_returns(start_date=dates["MTD"])
    d["YTD*TRet"] = ix.aggregate_total_returns(start_date=dates["YTD"])
    d["Mkt*Cap"] = ix_today.total_value().iloc[0] / bm_mv
    d[f"{config.horizon}m*Min"] = np.min(oas_horizon)
    d[f"{config.horizon}m*Med"] = np.median(oas_horizon)
    d[f"{config.horizon}m*Max"] = np.max(oas_horizon)
    d[f"{config.horizon}m*%tile"] = 100 * oas_horizon.rank(pct=True)[-1]
    d["z"] = z
    d["color"] = "blue" if z > 2 else "red" if z < -2 else None

    # Determine current portfolio overweight.
    if portfolio is not None:
        # Find all CUSIPs that fit current sector, whether they
        # are index eligible or not.
        unused_constraints = {
            "in_stats_index",
            "in_H4UN_index",
            "in_HUC3_index",
            "name",
        }
        constraints = {
            k: v
            for k, v in ix.constraints.items()
            if k not in unused_constraints
        }
        sector_portfolio = portfolio.subset(**constraints)
        d[ow_col] = np.sum(sector_portfolio.df["Weight_Diff"])

    else:
        d[ow_col] = np.nan

    return pd.DataFrame(d, index=[ix.name])


def get_h4un_gainers_losers_table(db, dates_d, n=10):
    """pd.DataFrame: top and worst ``n`` bonds and issuers in H4UN MTD."""
    ix = db.build_market_index(in_H4UN_index=True, start=dates_d["MTD"])
    cusip_trets = ix.accumulate_individual_total_returns().sort_values()

    issuer_ix = ix.issuer_index()
    issuer_trets = issuer_ix.accumulate_individual_total_returns().sort_values()

    df = pd.DataFrame()
    map = ix.cusip_to_bond_name_map()
    df["Top*Bonds"] = pd.Series(cusip_trets[::-1][:10].index).map(map)
    df["MTD Return"] = cusip_trets[::-1][:10].values
    df["Worst*Bonds"] = pd.Series(cusip_trets[:10].index).map(map)
    df[" MTD Return"] = cusip_trets[:10].values
    df["Top*Issuers"] = issuer_trets[::-1][:10].index
    df[" MTD Return "] = issuer_trets[::-1][:10].values
    df["Worst*Issuers"] = issuer_trets[:10].index
    df["MTD  Return"] = issuer_trets[:10].values
    df.index += 1
    return df


def get_cross_asset_table(db, dates_d):
    """pd.DataFrame: Cross asset table of values, ranges, and percentiles"""
    today = dates_d["~today"]
    security_d = {
        "credit": (["US_BBB", "US_BB", "US_B"], "OAS", "5y"),
        "rates": (["UST_10Y", "UST_30Y", "LIBOR"], "YTW", "5y"),
        "equities": (
            ["SP500", "DOWJONES", "RUSSELL_2000"],
            "PRICE",
            "1y",
        ),
        "vix": (["VIX"], "PRICE", "1y"),
    }
    d = defaultdict(list)
    for asset_class, (securities, field, horizon) in security_d.items():
        df = db.load_bbg_data(
            securities,
            field,
            start=db.date(horizon, reference_date=today),
            end=today,
            nan="ffill",
        )
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if asset_class == "credit":
            df["BBB-BB"] = df["US_BBB"] - df["US_BB"]
            df["BB-B"] = df["US_BB"] - df["US_B"]
            df = df[["BBB-BB", "BB-B"]]
        elif asset_class == "vix":
            df.columns = [db.bbg_names(df.columns)]
        else:
            df.columns = db.bbg_names(df.columns)
        for col in df.columns:
            s = df[col].dropna()
            pct_s = s.rank(pct=True)
            today_val = s.iloc[-1]
            today_pct = pct_s.iloc[-1]
            d["name"].append(col)
            d["asset_class"].append(asset_class)
            d["Close"].append(today_val)
            for label, date in dates_d.items():
                if label.startswith("~"):
                    continue
                if asset_class == "equities":
                    # Percent changes.
                    d[label].append((today_val / s.loc[date]) - 1)
                else:
                    # Absolute changes.
                    d[label].append(today_val - s.loc[date])
                d[f"{label}*%tile"].append(today_pct - pct_s.loc[date])

            d["Min"].append(np.min(s))
            d["Median"].append(np.median(s))
            d["Max"].append(np.max(s))
            d["%tile"].append(100 * pct_s.iloc[-1])
    df = pd.DataFrame(d).set_index("name", drop=True).rename_axis(None)
    return df


# %%

if __name__ == "__main__":
    date = "8/31/2020"
    date = Database().date("today")
    with Time():
        make_credit_snapshots(date)
        update_credit_snapshots()
