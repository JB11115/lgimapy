import warnings

import joblib
import numpy as np
import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm

import lgimapy.vis as vis
from lgimapy.data import Database, update_portfolio_history
from lgimapy.latex import Document, Page, merge_pdfs
from lgimapy.portfolios import AttributionIndex
from lgimapy.utils import load_json, root, replace_multiple, Time

# %%
def main():
    # %%
    vis.style()
    debug = False
    db = Database()
    db.update_portfolio_account_data()
    date = db.date("today")
    # date = db.date("MONTH_START")
    prev_date = db.date("1w")
    # prev_date = pd.to_datetime("4/1/2021")

    dated_fid = f"{date.strftime('%Y-%m-%d')}_Risk_Report"
    # dated_fid = f"{date.strftime('%Y-%m-%d')}_Risk_Report_Q3_2021"

    ignored_accounts = set(["SESIBNM", "SEUIBNM"])
    pdf_path = root("reports/strategy_risk")

    strategy = "US Corporate IG"
    strategy = "US Long Government/Credit"
    strategy = "US Long Corporate"
    strategy = "US Long Credit Plus"
    # strategy = "US Credit"
    # strategy = "Bloomberg LDI Custom - DE"
    # strategy = "US Long Credit"

    n_table_rows = 10

    strat_acnt = load_json("strategy_accounts")
    strat_df = pd.Series(
        {k: len(v) for k, v in strat_acnt.items()}
    ).sort_values(ascending=False)

    strategies = [
        "US Long Credit",
        "US Long Credit - Custom",
        "US Long Credit Plus",
        "US Long Corporate",
        "US Long Corp 2% Cap",
        "US Long Credit Ex Emerging Market",
        "US Corporate 1% Issuer Cap",
        "Global Agg USD Corp",
        "Custom RBS",
        "GM_Blend",
        "US Corporate IG",
        "US Intermediate Credit",
        "US Intermediate Credit A or better",
        "US Long GC 70/30",
        "US Long GC 75/25",
        "US Long GC 80/20",
        "US Long Government/Credit",
        "Liability Aware Long Duration Credit",
        "US Credit",
        "US Credit A or better",
        "US Credit Plus",
        "US Long A+ Credit",
        "80% US A or Better LC/20% US BBB LC",
        "Bloomberg LDI Custom - DE",
        "INKA",
        "US Long Corporate A or better",
    ]

    # %%

    res = []
    if debug:
        for strategy in tqdm(strategies):
            res.append(
                save_single_latex_risk_page(
                    strategy,
                    date,
                    prev_date,
                    ignored_accounts,
                    pdf_path,
                    n_table_rows,
                    debug,
                )
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = joblib.Parallel(n_jobs=6)(
                joblib.delayed(save_single_latex_risk_page)(
                    strategy,
                    date,
                    prev_date,
                    ignored_accounts,
                    pdf_path,
                    n_table_rows,
                    debug,
                )
                for strategy in strategies
            )

    # %%

    df = (
        pd.DataFrame(res, columns=["strategy", "summary", "fid"])
        .set_index("strategy")
        .reindex(strategies)
        .dropna(subset=["fid"])
    )

    summary_page_fid = "summary"
    build_summary_page(df, date, summary_page_fid, pdf_path)
    attribution_page_fid = "attribution"
    build_attribution_page("P-LD", attribution_page_fid, pdf_path)
    methodology_page_fid = "risk_report_methodology"
    build_methodology_page(methodology_page_fid, pdf_path)

    # %%
    # Merge all files together to build to final report.
    fids_to_merge = [summary_page_fid, attribution_page_fid]
    for fid in df["fid"]:
        fids_to_merge.append(fid)
        fids_to_merge.append(f"{fid}_history")
    fids_to_merge.append(methodology_page_fid)

    merge_pdfs(
        "Risk_Report",
        fids_to_merge,
        read_path=pdf_path,
        write_path="reports/current_reports",
        keep_bookmarks=True,
    )
    print("Risk Report Complete")

    merge_pdfs(dated_fid, fids_to_merge, path=pdf_path, keep_bookmarks=True)

    # Update history for every portfolio.
    update_portfolio_history()
    # %%


# %%
def build_summary_page(df, date, fid, pdf_path):
    """Make summary page table and save the pdf."""
    # Get summary table from parallel result.
    summary = pd.concat(df["summary"].values, axis=1, sort=False).T
    summary.index = [ix.replace("_", " ") for ix in summary.index]
    summary.index.rename("Strategy", inplace=True)
    # Remove any unnecessary columns.
    if len(summary.columns) > 15:
        summary = summary[list(summary)[:15]]

    # Create LaTeX Document with proper preabmle.
    summary_page = Document(fid, path=pdf_path)
    summary_page.add_preamble(
        bookmarks=True,
        orientation="landscape",
        bar_size=7,
        margin={
            "left": 0.5,
            "right": 0.5,
            "top": 0.3,
            "bottom": 0.2,
        },
        header=summary_page.header(
            left="Strategy Risk Report",
            right=f"EOD {date.strftime('%B %#d, %Y')}",
        ),
        footer=summary_page.footer(logo="LG_umbrella"),
    )
    summary_page.add_section("Summary")

    # Find midrule locations.
    saved_strategies = list(summary.index)
    midrule_strats = [
        "US Long Corporate",
        "US Corporate IG",
        "Liability Aware Long Duration Credit",
        "80% US A or Better LC/20% US BBB LC",
        "Barclays-Russell LDI Custom - DE",
    ]
    strategy_midrules = [
        saved_strategies.index(strat)
        for strat in midrule_strats
        if strat in saved_strategies
    ]

    # Get column precisions.
    prec = {}
    for col in summary.columns:
        if "OAD" in col:
            prec[col] = "2f"
        elif "Performance" in col:
            prec[col] = "+1f"
        elif "Tracking" in col:
            prec[col] = "0f"
        else:
            prec[col] = "2%"

    # Format table and save.
    summary_page.add_table(
        summary.reset_index(),
        col_fmt="ll|c|c|cc|c|r|cccc|ccccc",
        prec=prec,
        multi_row_header=True,
        adjust=True,
        hide_index=True,
        midrule_locs=strategy_midrules,
    )
    summary_page.save()


def build_attribution_page(account, fid, pdf_path):
    db = Database()
    attr = AttributionIndex(account, start=db.date("YTD"))
    attribution_page = Document(fid, path=pdf_path)
    attribution_page.add_preamble(
        bookmarks=True,
        orientation="landscape",
        bar_size=7,
        margin={
            "left": 0.5,
            "right": 0.5,
            "top": 0.3,
            "bottom": 0.2,
        },
        header=attribution_page.header(
            left="Strategy Risk Report",
            right=f"EOD {db.date('today').strftime('%B %#d, %Y')}",
        ),
        footer=attribution_page.footer(logo="LG_umbrella"),
        table_caption_justification="c",
    )
    attribution_page.add_section(f"Attribution ({account})")
    dates_d = {
        "today": {
            "name": f"Daily Attribution (for {db.date('today'):%b %-d})",
            "prec": 2,
        },
        "1M": {
            "name": f"1M Attribution (since {db.date('1M'):%b %-d})",
            "prec": 1,
        },
        "YEAR_START": {
            "name": f"YTD Attribution (since {db.date('YEAR_START'):%b %-d})",
            "prec": 1,
        },
    }
    table_widths = {"Ticker": 0.22, "Sector": 0.3, "Market Segment": 0.45}
    fontsize = "tiny"
    for date, d in dates_d.items():
        attribution_page.add_subsection(d["name"], bookmark=False)
        edits = attribution_page.add_minipages(
            n=len(table_widths), valign="t", widths=table_widths.values()
        )
        PnL_cols = ["PnL", "PnL "]
        col_prec = {col: f"+{d['prec']}f" for col in PnL_cols}
        with attribution_page.start_edit(edits[0]):
            table = attr.best_worst_df(
                attr.tickers(start=db.date(date)), prec=d["prec"]
            )
            attribution_page.add_table(
                table,
                caption="Tickers",
                font_size=fontsize,
                col_fmt="l|cr|cr",
                prec=col_prec,
            )
        with attribution_page.start_edit(edits[1]):
            table = attr.best_worst_df(
                attr.sectors(start=db.date(date)), prec=d["prec"]
            )
            attribution_page.add_table(
                table,
                caption="Sectors",
                font_size=fontsize,
                col_fmt="l|cr|cr",
                prec=col_prec,
            )
        with attribution_page.start_edit(edits[2]):
            table = attr.best_worst_df(
                attr.market_segments(start=db.date(date)), prec=d["prec"]
            )
            attribution_page.add_table(
                table,
                caption="Market Segments",
                font_size=fontsize,
                col_fmt="l|cr|cr",
                prec=col_prec,
            )
    attribution_page.save()


def build_methodology_page(fid, pdf_path):
    methodology_page = Document(fid, path=pdf_path, load_tex=fid)
    methodology_page.save()


def col_prec(df, ow_metric):
    prec = {}
    for col in df.columns:
        if ow_metric in col:
            if "Delta" in col:
                prec[col] = "3f"
            else:
                prec[col] = "2f"
    return prec


def _load_portfolio(strategy, date, universe, ignored_accounts):
    db = Database()
    port = db.load_portfolio(
        strategy=strategy,
        date=date,
        universe=universe,
        ignored_accounts=ignored_accounts,
    )
    port = port.drop_empty_accounts()
    port.expand_tickers()
    return port


def _create_account_dispersion_table(curr_strat):
    n_accounts = len(curr_strat.accounts)
    pivot = curr_strat.curve_pivot_point

    dispersion_metrics = {
        "DTS*(%)": curr_strat.account_dts(),
        "Barbell*(%)": curr_strat.account_barbell(),
        "Tracking*Error": curr_strat.account_tracking_error(),
        f"Curve ({pivot}yr)*Duration": curr_strat.account_curve_duration(pivot),
        "Tsy*OAD": curr_strat.account_tsy_oad(),
        "Cash*(%)": curr_strat.account_cash_pct(),
    }
    for col, vals in dispersion_metrics.items():
        dispersion_metrics[col] = vals.sort_values(ascending=False)

    if n_accounts > 10:
        # Limit to top 5 and bottom 5 accounts with midrule between.
        n_accounts = 10
        account_dispersion_midrules = "5"
        for col, vals in dispersion_metrics.items():
            dispersion_metrics[col] = pd.concat([vals.head(), vals.tail()])
    else:
        account_dispersion_midrules = None

    df = pd.DataFrame(index=range(n_accounts))
    for i, (col, vals) in enumerate(dispersion_metrics.items()):
        df["Account".center(7 + i)] = vals.index
        df[col] = vals.values

    return df, account_dispersion_midrules


def load_portfolios(strategy, date, prev_date, ignored_accounts):
    db = Database()
    curr_universe = prev_universe = "returns"
    if dt.today().month != date.month:
        # First day of the month. Use stats index.
        curr_universe = prev_universe = "stats"
    elif date.month != prev_date.month:
        # First week of the month, but not the first day. Use
        # current returns index but previous stats index for
        # fair comparison.
        prev_universe = "stats"

    try:
        curr_strat = _load_portfolio(
            strategy, date, curr_universe, ignored_accounts
        )
    except ValueError:
        # No data for strategy.
        return None, None

    prev_strat = None
    while prev_strat is None:
        try:
            prev_strat = _load_portfolio(
                strategy, prev_date, prev_universe, ignored_accounts
            )
        except ValueError:
            # No data on attempted previous date.
            if prev_date == date:
                # New strategy.
                prev_strat = current_strat
            else:
                # Increment one day forward
                prev_date = db.trade_dates(exclusive_start=prev_date)[0]

    return curr_strat, prev_strat


def save_single_latex_risk_page(
    strategy,
    date,
    prev_date,
    ignored_accounts,
    pdf_path,
    n_table_rows=10,
    debug=False,
):
    if debug:
        print(strategy, "-loading")

    curr_strat, prev_strat = load_portfolios(
        strategy, date, prev_date, ignored_accounts
    )

    if debug:
        print(strategy, "-starting")

    if curr_strat is None or not len(curr_strat.df):
        print(f"{strategy} is empty, skipping")
        return strategy, None, None

    date_fmt = date.strftime("%#m/%#d")
    prev_date_fmt = prev_date.strftime("%#m/%#d")

    # Build overview table.
    properties = [
        "dts_pct",
        "dts_abs",
        "dts_duration",
        "credit_pct",
        "curve_duration(5)",
        "curve_duration(7)",
        "curve_duration(10)",
        "curve_duration(20)",
    ]
    general_overview_midrules = ["Credit (\\%)", "Curve Dur (5yr)"]
    ow_metric = "OAD"
    if curr_strat.is_plus_strategy():
        ow_metric = "OASD"
        properties = [
            "dts_pct",
            "ig_dts_pct",
            "hy_dts_pct",
            "derivatives_dts_pct",
            "dts_abs",
            "ig_dts_abs",
            "hy_dts_abs",
            "derivatives_dts_abs",
            "credit_pct",
            "ig_mv_pct",
            "hy_mv_pct",
            "derivatives_mv_pct",
            "curve_duration(5)",
            "curve_duration(7)",
            "curve_duration(10)",
            "curve_duration(20)",
        ]
        general_overview_midrules = [
            "Credit (\\%)",
            "DTS OW (abs)",
            "Curve Dur (5yr)",
        ]

    overview_table = pd.concat(
        [
            curr_strat.properties(properties).rename(date_fmt),
            np.std(curr_strat.account_properties(properties)).rename("Std Dev"),
            prev_strat.properties(properties).rename(prev_date_fmt),
        ],
        axis=1,
        sort=False,
    )
    # Drop derivatives from table if there are none in the portfolio.
    overview_table = overview_table[
        (~overview_table.index.str.contains("Derivatives"))
        | (overview_table.abs().sum(axis=1) > 0)
    ]

    prop_formats = curr_strat.property_latex_formats(properties)
    for prop, fmt in zip(overview_table.index, prop_formats):
        overview_table.loc[prop] = [
            "{:.{}}".format(v, fmt) for v in overview_table.loc[prop]
        ]

    # Build rating overweight table.

    rating_ow_df = pd.concat(
        [
            curr_strat.rating_overweights(by="OAD"),
            curr_strat.rating_overweights(by="OASD"),
        ],
        axis=1,
        sort=False,
    )
    top_level_sector_ow_df = pd.concat(
        [
            curr_strat.top_level_sector_overweights(by="OAD"),
            curr_strat.top_level_sector_overweights(by="OASD"),
        ],
        axis=1,
        sort=False,
    )
    gen_ow_table = (
        pd.concat([rating_ow_df, top_level_sector_ow_df])
        .replace(0, np.nan)
        .dropna(how="all")
    )

    sub_cats = {
        "AAA",
        "AA",
        "A",
        "BBB",
        "BB",
        "B",
        "Industrials",
        "Financials",
        "Utilities",
    }
    gen_ow_table.index = [
        f"~{ix}" if ix in sub_cats else ix for ix in gen_ow_table.index
    ]

    # Build account dispersion table.
    (
        account_dispersion_table,
        account_dispersion_midrules,
    ) = _create_account_dispersion_table(curr_strat)

    # Build bond change in overweight table.
    n = n_table_rows
    bond_ow_df = pd.concat(
        [
            curr_strat.bond_overweights(ow_metric).rename("curr"),
            prev_strat.bond_overweights(ow_metric).rename("prev"),
        ],
        axis=1,
        sort=False,
    ).fillna(0)
    bond_ow_df["diff"] = bond_ow_df["curr"] - bond_ow_df["prev"]

    def clean_index(df):
        return (
            pd.Series(df.index)
            .apply(lambda x: str(x).split("#")[0].replace("$", "\\$"))
            .values
        )

    bond_ow_table = pd.DataFrame(index=range(n))
    bond_ow_df.sort_values("diff", inplace=True, ascending=False)
    bond_ow_table["Risk*Added"] = clean_index(bond_ow_df)[:n]
    bond_ow_table[f"$\\Delta${ow_metric}*(yrs)"] = bond_ow_df["diff"].values[:n]
    bond_ow_df.sort_values("diff", inplace=True, ascending=True)
    bond_ow_table["Risk*Reduced"] = clean_index(bond_ow_df)[:n]
    bond_ow_table[f"$\\Delta${ow_metric}*(yrs) "] = bond_ow_df["diff"].values[
        :n
    ]

    # Build sector overweight table.
    sector_ow_df = pd.concat(
        [
            curr_strat.sector_overweights(ow_metric).rename("curr"),
            prev_strat.sector_overweights(ow_metric).rename("prev"),
        ],
        axis=1,
        sort=False,
    ).fillna(0)

    sector_ow_df["diff"] = sector_ow_df["curr"] - sector_ow_df["prev"]

    sector_ow_table = pd.DataFrame(index=range(n))
    sector_ow_df.sort_values("curr", inplace=True, ascending=False)
    sector_ow_table["Largest*OW"] = sector_ow_df.index[:n]
    sector_ow_table[f"{ow_metric}*(yrs)"] = sector_ow_df["curr"].values[:n]
    sector_ow_df.sort_values("curr", inplace=True, ascending=True)
    sector_ow_table["Largest*UW"] = sector_ow_df.index[:n]
    sector_ow_table[f"{ow_metric}*(yrs) "] = sector_ow_df["curr"].values[:n]
    sector_ow_df.sort_values("diff", inplace=True, ascending=False)
    sector_ow_table["Risk*Added"] = sector_ow_df.index[:n]
    sector_ow_table[f"$\\Delta${ow_metric}*(yrs)"] = sector_ow_df[
        "diff"
    ].values[:n]
    sector_ow_table[f"Current*{ow_metric}"] = sector_ow_df["curr"].values[:n]
    sector_ow_df.sort_values("diff", inplace=True, ascending=True)
    sector_ow_table["Risk*Reduced"] = sector_ow_df.index[:n]
    sector_ow_table[f"$\\Delta${ow_metric}*(yrs) "] = sector_ow_df[
        "diff"
    ].values[:n]
    sector_ow_table[f"Current * {ow_metric} "] = sector_ow_df["curr"].values[:n]

    # Build ticker overweight table.
    ticker_ow_df = pd.concat(
        [
            curr_strat.ticker_overweights(ow_metric).rename("curr"),
            prev_strat.ticker_overweights(ow_metric).rename("prev"),
        ],
        axis=1,
        sort=False,
    ).fillna(0)
    ticker_ow_df["diff"] = ticker_ow_df["curr"] - ticker_ow_df["prev"]
    ticker_ow_table = pd.DataFrame(index=range(n))
    ticker_ow_df.sort_values("curr", inplace=True, ascending=False)
    ticker_ow_table["Largest*OW"] = ticker_ow_df.index[:n]
    ticker_ow_table[f"{ow_metric}*(yrs)"] = ticker_ow_df["curr"].values[:n]
    ticker_ow_df.sort_values("curr", inplace=True, ascending=True)
    ticker_ow_table["Largest*UW"] = ticker_ow_df.index[:n]
    ticker_ow_table[f"{ow_metric}*(yrs) "] = ticker_ow_df["curr"].values[:n]
    ticker_ow_df.sort_values("diff", inplace=True, ascending=False)
    ticker_ow_table["Risk*Added"] = ticker_ow_df.index[:n]
    ticker_ow_table[f"$\\Delta${ow_metric}*(yrs)"] = ticker_ow_df[
        "diff"
    ].values[:n]
    ticker_ow_table[f"Current*{ow_metric}"] = ticker_ow_df["curr"].values[:n]
    ticker_ow_df.sort_values("diff", inplace=True, ascending=True)
    ticker_ow_table["Risk*Reduced"] = ticker_ow_df.index[:n]
    ticker_ow_table[f"$\\Delta${ow_metric}*(yrs) "] = ticker_ow_df[
        "diff"
    ].values[:n]
    ticker_ow_table[f"Current * {ow_metric} "] = ticker_ow_df["curr"].values[:n]

    # Build HY ticker overweight table if necessry.
    if curr_strat.is_plus_strategy():
        hy_ticker_ow_df = pd.concat(
            [
                curr_strat.HY_ticker_overweights(ow_metric).rename("curr"),
                prev_strat.HY_ticker_overweights(ow_metric).rename("prev"),
                curr_strat.HY_ticker_overweights("P_Weight"),
            ],
            axis=1,
            sort=False,
        ).fillna(0)
        hy_ticker_ow_df["diff"] = (
            hy_ticker_ow_df["curr"] - hy_ticker_ow_df["prev"]
        )

        n_hy = int(n / 2)
        hy_ticker_ow_table = pd.DataFrame(index=range(n_hy))
        hy_ticker_ow_df.sort_values("curr", inplace=True, ascending=False)
        hy_ticker_ow_table["Largest*OW"] = hy_ticker_ow_df.index[:n_hy]
        hy_ticker_ow_table[f"{ow_metric}*(yrs)"] = hy_ticker_ow_df[
            "curr"
        ].values[:n_hy]
        hy_ticker_ow_table["MV*(%)"] = hy_ticker_ow_df["P_Weight"].values[:n_hy]
        hy_ticker_ow_df.sort_values("diff", inplace=True, ascending=False)
        hy_ticker_ow_table["Risk*Added"] = hy_ticker_ow_df.index[:n_hy]
        hy_ticker_ow_table[f"$\\Delta${ow_metric}*(yrs)"] = hy_ticker_ow_df[
            "diff"
        ].values[:n_hy]
        hy_ticker_ow_df.sort_values("diff", inplace=True, ascending=True)
        hy_ticker_ow_table["Risk*Reduced"] = hy_ticker_ow_df.index[:n_hy]
        hy_ticker_ow_table[f"$\\Delta${ow_metric}*(yrs) "] = hy_ticker_ow_df[
            "diff"
        ].values[:n_hy]

    # Create summary series for cover page.
    tsy_weights = curr_strat.tsy_weights()
    summary = pd.Series(name=strategy, dtype="float64")
    summary[f"Performance*(Est. bp) {date_fmt}"] = curr_strat.performance()
    summary[f"Tracking*Error (bp)"] = curr_strat.tracking_error()
    summary[f"DTS (%)*{date_fmt}"] = curr_strat.dts()
    summary[f"$\\Delta$DTS*{prev_date_fmt}"] = (
        curr_strat.dts() - prev_strat.dts()
    )
    summary["Barbell*(%)"] = curr_strat.barbell()
    summary["OAD*Total"] = curr_strat.total_oad()
    summary["Cash*(%)"] = curr_strat.cash_pct()
    summary["Tsy OAD*Total"] = curr_strat.tsy_oad()
    summary[f"Tsy $\\Delta$OAD*{prev_date_fmt}"] = (
        curr_strat.tsy_oad() - prev_strat.tsy_oad()
    )
    summary["Tsy (%)*Total"] = curr_strat.tsy_pct()
    summary["Tsy (%)*$\\leq$ 5y"] = tsy_weights.loc[[2, 3, 5]].sum()
    summary["Tsy (%)*7y"] = tsy_weights.loc[7]
    summary["Tsy (%)*10y"] = tsy_weights.loc[10]
    summary["Tsy (%)*20y"] = tsy_weights.loc[20]
    summary["Tsy (%)*30y"] = tsy_weights.loc[30]

    # Get data for curve risk tables.
    bm_tsy_table = curr_strat.bm_tsy_bucket_table()
    rating_risk_table = curr_strat.rating_risk_bucket_table()
    maturity_bucket_table = curr_strat.maturity_bucket_table()
    maturity_spread_heatmap = 100 * curr_strat.maturity_spread_bucket_heatmap()

    # Update stored properties and get stored data for historical tables.
    curr_strat.save_stored_properties()
    history_df = curr_strat.stored_properties_history_df
    history_table = curr_strat.stored_properties_history_table
    historic_percentile_df = 100 * curr_strat.stored_properties_percentile_table
    historic_range_df = curr_strat.stored_properties_range_table

    # Make LaTeX page for strategy.
    page = Document(fid=curr_strat.fid, path=pdf_path)
    page.add_section(curr_strat.latex_name)

    page.add_preamble(
        bookmarks=True,
        bar_size=7,
        margin={
            "left": 0.5,
            "right": 0.5,
            "top": 0.3,
            "bottom": 0.2,
            "paperheight": 31.5 if curr_strat.is_plus_strategy() else 28,
        },
        header=page.header(
            left="Strategy Risk Report",
            right=f"EOD {date.strftime('%B %#d, %Y')}",
        ),
        footer=page.footer(logo="LG_umbrella"),
    )

    row_precs = {
        "Total OAD OW": "+2f",
        "Tsy MV (%)": "1f",
        "Tsy OAD OW": "+2f",
        "Credit OAD OW": "+2f",
        "Corp OAD OW": "+2f",
        "DTS OW (%)": "+1f",
        "~Port DTS (%)": "1f",
        "~BM DTS (%)": "1f",
        "BM OAS": "0f",
        "DTS (%)": "1f",
        "DTS OW (abs)": "+1f",
        "DTS OW (dur)": "+2f",
        "Barbell (%)": "+1f",
        "Carry (bp)": "+1f",
        "Port Yield (%)": "2f",
        "Port OASD": "+2f",
        "BM OAS": "0f",
        "BM OAD": "1f",
        "Curve Dur (5yr)": "+2f",
        "Curve Dur (7yr)": "+2f",
        "Curve Dur (10yr)": "+2f",
        "Curve Dur (20yr)": "+2f",
        "AAA-AA OW": "+2f",
        "A OW": "+2f",
        "BBB OW": "+2f",
        "HY OW": "+2f",
        "Corp OW": "+2f",
        "Non-Corp OW": "+2f",
        "Tracking Error": "0f",
        "Normalized TE": "2f",
        "Performance": "+1f",
    }

    ticker_sector_ow_table_kwargs = {
        "col_fmt": "lrc|rc|rcc|rcc",
        "multi_row_header": True,
        "hide_index": True,
        "adjust": True,
        "font_size": "scriptsize",
    }

    # Excluded accounts.
    excluded_accounts = ignored_accounts & set(curr_strat._all_accounts)
    if excluded_accounts:
        page.add_text(
            f"\\small "
            f"Accounts Excluded from Analysis: "
            f"{', '.join(sorted(list(excluded_accounts)))}"
        )

    # Overview tables.
    if curr_strat.is_plus_strategy():
        overview_widths = [0.42, 0.55]
        rating_and_individual_bonds_widths = [0.29, 0.68]
    else:
        overview_widths = [0.38, 0.59]
        rating_and_individual_bonds_widths = [0.32, 0.65]
    edits = {}
    (
        edits["overview"],
        edits["rating_and_individual_bonds"],
    ) = page.add_minipages(n=2, valign="t", widths=overview_widths)
    with page.start_edit(edits["overview"]):
        page.add_table(
            overview_table,
            caption="General Overview",
            col_fmt="lccc",
            midrule_locs=general_overview_midrules,
            adjust=True,
        )
    with page.start_edit(edits["rating_and_individual_bonds"]):
        (
            edits["top_level"],
            edits["individual_bonds_change"],
        ) = page.add_minipages(
            n=2, valign="t", widths=rating_and_individual_bonds_widths
        )
        # Put HY issuer table below the other two.
        if curr_strat.is_plus_strategy():
            page.add_table(
                hy_ticker_ow_table,
                caption="HY Issuers",
                col_fmt="lrcc|rc|rc",
                multi_row_header=True,
                prec={
                    f"{ow_metric}*(yrs)": "2f",
                    "MV*(%)": "2%",
                    f"$\\Delta${ow_metric}*(yrs)": "3f",
                    f"$\\Delta${ow_metric}*(yrs) ": "3f",
                },
                font_size="scriptsize",
                hide_index=True,
                adjust=True,
                align="left",
            )

    with page.start_edit(edits["top_level"]):
        page.add_table(
            gen_ow_table,
            caption="\\scriptsize Rating/Top Level Sector Overweights",
            col_fmt="lcc",
            prec=2,
            indent_subindexes=True,
            midrule_locs=["Corp", "Non-Corp"],
            adjust=True,
            multi_row_header=True,
        )
    with page.start_edit(edits["individual_bonds_change"]):
        page.add_table(
            bond_ow_table,
            caption="Individual Bonds",
            col_fmt="lrc|rc",
            prec=3,
            align="left",
            multi_row_header=True,
            hide_index=True,
            adjust=True,
        )

    # Issuer and sector overweight and change.
    page.add_table(
        ticker_ow_table,
        caption="Issuers",
        align="left",
        prec=col_prec(ticker_ow_table, ow_metric),
        div_bar_col=[
            f"$\\Delta${ow_metric}*(yrs)",
            f"$\\Delta${ow_metric}*(yrs) ",
        ],
        center_div_bar_header=False,
        div_bar_kws={
            "cmax": "army",
            "cmin": "rose",
        },
        **ticker_sector_ow_table_kwargs,
    )
    page.add_table(
        sector_ow_table,
        caption="Sectors",
        prec=col_prec(sector_ow_table, ow_metric),
        **ticker_sector_ow_table_kwargs,
    )

    # Account dispersion table.
    page.add_table(
        account_dispersion_table,
        caption="Account Dispersion",
        col_fmt=("l" + "rc|" * (len(account_dispersion_table.columns) // 2))[
            :-1
        ],
        multi_row_header=True,
        midrule_locs=account_dispersion_midrules,
        prec={
            "DTS*(%)": "1%",
            "Barbell*(%)": "1%",
            "Tracking*Error": "0f",
            f"Curve ({curr_strat.curve_pivot_point}yr)*Duration": "2f",
            "Tsy*OAD": "2f",
            "Cash*(%)": "1%",
        },
        font_size="scriptsize",
        hide_index=True,
        align="left",
        adjust=True,
    )

    page.add_pagebreak()
    page.add_section(curr_strat.latex_name, bookmark=False)
    indents = {"Port DTS (%)", "BM DTS (%)"}

    # Rating bucket risk.
    rating_risk_table.columns = [
        c.replace(" ", "*") for c in rating_risk_table.columns
    ]

    (_, edits["rating_risk_buckets"], __) = page.add_subfigures(
        n=3, valign="t", widths=[0.2, 0.55, 0.2]
    )
    rating_risk_table.index = [
        f"~{idx}" if idx in indents else idx for idx in rating_risk_table.index
    ]
    with page.start_edit(edits["rating_risk_buckets"]):
        page.add_table(
            rating_risk_table,
            caption="Curve Risk by Rating Risk Bucket",
            col_fmt=f"l|{'r' * (len(rating_risk_table.columns) - 1)}|r",
            multi_row_header=True,
            indent_subindexes=True,
            midrule_locs=["Credit OAD OW", "DTS OW"],
            adjust=True,
            row_prec={idx: row_precs[idx] for idx in rating_risk_table.index},
        )

    # Benchmark treasury table.
    (_, edits["bm_treasury_buckets"], __) = page.add_subfigures(
        n=3, valign="t", widths=[0.1, 0.75, 0.1]
    )
    bm_tsy_table.index = [
        f"~{idx}" if idx in indents else idx for idx in bm_tsy_table.index
    ]
    with page.start_edit(edits["bm_treasury_buckets"]):
        page.add_table(
            bm_tsy_table,
            caption="Curve Risk by Respective Benchmark Treasury",
            col_fmt="l|rrrrrrr|r|r",
            indent_subindexes=True,
            midrule_locs=["Tsy MV", "Credit OAD OW", "DTS OW"],
            adjust=True,
            row_prec={idx: row_precs[idx] for idx in bm_tsy_table.index},
        )

    # Maturity bucket risk.
    maturity_bucket_table.index = [
        f"~{idx}" if idx in indents else idx
        for idx in maturity_bucket_table.index
    ]
    page.add_table(
        maturity_bucket_table,
        caption="Curve Risk by Maturity Bucket",
        col_fmt=f"l|{'r' * (len(maturity_bucket_table.columns) - 1)}|r",
        indent_subindexes=True,
        midrule_locs=["Credit OAD OW", "DTS OW"],
        adjust=True,
        row_prec={idx: row_precs[idx] for idx in maturity_bucket_table.index},
    )

    page.add_table(
        maturity_spread_heatmap,
        caption="DTS OW vs Benchmark (\\%) by Maturity Bucket and Spread Quintile",
        col_fmt=f"l|{'r' * (len(maturity_spread_heatmap.columns) - 1)}|r",
        row_prec={idx: "+1f" for idx in maturity_spread_heatmap.index},
        adjust=True,
        gradient_cell_col=maturity_spread_heatmap.columns,
        gradient_cell_kws={
            "cmax": "army",
            "cmin": "rose",
            "vmin": maturity_spread_heatmap.iloc[:, :-1].min().min(),
            "vmax": maturity_spread_heatmap.iloc[:, :-1].max().max(),
            "symmetric": True,
        },
    )
    page.save_as(curr_strat.fid, save_tex=False)

    history_fid = f"{curr_strat.fid}_history"
    history_page = Document(fid=history_fid, path=pdf_path)
    history_page.add_preamble(
        orientation="landscape",
        margin={
            "left": 0.5,
            "right": 0.5,
            "top": 0.3,
            "bottom": 0.2,
            "paperheight": 30,
        },
        header=history_page.header(
            left="Strategy Risk Report",
            right=f"EOD {date.strftime('%B %#d, %Y')}",
        ),
        footer=history_page.footer(logo="LG_umbrella"),
    )
    history_page.add_section(curr_strat.latex_name, bookmark=False)

    #  Strategy history.
    (
        edits["history"],
        edits["historic_percentiles"],
        edits["historic_range"],
    ) = history_page.add_subfigures(n=3, valign="t", widths=[0.48, 0.18, 0.28])

    midrule_locs = [
        "Barbell (\\%)",
        "Carry (bp)",
        "BM OAS",
        "Total OAD OW",
        "Curve Dur (5yr)",
        "AAA-AA OW",
        "Corp OW",
        "Performance",
    ]
    # Get proper formatting for history table columns
    history_col_fmt = "l|"
    for col in history_table.columns:
        if col.startswith("Q"):
            history_col_fmt += "r|"
        else:
            history_col_fmt += "r"
    history_col_fmt = history_col_fmt[:-1] + "|" + history_col_fmt[-1]

    history_table_kwargs = {
        "font_size": "footnotesize",
        "multi_row_header": True,
        "midrule_locs": midrule_locs,
    }
    footnote = """
        \\vspace{-2.5ex}
        \\scriptsize
        \\itemsep0.8em
        \\item
        Performance is a sum total of estimated performance over the
        respective periods (quarter, month, and MTD for the current month).

        \\item
        Snapshots are taken in the middle of the respective months in
        order to avoid issues with index changes near month end.
        For quarters, snapshots are from the middle of the first month of
        the quarter. For the current month, the snapshot is for
        the current date as noted in the column header.
        """
    with history_page.start_edit(edits["history"]):
        history_page.add_table(
            history_table,
            caption="Strategy snapshot history",
            col_fmt=history_col_fmt,
            row_prec={idx: row_precs[idx] for idx in history_table.index},
            table_notes=footnote,
            **history_table_kwargs,
        )
    with history_page.start_edit(edits["historic_percentiles"]):
        history_page.add_table(
            historic_percentile_df,
            caption="Historical Percentiles",
            col_fmt="l|rr",
            prec={col: "0f" for col in historic_percentile_df.columns},
            gradient_cell_col=historic_percentile_df.columns,
            gradient_cell_kws={
                "vmin": 0,
                "vmax": 100,
                "center": 50,
                "cmin": "rose",
                "cmax": "army",
            },
            **history_table_kwargs,
        )
    start_date = history_df.index[0]
    with history_page.start_edit(edits["historic_range"]):
        history_page.add_table(
            historic_range_df,
            caption=f"Historical Range (since {start_date:%#m/%d/%Y})",
            col_fmt="l|rrr",
            row_prec={idx: row_precs[idx] for idx in historic_range_df.index},
            **history_table_kwargs,
        )
    if debug:
        print(strategy, "-finished")

    history_page.save_as(history_fid, save_tex=False)
    return strategy, summary, curr_strat.fid


# %%

if __name__ == "__main__":
    main()
