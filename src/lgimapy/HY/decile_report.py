import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root

from lgimapy.HY.update_decile_report_data import update_decile_report_data


# %%


def build_decile_report(fid):
    args = parse_args()
    if args.update:
        update_decile_report_data()

    # Load data.
    data_dir = Database().local("HY/decile_report")
    historical_fields = ["spreads", "yields", "prices"]
    historical_fids = {f: data_dir / f"{f}.csv" for f in historical_fields}
    return_fids = {
        "Total Returns (Unhedged)": data_dir / "total_returns.csv",
        "Total Returns (USD Hedged)": data_dir / "total_returns_hedged.csv",
        "Excess Returns": data_dir / "excess_returns.csv",
    }
    historical_dfs = {
        f: pd.read_csv(
            fid, index_col=0, parse_dates=True, infer_datetime_format=True
        ).fillna(method="ffill")
        for f, fid in historical_fids.items()
    }
    return_dfs = {
        key: pd.read_csv(fid, index_col=0) for key, fid in return_fids.items()
    }
    date = historical_dfs["spreads"].index[-1]

    sections = {
        "Spreads Overview": {
            "df": historical_dfs["spreads"],
            "prec": "0f",
            "cols": [
                "H0A0*US HY",
                "HC1N*US BB",
                "HUC2*US B",
                "HUC3*US CCC",
                "HE00*EU HY",
                "HE1M*EU BB",
                "HE20*EU B",
                "EMUH*EM HY",
                "EM3B*EM BB",
                "EM6B*EM B",
                "H4UN*US HY",
                "HWXC*Global HY",
                "HL00*UK HY",
            ],
            "col_fmt": "l|rrrr|rrr|rrr|rrr",
        },
        "Spreads Fin/Non-Fin": {
            "df": historical_dfs["spreads"],
            "prec": "0f",
            "cols": [
                "CF40*US*BBB f",
                "C4NF*US*BBB nf",
                "EB40*EU*BBB f",
                "EN40*EU*BBB nf",
                "UF40*UK*BBB f",
                "U04N*UK*BBB nf",
                "ER00*EU*IG",
                "EB00*EU*IG f",
                "EN00*EU*IG nf",
                "COCO",
            ],
            "col_fmt": "l|rr|rr|rr|rrr|r",
        },
        "Spreads Differences": {
            "df": historical_dfs["spreads"],
            "prec": "0f",
            "cols": [
                "US*B-BB",
                "US nf*BB-BBB",
                "EU*BB-BBB",
                "US-EU nf*BBB",
                "US-EU*BB",
                "US-EU*B",
                "EM-US*BB",
                "EM-US*B",
            ],
            "col_fmt": "l|rr|r|rrr|rr",
        },
        "Yields Overview": {
            "df": historical_dfs["yields"],
            "prec": "2f",
            "cols": [
                "H0A0*US HY",
                "HC1N*US BB",
                "HUC2*US B",
                "HUC3*US CCC",
                "HE00*EU HY",
                "HE1M*EU BB",
                "HE20*EU B",
                "EMUH*EM HY",
                "EM3B*EM BB",
                "EM6B*EM B",
                "CF40*US*BBB f",
                "C4NF*US*BBB nf",
                "EB40*EU*BBB f",
                "EN40*EU*BBB nf",
            ],
            "col_fmt": "l|rrrr|rrr|rrr|rrrr",
        },
        "Prices Overview": {
            "df": historical_dfs["prices"],
            "prec": "2f",
            "cols": [
                "H0A0*US HY",
                "HC1N*US BB",
                "HUC2*US B",
                "HUC3*US CCC",
                "HE00*EU HY",
                "HE1M*EU BB",
                "HE20*EU B",
                "EMUH*EM HY",
                "EM3B*EM BB",
                "EM6B*EM B",
                "CF40*US*BBB f",
                "C4NF*US*BBB nf",
                "EB40*EU*BBB f",
                "EN40*EU*BBB nf",
            ],
            "col_fmt": "l|rrrr|rrr|rrr|rrrr",
        },
    }
    # Print indexes to scrape then exit.
    if args.print:
        print("\nYields")
        for i, index in enumerate(sections["Yields Overview"]["cols"]):
            print(f"  {i+1}) {index.split('*')[0]}")
        print("\nRemaining Spreads")
        remaining_indexes = (
            sections["Spreads Overview"]["cols"][-3:]
            + sections["Spreads Fin/Non-Fin"]["cols"][-6:]
        )
        for i, index in enumerate(remaining_indexes):
            print(f"  {i+1}) {index.split('*')[0]}")
        print()
        return

    doc = Document(fid, path="reports/HY")
    doc.add_preamble(
        orientation="landscape",
        bookmarks=True,
        font_size=15,
        table_caption_justification="c",
        margin={
            "left": 0.5,
            "right": 0.5,
            "top": 1.5,
            "bottom": 1,
            "paperwidth": 24,
        },
        header=doc.header(
            left="HY Decile Report",
            right=f"EOD {date:%B %#d, %Y}",
            height=0.5,
        ),
        footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.05),
    )

    # Add percentile tables for spreads and yields.
    doc.add_bookmark("Decile Tables")
    doc.add_bookmark("Spreads", level=1)
    prec_base = {"Percentile": "0f"}
    for section, kwargs in sections.items():
        if not section.startswith("Spread"):
            doc.add_bookmark(section.split()[0], level=1)

        df = kwargs["df"].copy()
        cols = kwargs["cols"]
        df
        table, color_locs, notes = make_table(df[cols], section)
        prec = prec_base.copy()
        for idx in table.index:
            if idx in {"Start Date", "Percentile"}:
                continue
            prec[idx] = kwargs["prec"]
        doc.add_table(
            table,
            caption=section,
            row_prec=prec,
            table_notes=notes,
            table_notes_justification="c",
            col_fmt=kwargs["col_fmt"],
            multi_row_header=True,
            font_size="large",
            int_vals=False,
            midrule_locs=["Median"],
            specialrule_locs=["0-9", "Sep-01"],
            loc_style=color_locs,
        )

    # Add total returns table.

    for title, df in return_dfs.items():
        doc.add_bookmark(title, level=1)
        table = df.copy()
        heatmap_cols = ["1WK", "YTD"]
        heatmap_kwargs = {}
        for col in heatmap_cols:
            mean = np.mean(df[col][:-3])
            color_vals = df[col].copy().values
            for i in range(1, 4):
                color_vals[-i] = mean

            heatmap_kwargs[col] = {
                "vals": color_vals,
                "center": mean,
                "cmax": "steelblue",
                "cmin": "firebrick",
            }
        doc.add_table(
            table,
            prec={col: "2f" for col in df.columns if col != "Index"},
            caption=title,
            col_fmt="l|l|rrrrr|rrr",
            font_size="Large",
            midrule_locs=["HE00", "EMUH", "C4NF", "LP01", "BEBG", "LDB1"],
            specialrule_locs=["SPX"],
            gradient_cell_col=heatmap_cols,
            gradient_cell_kws=heatmap_kwargs,
        )

    doc.save(save_tex=False)


def find_last_trade_date(df):
    """Find last date with no missing data."""
    df_nans = df.isna().sum(axis=1)
    return df_nans[df_nans == 0].index[-1]


def make_table(df, section):
    """
    Make table of summary stats and deciles.
    """
    current_date = df.index[-1]
    date_fmt = "%m/%d/%Y"

    table_colors = {
        "salmon": pd.to_datetime("3/23/2020"),
        "babyblue": pd.to_datetime("2/14/2020"),
        "mauve": current_date,
    }
    title = section.split()[0][:-1]
    table_notes = (
        "Percentile ranges display values for the high end of the range."
        "\\\\"
        f"{title} deciles highlited for \\color{{babyblue}}"
        f"\\textbf{{{table_colors['babyblue'].strftime(date_fmt)}}} "
        f"\\color{{black}}, \\color{{mauve}}"
        f"\\textbf{{{table_colors['mauve'].strftime(date_fmt)}}} "
        f"\\color{{black}}, and \\color{{salmon}}"
        f"\\textbf{{{table_colors['salmon'].strftime(date_fmt)}}} "
        f"\\color{{black}}."
    )

    table_dates = pd.to_datetime(
        [
            "9/3/2001",
            "11/3/2008",
            "2/2/2009",
            "6/2/2014",
            "2/1/2016",
            "12/1/2016",
            "12/1/2017",
            "6/1/2018",
            "9/3/2018",
            "12/3/2018",
            "3/1/2019",
            "6/3/2019",
            "9/2/2019",
            "12/2/2019",
        ]
    )
    color_locs = {}
    d = defaultdict(list)
    for i, col in enumerate(df.columns):
        if not section.startswith("Spread"):
            not_spreads = True
            vals = df[col].dropna().round(2)
        else:
            vals = df[col].dropna().astype("Int64")
            not_spreads = True

        d[current_date.strftime(date_fmt)].append(vals[-1])
        d["Percentile"].append(np.round(100 * vals.rank(pct=True)[-1]))
        d["Start Date"].append(vals.index[0].strftime("%b-%y"))
        med = np.median(vals)
        d["Median"].append(med if not_spreads else int(med))
        d["Wides"].append(np.max(vals))
        d["Tights"].append(np.min(vals))
        quantiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
        __, bins = pd.qcut(vals, np.array(quantiles) / 100, retbins=True)
        if not_spreads:
            bins = np.round(bins, 2)
        percentile_labels = [
            "0-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70-79",
            "80-89",
            "90-94",
            "95-98",
            "99+",
        ]
        for bin, label in zip(bins[1:], percentile_labels):
            d[label].append(bin if not_spreads else int(bin))

        for date in table_dates:
            try:
                d[date.strftime("%b-%y")].append(vals.loc[date])
            except KeyError:
                d[date.strftime("%b-%y")].append(None)

        for color, date in table_colors.items():
            j = list(vals.index).index(date)
            bin = np.digitize(vals, bins)[j]
            color_locs[(min(17, int(bin + 5)), i)] = f"\\cellcolor{{{color}}}"

    table = pd.DataFrame(d, index=df.columns).T
    if not_spreads:
        table = table.round(2)
    return table, color_locs, table_notes


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print Indexes"
    )
    parser.add_argument(
        "-u", "--update", action="store_true", help="Update Data Before Running"
    )
    return parser.parse_args()


# %%

if __name__ == "__main__":
    fid = "HY_Decile_Report"
    build_decile_report(fid)
