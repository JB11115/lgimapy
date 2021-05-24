import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.latex import Document
from lgimapy.utils import root

# %%


def update_spreads_yields_returns():
    # Load data.
    hy_dir = root("data/HY")
    spreads_fid = hy_dir / "report_spreads.csv"
    yields_fid = hy_dir / "report_yields.csv"
    return_fids = {
        "Total Returns (Unhedged)": hy_dir / "report_total_returns.csv",
        "Total Returns (USD Hedged)": hy_dir
        / "report_total_returns_hedged.csv",
        "Excess Returns": hy_dir / "report_excess_returns.csv",
    }
    spreads_df = pd.read_csv(
        spreads_fid, index_col=0, parse_dates=True, infer_datetime_format=True
    )
    yields_df = pd.read_csv(
        yields_fid, index_col=0, parse_dates=True, infer_datetime_format=True
    )
    return_dfs = {
        key: pd.read_csv(fid, index_col=0) for key, fid in return_fids.items()
    }
    sections = {
        "Spreads Overview": {
            "df": spreads_df,
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
            "df": spreads_df,
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
            "df": spreads_df,
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
            "df": yields_df,
            "yields": True,
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
    args = parse_args()
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

    doc = Document("HY_spreads_yields_returns", path="reports/HY")
    date = spreads_df.index[-1]
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
            left="HY Spreads, Yields, and Returns",
            right=f"EOD {date.strftime('%B %#d, %Y')}",
            height=0.5,
        ),
        footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.05),
    )

    # Add percentile tables for spreads and yields.
    doc.add_section("Decile Tables")
    doc.add_subsection("Spreads")
    for section, kwargs in sections.items():
        if section.startswith("Yield"):
            doc.add_subsection("Yields")
        df, cols, yields = (
            kwargs["df"],
            kwargs["cols"],
            kwargs.get("yields", False),
        )
        table, color_locs, notes = make_table(df[cols], yields=yields)
        doc.add_table(
            table,
            caption=section,
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
        doc.add_subsection(title)
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
            table.round(2),
            caption=title,
            col_fmt="l|l|rrrrr|rrr",
            font_size="Large",
            midrule_locs=["HE00", "EMUH", "C4NF", "LP01", "BEBG", "LDB1"],
            specialrule_locs=["SPX"],
            gradient_cell_col=heatmap_cols,
            gradient_cell_kws=heatmap_kwargs,
        )
    doc.save(save_tex=False)


def make_table(df, yields=False, highlight_dates=None):
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
    title = "Yield" if yields else "Spread"
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
        if yields:
            vals = df[col].dropna().round(2)
        else:
            vals = df[col].dropna().astype("Int64")
        d[current_date.strftime(date_fmt)].append(vals[-1])
        d["Percentile"].append(np.round(100 * vals.rank(pct=True)[-1]))
        d["Start Date"].append(vals.index[0].strftime("%b-%y"))
        med = np.median(vals)
        d["Median"].append(med if yields else int(med))
        d["Wides"].append(np.max(vals))
        d["Tights"].append(np.min(vals))
        quantiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
        __, bins = pd.qcut(vals, np.array(quantiles) / 100, retbins=True)
        if yields:
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
            d[label].append(bin if yields else int(bin))

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
    if yields:
        table = table.round(2)
    return table, color_locs, table_notes


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print Indexes"
    )
    return parser.parse_args()


# %%

if __name__ == "__main__":
    update_spreads_yields_returns()