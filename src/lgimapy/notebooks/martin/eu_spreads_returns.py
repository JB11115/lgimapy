from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.latex import Document
from lgimapy.utils import root

# %%


def main():
    # Load data.
    spreads_fid = root("data/HY/report_spreads.csv")
    tret_fid = root("data/HY/report_total_returns.csv")

    spreads_df = pd.read_csv(
        spreads_fid, index_col=0, parse_dates=True, infer_datetime_format=True
    )
    tret_df = pd.read_csv(tret_fid, index_col=0)

    sections = {
        "Spreads Overview": {
            "df": spreads_df,
            "cols": ["HE00*EU HY", "HE1M*EU BB", "HE20*EU B",],
            "col_fmt": "l|rrr",
        },
    }
    # %%
    doc = Document("EU_spreads_returns", path="reports/HY")
    doc.add_preamble(
        orientation="landscape",
        font_size=15,
        table_caption_justification="c",
        margin={"left": 0.5, "right": 0.5, "top": 1.5, "bottom": 1},
    )

    # Add percentile tables for spreads and yields.
    for section, kwargs in sections.items():
        df, cols, yields = (
            kwargs["df"],
            kwargs["cols"],
            kwargs.get("yields", False),
        )
        table, color_locs, notes = make_table(df[cols], yields=False)
        doc.add_table(
            table.iloc[:17],
            caption=section,
            table_notes=notes,
            table_notes_justification="c",
            col_fmt=kwargs["col_fmt"],
            multi_row_header=True,
            font_size="large",
            int_vals=True,
            midrule_locs=["Median"],
            specialrule_locs=["0-9"],
            loc_style=color_locs,
        )

    # Add total returns table.
    tret_table = tret_df.loc[
        ["HE00", "HE1M", "HE20"], ["Index", "1MO", "YTD", "1YR"]
    ]
    doc.add_table(
        tret_table.round(2),
        caption="Total Returns",
        col_fmt="l|l|rrr",
        font_size="Large",
    )
    doc.save()
    # %%


def make_table(df, yields=False, highlight_dates=None):
    """
    Make table of summary stats and deciles.
    """
    current_date = df.index[-1]
    date_fmt = "%m/%d/%Y"

    table_colors = {
        "salmon": pd.to_datetime("3/23/2020"),
        "babyblue": pd.to_datetime("2/14/2020"),
        "eggplant": current_date,
    }
    title = "Yield" if yields else "Spread"
    table_notes = (
        f"{title} deciles highlited for \\\\ \\color{{babyblue}}"
        f"\\textbf{{{table_colors['babyblue'].strftime(date_fmt)}}} "
        f"\\color{{black}}, \\color{{eggplant}}"
        f"\\textbf{{{table_colors['eggplant'].strftime(date_fmt)}}} "
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
            color_locs[(min(16, int(bin + 4)), i)] = f"\\cellcolor{{{color}}}"

    table = pd.DataFrame(d, index=df.columns).T
    if yields:
        table = table.round(2)
    return table, color_locs, table_notes


if __name__ == "__main__":
    main()
