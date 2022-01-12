import pandas as pd
import numpy as np

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.latex import Document

# %%


def update_valuations(fid, db):
    vis.style()
    doc = Document(fid, path="reports/HY", fig_dir=True)
    date = db.date("today").strftime("%B %#d, %Y")
    doc.add_preamble(
        margin={
            "paperheight": 32,
            "left": 0.5,
            "right": 0.5,
            "top": 0.5,
            "bottom": 0.2,
        },
        bookmarks=True,
        table_caption_justification="c",
        header=doc.header(left="HY Valuations", right=f"EOD {date}"),
        footer=doc.footer(logo="LG_umbrella"),
    )
    start = "1/1/2020"
    total_ix = db.build_market_index(
        in_hy_stats_index=True, OAS=(-10, 5000), start=start
    )

    energy_sectors = [
        "INDEPENDENT",
        "REFINING",
        "OIL_FIELD_SERVICES",
        "INTEGRATED",
        "MIDSTREAM",
    ]
    ix_ex_energy = total_ix.subset(
        sector=energy_sectors, special_rules="~Sector"
    )
    ix_energy = total_ix.subset(sector=energy_sectors)

    ixs = {"Overall": total_ix, "Ex-Energy": ix_ex_energy, "Energy": ix_energy}

    # Get summary table dates.
    all_dates = db.trade_dates(start=start)
    tights = pd.to_datetime("1/17/2020")
    wides = pd.to_datetime("3/23/2020")
    month_starts = set(
        [db.date("month_start", reference_date=d) for d in all_dates]
    )
    dates = sorted(list(month_starts.union([tights, wides, all_dates[-1]])))

    rating_buckets = {
        "HY": (None, None),
        "BB": ("BB+", "BB-"),
        "B": ("B+", "B-"),
        # "CCC": ("CCC+", "CCC-"),
    }
    colors = ["#04060F", "#0294A5", "#C1403D"]
    cols = {"OAS": "OAS ", "DirtyPrice": "Price ", "YieldToWorst": "YTW"}
    mv_fmt = "{}{} ($ Weighted)"
    i_fmt = "{}{} (Issuer Weighted)"

    writer = pd.ExcelWriter(doc.path / "HY_Valuations.xlsx")
    doc.add_section("Valuations")
    for section, ix in ixs.items():
        doc.add_subsection(section)

        df_list = []
        fig, axes = vis.subplots(3, 1, figsize=(16, 12), sharex=True)
        for (rating, rating_kwargs), color in zip(
            rating_buckets.items(), colors
        ):
            ix_rating = ix.subset(rating=rating_kwargs)
            for i, (col, name) in enumerate(cols.items()):
                mv = ix_rating.market_value_weight(col).rename(
                    mv_fmt.format(name, rating)
                )
                iss = (
                    ix_rating.df.groupby("Date")[col]
                    .mean()
                    .rename(i_fmt.format(name, rating))
                )
                if i == 2:
                    mv /= 100
                    iss /= 100
                ytickfmt = {0: None, 1: "${x:.0f}", 2: "{x:.0%}"}[i]
                vis.plot_timeseries(
                    mv,
                    color=color,
                    alpha=0.8,
                    ls="-",
                    label=mv_fmt.format("", rating),
                    xtickfmt="auto",
                    ytickfmt=ytickfmt,
                    ax=axes[i],
                    legend=i == 0,
                )
                vis.plot_timeseries(
                    iss,
                    color=color,
                    alpha=0.8,
                    ls="--",
                    label=i_fmt.format("", rating),
                    xtickfmt="auto",
                    ytickfmt=ytickfmt,
                    legend=i == 0,
                    ax=axes[i],
                )
                axes[i].set_ylabel(name)
                df_list.extend([mv, iss])

        df = pd.concat(df_list, axis=1, sort=True)

        # Make Summary Table
        selected_cols = [
            col for col in df.columns if "$" in col and "Price" not in col
        ]
        table = df.loc[dates, selected_cols].rename_axis(None)
        table.index = [ix.strftime("%m/%d/%Y") for ix in table.index]
        table.columns = [
            "HY*OAS",
            "HY*YTW",
            "BB*OAS",
            "BB*YTW",
            "B*OAS",
            "B*YTW",
        ]
        prec = {col: "0f" if "OAS" in col else "2%" for col in table.columns}

        cap = f"{section} \\$ weighted summary."
        doc.add_table(
            table,
            col_fmt="l|cc|cc|cc",
            font_size="scriptsize",
            prec=prec,
            multi_row_header=True,
            caption=cap,
        )
        doc.add_figure(section, width=0.95, savefig=True)

        # Round spreads.
        for col in df.columns:
            if "OAS" in col:
                df[col] = df[col].round(0).astype(int)
        df.index = pd.Series(df.index).dt.strftime("%m/%d/%Y")
        df.rename_axis(None, inplace=True)

        df.to_excel(writer, sheet_name=section)

    doc.save()
    writer.save()


if __name__ == "__main__":
    fid = "HY_Valuations"
    db = Database()
    db.load_market_data(start="1/1/2020")
    update_valuations(fid, db)
