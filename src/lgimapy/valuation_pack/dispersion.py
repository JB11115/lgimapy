from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document, Page
from lgimapy.models import Dispersion

# %%


def update_dispersion(fid):
    db = Database()
    disp = Dispersion("IG", db)
    disp.update()
    doc = Document(
        fid, path="reports/valuation_pack", fig_dir=True, load_tex=True
    )
    doc = edit_dispersion_tables(doc, disp)
    doc.save_tex()


def build_dispersion_sheet(fid, disp):
    date = db.date("today")
    doc = Document(fid, path="reports/valuation_pack", fig_dir=True)
    doc.add_preamble(
        orientation="landscape",
        bookmarks=True,
        table_caption_justification="c",
        margin={"left": 0.5, "right": 0.5, "top": 2, "bottom": 1},
        header=doc.header(
            left="Dispersion",
            right=f"EOD {date:%B %#d, %Y}",
            height=0.5,
        ),
        footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.05),
    )
    doc.add_section("Dispersion")
    doc = add_dispersion_tables(doc, disp)
    doc.save()


def add_dispersion_tables(doc, disp):
    page_columns = doc.add_subfigures(n=2, valign="t")
    for maturity, page_col in zip([10, 30], page_columns):
        with doc.start_edit(page_col):
            doc.add_text(f"\\Huge \\textbf \u007b{maturity} Year\u007d")
            overview_table = disp.overview_table(maturity=maturity)
            doc.add_table(
                overview_table,
                adjust=True,
                caption="Dispersion Overview \\%tiles",
                prec={col: "0f" for col in overview_table.columns},
                gradient_cell_col=overview_table.columns,
                gradient_cell_kws={"vmin": 0, "vmax": 100, "center": 50},
            )

            intra_sector_table = disp.intra_sector_table(maturity=maturity)
            doc.add_table(
                intra_sector_table,
                adjust=True,
                caption="Intra-Sector Dispersion \\%tiles",
                col_fmt="l|rrr|rrr",
                multi_row_header=True,
                prec={col: "0f" for col in intra_sector_table.columns},
                gradient_cell_col=[
                    col for col in intra_sector_table.columns if "Rel" in col
                ],
                gradient_cell_kws={"vmin": 0, "vmax": 100, "center": 50},
            )

    return doc


def edit_dispersion_tables(doc, disp):
    for maturity in [10, 30]:
        with doc.start_edit(f"dispersion_{maturity}y"):
            doc.add_text(f"\\Huge \\textbf \u007b{maturity} Year\u007d")
            overview_table = disp.overview_table(maturity=maturity)
            doc.add_table(
                overview_table,
                adjust=True,
                caption="Dispersion Overview \\%tiles",
                prec={col: "0f" for col in overview_table.columns},
                gradient_cell_col=overview_table.columns,
                gradient_cell_kws={"vmin": 0, "vmax": 100, "center": 50},
            )

            intra_sector_table = disp.intra_sector_table(maturity=maturity)
            doc.add_table(
                intra_sector_table,
                adjust=True,
                caption="Intra-Sector Dispersion \\%tiles",
                col_fmt="l|rrr|rrr",
                multi_row_header=True,
                prec={col: "0f" for col in intra_sector_table.columns},
                gradient_cell_col=[
                    col for col in intra_sector_table.columns if "Rel" in col
                ],
                gradient_cell_kws={"vmin": 0, "vmax": 100, "center": 50},
            )

    return doc


if __name__ == "__main__":
    db = Database()
    disp = Dispersion("IG", db)
    disp.update()
    fid = "Dispersion"
    build_dispersion_sheet(fid, disp)
