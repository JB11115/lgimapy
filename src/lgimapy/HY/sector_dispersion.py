from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document, Page
from lgimapy.models import Dispersion

# %%


def update_sector_dispersion(fid):
    disp = Dispersion("HY", Database())
    disp.update()
    doc = build_dispersion_doc(fid)
    doc = add_dispersion_tables(doc, disp)
    doc.save()


def build_dispersion_doc(fid):
    date = Database().date("today")
    doc = Document(fid, path="reports/HY", fig_dir=True)
    doc.add_preamble(
        bookmarks=True,
        table_caption_justification="c",
        margin={"left": 0.5, "right": 0.5, "top": 2, "bottom": 1},
        header=doc.header(
            left="HY Sector Dispersion",
            right=f"EOD {date:%B %#d, %Y}",
            height=0.5,
        ),
        footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.05),
    )
    doc.add_section("Sector Dispersion")
    return doc


def add_dispersion_tables(doc, disp):
    overview_table = disp.overview_table(maturity=None)
    doc.add_table(
        overview_table,
        # adjust=True,
        caption="Dispersion Overview \\%tiles",
        prec={col: "0f" for col in overview_table.columns},
        gradient_cell_col=overview_table.columns,
        gradient_cell_kws={"vmin": 0, "vmax": 100, "center": 50},
    )

    intra_sector_table = disp.intra_sector_table(maturity=None)
    doc.add_table(
        intra_sector_table,
        # adjust=True,
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
    fid = "HY_Sector_Dispersion"
    update_sector_dispersion(fid)
