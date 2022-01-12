from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document, Page
from lgimapy.models import Dispersion


fid = "Dispersion"
maturity = 10

# %%


def update_dispersion(fid):
    # %%
    vis.style()

    db = Database()
    disp = Dispersion("IG", db)
    disp.update()

    # %%
    date = db.date("today")
    doc = Document(fid, path="reports/valuation_pack", fig_dir=True)
    # doc = Page(path="reports/valuation_pack", fig_dir=True)
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
    doc.save()


if __name__ == "__main__":
    fid = "Dispersion"
    maturity = 10
    update_dispersion(fid)
