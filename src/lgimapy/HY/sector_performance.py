from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import BetaAdjustedPerformance
from lgimapy.utils import get_ordinal

# %%


def update_sector_performance(fid):

    doc = Document(fid, path="reports/HY", fig_dir=True)
    doc.add_section("Sector Performance")
    today = Database().date("today").strftime("%B %#d, %Y")
    doc.add_preamble(
        margin={
            "paperheight": 28,
            "left": 0.5,
            "right": 0.5,
            "top": 0.5,
            "bottom": 0.2,
        },
        bookmarks=True,
        table_caption_justification="c",
        header=doc.header(left="Sector Breakdown", right=f"EOD {today}"),
        footer=doc.footer(logo="LG_umbrella"),
    )
    make_sector_performance_table(doc)
    doc.save()


def make_sector_performance_table(doc):
    mod = BetaAdjustedPerformance(Database())
    mod.train(universe="HY", forecast="1M")
    H4UN_oas = mod._db.build_market_index(in_H4UN_index=True).OAS()
    table = mod.get_sector_table(
        add_index_row=True,
        index_name="H4UN Index",
        index_oas=H4UN_oas,
        return_type="TRet",
    )
    table.index += 1  ## start count at 1
    sector_row_colors = {
        "ENERGY": "magicmint",
        "COMMUNICATIONS": "opal",
        "CONSUMER_NON_CYCLICAL": "sage",
    }
    sector_locs = {}
    for sector, color in sector_row_colors.items():
        bottom_sector_names = {
            "COMMUNICATIONS": {
                "CABLE_SATELLITE",
                "MEDIA_CONTENT",
                "TELECOM_SATELLITE",
                "TELECOM_WIRELESS",
            },
            "ENERGY": {
                "ENERGY_EXPLORATION_AND_PRODUCTION",
                "GAS_DISTRIBUTION",
                "OIL_REFINING_AND_MARKETING",
            },
            "CONSUMER_NON_CYCLICAL": {
                "BEVERAGE",
                "FOOD",
                "PERSONAL_AND_HOUSEHOLD_PRODUCTS",
                "HEALTH_FACILITIES",
                "MANAGED_CARE",
                "PHARMA",
            },
        }[sector]

        locs = tuple(
            table.loc[table["raw_sector"].isin(bottom_sector_names)].index
        )
        if locs:
            sector_locs[locs] = color

    index_loc = tuple(table[table["TopLevelSector"] == "-"].index)
    table.drop(["raw_sector", "TopLevelSector"], axis=1, inplace=True)
    rsquared = mod.rsquared()
    rsquared_pctile = mod.rsquared(pctile=True)
    ordinal = get_ordinal(rsquared_pctile)
    fnote = (
        f"Performance since {mod.predict_from_date:%m/%d/%Y}. "
        f"Model $R^2$ was {rsquared:.2f}, "
        f"the {rsquared_pctile}{ordinal} \%tile of model history."
    )
    doc.add_table(
        table,
        prec=mod.table_prec(table),
        col_fmt="llc|cc|ccc",
        caption="HY Beta-Adjusted 1M Sector Performance",
        table_notes=fnote,
        table_notes_justification="c",
        font_size="scriptsize",
        multi_row_header=True,
        row_font={index_loc: "\\bfseries"},
        row_color=sector_locs,
        gradient_cell_col="Out*Perform",
        gradient_cell_kws={"cmax": "steelblue", "cmin": "firebrick"},
    )


if __name__ == "__main__":
    fid = "HY_Sector_Performance"
    update_sector_performance(fid)
