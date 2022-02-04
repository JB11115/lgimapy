from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.bloomberg import bdp
from lgimapy.data import Database, Index
from lgimapy.latex import Document

# %%
def get_fallen_angel_impact():
    db = Database()
    db.load_market_data()

    sectors = [
        "CHEMICALS",
        "METALS_AND_MINING",
        "COMMUNICATIONS",
        "AUTOMOTIVE",
        "RETAILERS",
        "CONSUMER_CYCLICAL_EX_AUTOS_RETAILERS",
        "HEALTHCARE_PHARMA",
        "FOOD_AND_BEVERAGE",
        "CONSUMER_NON_CYCLICAL_OTHER",
        "ENERGY",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "BANKS",
        "INSURANCE",
        "REITS",
        "UTILITY",
    ]
    ig_ix = db.build_market_index(in_stats_index=True)
    bbb_ix = ig_ix.subset(rating=("BBB", None))
    near_fa_ix = get_near_fallen_angels(bbb_ix)
    hy_ix = db.build_market_index(in_H0A0_index=True)

    ig_mv = ig_ix.total_value().iloc[0]
    hy_mv = hy_ix.total_value().iloc[0]

    d = defaultdict(list)
    for sector in sectors:
        kwargs = db.index_kwargs(
            sector, unused_constraints=["in_stats_index", "OAS"]
        )
        # Find value of sector that is IG near a fallen angel.
        near_fa_sector = near_fa_ix.subset(**kwargs)
        if len(near_fa_sector.df):
            near_fa_sector_mv = near_fa_sector.total_value().iloc[0]
        else:
            near_fa_sector_mv = 0
        # Find value of sector that is HY.
        hy_sector = hy_ix.subset(**kwargs)
        if len(hy_sector.df):
            hy_sector_mv = hy_sector.total_value().iloc[0]
            potential_increase = near_fa_sector_mv / hy_sector_mv
        else:
            hy_sector_mv = 0
            potential_increase = np.inf
        d["Sector"].append(kwargs["name"])
        d["PFA MV*(\\$B)"].append(near_fa_sector_mv / 1e3)
        d["PFA*% of IG"].append(near_fa_sector_mv / ig_mv)
        d["HY MV*(\\$B)"].append(hy_sector_mv / 1e3)
        d["Sector*% of HY"].append(hy_sector_mv / hy_mv)
        d["Potential*Increase"].append(potential_increase)

    return pd.DataFrame(d)


def get_near_fallen_angels(ix):
    """
    Scrape rating outlooks from the major agencies and subset
    :class:`Index` to bonds that are on negative outlook at a
    rating agency where a downgrade from said agency would
    result in a transition to the HY index.
    """
    db = Database()
    rating_agencies = ["SP", "Moody", "Fitch"]
    for agency in rating_agencies:
        bbg_agency = {"Moody": "MDY"}.get(agency, agency).upper()
        outlook = bdp(ix.cusips, "Corp", f"RTG_{bbg_agency}_OUTLOOK").squeeze()
        ix.df[f"{agency}Outlook"] = (
            outlook.str.contains("NEG").fillna(0).astype(int)
        )
        ix.df[f"{agency}NumericRating"] = db._get_numeric_ratings(
            ix.df, [f"{agency}Rating"]
        )

    rating_cols = [f"{agency}NumericRating" for agency in rating_agencies]

    # %%
    potential_fa_d = defaultdict(lambda: np.zeros(len(ix.df), dtype=bool))
    for i, ratings in ix.df[rating_cols].reset_index(drop=True).iterrows():
        sorted_ratings = ratings.sort_values().dropna()
        # Find bonds that are one downgrade away from becoming HY.
        # - rated by one agency and is BBB-
        # - rated by two agencies and the lower is BBB-
        # - rated by three agencies and the middle is BBB-
        if (
            (len(sorted_ratings) == 1 and sorted_ratings.iloc[0] == 10)
            or (len(sorted_ratings) == 2 and sorted_ratings.iloc[-1] == 10)
            or (len(sorted_ratings) == 3 and sorted_ratings.iloc[1] == 10)
        ):
            for agency_col, rating in sorted_ratings.items():
                if rating == 10:
                    # Agency has BBB- rating.
                    agency = agency_col.split("Num")[0]
                    if ix.df[f"{agency}Outlook"].iloc[i]:
                        # Agency has a negative outlook.
                        is_near_fa[i] = 1

    near_fa_df = ix.df.loc[is_near_fa.astype(bool)].copy()
    return Index(near_fa_df)


# %%


def main():
    df = get_fallen_angel_impact()
    # %%
    fid = "Potential_Fallen_Angel_Impact"
    doc = Document(fid, path="reports/HY")
    prec = {col: "1%" for col in df.columns[1:]}
    for key in prec.keys():
        if "$" in key:
            prec[key] = "0f"
    cap = "Potential Fallen Angel Impact"
    footnote = """
        Potential fallen angel (PFA) for the IG index is defined as
        any issuer that currently has a negative outlook from a
        rating agency where a downgrade from said agency
        would cause the issuer to become a fallen angel.
        """
    # edit = doc.add_subfigures(1, widths=[0.75])
    # with doc.start_edit(edit):
    doc.add_table(
        df.sort_values("Potential*Increase", ascending=False),
        caption=cap,
        table_notes=footnote,
        table_notes_justification="l",
        adjust=True,
        hide_index=True,
        multi_row_header=True,
        col_fmt="ll|cr|cr|r",
        prec=prec,
        gradient_cell_col="Potential*Increase",
        gradient_cell_kws={"vmax": 1},
    )

    doc.save()
