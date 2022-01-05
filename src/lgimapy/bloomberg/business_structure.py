import json

import numpy as np
import pandas as pd

from lgimapy.utils import load_json, dump_json
from lgimapy.bloomberg import bdp
from lgimapy.utils import to_sql_list

# %%


def update_cusip_business_structure_json():
    """Update bloomberg business_structure json file for all cusips."""
    fid = "cusip_holdco_opco"
    business_structures = load_json(fid)
    scrape_cusip_bloomberg_business_structures(list(business_structures.keys()))


def get_cusip_business_structure(cusips):
    """
    Get bloomberg business_structure for list of cusips. First attempts
    to use saved `cusip_holdco_opco.json` file, updating the file
    for any cusip which is unsuccesful.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for business_structures.

    Returns
    -------
    List[str]:
        List of bloomberg business_structures matching input cusips.
    """
    if isinstance(cusips, str):
        cusips = [cusips]  # convert to list

    # Find any cusips which are not in saved file.
    fid = "cusip_holdco_opco"
    business_structures = load_json(fid, empty_on_error=True)
    business_structures[np.nan] = np.nan
    missing = []
    for c in list(set(cusips)):
        if c not in business_structures:
            missing.append(c)

    # Scrape missing cusips and reload file.
    if missing:
        scrape_cusip_bloomberg_business_structures(missing)
        business_structures = load_json(fid)
        business_structures[np.nan] = np.nan

    return [business_structures[c] for c in cusips]


def scrape_cusip_bloomberg_business_structures(cusips):
    """
    Scrapes specified cusips and updates
    `cusip_holdco_opco.json` with scraped
    cusips and their business_structures.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for business_structures.
    """
    # Build dict of cusip: scraped business_structures.
    field = "INDUSTRY_SUBGROUP"
    df = bdp(cusips, "Corp", fields=field)
    scraped_business_structures = {c: s for (c, s) in zip(cusips, df[field])}

    # Load `cusip_holdco_opco.json`, add new cusips, and save.
    fid = "cusip_holdco_opco"
    business_structures = load_json(fid, empty_on_error=True)
    updated_s = (
        pd.Series({**business_structures, **scraped_business_structures})
        .str.upper()
        .replace("INSUFFICIENT INFORMATION", np.nan)
        .replace("NOT APPLICABLE", np.nan)
    )
    dump_json(updated_s.to_dict(), fid)


def update_issuer_business_strucure_json(df, db):
    fid = "issuer_holdco_opco"
    business_structures = load_json(fid, empty_on_error=True)
    df = df[~df["Issuer"].isin(business_structures.keys())]

    # issuer_df.head()
    # issuer_df[issuer_df["long_comp_name"].str.contains("American")]
    # issuer_df[issuer_df["risk_parent_name"].str.contains("American")]
    parent_df[parent_df["risk_parent_name"].str.contains("American Water")]
    s = pd.Series(business_structures)
    s[s == "Eff OpCo"]
    df[df["Ticker"] == "WTRG"]
    len(df["Issuer"].unique())
    len(df["Ticker"].unique())
    df.to_csv("utes.csv")


# %%

if __name__ == "__main__":
    update_cusip_business_structure_json()


# %%
