import json
import numpy as np

from lgimapy.utils import root

# R specific packages are installed only if needed.
# Import statements are located in `scrape_bloomberg_subsectors`.


def update_subsector_json():
    """Update bloomberg subsector json file."""
    subsector_json_fid = root("data/bloomberg_subsectors.json")
    with open(subsector_json_fid, "r") as fid:
        old_json = json.load(fid)
    scrape_bloomberg_subsectors(list(old_json.keys()))


def get_bloomberg_subsector(cusips):
    """
    Get bloomberg subsector for list of cusips. First attempts
    to use saved `bloomberg_subsectors.json` file, updating the file
    for any cusip which is unsuccesful.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for subsectors.

    Returns
    -------
    subsectors: List[str].
        List of bloomberg subsectors matching input cusips.
    """
    if isinstance(cusips, str):
        cusips = [cusips]  # convert to list

    # Find any cusips which are not in saved file.
    subsector_json_fid = root("data/bloomberg_subsectors.json")
    with open(subsector_json_fid, "r") as fid:
        subsectors_json = json.load(fid)

    missing = []
    for c in cusips:
        if c not in subsectors_json:
            missing.append(c)

    # Scrape missing cusips and reload file.
    if missing:
        scrape_bloomberg_subsectors(missing)
        with open(subsector_json_fid, "r") as fid:
            subsectors_json = json.load(fid)

    subsectors = [subsectors_json[c] for c in cusips]
    return subsectors


def scrape_bloomberg_subsectors(cusips):
    """
    Scrapes specified cusips and updates
    `bloomberg_subsectors.json` with scraped cusips
    and their subsectors.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for subsectors.
    """
    # End function if cusips is null or empty list.
    if not cusips:
        return

    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    importr("Rblpapi")  # R package

    # Make cusips a unique list and append 'Corp' for search.
    if isinstance(cusips, str):
        cusips = [cusips]
    cusips = list(set(cusips))
    blp_searches = [f"{c} Corp" for c in cusips]
    r_blp_searches = robjects.StrVector(blp_searches)  # convert list to R

    # Scrape bloomberg subsectors in R.
    r_bloom_scrape = robjects.r(
        r"""
        function(blp_search_list) {
            con = blpConnect()
            blp_subsectors = bdp(
                securities=blp_search_list,
                fields='INDUSTRY_SUBGROUP'
                )
            return(blp_subsectors$INDUSTRY_SUBGROUP)
            }
        """
    )
    bloomberg_subsectors = np.asarray(r_bloom_scrape(r_blp_searches))

    # Build dict of cusip: scraped subsectors.
    scraped_subsectors = {
        cusip: sector for (cusip, sector) in zip(cusips, bloomberg_subsectors)
    }

    # Load `bloomberg_subsectors.json`, add new cusips, and save.
    subsector_json_fid = root("data/bloomberg_subsectors.json")
    with open(subsector_json_fid, "r") as fid:
        subsectors_json = json.load(fid)
    subsectors_json = {**subsectors_json, **scraped_subsectors}
    with open(subsector_json_fid, "w") as fid:
        json.dump(subsectors_json, fid, indent=4)
