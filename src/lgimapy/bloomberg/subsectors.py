import json

import numpy as np

from lgimapy.utils import load_json, dump_json
from lgimapy.bloomberg import bdp


def update_subsector_json():
    """Update bloomberg subsector json file for all cusips."""
    fid = "cusip_bloomberg_subsectors"
    subsectors = load_json(fid)
    scrape_bloomberg_subsectors(list(subsectors.keys()))


def get_bloomberg_subsector(cusips):
    """
    Get bloomberg subsector for list of cusips. First attempts
    to use saved `cusip_bloomberg_subsectors.json` file, updating the file
    for any cusip which is unsuccesful.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for subsectors.

    Returns
    -------
    List[str]:
        List of bloomberg subsectors matching input cusips.
    """
    if isinstance(cusips, str):
        cusips = [cusips]  # convert to list

    # Find any cusips which are not in saved file.
    fid = "cusip_bloomberg_subsectors"
    subsectors = load_json(fid, empty_on_error=True)
    missing = []
    for c in list(set(cusips)):
        if c not in subsectors:
            missing.append(c)

    # Scrape missing cusips and reload file.
    if missing:
        scrape_bloomberg_subsectors(missing)
        subsectors = load_json(fid)

    return [subsectors[c] for c in cusips]


def scrape_bloomberg_subsectors(cusips):
    """
    Scrapes specified cusips and updates
    `cusip_bloomberg_subsectors.json` with scraped
    cusips and their subsectors.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for subsectors.
    """
    # Build dict of cusip: scraped subsectors.
    subsectors = bdp(cusips, "Corp", field="INDUSTRY_SUBGROUP")
    scraped_subsectors = {c: s for (c, s) in zip(cusips, subsectors)}

    # Load `cusip_bloomberg_subsectors.json`, add new cusips, and save.
    fid = "cusip_bloomberg_subsectors"
    subsectors = load_json(fid, empty_on_error=True)
    dump_json({**subsectors, **scraped_subsectors}, fid)
