import json

import numpy as np

from lgimapy.utils import root
from lgimapy.bloomberg import bdp


def id_to_cusip(ids):
    """
    Get bloomberg cusip for list of member IDs. First attempts
    to use saved `id_to_cusip.json` file, updating the file
    for any id which is unsuccesful.

    Parameters
    ----------
    ids: str, List[str].
        Member ID(s) to scrape Bloomberg for cusips.

    Returns
    -------
    List[str]:
        List of Bloomberg cusips matching input member IDs.
    """
    if isinstance(ids, str):
        ids = [ids]  # convert to list

    # Find any ids which are not in saved file.
    fid = "id_to_cusip"
    cusips = load_json(fid, empty_on_error=True)
    missing = []
    for i in list(set(ids)):
        if i not in cusips:
            missing.append(i)

    # Scrape missing ids and reload file.
    if missing:
        scrape_id_to_cusip(missing)
        cusips = load_json(fid)

    return [cusips[i] for i in ids]


def scrape_id_to_cusip(ids):
    """
    Scrapes specified cusips and updates
    `id_to_cusip.json` with scraped ids
    and their cusips.

    Parameters
    ----------
    ids: str, List[str].
        Member ID(s) to search Bloomberg for cuisps.
    """
    # Build dict of id: scraped cusips.
    scraped_cusips = bdp(ids, "Corp", field="ID_CUSIP")
    scraped_cusips = {i: c for (i, c) in zip(ids, scraped_cusips)}

    # Load `id_to_cusip.json`, add new ids, and save.
    fid = "id_to_cusip"
    cusips = load_json(fid, empty_on_error=True)
    dump_json({**cusips, **scraped_cusips}, fid)
