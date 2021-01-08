from lgimapy.utils import root, load_json, dump_json
from lgimapy.bloomberg import bdp


def id_to_cusip(ids):
    """
    Get bloomberg cusip for list of member IDs. First attempts
    to use saved `id_to_cusip.json` file, updating the file
    for any id which is unsuccesful.

    Parameters
    ----------
    ids: str, List[str].
        Member ID(s) to scrape Bloomberg for CUSIPs.

    Returns
    -------
    List[str]:
        List of Bloomberg CUSIPs matching input member IDs.
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
        scrape_ids(missing, "CUSIP")
        cusips = load_json(fid)

    return [cusips[i] for i in ids]


def id_to_isin(ids):
    """
    Get bloomberg cusip for list of member IDs. First attempts
    to use saved `id_to_isin.json` file, updating the file
    for any id which is unsuccesful.

    Parameters
    ----------
    ids: str, List[str].
        Member ID(s) to scrape Bloomberg for ISINs.

    Returns
    -------
    List[str]:
        List of Bloomberg ISINs matching input member IDs.
    """
    if isinstance(ids, str):
        ids = [ids]  # convert to list

    # Find any ids which are not in saved file.
    fid = "id_to_isin"
    isins = load_json(fid, empty_on_error=True)
    missing = []
    for i in list(set(ids)):
        if i not in isins:
            missing.append(i)

    # Scrape missing ids and reload file.
    if missing:
        scrape_ids(missing, "ISIN")
        isins = load_json(fid)

    return [isins[i] for i in ids]


def scrape_ids(ids, new_id):
    """
    Scrapes specified cusips and updates
    `id_to_cusip.json` with scraped ids
    and their cusips.

    Parameters
    ----------
    ids: str, List[str].
        Member ID(s) to search Bloomberg for cuisps.
    new_id: 'isin' or 'cusip'
        New identification format.
    """
    # Build dict of id: scraped cusips.
    field = f"ID_{new_id.upper()}"
    df = bdp(ids, "Corp", fields=field)
    scraped_ids = {i: c for (i, c) in zip(ids, df[field])}

    # Load `id_to_cusip.json`, add new ids, and save.
    fid = f"id_to_{new_id.lower()}"
    saved_ids = load_json(fid, empty_on_error=True)
    dump_json({**saved_ids, **scraped_ids}, fid)
