from lgimapy.utils import load_json, dump_json
from lgimapy.bloomberg import bdp


def get_issue_price(cusips):
    """
    Get issue price for list of cusips. First attempts
    to use saved `cusip_issue_price.json` file, updating the file
    for any cusip which is unsuccesful.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to retrieve issue price for.

    Returns
    -------
    List[str]:
        List of issue prices matching input cusips.
    """
    if isinstance(cusips, str):
        cusips = [cusips]  # convert to list

    # Find any cusips which are not in saved file.
    fid = "cusip_issue_price"
    issue_prices = load_json(fid, empty_on_error=True)
    missing = []
    for c in list(set(cusips)):
        if c not in issue_prices:
            missing.append(c)

    # Scrape missing cusips and reload file.
    if missing:
        scrape_issue_prices(missing)
        issue_prices = load_json(fid)

    return [issue_prices[c] for c in cusips]


def scrape_issue_prices(cusips):
    """
    Scrapes specified cusips and updates
    `cusip_issue_price.json` with scraped
    cusips and their issue prices.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to retrieve issue prices for.
    """
    # Build dict of cusip: scraped issue_prices.
    field = "ISSUE_PX"
    df = bdp(cusips, "Corp", fields=field)
    scraped_issue_prices = {c: s for (c, s) in zip(cusips, df[field])}

    # Load `cusip_issue_price.json`, add new cusips, and save.
    fid = "cusip_issue_price"
    issue_prices = load_json(fid, empty_on_error=True)
    dump_json({**issue_prices, **scraped_issue_prices}, fid)
