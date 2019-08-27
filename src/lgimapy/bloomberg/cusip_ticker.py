from lgimapy.utils import load_json, dump_json
from lgimapy.bloomberg import bdp


def get_bloomberg_ticker(cusips):
    """
    Get bloomberg ticker for list of cusips. First attempts
    to use saved `cusip_bloomberg_tickers.json` file, updating the file
    for any cusip which is unsuccesful.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for tickers.

    Returns
    -------
    List[str]:
        List of bloomberg tickers matching input cusips.
    """
    if isinstance(cusips, str):
        cusips = [cusips]  # convert to list

    # Find any cusips which are not in saved file.
    fid = "cusip_bloomberg_tickers"
    tickers = load_json(fid, empty_on_error=True)
    missing = []
    for c in list(set(cusips)):
        if c not in tickers:
            missing.append(c)

    # Scrape missing cusips and reload file.
    if missing:
        scrape_bloomberg_tickers(missing)
        tickers = load_json(fid)

    return [tickers[c] for c in cusips]


def scrape_bloomberg_tickers(cusips):
    """
    Scrapes specified cusips and updates
    `cusip_bloomberg_tickers.json` with scraped
    cusips and their tickers.

    Parameters
    ----------
    cusips: str, List[str].
        Cusip(s) to search bloomberg for tickers.
    """
    # Build dict of cusip: scraped tickers.
    field = "TICKER"
    df = bdp(cusips, "Corp", fields=field)
    scraped_tickers = {c: s for (c, s) in zip(cusips, df[field])}

    # Load `cusip_bloomberg_tickers.json`, add new cusips, and save.
    fid = "cusip_bloomberg_tickers"
    tickers = load_json(fid, empty_on_error=True)
    dump_json({**tickers, **scraped_tickers}, fid)
