import datetime as dt
import json
from collections import defaultdict

from lgimapy.bloomberg import bdp
from lgimapy.utils import load_json, dump_json
from tqdm import tqdm


def scrape_rating_changes(cusip):
    """
    Scrape effective dates and rating changes from
    Bloomberg for a specified cusip and update
    `rating_changes.json` with result.

    Parameters
    ----------
    cusip: str
        Cusip to scrape ratings changes for.
    """
    # Build empty rating dict.
    agencies = ["SP", "Moody", "Fitch"]
    rating_changes = {a: defaultdict(list) for a in agencies}

    # Format Bloomberg field calls.
    date_field = {a: f"{a.upper()}_EFF_DT" for a in agencies}
    rating_field = {a: f"RTG_{a.upper()}" for a in agencies}

    for a in agencies:
        rating = "TEMP"
        ovrd = None
        while True:
            # Find current rating and its effective date.
            rating = bdp(cusip, "Corp", rating_field[a], ovrd=ovrd)[0]
            # Clean rating (remove Bloomberg Watch tag).
            rating = rating.split(" *")[0].strip("u")
            if rating == "":
                break  # All ratings have been scraped.

            eff_date = bdp(cusip, "Corp", date_field[a], ovrd, date=True)[0]
            eff_date = dt.datetime.strptime(str(eff_date), "%Y%m%d")

            # Append results to cusip dict.
            rating_changes[a]["date"].append(eff_date.strftime("%m/%d/%Y"))
            rating_changes[a]["rating"].append(rating)

            # Find new date and repeat.
            new_date = (eff_date - dt.timedelta(1)).strftime("%m/%d/%Y")
            ovrd = {"RATING_AS_OF_DATE_OVERRIDE": new_date}

    # Load `ratings_changes.json`, add new cusips, and save.
    fid = "ratings_changes.json"
    ratings = load_json(fid, empty_on_error=True)
    ratings[cusip] = rating_changes
    dump_json(ratings, fid)


def main():
    from lgimapy.index import IndexBuilder

    ixb = IndexBuilder()
    ixb.load(local=True)
    ix = ixb.build(rating="IG", start="1/1/2018")
    cusips = list(ix.cusips)
    del ixb
    del ix

    rating_changes_json = root("data/ratings_changes.json")
    with open(rating_changes_json, "r") as fid:
        rating_changes_dict = json.load(fid)

    cusips_to_scrape = []
    for cusip in cusips:
        if cusip not in rating_changes_dict:
            cusips_to_scrape.append(cusip)

    for cusip in tqdm(cusips_to_scrape):
        try:
            scrape_rating_changes(cusip)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            print(cusip, "failed")
            continue


if __name__ == "__main__":
    main()
