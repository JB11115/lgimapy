import datetime as dt
from collections import defaultdict

from blpapi import NotFoundException
from tqdm import tqdm

from lgimapy.bloomberg import bds
from lgimapy.utils import load_json, dump_json


def scrape_amount_outstanding(cusip):
    """
    Scrape amount outstanding from Bloomberg
    for a specified cusip and update
    `amount_outstanding.json` with result.

    Parameters
    ----------
    cusip: str
        Cusip to scrape ratings changes for.
    """
    # Build empty rating dict.
    d = defaultdict(list)
    cusips_w_no_history = []

    # Find current rating and its effective date.
    df = bds(cusip, "Corp", "AMOUNT_OUTSTANDING_HISTORY")

    # Clean date and rating (remove Bloomberg Watch tag).
    try:
        date = df[fields[0]][0]
        rating = df[fields[1]][0].split(" *")[0].strip("u")
    except AttributeError:
        # NaN encountered, scrape finished.
        break

    # Append results to cusip dict.
    rating_changes[a]["date"].append(date.strftime("%m/%d/%Y"))
    rating_changes[a]["amount_outstanding"].append(rating)

    # Load `ratings_changes.json`, add new cusips, and save.
    fid = "amount_outstanding"
    ratings = load_json(fid, empty_on_error=True)
    ratings[cusip] = rating_changes
    dump_json(ratings, fid)


def main():
    import pandas as pd
    from lgimapy.data import Database, Index, concat_index_dfs

    db = Database()
    # db.load_market_data(local=True)
    df_list = [][""]
    for y in tqdm(range(2004, 2020)):
        for m in [1, 4, 8]:
            d = 1
            while True:
                date = f"{m}/{d}/{y}"
                print(date)
                try:
                    df = db.load_market_data(
                        start=date, end=date, clean=False, ret_df=True
                    )
                except ValueError:
                    d += 1
                else:
                    break
                if d > 8:
                    break
            df_list.append(df)

    df = pd.concat(df_list, join="outer", sort=False)
    cusips = list(set(df["CUSIP"]))

    del db
    # del ix

    ratings = load_json("amount_outstanding")
    cusips_to_scrape = []
    for cusip in cusips:
        if cusip not in ratings:
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
    cusips = ["172967GD7"]
    cusip = cusips[0]
    for cusip in cusips:
        try:
            df = bds(cusip, "Corp", "AMOUNT_OUTSTANDING_HISTORY")
            print(df)
            break
        except NotFoundException:
            continue
