import datetime as dt
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

    for a in agencies:
        ovrd = None
        fields = [f"{a.upper()}_EFF_DT", f"RTG_{a.upper()}"]
        while True:
            # Find current rating and its effective date.
            df = bdp(cusip, "Corp", fields, ovrd=ovrd)

            # Clean date and rating (remove Bloomberg Watch tag).
            try:
                date = df[fields[0]][0]
                rating = df[fields[1]][0].split(" *")[0].strip("u")
            except AttributeError:
                # NaN encountered, scrape finished.
                break

            # Append results to cusip dict.
            rating_changes[a]["date"].append(date.strftime("%m/%d/%Y"))
            rating_changes[a]["rating"].append(rating)

            # Find new date and repeat.
            new_date = (date - dt.timedelta(1)).strftime("%Y%m%d")
            ovrd = {"RATING_AS_OF_DATE_OVERRIDE": new_date}

    # Load `ratings_changes.json`, add new cusips, and save.
    fid = "ratings_changes"
    ratings = load_json(fid, empty_on_error=True)
    ratings[cusip] = rating_changes
    dump_json(ratings, fid)


def main():
    import pandas as pd
    from lgimapy.data import Database, Index, concat_index_dfs

    db = Database()
    # db.load_market_data(local=True)
    df_list = []
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

    ratings = load_json("ratings_changes")
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
    main()
