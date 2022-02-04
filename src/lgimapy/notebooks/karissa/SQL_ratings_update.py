import pandas as pd
from tqdm import tqdm

from lgimapy.utils import load_json, root
from lgimapy.data import Database


def main():
    filename = root("data/SQL_ratings_updates.txt")
    ratings_json = load_json("ratings_changes")

    # Get 9 digit cusips to build updates for build update file and save.
    cusips = [cusip for cusip in ratings_json if len(cusip) == 9]
    updates = "\n".join(make_statements(cusips, ratings_json))
    with open(filename, "w") as fid:
        fid.write(updates)


def convert_json_to_df(cusip, ratings_dict):
    """Convert json format for each cusip to a DataFrame."""
    agencies = ["SP", "Moody", "Fitch"]
    series_list = []
    for agency in agencies:
        try:
            dates = ratings_dict[cusip][agency]["date"]
            ratings = ratings_dict[cusip][agency]["rating"]
            series_list.append(pd.Series(ratings, index=dates, name=agency))
        except KeyError:
            # Make emtpy column for agency.
            series_list.append(pd.Series([], index=[], name=agency))

    # Combine all agencies, sort by date, and then fill missing
    # values by a forward fill first, followed by filling
    # missing values with a `NR` classification.
    df = pd.concat(series_list, axis=1, sort=False)[agencies]
    df.index = pd.to_datetime(df.index)
    df = (
        df[df.index <= pd.to_datetime("10/10/2019")]
        .sort_index()
        .fillna(method="ffill")
        .fillna("NR")
    )
    df.index = df.index.strftime("%Y%m%d")  # formated for SQL
    return df


def make_statements(cusips, ratings):
    """
    Make generator of single SQL update statements for each
    specified cusip.

    Yields
    ------
    statement: str
        Single line SQL update.
    """
    DATE, SP, MOODY, FITCH = range(4)
    statement = (
        "update ia set SPRating = '{}', FitchRating = '{}', "
        "MoodyRating = '{}' from LGIMADatamart.DBO.instrumentanalytics ia "
        "inner join LGIMADatamart.DBO.diminstrument"
        " i on ia.instrumentkey = i.instrumentkey where "
        "cusip = '{}' and effectivedatekey{}{}{}{}{};"
    )
    edk = " and effectivedatekey"
    for cusip in tqdm(cusips):
        df = convert_json_to_df(cusip, ratings)
        if not len(df):
            continue  # No history to update
        for i, row in enumerate(df.itertuples()):
            date = row[DATE]
            cur_ratings = (row[SP], row[FITCH], row[MOODY], cusip)
            if i == 0:
                # First rating update.
                yield statement.format(*cur_ratings, " < ", date, "", "", "")
            else:
                # Intermediate rating update.
                yield statement.format(
                    *prev_ratings, " > ", prev_date, edk, " < ", date
                )
            yield statement.format(*cur_ratings, " = ", date, "", "", "")
            # Store previous values.
            prev_date = date
            prev_ratings = cur_ratings


if __name__ == "__main__":
    main()
