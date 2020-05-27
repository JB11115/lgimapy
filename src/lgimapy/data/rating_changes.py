import pandas as pd
from tqdm import tqdm

from lgimapy.bloomberg import get_bloomberg_subsector
from lgimapy.data import Database, clean_dtypes, convert_sectors_to_fin_flags
from lgimapy.utils import root


# %%
def update_rating_changes():
    fid = root("data/rating_changes.csv")

    db = Database()
    saved_df, last_saved_date = read_saved_data(fid)
    dates = db.trade_dates(start=last_saved_date)
    curr_date = dates[-1]  # init in case loop doesn't run
    df_list = [saved_df]

    for i, curr_date in enumerate(dates[1:]):
        if i == 100:
            break
        prev_date = dates[i]
        df = db.load_market_data(
            start=prev_date, end=curr_date, clean=False, ret_df=True
        )
        # Remove holidays.
        df = df[df["Date"].isin({prev_date, curr_date})].copy()
        df_list.append(get_rating_changes(df, db))

    # Combine saved data with newly computed data, and append
    # blank row with last scraped date.
    updated_df = pd.concat(df_list, axis=0, sort=False).append(
        pd.Series(dtype="float64"), ignore_index=True
    )
    updated_df.iloc[-1, 1] = curr_date
    updated_df.to_csv(fid)


def read_saved_data(fid):
    try:
        df = pd.read_csv(fid, index_col=0)
    except FileNotFoundError:
        df = None
        last_date = Database().date("MARKET_START")
    else:
        # Get last scraped date and remove blank row.
        last_date = df["Date_NEW"].iloc[-1]
        df = df.iloc[:-1, :]
        for date_col in ["Date_PREV", "Date_NEW", "MaturityDate"]:
            df[date_col] = pd.to_datetime(
                df[date_col], format="%Y-%m-%d", errors="coerce"
            )
    return df, last_date


def rating_change_locs(df, col):
    """
    Return locations where rating data is available for
    both days and rating change occured for given column.
    """
    return (
        (~df[f"{col}_PREV"].isna())
        & (~df[f"{col}_NEW"].isna())
        & (df[f"{col}_PREV"] != df[f"{col}_NEW"])
    )


def get_rating_changes(df, db):
    """
    Find rating changes over two day period, and return
    DataFrame with pertinent columns.
    """
    # Get middle-or-lower numeric rating for each cusip
    # and convert ratings to numeric values.
    rating_cols = ["MoodyRating", "SPRating", "FitchRating"]
    df["NumericRating"] = db._get_numeric_ratings(df, rating_cols)
    for col in rating_cols:
        df[col] = db._get_numeric_ratings(df, [col])

    # Define maturites as number of years until maturity.
    day = "timedelta64[D]"
    df["MaturityYears"] = (df["MaturityDate"] - df["Date"]).astype(day) / 365

    # Calculate market value.
    df["AmountOutstanding"] /= 1e6
    df.eval("DirtyPrice = CleanPrice + AccruedInterest", inplace=True)
    df.eval("MarketValue = AmountOutstanding * DirtyPrice / 100", inplace=True)

    # Get subsector and financial flag.
    df["Subsector"] = get_bloomberg_subsector(df["CUSIP"].values)
    df["FinancialFlag"] = convert_sectors_to_fin_flags(df["Sector"])

    # Split data into separate days, and keep intersetcion of bonds
    # which have data for both days.
    df_prev = df[df["Date"] == df["Date"].iloc[0]]
    df_curr = df[df["Date"] == df["Date"].iloc[-1]]
    cusips = set(df_prev["CUSIP"]) & set(df_curr["CUSIP"])

    # Combine dates and subset to dates with rating changes.
    comb_df = pd.merge(df_prev, df_curr, on="CUSIP", suffixes=("_PREV", "_NEW"))
    col_map = {
        "Date_PREV": "Date_PREV",
        "Date_NEW": "Date_NEW",
        "CUSIP": "CUSIP",
        "Ticker_NEW": "Ticker",
        "Issuer_NEW": "Issuer",
        "Sector_NEW": "Sector",
        "Subsector_NEW": "Subsector",
        "MaturityDate_NEW": "MaturityDate",
        "MaturityYears_NEW": "MaturityYears",
        "USCreditReturnsFlag_PREV": "USCreditReturnsFlag",
        "USHYReturnsFlag_PREV": "USHYReturnsFlag",
        "FinancialFlag_PREV": "FinancialFlag",
        "MarketValue_PREV": "MarketValue",
        "NumericRating_PREV": "NumericRating_PREV",
        "NumericRating_NEW": "NumericRating_NEW",
        "MoodyRating_PREV": "MoodyRating_PREV",
        "MoodyRating_NEW": "MoodyRating_NEW",
        "SPRating_PREV": "SPRating_PREV",
        "SPRating_NEW": "SPRating_NEW",
        "FitchRating_PREV": "FitchRating_PREV",
        "FitchRating_NEW": "FitchRating_NEW",
    }
    cols = list(col_map.keys())
    rating_change_df = comb_df.loc[
        rating_change_locs(comb_df, "NumericRating")
        | rating_change_locs(comb_df, "MoodyRating")
        | rating_change_locs(comb_df, "SPRating")
        | rating_change_locs(comb_df, "FitchRating"),
        cols,
    ].rename(columns=col_map)

    # Calculate change in each rating. Use inverse sign (old - new),
    # as opposed to (new - old), in order to make the change
    # negative for downgrades and positive for upgrades.
    df_columns = list(rating_change_df.columns)
    rating_cols.append("NumericRating")
    for col, ix in zip(rating_cols, [-4, -2, 999, -9]):
        rating_change_df[f"{col}_CHANGE"] = (
            rating_change_df[f"{col}_PREV"] - rating_change_df[f"{col}_NEW"]
        )
        # Insert new columns by respective PREV and NEW values.
        df_columns.insert(ix, f"{col}_CHANGE")

    return rating_change_df[df_columns]


if __name__ == "__main__":
    update_rating_changes()
