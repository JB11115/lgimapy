import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root


def update_trade_dates(dates=None):
    """
    Update `trade_dates.csv` file with all trade dates
    and holidays.

    Parameters
    ----------
    dates: List[datetime].
        List of all trade dates available in DataMart.
    """
    # Create DataFrame with trade date index and boolean value for holiday.
    fid = root("data/trade_dates.csv")
    trade_dates = Database().load_trade_dates() if dates is None else dates
    df = pd.DataFrame(index=trade_dates)
    df["holiday"] = 0

    try:
        saved_df = pd.read_csv(
            fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )
    except FileNotFoundError:
        # No file, find all holidays.
        start = pd.to_datetime("1/2/1998")
        holidays = find_holidays(trade_dates, start_date=start)
    else:
        # Succesfully loaded file, find holidays since last save.
        saved_holidays = list(saved_df[saved_df["holiday"] == 1].index)
        new_holidays = find_holidays(trade_dates, start_date=saved_df.index[-3])
        holidays = list(set(saved_holidays + new_holidays))
        if new_holidays:
            # Print new holidays to the screen.
            fout = [f"  {h.strftime('%m/%d/%Y')}" for h in new_holidays]
            print("New Holidays:")
            print(*fout, sep="\n")

    # Set boolean value to 1 for holidays and save to .csv file.
    df.loc[df.index.isin(holidays), "holiday"] = 1
    df.to_csv(fid)
    print("Updated trade dates.")


def find_holidays(trade_dates, start_date=None):
    """List[datetime]: Find all holidays after given start date."""
    db = Database()
    last_date = trade_dates[-1]
    holidays = []
    for year in range(start_date.year, last_date.year + 1):
        # Load either full year or correct portion of the year.
        start = f"12/31/{year-1}" if year != start_date.year else start_date
        end = f"1/5/{year+1}" if year != last_date.year else last_date
        db.load_market_data(start=start, end=end)
        ix = db.build_market_index(in_returns_index=True)

        # Find days where price, duration, and spread don't change
        # for at least 100 cusips. These are assumed to be holidays.
        # **On a non-holiday the number is never greater than 6.
        cols = ["CleanPrice", "OAD", "OAS"]
        dupe_locs = ix.df.duplicated(subset=cols, keep="first")
        df = ix.df[dupe_locs]
        counts = df["Date"].value_counts()
        # Add new holidays to holiday list.
        holidays.extend(list(counts[counts > 100].index))

    return holidays


if __name__ == "__main__":
    update_trade_dates()
