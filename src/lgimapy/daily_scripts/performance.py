import argparse
from pathlib import Path

import pandas as pd

from lgimapy.data import Database

# %%
def main():
    args = parse_args()
    accounts = [
        "P-LD",
        "CHLD",
        "FLD",
        "GSKLC",
        "CITMC",
        "JOYLA",
        "USBGC",
    ]
    if args.print:
        for account in accounts:
            print(account)
        return

    if args.date is not None:
        date = pd.to_datetime(args.date)
    else:
        date = Database().date("today")
    print(date.strftime("%m/%d/%Y"))
    if args.estimate:
        print("\n***Estimates***")
    for account in accounts:
        if args.estimate:
            performance = estimate_performance(account, date)
        else:
            performance = read_jmgr_file(account, date)
        print(f"{account}: {performance:+.1f}")


def estimate_performance(account, date):
    # Load data for current day.
    db = Database()
    curr_df = db.load_portfolio(account=account, date=date).df.set_index("ISIN")

    # Use stats index for previous date if it's the 1st of the month.
    prev_date = db.date("yesterday", date)
    prev_universe = "stats" if prev_date == db.date("MTD", date) else "returns"
    prev_df = db.load_portfolio(
        account=account, date=prev_date, universe=prev_universe
    ).df.set_index("ISIN")

    # Estimate performance from spread change and average duration over
    # the two day period.
    oas_change = curr_df["OAS"] - prev_df["OAS"]
    oad_avg = (curr_df["OAD_Diff"] + prev_df["OAD_Diff"]) / 2
    return (-oas_change * oad_avg).sum()


def read_jmgr_file(account, date):
    path = Path("C:/blp/data")
    fids = list(path.glob(f"*Daily*{account}.{date.strftime('%Y%m%d')}.xls"))
    try:
        fid = fids[0]
    except IndexError:
        raise FileNotFoundError(f"No file found for {account}.")

    df = pd.read_excel(fid, usecols=[1, 2])
    df.columns = ["ix", "val"]
    df.set_index("ix", inplace=True)
    return df.loc["Outperformance (bps)"].squeeze()


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help="Performance Date")
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print Accounts"
    )
    parser.add_argument(
        "-e", "--estimate", action="store_true", help="Estimate Performance"
    )
    parser.set_defaults(date=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

# %%
