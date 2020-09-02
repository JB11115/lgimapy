import argparse
from pathlib import Path

import pandas as pd

from lgimapy.data import Database

# %%
def main():
    args = parse_args()
    accounts = [
        "P-LD",
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

    path = Path("C:/blp/data")
    for account in accounts:
        fids = list(
            path.glob(f"*Daily*{account}.{date.strftime('%Y%m%d')}.xls")
        )
        try:
            fid = fids[0]
        except IndexError:
            raise FileNotFoundError(f"No file found for {account}.")

        df = pd.read_excel(fid, usecols=[1, 2])
        df.columns = ["ix", "val"]
        df.set_index("ix", inplace=True)
        performance = df.loc["Outperformance (bps)"].squeeze()
        print(f"{account}: {performance:+.1f}")


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help="Performance Date")
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print Accounts"
    )
    parser.set_defaults(date=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
