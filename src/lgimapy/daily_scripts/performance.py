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

    db = Database()
    if args.date is not None:
        date = pd.to_datetime(args.date)
    else:
        date = db.date("today")
    print(date.strftime("%m/%d/%Y"))

    if args.estimate:
        print("\n***Estimates***  (YC)")

    for account in accounts:
        if args.estimate:
            port = db.load_portfolio(account=account, date=date)
            tret = 1e4 * port.total_return()
            yc = 1e4 * port.total_return("tsy")
        else:
            tret, yc = read_jmgr_file(account, date)

        print(f"{account: <6}:  {tret:+.1f}   ({yc:+.1f})")


def read_jmgr_file(account, date):
    path = Path("C:/blp/data")
    fids = list(path.glob(f"*Daily*{account}.{date.strftime('%Y%m%d')}.xls"))
    try:
        fid = fids[0]
    except IndexError:
        raise FileNotFoundError(f"No file found for {account}.")

    df = pd.read_excel(fid, usecols=[1, 2, 4, 8]).head(15)
    df.columns = ["total_idx", "total_val", "detail_idx", "detail_val"]
    tret = df.set_index("total_idx").loc["Outperformance (bps)", "total_val"]
    yc = df.set_index("detail_idx").loc["Yield Curve", "detail_val"]
    return tret, yc


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
