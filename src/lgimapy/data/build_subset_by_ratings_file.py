import argparse
import datetime as dt

import pandas as pd

from lgimapy.index import IndexBuilder
from lgimapy.utils import Time


def main():
    """
    Build DataFrames for a specific column subset by
    rating numeric value and maturity. Save to
    `../data/subset_by_rating/{col}_{maturity range}.csv`.
    """
    # args = parse_args()
    col = "OAS"
    maturities = [(0, 5), (5, 10), (10, 25), (25, 32)]
    fid = f"{col}_{{}}-{{}}"

    ixb = IndexBuilder()

    with Time() as t:
        ixb.load(local=True)
        t.split("  load")
        for mats in maturities:
            ix = ixb.build(rating=("AAA", "D"), maturity=mats)
            ix.subset_value_by_rating("OAS", save=True, fid=fid.format(*mats))
            t.split(f"  {mats} stored")


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--column", help="Column name")
    parser.add_argument(
        "-u", "--update", action="store_true", help="Update File"
    )
    this_year = dt.date.today().year
    return parser.parse_args()


if __name__ == "__main__":
    main()
