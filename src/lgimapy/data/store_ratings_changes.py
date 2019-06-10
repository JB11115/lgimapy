import datetime as dt

import pandas as pd

from lgimapy.index import IndexBuilder
from lgimapy.utils import root, Time


def main():
    """
    Build DataFrames for each rating agency of rating changes.
    """
    fid = "data/{}_rating_changes.csv"

    ixb = IndexBuilder()

    agencies = ["SP", "Moody"]
    with Time("t") as t:
        ixb.load(local=True)
        t.split("  load")
        ix = ixb.build(rating=("AAA", "D"), start="1/1/2018", end=None)
        t.split("  ix build")

        # rating_df = ix.get_value_history('SPRating')
        # for i, col in enumerate(list(rating_df)):
        #     vals = rating_df.iloc[:, i].dropna()
        #     v0, v1 = vals.iloc[0], vals.iloc[-1]
        #     if v0 != v1:
        #         print(col, v0, v1)

        for agency in agencies:
            change_df = ix.find_rating_changes(agency)
            change_df.to_csv(root(fid.format(agency)))
            t.split(f"  {agency} stored")


if __name__ == "__main__":
    main()
