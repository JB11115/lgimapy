import argparse
import json
import pandas as pd

from index_functions import IndexBuilder


def main():
    args = parse_args()
    print("start", args.start)
    print("end", args.end)
    print("indexes", args.indexes)

    ix = IndexBuilder()

    ix.load(args.start, args.end)


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="Start Date")
    parser.add_argument("-e", "--end", help="End Date")
    parser.add_argument("-i", "--indexes", help="Indexes to Build")
    parser.set_defaults(start=None, end=None, indexes=None)
    args = parser.parse_args()

    # Verify all indexes exist before attempting to load/save data.
    indexes_fid = "../data/indexes.json"
    with open(indexes_fid, "r") as fid:
        indexes_json = json.load(fid)
    if isinstance(args.indexes, str):
        index_names = args.indexes.split()
        for ix in index_names:
            if ix not in indexes_json:
                raise ValueError(f'"{ix}" is not a valid index')
        args.indexes = index_names
    else:
        args.indexes = list(indexes_json.keys())

    return args


if __name__ == "__main__":
    main()
