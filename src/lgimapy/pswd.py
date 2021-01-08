import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    args = parse_args()
    # Load passwords.
    pswd_fid = Path("P:/pswd.json")
    with open(pswd_fid, "r") as fid:
        pswd_d = json.load(fid)

    # Print specified username and password.
    website = str(args.website).upper()
    usr_pswd = pswd_d[website]
    print(website)
    if args.print:
        for key, val in usr_pswd.items():
            print(f"  {key}: {val}")
        print()

    # Add username, password, token, etc. to clipboard.
    for key, val in usr_pswd.items():
        if key == "desc":
            continue
        try:
            pd.Series([val]).to_clipboard(index=False, header=False)
        except KeyError:
            print(f"{key} does not exist in entry for {website}")
        else:
            print(f"{key} has been copied to the clipboard.", end=" ")
            input(f"Press any key to continue.")


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("website", help="Name of Website")
    parser.add_argument(
        "-p", '--print', action='store_true', help="Print values."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
