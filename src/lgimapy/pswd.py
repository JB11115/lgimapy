import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from fuzzywuzzy import process

# %%
def main():
    args = parse_args()
    # Load passwords.
    if sys.platform == "linux":
        pswd_fid = Path.home() / "pswd.json"
    elif sys.platform == "win32":
        pswd_fid = Path("P:/pswd.json")
    else:
        raise OSError(f"Unknown platform: {sys.platform}")

    with open(pswd_fid, "r") as fid:
        pswd_d = json.load(fid)

    # Print specified username and password.
    website = str(args.website).upper()
    try:
        usr_pswd = pswd_d[website]
    except KeyError:
        website_names = get_all_website_names(pswd_d)
        closest_matches = process.extract(
            args.website, list(website_names.keys()), limit=5
        )
        print(f"Error. {website} is not present. Input number of correct key.")
        for i, match in enumerate(closest_matches):
            print(f"  {i}) {match[0]}")
        int_key = int(input())
        website_key = closest_matches[int_key][0]
        pswd_key = website_names[website_key]
        usr_pswd = pswd_d[pswd_key]

    if args.print:
        for key, val in usr_pswd.items():
            print(f"  {key}: {val}")
        print()

    # Add username, password, token, etc. to clipboard.
    for key, val in usr_pswd.items():
        if key in {"desc", "name"}:
            continue
        try:
            pd.Series([val]).to_clipboard(index=False, header=False)
        except KeyError:
            print(f"{key} does not exist in entry for {website}")
        else:
            print(f"{key} has been copied to the clipboard.", end=" ")
            input(f"Press any key to continue.")


def get_all_website_names(d):
    names = {}
    for key, val in d.items():
        for name in val["name"]:
            names[name] = key
    return names


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("website", help="Name of Website")
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print values."
    )

    return parser.parse_args()


# %%
if __name__ == "__main__":
    main()
