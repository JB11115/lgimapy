import pandas as pd
from blpapi import NotFoundException, InvalidArgumentException
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.bloomberg import bds
from lgimapy.utils import root, mkdir

# %%
cusip_dir = root("data/amt_outstanding_history")

# Get all cusips.
print("Collecting CUSIPs")
db = Database()
cusips = set()
for year in tqdm(range(1998, 2021)):
    for month in (1, 13):
        df = db.load_market_data(
            date=db.nearest_date(f"{month}/1/{year}"), clean=False, ret_df=True
        )
        cusips = cusips | set(df["CUSIP"].unique())

# Filter CUSIPs that have already been scraped or have failed.
failed_fid = root("data/failed_amt_outstanding_cusips.txt")
with open(failed_fid, "a") as f:
    f.write("\n")

with open(failed_fid, "r") as f:
    failed_cusips = set(f.read().splitlines())

scraped_fids = cusip_dir.glob("*.csv")
scraped_cusips = set([fid.stem for fid in scraped_fids])
cusips_to_be_scraped = cusips - scraped_cusips - failed_cusips
print(f"Total Cusips: {len(cusips_to_be_scraped):,.0f}")

# %%
# Scrape and save amount outstanding history for each Cusip.
print("Scraping CUSIPs")
mkdir(cusip_dir)
for cusip in tqdm(cusips_to_be_scraped):
    try:
        df = bds(cusip, "Corp", "AMOUNT_OUTSTANDING_HISTORY")
    except NotFoundException:
        with open(failed_fid, "a") as f:
            f.write(cusip + "\n")
    except InvalidArgumentException:
        raise ValueError("BLOOMBERG DATA LIMIT REACHED")
    else:
        try:
            df.to_csv(cusip_dir / f"{cusip}.csv")
        except OSError:
            with open(failed_fid, "a") as f:
                f.write(cusip + "\n")
