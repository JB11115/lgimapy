import os
from time import sleep

import awswrangler as wr
import boto3
from botocore.exceptions import ProxyConnectionError
from tqdm import tqdm

from lgimapy.data import Database

# %%

db = Database()

sess = boto3.Session(
    aws_access_key_id=db._passwords["AWS"]["dev_access_key"],
    aws_secret_access_key=db._passwords["AWS"]["dev_secret_access_key"],
)

# %%
s3_dir = "s3://lgima-prod-3pdh-data-bucket/qws-inbound/qws-rds-history/"
markets = ["US", "EUR", "GBP"]
markets = ["EUR", "GBP"]

for market in markets:
    db = Database(market=market)
    mkt = db.market.lower()
    dates = db.trade_dates(start=db.date("MARKET_START"))
    print(f"\n{market}")
    for i, date in enumerate(tqdm(dates)):
        # print(date)
        df = db.load_market_data(date=date, ret_df=True)
        filename = f"security_analytics_{mkt}_{date:%Y%m%d}"
        s3_fid = f"{s3_dir}/{mkt}/{filename}.parquet"
        while True:
            try:
                wr.s3.to_parquet(
                    df, path=s3_fid, index=False, boto3_session=sess
                )
            except (OSError, ProxyConnectionError):
                sleep(2)
                continue
            else:
                sleep(0.5)
                break
