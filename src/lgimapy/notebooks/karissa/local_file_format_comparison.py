from lgimapy.data import Database
from lgimapy.utils import Time


# %%
db = Database()

with Time() as t:
    for fmt in ["feather", "parquet"]:
        db.load_market_data(date="1/5/2002", local=True, local_file_fmt=fmt)
        t.split(f"{fmt} 1 day")
        db.load_market_data(
            start="1/1/2002", end="1/31/2002", local=True, local_file_fmt=fmt
        )
        t.split(f"{fmt} 1 month")
        db.load_market_data(
            start="1/1/2002", end="12/31/2002", local=True, local_file_fmt=fmt
        )
        t.split(f"{fmt} 1 year")

db.display_all_columns()
db.load_market_data(date="1/7/1999", local=True, local_file_fmt="parquet")
df_parquet = db.build_market_index().df

db.load_market_data(date="1/7/1999", local=True, local_file_fmt="feather")
df_feather = db.build_market_index().df

subset_cols = [
    "Date",
    "Ticker",
    "Issuer",
    "CouponRate",
    "MaturityDate",
    "IssueDate",
    "Sector",
    "MoodyRating",
    "SPRating",
    "FitchRating",
    "CollateralType",
    "AmountOutstanding",
    "MarketOfIssue",
    "NextCallDate",
    "CouponType",
    "CallType",
    "CleanPrice",
    "OAD",
    "OAS",
    "OASD",
]
import pandas as pd

pd.concat([df_parquet, df_feather]).drop_duplicates(keep=False)



df_parquet[subset_cols].equals(df_feather[subset_cols])
df_parquet.head()
df_feather.head()
