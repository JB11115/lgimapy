import awswrangler as wr
import numpy as np
import pandas as pd

bucket = "lgima-qa-3pdh-data-bucket"
path = f"s3://{bucket}/idh"
fid = f"s3://{bucket}/idh/test_parquet.parquet"

# Make small DataFrame.
write_df = pd.DataFrame(
    np.random.randn(5, 5), columns="a b c d e".split()
).round(2)

# Write df to parquet file on s3.
wr.s3.to_parquet(write_df, path=fid)

# Read the file back.
read_df = wr.s3.read_parquet(fid)

if read_df.equals(write_df):
    print("True")


# %%
import boto3
import sys

s3 = boto3.resource("s3")


def ls(env, path):
    bucket = s3.Bucket(f"lgima-{env}-3pdh-data-bucket")
    for object_summary in bucket.objects.filter(Prefix=path):
        print(object_summary.key)


env = "prod"
path = "idh/"

ls(env, path)
