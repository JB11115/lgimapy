import matplotlib.pyplot as plt
import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdp
from lgimapy.data import Database


# %%
# Load data and compare amount outstanding to Bloomberg.
db = Database()
db.load_market_data()
ix = db.build_market_index()
act_amt_out = bdp(ix.cusips, "Corp", "AMT_OUTSTANDING") / 1e6
df = pd.concat([ix.df["AmountOutstanding"], act_amt_out], join="inner", axis=1)
df.columns = ["LGIMA", "Bloomberg"]
df["diff"] = df.eval("LGIMA - Bloomberg")
df.dropna(inplace=True)
df_wrong = df[df["diff"] != 0].sort_values("diff")
print(f"Total Incorrect: {len(df_wrong)}")
len(ix.df)

# %%
# Raw histogram of incorrect values.
fig, ax = vis.subplots()
vis.plot_hist(df_wrong["diff"], bins=25, ax=ax)
ax.set_xlabel("$M")
vis.show()

# %%
# Subset incorrect values to those that are off by more than $1M.
df_wrong_int = df_wrong.round(0).astype(int)
df_wrong_int = df_wrong_int[df_wrong_int["diff"] != 0]
print(f"Total Incorrect: {len(df_wrong_int)}")

# %%
# Histogram of incorrect values that are more than $1M off.
fig, ax = vis.subplots()
vis.plot_hist(df_wrong_int["diff"].values, bins=40, ax=ax)
ax.set_xlabel("$M")
vis.show()

# %%
date = db.date("today")
df_wrong_large = df_wrong_int.join(
    ix.df[
        [
            "Ticker",
            "Issuer",
            "MaturityDate",
            "USCreditStatisticsFlag",
            "NumericRating",
        ]
    ]
)


# df_wrong_large.to_csv(
#     f"incorrect_amount_outstanding_{date.strftime('%Y%m%d')}.csv"
# )

df_wrong_large
