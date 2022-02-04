import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sms

from lgimapy.data import Database, spread_diff
from lgimapy.utils import root

plt.style.use("fivethirtyeight")
# %matplotlib qt
# %%
fid = root("src/lgimapy/notebooks/pld.csv")

db = Database()
db.load_market_data(start="12/31/2018", local=True)
ix = db.build_market_index()

df = pd.read_csv(fid, index_col=0)
df = df[df.index != np.nan]
df.sort_values("OAD_diff", inplace=True)

# %%


# %%
plt.figure()
plt.hist(df["OAD_diff"], bins=100)
plt.yscale("log")
plt.show()


# %%
# Find OAS for q bins of OAD_diff.
q = 10
vals, raw_bins = pd.qcut(df["OAD_diff"], q=q, retbins=True, labels=range(q))
raw_bins *= 1000
bins = [f"({raw_bins[i]:.1f}, {raw_bins[i+1]:.1f}]" for i in range(q)]

oas_ts = np.zeros([len(bins), len(ix.dates)])
for i in range(q):
    cusips_i = list(vals[vals == i].index)
    ix_i = ix.subset(cusip=cusips_i)
    oas_ts[i, :] = ix_i.market_value_weight("OAS").values

change = [oas_ts[i, -1] - oas_ts[i, 0] for i in range(q)]

# %%
# Plot OAS for each bin ytd.
fig, ax = plt.subplots(1, 1)
sns.set_palette("coolwarm", n_colors=len(bins))
for i in range(q):
    ax.plot(
        ix.dates, oas_ts[i, :], label=f"{bins[i]}: $\Delta$={change[i]:.1f}"
    )
ax.legend(title="OAD Overweight (bp)")
fig.autofmt_xdate()
ax.set_xlabel("Date")
ax.set_ylabel("OAS")
plt.show()

# %%
# Plot OAS deltas for each bin ytd with same endpoint.
fig, ax = plt.subplots(1, 1)
sns.set_palette("coolwarm", n_colors=len(bins))
for i in range(q):
    ax.plot(
        ix.dates,
        oas_ts[i, :] - oas_ts[i, -1],
        lw=3,
        alpha=0.8,
        label=f"{bins[i]}: $\Delta$={change[i]:.1f}",
    )
ax.legend(title="OAD Overweight (bp)")
fig.autofmt_xdate()
ax.set_xlabel("Date")
ax.set_ylabel("OAS")
plt.show()

# %%
# Find difference and percent difference of OAS ytd for each bond.
ix_pld = ix.subset(cusip=list(df.index))
diff_df = spread_diff(ix_pld.day("12/31/2018"), ix_pld.day("7/15/2019"))
diff_df.set_index("CUSIP", inplace=True)
diff_df = diff_df.join(df)

# %%
# Regress OAD diff on OAS percent change.
x = diff_df["OAD_diff"].values * 1000
y = diff_df["OAS_pct_change"].values
ols = sms.OLS(y, sms.add_constant(x)).fit()
fig, ax = plt.subplots(1, 1, figsize=[8, 8])
ax.plot(x, y, "o", alpha=0.5)
ax.plot(x, x * ols.params[1] + ols.params[0], lw=2, c="firebrick")
ax.set_xlabel("OAD Overweight (bp)")
ax.set_ylabel("OAS pct Change")
plt.show()

# %%
# Regress OAD diff on OAS absolute change.
x = diff_df["OAD_diff"].values * 1000
y = diff_df["OAS_change"].values
ols = sms.OLS(y, sms.add_constant(x)).fit()
fig, ax = plt.subplots(1, 1, figsize=[8, 8])
ax.plot(x, y, "o", ms=3, alpha=0.4)
ax.plot(x, x * ols.params[1] + ols.params[0], lw=2, c="firebrick")
ax.set_xlabel("OAD Overweight (bp)")
ax.set_ylabel("OAS Change (bp)")
plt.show()

# %%
i = 5
bins[i]
cusips_i = list(vals[vals == i].index)
ix_i = ix.subset(cusip=cusips_i)


jul_df = ix_i.day(ix.dates[-1])[
    ["Issuer", "AmountOutstanding", "IssueDate", "IssueYears"]
]
jul_df.sort_values("IssueYears")
diff_df.loc["892331AG4", "OAD_diff"].head()
