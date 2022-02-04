from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import Time, load_json, dump_json, savefig, root

plt.style.use("fivethirtyeight")

# %%
# Load validation data.
df = pd.read_csv("BondLevelReturns.csv")
df.columns = ["Date", "TRet", "XSRet"]

# Make list of DataFrames grouped by unique cusips.
x = np.zeros(len(df))
group = 0
for i, tret in enumerate(df["TRet"]):
    if np.isnan(tret):
        group += 1
    x[i] = group
df["group"] = x
dfs = [df for _, df in df.groupby("group")]

# Make dictionary containing start date, end date, cumulative
# total and cumulative excess returns for each cusip.
d = defaultdict(dict)
df = dfs[0]
for i, df in enumerate(dfs):
    cusip = df.iloc[0, 0]
    df = df.iloc[1:].copy()
    df.index = pd.to_datetime([date.split("-")[1] for date in df["Date"]])
    d[cusip]["start"] = np.min(df.index)
    d[cusip]["end"] = np.max(df.index)
    d[cusip]["i"] = i

# Load LGIMA data and save error between the two to a dict.
db = Database()
db.load_market_data(
    start=np.min([d[cusip]["start"] for cusip in d.keys()]),
    end=np.max([d[cusip]["end"] for cusip in d.keys()]),
    local=True,
)
ix = db.build_market_index()

# %%
cusip = list(d.keys())[0]
cusip = "369604BH"
err = defaultdict(list)
cusip_map = {cusip[:-1]: cusip for cusip in ix.cusips}
cusips_with_diff_n_days_in_sample = []
for cusip in tqdm(d.keys()):

    if cusip in cusip_map.keys():
        full_cusip = cusip_map[cusip]
    else:
        continue

    # Compute total and excess returns using LGIMA data.
    cusip_ix = ix.subset(
        cusip=full_cusip, start=d[cusip]["start"], end=d[cusip]["end"]
    )
    date_df = cusip_ix.df.dropna(subset=["TRet"], how="any")

    total_ret = np.prod(1 + cusip_ix.df["TRet"]) - 1
    tsy_t_rets = cusip_ix.df["TRet"] - cusip_ix.df["XSRet"]
    tsy_total_ret = np.prod(1 + tsy_t_rets) - 1
    xs_ret = total_ret - tsy_total_ret

    # Compute total and excess returns using Bloomberg data.
    df = dfs[d[cusip]["i"]].iloc[1:]
    df.index = pd.to_datetime([date.split("-")[1] for date in df["Date"]])

    try:
        df = df[
            (df.index >= date_df["Date"].iloc[0])
            & (df.index <= date_df["Date"].iloc[-1])
        ]
    except IndexError:
        continue

    if len(df) != len(cusip_ix.df):
        cusips_with_diff_n_days_in_sample.append(full_cusip)

    total_ret_bloom = np.prod(1 + df["TRet"] / 100) - 1
    tsy_t_rets_bloom = (df["TRet"] - df["XSRet"]) / 100
    tsy_total_ret_bloom = np.prod(1 + tsy_t_rets_bloom) - 1
    xs_ret_bloom = total_ret_bloom - tsy_total_ret_bloom

    err["cusip"].append(full_cusip)
    err["tret"].append(total_ret - total_ret_bloom)
    err["tret_pct"].append(
        100 * (total_ret - total_ret_bloom) / total_ret_bloom
    )
    err["xsret"].append(xs_ret - xs_ret_bloom)
    err["xsret_pct"].append(100 * (xs_ret - xs_ret_bloom) / xs_ret_bloom)

len(cusips_with_diff_n_days_in_sample)
# %%
fid = "tret_xsret_validation"
dump_json(err, fid)
err = load_json(fid)


# %%

df_err = pd.DataFrame(err)
df_err.sort_values("tret", inplace=True)
df_err.head()
df_tret = ix.get_value_history("TRet")

# %%
# cusip = df_err['cusip'].iloc[0]
cusip = "25470DBC2"
missing_day_cusips = {}
for cusip in cusips_with_diff_n_days_in_sample:
    df_bloomberg = dfs[d[cusip[:-1]]["i"]].iloc[1:].copy()
    df_bloomberg.index = pd.to_datetime(
        [date.split("-")[1] for date in df_bloomberg["Date"]]
    )
    df_bloomberg["bloomberg"] = df_bloomberg["TRet"]

    df_cusip_comp = pd.DataFrame(df_tret[cusip]).join(
        df_bloomberg["bloomberg"] / 100
    )
    temp_ix = df_cusip_comp[cusip].dropna().index
    df_cusip_comp = df_cusip_comp[
        (df_cusip_comp.index >= temp_ix[0])
        & (df_cusip_comp.index <= temp_ix[-1])
    ]
    df_cusip_comp.dropna(subset=["bloomberg"], inplace=True)
    df_cusip_comp["diff"] = df_cusip_comp[cusip] - df_cusip_comp["bloomberg"]
    df_cusip_comp
    missing_dates = list(df_cusip_comp[df_cusip_comp.isna().any(axis=1)].index)
    if missing_dates:
        missing_day_cusips[cusip] = [
            md.strftime("%Y-%m-%d") for md in missing_dates
        ]

dump_json(missing_day_cusips, "missing_date_period_examples")

# %%
df_cusip_comp.sort_values("diff", ascending=False)
# df_cusip_comp.iloc[50:, :]

ix_sub = ix.subset(cusip=cusip, start="3/15/2019", end="3/24/2019")
ix_sub.df[["Date", "CleanPrice", "DirtyPrice"]]


list(ix_sub.df)

# %%

path = root("latex/docs/excess_returns")

var = "tret"
x = np.array(err[var]) * 1e4
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(x, bins=150, color="steelblue", alpha=0.8, label="_nolegend_")
mean = np.mean(x)
ax.axvline(mean, c="firebrick", ls="--", lw=1, label=f"Mean: {mean:.0f} bp")
med = np.median(x)
ax.axvline(med, color="firebrick", lw=1, label=f"Median: {med:.0f} bp")
ax.axvline(med, alpha=0, label=f"MAE: {np.mean(np.abs(x)):.0f} bp")
ax.axvline(med, alpha=0, label=f"RMSE: {np.mean(x**2)**0.5:.0f} bp")
ax.set_yscale("log")
ax.set_xlabel("LGIMA - Validation (bp)")
ax.set_title("Total Return")
ax.legend()
plt.tight_layout()
# savefig("total_returns", path)
plt.show()


# %%
var = "xsret"
x = np.array(err[var]) * 1e4
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(x, bins=150, color="steelblue", alpha=0.8, label="_nolegend_")
mean = np.mean(x)
ax.axvline(mean, c="firebrick", ls="--", lw=1, label=f"Mean: {mean:.0f} bp")
med = np.median(x)
ax.axvline(med, color="firebrick", lw=1, label=f"Median: {med:.0f} bp")
ax.axvline(med, alpha=0, label=f"MAE: {np.mean(np.abs(x)):.0f} bp")
ax.axvline(med, alpha=0, label=f"RMSE: {np.mean(x**2)**0.5:.0f} bp")
ax.set_yscale("log")
ax.set_xlabel("LGIMA - Validation (bp)")
ax.set_title("Excess Return")
ax.legend()
plt.tight_layout()
# savefig("excess_returns", path)
plt.show()
