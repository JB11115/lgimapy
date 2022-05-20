import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database, TreasuryCurve, Bond, SyntheticBond
from lgimapy.models import CreditCurve

vis.style()
# %%
db = Database()
ticker = "UNH"
date = db.date("today")

treasury_curve = TreasuryCurve(date)

db.load_market_data(date)
issuer_ix = db.build_market_index(ticker=ticker, maturity=(None, 32))
full_issuer_df = issuer_ix.df.copy()
otr_issuer_ix = issuer_ix.subset_on_the_runs()
cols = [
    "IssueDate",
    "MaturityDate",
    "CouponRate",
    "MaturityYears",
    "OriginalMaturity",
    "DirtyPrice",
    "YieldToWorst",
    "OAS",
]
issuer_df = otr_issuer_ix.df.sort_values("OriginalMaturity")
issuer_df[cols]

# %%

mod = CreditCurve(issuer_df, treasury_curve)
res = mod.fit(s)
# %%

oas_error = 99
n = 100
i = 0
params = {}
resids = {}
while oas_error > 1 and i < n:
    res = mod.fit()
    oas_error = (res.fit_OAS(issuer_df) - issuer_df["OAS"]).abs().max()
    print(i, int(oas_error))
    resids[i] = oas_error
    params[i] = res.params
    i += 1

best_param_key = min(resids, key=resids.get)
res = mod.fit(params=params[best_param_key])


# %%
new_issue_maturities = [5, 7, 10, 30, 40]
new_issue_yields = res.par_yield_curve(new_issue_maturities)
new_issue_bonds = [
    SyntheticBond(maturity=t, coupon=c) for t, c in new_issue_yields.items()
]
new_issue_spreads = pd.Series(
    res.fit_Zspreads(new_issue_bonds), index=new_issue_yields.index
).round(0)
new_issue_spreads

par_yields = res.par_yield_curve()
par_bonds = [SyntheticBond(maturity=t, coupon=c) for t, c in par_yields.items()]
t = np.arange(0.5, 30.5, 0.5)
oas_curve = pd.Series(res.fit_Zspreads(par_bonds), index=t)
# %%


def add_model_columns(df, res):
    df["ModelOAS"] = res.fit_OAS(df)
    df["LiquidtyDiscount"] = df["OAS"] - df["ModelOAS"]


add_model_columns(issuer_df, res)
add_model_columns(full_issuer_df, res)

# %%

fig, ax = vis.subplots(figsize=(12, 8))

ax.plot(
    full_issuer_df["MaturityYears"],
    full_issuer_df["OAS"],
    "o",
    color="skyblue",
    ms=5,
    label="Off-the-run",
)
ax.plot(
    issuer_df["MaturityYears"],
    issuer_df["OAS"],
    "o",
    color="navy",
    ms=6,
    label="On-the-run\n(used for calibration)",
)
ax.plot(
    oas_curve.index,
    oas_curve.values,
    color="firebrick",
    alpha=0.8,
    lw=1.7,
    label="Fit Par OAS Curve",
)

vis.legend(bbox_to_anchor=(1, 1), ax=ax)
ax.set_ylim(0, None)
ax.set_ylabel("OAS")
ax.set_xlabel("Maturity")
vis.show()
# vis.savefig(f"{ticker}_par_OAS_curve")

# %%
fig, ax = vis.subplots(figsize=(12, 7))
ld_df = full_issuer_df[full_issuer_df["MaturityYears"] > 1]
ax.bar(
    ld_df["MaturityYears"],
    ld_df["LiquidtyDiscount"],
    width=0.2,
    color="skyblue",
    label="Off-the-run",
)
ax.bar(
    issuer_df["MaturityYears"],
    issuer_df["LiquidtyDiscount"],
    width=0.2,
    color="navy",
    label="On-the-run\n(used for calibration)",
)
vis.legend(ax=ax)
ax.set_ylabel("Liqudity OAS Discount (bp)")
ax.set_xlabel("Maturity")
vis.savefig(f"{ticker}_liquidity_discount")

# %%
save_cols = ["Ticker"] + cols + ["OAS", "ModelOAS", "LiquidtyDiscount"]
save_df = full_issuer_df[save_cols].sort_values("MaturityYears")
save_df.to_csv(f"{ticker}_curve_model_results.csv")
save_df
