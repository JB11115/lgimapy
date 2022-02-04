import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import (
    mark_inset,
    inset_axes,
    InsetPosition,
)

from lgimapy import vis
from lgimapy.data import Bond, Database, TreasuryCurve, SyntheticBond
from lgimapy.models import CreditCurve
from lgimapy.utils import root, mkdir

vis.style()
# %%
db = Database()

date = db.date("today")
# date = "7/17/2020"
date = db.nearest_date("3/9/2021")
db.load_market_data(date=date)
treasury_curve = TreasuryCurve(date)
__ = treasury_curve._curves_df

path = root("reports/quant_showcases/2021-03")
mkdir(path)
# %%
cols = ["ISIN", "OriginalMaturity", "IssueYears", "YieldToWorst", "DirtyPrice"]
ticker = "AAPL"
fit_isins = [
    "US037833DV96",
    "US037833EB24",
    "US037833ED89",
    "US037833EF38",
    "US037833EC07",
    "US037833EE62",
]
# fit_ix = db.build_market_index(isin=fit_isins)
#
# ticker = 'DIS'
# fit_ix = db.build_market_index(ticker=ticker, issue_years=(None, 0.5), maturity=(None, 31))
#
# ticker = 'T'
# fit_ix = db.build_market_index(ticker=ticker, issue_years=(None, 5), maturity=(None, 31))
# fit_ix.df[].sort_values('OriginalMaturity')
#
# fit_isins = [
#     "US254687CL89",
#     "US254687CS33",
#     "US254687DC71",
#     "US254687DS24",
#     "US254687EN28",
#     "US254687FA97",
# ]
fit_ix = db.build_market_index(isin=fit_isins)
fit_df = fit_ix.df.sort_values("MaturityYears")
# fig, ax = vis.subplots(figsize=(8, 6))
# ax.plot(fit_df["MaturityYears"], fit_df["YieldToWorst"], "-o", lw=2)
# vis.savefig(f"{ticker}_calibration_bonds")
fit_df[cols]
# %%

ticker_df = db.build_market_index(
    ticker=ticker, isin=fit_df["ISIN"], special_rules="~ISIN"
).df
ticker_df = ticker_df[ticker_df["MaturityYears"] <= 30]
mod = CreditCurve(fit_df, treasury_curve)
# %%
oas_error = 999
param_fid = root(f"data/{ticker}_params.csv")
params = pd.read_csv(param_fid, index_col=0).squeeze().values
while oas_error > 8:
    res = mod.fit(params=params)
    oas_error = (res.fit_OAS(fit_df) - fit_df["OAS"]).abs().max()
    print(oas_error)

mod.bond_errors
# pd.DataFrame(mod.params).to_csv(param_fid)

# %%


def add_model_columns(df, res):
    df["ModelOAS"] = res.fit_OAS(df)
    df["LiquidtyDiscount"] = df["OAS"] - df["ModelOAS"]


add_model_columns(fit_df, res)
add_model_columns(ticker_df, res)

# %%
t = np.arange(0.5, 30.5, 0.5)
par_yields = res.par_yield_curve()
par_bonds = [SyntheticBond(maturity=t, coupon=c) for t, c in par_yields.items()]
oas_curve = pd.Series(res.fit_Zspreads(par_bonds), index=t)


bond = SyntheticBond(maturity=30, coupon=0)
res.fit_prices([bond])
Q = res.beta @ res._mod._SSpline(res.alpha).squeeze()
Q = np.concatenate(([1], Q))
t_full = np.arange(0, 30.5, 0.5)
rf_curve = treasury_curve.yields(t_full)
rf_discount_curve = np.exp(-rf_curve * t_full)
risky_discount_curve = rf_discount_curve * Q
risky_zero_curve = -np.log(risky_discount_curve) / t_full

# %%

fig, ax = vis.subplots(figsize=(12, 8))

ax.plot(
    ticker_df["MaturityYears"],
    ticker_df["OAS"],
    "o",
    color="skyblue",
    ms=5,
    label="Off-the-run",
)
ax.plot(
    fit_df["MaturityYears"],
    fit_df["OAS"],
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

ax.legend(fancybox=True, shadow=True)
ax.set_title("AAPL", fontweight="bold", fontsize=14)
ax.set_ylabel("OAS")
ax.set_xlabel("Maturity")
vis.savefig(path / "OAS_curve")

# %%
fig, ax = vis.subplots(figsize=(12, 8))

ax.plot(
    ticker_df["MaturityYears"],
    ticker_df["OAS"],
    "o",
    color="skyblue",
    ms=5,
    label="Off-the-run",
)
ax.plot(
    fit_df["MaturityYears"],
    fit_df["OAS"],
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
ax.plot(
    oas_curve.index[-1],
    oas_curve.values[-1],
    "o",
    ms=15,
    color="darkorchid",
    label="New Issue Par 30y\nModel Implied Fair Value",
)
ax.legend(fancybox=True, shadow=True, loc="upper left")
ax.set_ylabel("OAS")
ax.set_xlabel("Maturity")
ax.set_xlim(25, 30.15)
ax.set_ylim(82, 92)
vis.savefig(path / "new_issue_concession")

# %%
fig, ax = vis.subplots(figsize=(12, 7))
ld_df = ticker_df[ticker_df["MaturityYears"] > 1.8]
fit_df
ax.bar(
    ld_df["MaturityYears"],
    ld_df["LiquidtyDiscount"],
    width=0.2,
    color="skyblue",
    label="Off-the-run",
)
ax.bar(
    fit_df["MaturityYears"],
    fit_df["LiquidtyDiscount"],
    width=0.2,
    color="navy",
    label="On-the-run\n(used for calibration)",
)
ax.legend(fancybox=True, shadow=True, loc="upper right")
ax.set_ylabel("Liqudity Discount (OAS)")
ax.set_xlabel("Maturity")
vis.savefig(path / "liquidty_discount")

# %%
fig, axes = vis.subplots(1, 2, figsize=(14, 7))

axes[1].plot(
    risky_zero_curve, color="navy", lw=2, label="AAPL Zero Coupon Curve"
)
axes[1].plot(
    par_yields / 100, color="darkorchid", lw=2, label="AAPL Par Coupon Curve"
)
axes[1].plot(rf_curve, color="k", lw=2, label="Risk Free Zero Coupon Curve")
axes[0].plot(
    risky_discount_curve, color="navy", lw=2, label="AAPL Discount Curve"
)
axes[0].plot(rf_discount_curve, color="k", lw=2, label="Risk Free Curve")
vis.format_yaxis(axes[1], ytickfmt="{x:.1%}")
axes[1].set_ylabel("Yield")
axes[0].set_ylabel("Discount Factor")
axes[1].legend(fancybox=True, shadow=True, loc="upper left", fontsize=10)
axes[0].legend(fancybox=True, shadow=True, fontsize=10)

for ax in axes:
    ax.set_xlabel("Maturity")
vis.savefig(path / "discount_curves")

# %%

fig, ax = vis.subplots(figsize=(12, 8))
bonds = [Bond(row) for _, row in fit_df.iterrows()]
axins = plt.axes([0, 0, 1, 1])
axins.set_xlim(4.86, 4.96)
axins.set_ylim(0.0113, 0.0117)
ip = InsetPosition(ax, [0.5, 0.2, 0.4, 0.4])
axins.set_axes_locator(ip)
colors = sns.color_palette("Spectral", len(bonds)).as_hex()
for bond, color in zip(bonds, colors):
    curve = res.spot_curve(bond)
    ax.plot(
        bond.MaturityYears,
        curve.iloc[-1],
        "o",
        color=color,
        ms=12,
        zorder=10,
        label=f"{bond.Ticker} {bond.CouponRate:.2f} `{bond.MaturityDate:%y}",
    )
    ax.plot(curve.index, curve.values, color=color, lw=2, alpha=0.8)
    axins.plot(bond.MaturityYears, curve.iloc[-1], "o", color=color, ms=15)
    axins.plot(curve.index, curve.values, color=color, lw=2, alpha=0.8)
vis.format_yaxis(ax, ytickfmt="{x:.1%}")
vis.format_yaxis(axins, ytickfmt="{x:.2%}")

for key in axins.spines.keys():
    axins.spines[key].set_color("grey")

ax.grid(False)
axins.grid(False)
mark_inset(ax, axins, loc1=2, loc2=3, fc="grey", ec="grey", lw=2, alpha=0.4)
axins.tick_params(axis="x", which="major", pad=4)
ax.set_xlabel("Maturity")
ax.set_ylabel("Yield")
ax.legend(fancybox=True, shadow=True)
vis.savefig(path / "rolldown")

#%%
fig, ax = vis.subplots(figsize=(12, 8))
db_2 = Database()
db_2.load_market_data(start=db.date("2y"))
# %%
col = "MaturityYears"
df_list = []
end = None
for isin in ["US037833EF38", "US037833DZ01", "US037833DW79", "US037833DQ02"]:
    df_30 = db_2.build_market_index(isin=isin, end=end).df.set_index("Date")[
        col
    ]
    df_list.append(df_30)
    end = db.trade_dates(exclusive_end=df_30.index[0])[-1]

market_30 = pd.concat(df_list).sort_index()
const_30 = pd.Series(np.zeros(len(market_30)) + 30, index=market_30.index)
# %%
fig, ax = vis.subplots(figsize=(12, 8))
vis.plot_timeseries(market_30, color="k", label="AAPL on-the-run 30y", ax=ax)
vis.plot_timeseries(const_30, color="navy", label="Model Implied 30y", ax=ax)
ax.set_ylabel("Years to Maturity")
ax.set_ylim(29, 30.5)
vis.savefig(path / "constant_30")

# %%
