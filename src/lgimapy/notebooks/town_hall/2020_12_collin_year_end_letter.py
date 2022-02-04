from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import get_ordinal

vis.style()

# %%

db = Database()
market = "US"
df_list = []
for market in ["US", "GBP", "EU"]:
    name = f"{market}_CORP" if market == "GBP" else f"{market}_IG"
    oas = db.load_bbg_data(name, "OAS", start="1/1/2010").rename(market)
    df_list.append(oas)
    pctile = oas.rank(pct=True).round(2)
    label = f"OAS\nCurrent: {oas[-1]:.0f} bp, ({pctile[-1]:.0%}tile)"
    fig, ax = vis.subplots()
    vis.plot_timeseries(
        oas,
        title=f"{market} OAS",
        ax=ax,
        median_line=True,
        label=label,
        color="navy",
    )
    low = np.percentile(oas, 5)
    high = np.percentile(oas, 95)
    label = f"5/95 %tiles [{low:.0f}, {high:.0f}]"
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    ax.axhline(low, label=label, **pct_line_kwargs)
    ax.axhline(high, label="_nolegend_", **pct_line_kwargs)
    ax.legend(fancybox=True, shadow=True)
    vis.savefig(f"{market}_10y_OAS")


# %%
def aggregate_excess_returns(xsret, tret, start, end=None, derivative=False):
    # Combine data and drop any missing dates.
    data_df = pd.concat(
        [tret.rename("tret"), xsret.rename("xsret")], axis=1, sort=True
    ).dropna()

    if derivative:
        xsret_s = xsret.dropna()
        start_date = nearest_date(
            start, xsret_s.index, inclusive=False, after=False
        )
        start_val = xsret_s.loc[start_date]
        cur_val = xsret_s[-1] if end is None else xsret_s.loc[end]
        return 1e4 * (cur_val / start_val - 1)

    # Split DataFrame into months.
    month_dfs = [df for _, df in data_df.groupby(pd.Grouper(freq="M"))]
    tret_ix_0 = month_dfs[0]["tret"][-1]  # last value of prev month
    xs_col_month_list = []
    for df in month_dfs[1:]:  # first month is likely incomplete
        a = np.zeros(len(df))
        for i, row in enumerate(df.itertuples()):
            tret_ix, cum_xsret = row[1], row[2] / 100
            if i == 0:
                a[i] = cum_xsret
                tret = (tret_ix - tret_ix_0) / tret_ix_0
                prev_tret_ix = tret_ix
                prev_cum_rf_ret = 1 + tret - cum_xsret
            else:
                cum_tret = (tret_ix - tret_ix_0) / tret_ix_0
                tret = tret_ix / prev_tret_ix - 1
                rf_ret = (cum_tret - cum_xsret + 1) / prev_cum_rf_ret - 1
                a[i] = tret - rf_ret
                prev_tret_ix = tret_ix
                prev_cum_rf_ret *= 1 + rf_ret

        tret_ix_0 = tret_ix
        xs_col_month_list.append(pd.Series(a, index=df.index))

    xsret_s = pd.concat(xs_col_month_list, sort=True)
    tret_s = (data_df["tret"] / data_df["tret"].shift(1) - 1)[1:]
    xsret_s = xsret_s[xsret_s.index >= start]
    tret_s = tret_s[tret_s.index >= start]
    if end is not None:
        xsret_s = xsret_s[xsret_s.index <= end]
        tret_s = tret_s[tret_s.index <= end]

    rf_ret_s = tret_s - xsret_s
    total_ret = np.prod(1 + tret_s) - 1
    rf_total_ret = np.prod(1 + rf_ret_s) - 1
    return total_ret - rf_total_ret


# %%
market = "US"
db = Database()
for market in ["US", "GBP", "EU"]:
    name = f"{market}_CORP" if market == "GBP" else f"{market}_IG"
    df = db.load_bbg_data(
        name, ["OAS", "OAD", "TRET", "XSRET"], start="1/1/1990", nan="drop"
    )
    df["year"] = pd.Series(df.index).dt.strftime("%Y").astype("int32").values
    years = sorted(df["year"].unique())[1:-1]
    d = defaultdict(list)
    year = 2018
    for year in years:
        df_year = df[df["year"] == year]
        start_date = df_year.index[0]
        end_date = df_year.index[-1]
        xsret = aggregate_excess_returns(
            df["XSRET"], df["TRET"], start_date, end_date
        )
        start_oas = df.loc[start_date, "OAS"]
        end_oas = df.loc[end_date, "OAS"]
        start_oad = df.loc[start_date, "OAD"]
        breakeven = start_oas / start_oad
        d["date"].append(start_date)
        d["breakeven"].append(breakeven)
        d["xsret"].append(xsret)
        d["starting_oas"].append(start_oas)

    yearly_df = pd.DataFrame(d, index=years)
    current_breakeven = df["OAS"].iloc[-1] / df["OAD"].iloc[-1]
    current_spread = df["OAS"].iloc[-1]

    fig, axes = vis.subplots(1, 2, figsize=[12, 6])
    axes[0].plot(yearly_df["breakeven"], yearly_df["xsret"], "o", color="navy")
    axes[0].axvline(current_breakeven, lw=1.5, color="firebrick")
    axes[0].set_title(
        f"{market} Breakevens vs Forward Excess Returns",
        fontweight="bold",
        fontsize=14,
    )
    axes[0].set_xlabel("Excess Return Breakeven on 12/31 (bp)")
    axes[0].set_ylabel("Excess Return for following Year")
    axes[0].axhline(0, color="gray", lw=1)
    vis.format_yaxis(axes[0], ytickfmt="{x:.0%}")

    axes[1].plot(
        yearly_df["starting_oas"], yearly_df["xsret"], "o", color="navy"
    )
    axes[1].axvline(current_spread, lw=1.5, color="firebrick")
    axes[1].set_title(
        f"{market} Starting Spread vs Forward Excess Returns",
        fontweight="bold",
        fontsize=14,
    )
    axes[1].set_xlabel("OAS on 12/31 (bp)")
    axes[1].set_ylabel("Excess Return for following Year")
    axes[1].axhline(0, color="gray", lw=1)
    vis.format_yaxis(axes[1], ytickfmt="{x:.0%}")
    vis.savefig(f"{market}_start_point")


yearly_df
# %%

db.load_market_data(start="1/1/2020")

ig_ix = db.build_market_index(in_stats_index=True)
ig_raw_oas = ig_ix.OAS()
ig_corr_ix = ig_ix.drop_ratings_migrations()
ig_oas = ig_corr_ix.get_synthetic_differenced_history("OAS")

hy_ix = db.build_market_index(in_hy_stats_index=True)
hy_raw_oas = hy_ix.OAS()
hy_corr_ix = hy_ix.drop_ratings_migrations(allowable_ratings="HY")
hy_oas = hy_corr_ix.get_synthetic_differenced_history("OAS")

# %%
fig, axes = vis.subplots(1, 2, figsize=(12, 6))
vis.plot_timeseries(
    ig_raw_oas, label="Historical OAS", ax=axes[0], pct_lines=(5, 95)
)
vis.plot_timeseries(
    ig_oas,
    label="Corrected OAS",
    ax=axes[0],
    color="navy",
    title="US IG Credit",
)

vis.plot_timeseries(
    hy_raw_oas, label="Historical OAS", ax=axes[1], pct_lines=(5, 95)
)
vis.plot_timeseries(
    hy_oas,
    label="Corrected OAS",
    ax=axes[1],
    color="navy",
    title="US HY Credit",
)
vis.savefig("corrected_US_1y_OAS")


low = np.percentile(ig_raw_oas, 5)
high = np.percentile(ig_raw_oas, 95)
print(f"IG: ({low:.0f}, {high:.0f})")
low = np.percentile(hy_raw_oas, 5)
high = np.percentile(hy_raw_oas, 95)
print(f"HY: ({low:.0f}, {high:.0f})")

df = pd.concat(
    (
        ig_raw_oas.rename("US_IG_Historical_YTD_OAS"),
        ig_oas.rename("US_IG_Corrected_YTD_OAS"),
        hy_raw_oas.rename("US_HY_Historical_YTD_OAS"),
        hy_oas.rename("US_HY_Corrected_YTD_OAS"),
    ),
    axis=1,
)
df.to_csv("Corrected_US_Credit_YTD_OAS.csv")


# %%
market = "US"
fig, ax = vis.subplots()
df_list = []
for market, color in zip(["US", "GBP", "EU"], ["navy", "#752333", "#8E5C1D"]):
    name = f"{market}_CORP" if market == "GBP" else f"{market}_IG"
    oas = db.load_bbg_data(name, "OAS", start="1/1/2010")
    df_list.append(oas)
    med = np.median(oas)
    print(f"{market}: {med:.0f}")
    pctile = np.round(100 * oas.rank(pct=True).iloc[-1], 0)
    ord = get_ordinal(pctile)
    market_label = {"US": "USD", "GBP": "GBP", "EU": "EUR"}[market]
    vis.plot_timeseries(
        oas,
        ax=ax,
        label=f"{market_label} ({pctile:.0f}{ord} %tile)",
        color=color,
        lw=2.5,
        end_point_kws={"s": 50},
    )
    lbl = "10y Median" if market == "US" else "_nolegend_"
    ax.axhline(med, lw=1.5, ls="--", color=color, label=lbl)
ax.legend(fancybox=True, shadow=True)
ax.set_title("OAS Spread History", fontweight="bold")
# vis.show()
vis.savefig(f"All_markets_10y_OAS")
oas_df = pd.concat(df_list, axis=1)
oas_df.to_csv("10y_all_markets_OAS.csv")
