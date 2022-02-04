import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdh

vis.style()
# %%

start = "1/1/2010"
df = (
    pd.concat(
        (
            bdh("BNYDMVU", "Index", "PX_LAST", start=start)
            .squeeze()
            .rename("NegYieldDebt"),
            bdh("LEGATRUU", "Index", "INDEX_MARKET_VALUE", start=start)
            .squeeze()
            .rename("AggDebt"),
        ),
        axis=1,
    )
    .fillna(method="ffill")
    .dropna(how="any")
)
df['% Negative Yielding Debt'] = df['NegYieldDebt'] / df['AggDebt']
df = df.astype(float)
# %%
vis.plot_double_y_axis_timeseries(
    df['% Negative Yielding Debt'],
    df['NegYieldDebt'].rename('Nominal Negative Yielding Debt (USD)') / 1e6,
    ytickfmt_left='{x:.0%}',
    ytickfmt_right='${x:.0f}T',
    color_left='navy',
    color_right='dodgerblue',
    lw=2,
)
vis.savefig('negative_yielding_debt')
