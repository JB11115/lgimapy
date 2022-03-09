from lgimapy import vis
from lgimapy.bloomberg import bdh

vis.style()
# %%

df = bdh(["RX1", "RXU2"], "Comdty", "PX_LAST", start="1/1/2021").dropna()

# %%
vis.plot_timeseries(df.iloc[:, 1] - df.iloc[:, 0])
vis.show()

# %%
df_ecb = bdh("EZ0BFR DEC2022", "Index", "PX_LAST", start="1/1/2021")
df_us = bdh("US0AFR DEC2022", "Index", "PX_LAST", start="1/1/2021")
# %%
vis.plot_timeseries(
    df_ecb.squeeze(),
    ylabel="Implied ECB Policy Rate",
    ytickfmt="{x:.1f}%",
    color="navy",
)
vis.show()
