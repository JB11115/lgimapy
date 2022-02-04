import pandas as pd
import statsmodels.api as sms

from lgimapy import vis
from lgimapy.bloomberg import bdh

vis.style()

# %%
start = "1/1/2010"
forward = "5y1m"
df = pd.concat(
    (
        bdh("NAPMPMI", "Index", "PX_LAST", start).squeeze().rename("PMI"),
        bdh(f"S0042FS {forward.upper()} BLC", "Curncy", "PX_LAST", start)
        .squeeze()
        .rename(f"OIS {forward}"),
    ),
    axis=1,
).dropna()

# %%
vis.plot_double_y_axis_timeseries(
    df["PMI"],
    df[f"OIS {forward}"],
)
vis.savefig(f"PMI_vs_OIS_{forward}")

# %%
x = sms.add_constant(df[f"OIS {forward}"])
ols = sms.OLS(df["PMI"], x).fit()
ols.predict(x)
ols.summary()
