import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import root, mkdir

vis.style()

path = root("reports/quant_showcases/2021-02")
mkdir(path)

# %%

db = Database()
db.load_market_data(start="2/1/2020", end="6/1/2020")

# %%
ix = db.build_market_index(
    sector="OWNED_NO_GUARANTEE", maturity=(10, None), in_stats_index=True
)
oas = ix.OAS()
oas_raw_corr = ix.get_synthetic_differenced_history("OAS")
ix_corrected = ix.drop_ratings_migrations()
oas_corr = ix_corrected.get_synthetic_differenced_history("OAS")

# %%
fig, ax = vis.subplots()
ax.axhline(0, color="k", lw=1)
kwargs = {"ax": ax, "lw": 1.5, "alpha": 0.8}
vis.plot_timeseries(oas, color="k", label="Historical", ylabel="OAS", **kwargs)
vis.plot_timeseries(
    oas_raw_corr, color="navy", ls="--", label="Corrected", **kwargs
)
kwargs.update({"lw": 4})
vis.plot_timeseries(
    oas_corr, color="navy", label="Corrected\nPEMEX\nRemoved", **kwargs
)
ax.set_title(
    "Owned No Guarantee Sector Corrected Spreads",
    fontweight="bold",
    fontsize=12,
)
date = pd.to_datetime("4/17/2020")
ax.axvline(date, lw=1, color="firebrick", ls="--")
ax.annotate(
    "PEMEX\nDowngrade",
    xy=(date, 25),
    xytext=(80, 10),
    textcoords="offset points",
    color="firebrick",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.3),
    arrowprops=dict(
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.2",
        color="firebrick",
        mutation_scale=20,
    ),
)
ax.legend(loc="upper left", fancybox=True, shadow=True, fontsize=12)
vis.savefig(path / "PEMEX")

# %%

db.load_market_data(start="7/15/2013", end="10/1/2013")
# %%
ix = db.build_market_index(sector="WIRELINES")
oas = ix.OAS()
oas_raw_corr = ix.get_synthetic_differenced_history("OAS")
ix_corrected = ix.drop_ratings_migrations()
oas_corr = ix_corrected.get_synthetic_differenced_history("OAS")

# %%
fig, ax = vis.subplots()
kwargs = {"ax": ax, "lw": 1.5, "alpha": 0.8}
vis.plot_timeseries(oas, color="k", label="Historical", ylabel="OAS", **kwargs)
kwargs.update({"lw": 4})
vis.plot_timeseries(oas_corr, color="navy", label="Corrected", **kwargs)
ax.set_title(
    "Wirelines Sector Corrected Spreads", fontweight="bold", fontsize=12
)
date = pd.to_datetime("9/12/2013")
ax.axvline(date, lw=1, color="firebrick", ls="--")
ax.annotate(
    "VZ\n$49B Deal",
    xy=(date, 210),
    xytext=(70, 20),
    textcoords="offset points",
    color="firebrick",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.3),
    arrowprops=dict(
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.2",
        color="firebrick",
        mutation_scale=20,
    ),
)
ax.legend(loc="upper left", fancybox=True, shadow=True)
vis.savefig(path / "VZ")
