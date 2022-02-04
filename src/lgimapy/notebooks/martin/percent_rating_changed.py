import lgimapy.vis as vis
from lgimapy.data import Database

vis.style()
# %%

db = Database()
db.load_market_data(start="1/1/2020", local=True)

ratings = {"BB": ("BB+", "BB-"), "B": ("B+", "B-")}

for rating, rating_kws in ratings.items():
    ix_start = db.build_market_index(
        date=db.nearest_date("2/29/2020", after=False),
        in_hy_stats_index=True,
        rating=rating_kws,
    )
    start_cusips = set(ix_start.cusips)
    n_start = len(start_cusips)
    ix_now = db.build_market_index(
        date=db.date("today"), in_hy_stats_index=True, rating=rating_kws
    )
    cusips = set(ix_now.cusips)
    n_now = len(cusips & start_cusips)
    print(f"{rating}: {n_now/n_start:.1%}")

# %%
fig, axes = vis.subplots(2, 1, figsize=(10, 10))

energy_sectors = [
    "INDEPENDENT",
    "REFINING",
    "OIL_FIELD_SERVICES",
    "INTEGRATED",
    "MIDSTREAM",
]

colors = ("navy", "darkorchid")
for color, (rating, rating_kws) in zip(colors, ratings.items()):
    ix = db.build_market_index(
        in_hy_stats_index=True,
        rating=rating_kws,
        OAS=(1, 3000),
        sector=energy_sectors,
    )
    rsd = ix.RSD("OAS")
    qcd = ix.QCD("OAS")
    vis.plot_timeseries(
        rsd, color=color, xtickfmt="auto", label=f"{rating} Energy", ax=axes[0]
    )
    vis.plot_timeseries(
        qcd, color=color, xtickfmt="auto", label=f"{rating} Energy", ax=axes[1]
    )
axes[0].set_ylabel("RSD")
axes[1].set_ylabel("QCD")
vis.savefig('HY_energy_dispersion_ytd')

# %%
