from lgimapy.vis import vis
from lgimapy.data import Database

vis.style()
# %%

db = Database()
db.load_market_data(start="2/1/2020", local=True)

# %%

ix = db.build_market_index(
    sector="OWNED_NO_GUARANTEE", maturity=(10, None), in_stats_index=True
)
ix_drop = ix.drop_ratings_migrations()
# %%

oas = ix.OAS()
oas_synth = ix.get_synthetic_differenced_history("OAS",)
oas_drop = ix_drop.OAS()
oas_drop_synth = ix_drop.get_synthetic_differenced_history("OAS")

# oas_mv = ix.get_synthetic_differenced_history(
#     "OAS", col2="MarketValue", force_calculations=True, dropna=True
# )

# %%

vis.plot_multiple_timeseries(
    [
        oas.rename("Actual OAS"),
        oas_synth.rename("Corrected OAS"),
        oas_drop.rename("Actual OAS Ex-Ratings Migrants"),
        oas_drop_synth.rename("Corrected OAS Ex-Ratings Migrants"),
    ],
    figsize=(10, 8),
    lw=2,
    alpha=0.8,
)
# vis.savefig("Gov_owned_no_guar_OAS_2020")
vis.show()

# %%
