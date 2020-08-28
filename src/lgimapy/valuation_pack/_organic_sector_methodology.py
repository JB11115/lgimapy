from lgimapy.vis import vis
from lgimapy.data import Database

vis.style()
# %%

db = Database()
db.load_market_data(start="2/1/2020", local=True)
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
    alpha=0.6,
    ls_list=["-", "-", "--", "--"],
)
# vis.savefig("Gov_owned_no_guar_OAS_2019")
vis.show()


cols = ["ISIN", "Ticker", "Issuer", "OAS", "MarketValue"]
ix_sub = ix.subset(date="4/17/2020")
ix_sub_20 = ix.subset(date="4/20/2020")

ix_sub.df[cols].sort_values("OAS", ascending=False)
ix_sub_20.df[cols].sort_values("OAS", ascending=False)


df = ix._ratings_changes_df

df[df["Ticker"] == "PEMEX"]
