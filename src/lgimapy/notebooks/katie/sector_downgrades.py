import pickle
from collections import defaultdict

import empiricalutilities as eu
import pandas as pd
import matplotlib.pyplot as plt

from lgimapy.utils import load_json, dump_json, savefig, root
from lgimapy.bloomberg import bdh

plt.style.use("fivethirtyeight")
pd.plotting.register_matplotlib_converters()

# %%
# Load data.
rating_changes = load_json("ratings_changes")
sectors_json = load_json("cusip_sectors")
risk_entities = load_json("risk_entities")
ratings_map = load_json("ratings")
with open("cusips.pickle", "rb") as fid:
    cusips = pickle.load(fid)

luac = bdh("LUACOAS", "Index", field="PX_BID", start="1/1/2007") * 100

# %%
# Settings
# --------------------------------------------------------------------------- #

agency = "Fitch"

# --------------------------------------------------------------------------- #

risk_entity_duplicates = {}
sector_downgrades = defaultdict(lambda: defaultdict(int))
for cusip in cusips:
    try:
        dates = rating_changes[cusip][agency]["date"][::-1]
        ratings = rating_changes[cusip][agency]["rating"][::-1]
    except KeyError:
        continue
    years = [d.rsplit("/", 1)[1] for d in dates]
    while True:
        try:
            num_ratings = [ratings_map[r] for r in ratings]
        except KeyError as e:
            # Ask user to provide value for missing key.
            key = e.args[0]
            val = int(
                input(
                    (
                        f"KeyError: {cusip} '{key}' is not in `ratings.json`.\n"
                        "Please provide the appropriate numeric value:\n"
                    )
                )
            )
            # Update `ratings.json` with user provided key:val pair.
            ratings_map[key] = val
            dump_json(ratings_map, "ratings")
        else:
            break

    first_nonzero = next((i for i, r in enumerate(num_ratings) if r), None)
    try:
        r_prev = num_ratings[first_nonzero]
    except TypeError:
        continue
    for d, r, y in zip(
        dates[first_nonzero + 1 :],
        num_ratings[first_nonzero + 1 :],
        years[first_nonzero + 1 :],
    ):
        if r == 0:
            continue
        change = r - r_prev
        r_prev = r
        if change <= 0:
            continue
        if (d, *risk_entities[cusip]) in risk_entity_duplicates:
            continue
        else:
            risk_entity_duplicates[(d, *risk_entities[cusip])] = 1
        sector_downgrades[y][sectors_json[cusip]] -= change


# Create DataFrame of downgrades.
years = sorted(list(sector_downgrades.keys()))
pd.DataFrame(sector_downgrades[y], index=[int(y)]).T

df = pd.DataFrame()
df = df.join(
    (pd.DataFrame(sector_downgrades[y], index=[int(y)]).T for y in years),
    how="outer",
    sort=True,
)

sector_abbvs = {
    "AEROSPACE_DEFENSE": "AERO/DEF",
    "APARTMENT_REITS": "APT REITS",
    "AUTOMOTIVE": "AUTOS",
    "BROKERAGE_ASSETMANAGERS_EXCHANGES": "BROKERS",
    "BUILDING_MATERIALS": "BUILD MAT",
    "CONSTRUCTION_MACHINERY": "CONST MACHINERY",
    "CONSUMER_CYCLICAL_SERVICES": "CONS CYC SRVS",
    "CONSUMER_PRODUCTS": "CONS PRODS",
    "DIVERSIFIED_MANUFACTURING": "DIV MFG",
    "FINANCE_COMPANIES": "FIN COMPANIES",
    "FINANCIAL_OTHER": "FIN OTHER",
    "FOOD_AND_BEVERAGE": "FOOD & BEV",
    "GOVERNMENT_GUARANTEE": "GOVT GTD",
    "HEALTH_INSURANCE": "HEALTH INS",
    "INDUSTRIAL_OTHER": "INDST OTHER",
    "LOCAL_AUTHORITIES": "LOCAL AUTH",
    "MEDIA_ENTERTAINMENT": "MEDIA/ENTMT",
    "METALS_AND_MINING": "METALS/MINING",
    "NON_CAPTIVE_CONSUMER": "NON CAP CONS",
    "NON_CAPTIVE_DIVERSIFIED": "NON CAP DIV",
    "OWNED_NO_GUARANTEE": "OWNED NO GTD",
    "PHARMACEUTICALS": "PHARMA",
    "TRANSPORTATION_SERVICES": "TRANSPORT SRVS",
}

# df.index = [sector_abbvs.get(s, s).replace("_", " ") for s in df.index]
table = pd.DataFrame()
for y in years:
    if int(y) < 2007:
        continue
    ydf = df.sort_values(int(y))[int(y)][:3]
    table[y] = [f"{i}: {abs(v):.0f}" for i, v in zip(ydf.index, ydf.values)]

col_fmt = (len(table.columns) + 1) * "r"


# Save results to csv.
# df.abs().to_csv(f'{agency}_sector_downgrades.csv')

# %%
# Print table.
eu.latex_print(table, col_fmt=col_fmt, hide_index=True, adjust=True)


# %%
# Save figure.
fid = root("latex/ratings/sector_downgrades")
fig, ax = plt.subplots(1, 1, figsize=[12, 1.5])
ax.plot(luac, color="steelblue", lw=1)
ax.set_ylabel("OAS")
ax.set_xticks([pd.to_datetime(f"1/1/{y}") for y in years if int(y) >= 2007])
savefig(f"{agency}_sectors", fid)
plt.show()
