from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.models import BetaAdjustedPerformance

vis.style()

# %%
universe = "IG"
universe_kwargs = {}
universe_kwargs = {"maturity": (5, 10)}

db = Database()

forecast_dates = db.date("MONTH_STARTS", start="1/1/2018")
forecast_horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
db.load_market_data(
    start=db.date(f"{max(forecast_horizons)}w", forecast_dates[0])
)

# %%
mod = BetaAdjustedPerformance(db, universe=universe)

d = defaultdict(list)
for date in tqdm(forecast_dates):
    for horizon in forecast_horizons:
        mod.train(forecast=f"{horizon}w", date=date, **universe_kwargs)
        d[horizon].append(mod.rsquared())


df = pd.DataFrame(d, index=forecast_dates)

# %%

ax = vis.boxplot(
    df[range(1, 9)],
    title=f"Accuracy of {universe} 5-10yr Beta Forecast by Horizon",
    ylabel="$R^2$",
    xlabel="Forecast Horizon (weeks)",
    figsize=(16, 6),
)
ax.set_ylim(-0.1, 1.1)
vis.show()
vis.savefig(f"Beta_adjusted_performance_{universe}_5-10_model_horizons")
