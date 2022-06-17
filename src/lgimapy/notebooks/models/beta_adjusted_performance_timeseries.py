from lgimapy import vis
from lgimapy.data import Database
from lgimapy.models import BetaAdjustedPerformance

vis.style()

# %%

mod = BetaAdjustedPerformance(Database())
df = mod._model_history_df
df.columns
vis.plot_timeseries(df['IG_maturity=(5, 10)'].rolling(100).mean())
vis.show()
