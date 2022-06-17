from lgimapy import vis
from lgimapy.data import Database
# %%
vis.style()
db = Database()
df = db.load_bbg_data(['VIX', 'MOVE'], 'Price', start='1/1/2010', nan='ffill').dropna()
vis.plot_double_y_axis_timeseries(
    df['VIX'],
    df['MOVE'],
)
vis.show()

# %%
vis.plot_timeseries(df['MOVE'] / df['VIX'])
vis.show()
