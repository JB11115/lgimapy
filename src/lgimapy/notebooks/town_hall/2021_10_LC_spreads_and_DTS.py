import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%
account = 'P-LD'

db = Database()
dates = db.date("MONTH_STARTS", start='9/15/2020')
dates.append(db.date('today'))

dts_list = []
for date in dates:
    port = db.load_portfolio(account=account, date=date)
    dts_list.append(port.dts())


dts = pd.Series(dts_list, index=dates)
lc_oas = db.load_bbg_data('US_IG_10+', 'OAS', start=dts.index[0])
# %%
vis.plot_double_y_axis_timeseries(
    dts.rename('Long Duration DTS'),
    lc_oas.rename("Long Credit Index OAS (bp)"),
    ytickfmt_left='{x:.0%}',
)
vis.savefig("Long_duration_credit_DTS_vs_OAS")
