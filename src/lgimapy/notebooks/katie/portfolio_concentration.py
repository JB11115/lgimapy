from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database

vis.style()
# %%

db = Database()
dates = db.date("MONTH_STARTS", start=db.date("PORTFOLIO_START"))

# %%
d = defaultdict(list)
for date in tqdm(dates):
    port = db.load_portfolio(account="P-LD", date=date)
    p_df = port.df[port.df["P_Weight"] > 0]
    n_issuers_port = len(p_df["Ticker"].unique())
    n_issuers_bm = len(port.df["Ticker"].unique())
    d["# Issuers Held"].append(n_issuers_port)
    d["% Issuers Held"].append(np.round(100 * n_issuers_port / n_issuers_bm, 1))
    ow = port.ticker_df.sort_values("P_OAD", ascending=False)["P_OAD"]
    for n in [10, 20, 30]:
        pct = np.round(100 * ow.iloc[:n].sum() / ow.sum(), 1)
        d[f"% of Portfolio Duration in top {n} Holdings"].append(pct)

df = pd.DataFrame(d, index=dates)
# %%
df.to_csv("LD_portfolio_concentration_timeseries.csv")
df

# %%
db = Database()
port = db.load_portfolio(account="P-LD")
# %%
s = port.ticker_df.sort_values("P_OAD", ascending=False).reset_index()["P_OAD"]
s = s[s > 0]
s /= s.sum()
s.index += 1
s_cum = np.cumsum(s)

# %%
# fig, ax = vis.subplots()
# ax.plot(s_cum, color='steelblue')
# ax.set_title('Cumulative Portfolio Duration', fontweight='bold')
# ax.set_xlabel('$n^{th}$ Largest Issuer Holding')
# vis.format_yaxis(ax, '{x:.0%}')
# vis.savefig('cumulative_portfolio_duration_by_issuer')
