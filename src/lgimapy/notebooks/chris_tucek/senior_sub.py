from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%

account = "CITLD"

snr = ["UNSECURED"]
sec = ["1ST MORTGAGE", "SECURED", "COLLATERAL TRUST", "PASS THRU CERTS"]
sub = ["SUBORDINATED", "JR SUBORDINATED"]
# %%
db = Database()
dates = [
    db.nearest_date(d)
    for d in pd.date_range("1/1/2019", db.date("today"), freq="M")
]
d = defaultdict(list)
for date in dates:
    port = db.load_portfolio(account=account, date=date)
    port_snr = port.subset(collateral_type=snr)
    port_sec = port.subset(collateral_type=sec)
    port_sub = port.subset(collateral_type=sub)
    d["BM Snr %"].append(port_snr.df["BM_Weight"].sum())
    d["BM Secured %"].append(port_sec.df["BM_Weight"].sum())
    d["BM Sub %"].append(port_sub.df["BM_Weight"].sum())
    d[f"{account} Snr %"].append(port_snr.df["P_Weight"].sum())
    d[f"{account} Secured %"].append(port_sec.df["BM_Weight"].sum())
    d[f"{account} Sub %"].append(port_sub.df["P_Weight"].sum())

df = pd.DataFrame(d, index=dates)


# %%
df

# %%
fig, ax = vis.subplots()
colors = ['navy', 'navy', 'navy', 'darkorange', 'darkorange', 'darkorange']
lines = ['-', '--', ':', '-', '--', ":"]
for col, color, ls, in zip(df.columns, colors, lines):
    label = f"{col}, Current={df[col].iloc[-1]:.1%}"
    vis.plot_timeseries(df[col], color=color, ls=ls, label=label, ax=ax)
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
vis.savefig(f'{account}_snr_sec_sub_timeseries')


# %%
db = Database()
dates = [
    db.nearest_date(d)
    for d in pd.date_range("1/1/2000", db.date("today"), freq="3M")
]
d = defaultdict(list)
for date in dates:
    db.load_market_data(date=date)
    ix = db.build_market_index(in_returns_index=True, maturity=(10, None))
    ix_mv = ix.total_value()[0]
    ix_snr = ix.subset(collateral_type=snr).total_value()[0]
    ix_sec = ix.subset(collateral_type=sec).total_value()[0]
    ix_sub = ix.subset(collateral_type=sub).total_value()[0]
    d["LDC Snr %"].append(ix_snr / ix_mv)
    d["LDC Secured %"].append(ix_sec / ix_mv)
    d["LDC Sub %"].append(ix_sub / ix_mv)

df = pd.DataFrame(d, index=dates)

# %%
fig, ax = vis.subplots()
lines = ['-', '--', ':']
for col, ls, in zip(df.columns, lines):
    label = f"{col}, Current={df[col].iloc[-1]:.1%}"
    vis.plot_timeseries(df[col], color='navy', ls=ls, label=label, ax=ax)
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
vis.savefig(f'LDC_snr_sec_sub_timeseries')
