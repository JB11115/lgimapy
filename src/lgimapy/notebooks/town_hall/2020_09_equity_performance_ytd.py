import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdh, bdp, bds

vis.style()
# %%
sp_tickers = bds("SPX", "Index", "INDX_MEMBERS").squeeze()
price_df = bdh(sp_tickers, "Equity", "PX_LAST", "1/1/2020")
market_cap_df = bdh(sp_tickers, "Equity", "CUR_MKT_CAP", "1/1/2020")

russ_tret_raw = bdh("RU20INTR", "Index", "PX_LAST", "1/1/2020").squeeze()
russ_tret = russ_tret_raw / russ_tret_raw[0]
# %%
n = 10
top_n = set(market_cap_df.iloc[-1].sort_values().index[-n:])
ex_top_n = set(sp_tickers) - top_n


def get_returns(p_df, mc_df, tickers=None):
    if tickers is None:
        sorted_tickers = sorted(list(p_df.columns))
    else:
        sorted_tickers = sorted(list(tickers))

    mc_df = mc_df[sorted_tickers]
    weight_df = mc_df.divide(mc_df.sum(axis=1), axis=0)
    # Change index so weights are from previous day.
    weights_index = weight_df.index[1:]
    weight_df = weight_df.iloc[:-1]
    weight_df.index = weights_index

    ret_df = price_df[sorted_tickers].pct_change()[1:]
    weighted_ret = ret_df * weight_df
    port_ret = weighted_ret.sum(axis=1)
    return np.cumprod(port_ret + 1)


mega_caps = get_returns(price_df, market_cap_df, top_n)
large_caps = get_returns(price_df, market_cap_df, ex_top_n)

# %%
fig, ax = vis.subplots(figsize=(8, 6))
ax.axhline(1, color="k", ls="--", lw=1, label="_nolegend_")
kws = {"alpha": 0.9, "lw": 3, "ax": ax}
vis.plot_timeseries(mega_caps, color="#003D73", label="Mega Caps", **kws)
vis.plot_timeseries(
    large_caps, color="#1ECFD6", label="Large Caps (ex-Mega)", **kws
)
vis.plot_timeseries(russ_tret, color="#C05640", label="Small Caps", **kws)
ax.legend(shadow=True, fancybox=True)
ax.set_ylabel("YTD Return Ratio")
vis.savefig("equities_ytd")
# vis.show()


# %%
