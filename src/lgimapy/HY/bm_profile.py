from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document

# %%
db = Database()
dates = {
    "YE 2007": db.date("YTD", "1/15/2008"),
    "YE 2014": db.date("YTD", "1/15/2015"),
    "YE 2018": db.date("YTD", "1/15/2019"),
    "Feb 2020": db.date("MTD", "2/15/2020"),
    "YE 2020": db.date("YTD", "1/15/2021"),
    "Current": db.date("today"),
}
ixs = {}
for name, date in dates.items():
    db.load_market_data(date)
    ixs[name] = db.build_market_index(in_H0A0_index=True)

# %%
ix = ixs["Current"]


def make_overview_table(ixs):
    d = defaultdict(list)
    for date, ix in ixs.items():
        d["Issuers"].append(len(ix.tickers))
        mv = ix.total_value().iloc[0] / 1e3
        d["MV ($B)"].append(mv)
        d["OAS"].append(ix.market_value_weight("OAS").iloc[0])
        d["YTW"].append(ix.market_value_weight("YieldToWorst").iloc[0])
        d["OAD"].append(ix.market_value_weight("OAD").iloc[0])
        for rating in ["BB", "B", "CCC"]:
            ix_sub = ix.subset(rating=(f"{rating}+", f"{rating}-"))
            mv_sub = ix_sub.total_value().iloc[0] / 1e3
            d[f"{rating} %"].append(mv_sub / mv)

    return pd.DataFrame(d, index=ixs.keys())


overview_table = make_overview_table(ixs)
overview_table
