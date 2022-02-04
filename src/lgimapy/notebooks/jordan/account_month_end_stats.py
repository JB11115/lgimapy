from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database

# %%

account = "CNPIC"

db = Database()
dates = db.date("month_ends")
dates = [d for d in dates if d >= pd.to_datetime("1/1/2018")]
d = defaultdict(list)
for date in tqdm(dates):
    try:
        port = db.load_portfolio(account=account, date=date, universe="stats")
    except ValueError:
        continue

    d["Date"].append(date)
    cols = ["OAS", "OASD", "OAD", "DTS"]
    for col in cols:
        if col == "OAD":
            p_val = port.full_df[f"P_{col}"].sum()
        else:
            p_val = port.df[f"P_{col}"].sum()
        bm_val = port.df[f"BM_{col}"].sum()

        d[f"Port_{col}"].append(p_val)
        d[f"BM_{col}"].append(bm_val)
        d[f"{col}_Diff"].append(p_val - bm_val)
        d[f"{col}_Ratio"].append(p_val / bm_val)

df = pd.DataFrame(d).set_index("Date").rename_axis(None).sort_index()
# %%
df.to_csv(f"{account}_stats_index_month_end_stats.csv")
