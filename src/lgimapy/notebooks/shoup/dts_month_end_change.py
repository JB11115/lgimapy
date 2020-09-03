from collections import defaultdict

import pandas as pd

from lgimapy.data import Database

# %%
db = Database()
strats = {
    "Long Credit": "P-LD",
    "Long Corp": "GSKLC",
    "Liability Aware": "JOYLA",
    "Long Corp A": "CARGLC",
    "Long Gov/Credit": "USBGC",
}

date = db.date("month_end", "6/1/2020")
print(date.strftime("%m/%d/%Y"))


d = defaultdict(list)
for name, account_name in strats.items():
    ret_acnt = db.load_portfolio(account=account_name, date=date)
    stats_acnt = db.load_portfolio(
        account=account_name, date=date, universe="stats"
    )
    d["Returns DTS"].append(ret_acnt.df["BM_DTS"].sum())
    d["Stats DTS"].append(stats_acnt.df["BM_DTS"].sum())

df = pd.DataFrame(d, index=strats.keys())
df["Difference (abs)"] = df["Stats DTS"] - df["Returns DTS"]
df["Difference (%)"] = 100 * (df["Stats DTS"] / df["Returns DTS"] - 1)
df.round(1)
