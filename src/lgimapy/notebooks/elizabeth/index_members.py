from collections import defaultdict
from unicodedata import numeric

import pandas as pd

from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root

# %%

index = "h4un"
start = "2020-07-01"
end = "2020-08-03"

start_fid = root(f"data/HY/{index}_{start}.csv")
end_fid = root(f"data/HY/{index}_{end}.csv")

# %%
def load_csv(fid):
    df = pd.read_csv(fid, index_col=0, engine="python", encoding="utf_8_sig")
    df = df[df["Weight (%)"] > 0]
    df["Weight"] = df["Weight (%)"] / 100
    df["Ticker"] = [desc.split()[0] for desc in df["Description"]]
    return df.rename_axis(None)


start_df = load_csv(start_fid)
end_df = load_csv(end_fid)

dropped_cusips = set(start_df.index) - set(end_df.index)
dropped_df = start_df[start_df.index.isin(dropped_cusips)]

new_cusips = set(end_df.index) - set(start_df.index)
new_df = end_df[end_df.index.isin(new_cusips)]


new_issues = (
    new_df.groupby("Ticker").sum().sort_values("Weight (%)", ascending=False)
)
dropped_issues = (
    dropped_df.groupby("Ticker")
    .sum()
    .sort_values("Weight (%)", ascending=False)
)


def add_issuer_to_df(df):
    db = Database()
    db.load_market_data(local=True)
    ix = db.build_market_index(in_hy_stats_index=True)
    issuer_mv = (
        ix.df[["Issuer", "MarketValue"]].groupby("Issuer").sum().squeeze()
    )
    ticker_to_issuer = {}
    for ticker, gdf in ix.df.groupby("Ticker"):
        if len(gdf) < 1:
            continue
        issuer_df = (
            gdf.groupby("Issuer", observed=True)
            .sum()
            .sort_values("MarketValue", ascending=False)
        )
        ticker_to_issuer[ticker] = issuer_df.index[0]
    df["Issuer"] = df.index.map(ticker_to_issuer)
    df.rename_axis(None, inplace=True)
    return df[["Issuer", "Weight"]].copy()


new_issue_df = add_issuer_to_df(new_issues)
total = new_issue_df.sum(numeric_only=True)
table_new_issues = new_issue_df.head(10).append(pd.Series(total, name="Total"))

out_index_df = add_issuer_to_df(dropped_issues)
total = out_index_df.sum(numeric_only=True)
table_dropped = out_index_df.head(10).append(pd.Series(total, name="Total"))


# %%
fid = "new_issues"
doc = Document(fid, path="reports/HY")
doc.add_preamble(table_caption_justification="c")
doc.add_table(
    table_new_issues,
    prec={"Weight": "2%"},
    col_fmt="llc",
    midrule_locs=["Total"],
    alternating_colors=(None, "lightgray"),
    caption="Largest Issuers Entering Index",
)
doc.add_table(
    table_dropped,
    prec={"Weight": "2%"},
    col_fmt="llc",
    midrule_locs=["Total"],
    alternating_colors=(None, "lightgray"),
    caption="Largest Issuers Exiting Index",
)
doc.save()


# %%


# %%
text = start_df["Description"][0]
str(text)
text.split()
from tqdm import tqdm

# %%
df = start_df.copy()
d = defaultdict(list)


temp = pd.DataFrame(d)
temp
fid = start_fid

# %%

fid = end_fid
raw_df = pd.read_csv(fid, index_col=0, engine="python", encoding="utf_8_sig")


def decode_coupon(x):
    coupon = 0
    for val in x:
        try:
            coupon += float(val)
        except ValueError:
            coupon += numeric(val)
    return coupon


def decode_date(x):
    if x == "PERP":
        return pd.to_datetime(pd.Timestamp.max.date())
    else:
        return pd.to_datetime(x)


d = defaultdict(list)
for text in raw_df["Description"]:
    ticker, *raw_coup, raw_date = text.split()
    d["Desc"].append(text)
    d["Ticker"].append(ticker)
    d["Coupon"].append(decode_coupon(raw_coup))
    d["MaturityDate"].append(decode_date(raw_date))

df = pd.DataFrame(d)
df
