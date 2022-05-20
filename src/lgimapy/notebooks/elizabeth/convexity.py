import warnings
from collections import defaultdict

import matplotlib as mpl
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document

vis.style()

# %%

db = Database()
db.load_market_data(start=db.date("1Y"))
today_ix = db.build_market_index(in_H0A0_index=True, date=db.date("today"))

port = db.load_portfolio(strategy="US High Yield")
bond_holdings = (
    port.bond_overweights(by="P_Notional", index="CUSIP").fillna(0) / 1e6
)

# %%
def plot_convexity(df, ticker):
    fig, ax = vis.subplots()
    im = ax.scatter(
        df["short_cusip_ytw"].iloc[:-1],
        df["px_ratio"].iloc[:-1],
        c=df["short_cusip_px"].iloc[:-1],
        alpha=0.6,
        vmin=60,
        vmax=100,
        cmap="viridis",
    )
    ax.scatter(
        df["short_cusip_ytw"].iloc[-1],
        df["px_ratio"].iloc[-1],
        s=60,
        c="firebrick",
        label=f"Last: ${df['long_cusip_px'].iloc[-1]:.2f}",
    )
    cmap = vis.cmap("viridis")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Long Bond \$px")

    ax.set_ylabel("Long Bond \$px / Short Bond \$px")
    ax.set_xlabel("Short Bond Yield")
    vis.legend(ax)
    vis.format_xaxis(ax, xtickfmt="{x:.1%}")
    ax.set_title(f"{ticker} Convexity")


def plot_bond(df, bond, title):
    fig, ax = vis.subplots()
    ax.scatter(
        df[f"{bond}_cusip_ytw"], df[f"{bond}_cusip_px"], color="navy", alpha=0.6
    )
    ax.scatter(
        df[f"{bond}_cusip_ytw"].iloc[-1],
        df[f"{bond}_cusip_px"].iloc[-1],
        color="firebrick",
        s=50,
        label="Last",
    )
    ax.set_ylabel("\$px")
    ax.set_xlabel("Yield to Worst")
    vis.format_xaxis(ax, xtickfmt="{x:.1%}")
    vis.legend(ax)
    ax.set_title(title)
    vis.show()


# %%
tickers = today_ix.tickers

doc = Document("HY_convexity", path="latex/HY/2022/05", fig_dir=True)
tickers = [
    "BHCCN",
    "CHTR",
    "DISH",
    "FE",
    "NAVI",
    "OMF",
    "QVCN",
    "RKTRM",
    "ROCKIE",
    "RRR",
    "SVC",
    "VOD",
]
fids = {}
d = defaultdict(list)
for ticker in tickers:
    ticker_today_ix = today_ix.subset(
        ticker=ticker, OAD=(3, None), dirty_price=(None, 90)
    )
    if ticker_today_ix.is_empty():
        continue

    ticker_today_ix.add_bond_description()
    ticker_oad = ticker_today_ix.df["OAD"]
    if ticker_oad.max() - ticker_oad.min() < 2:
        continue
    long_cusip = ticker_oad.idxmax()
    short_cusip = ticker_oad.idxmin()

    ticker_ix = db.build_market_index(cusip=[long_cusip, short_cusip])
    price_df = ticker_ix.get_value_history("DirtyPrice")
    price_df["px_ratio"] = price_df[long_cusip] / price_df[short_cusip]
    yield_df = ticker_ix.get_value_history("YieldToWorst") / 100

    df = pd.concat(
        (
            price_df[short_cusip].rename("short_cusip_px"),
            price_df[long_cusip].rename("long_cusip_px"),
            yield_df[short_cusip].rename("short_cusip_ytw"),
            yield_df[long_cusip].rename("long_cusip_ytw"),
            price_df["px_ratio"],
        ),
        axis=1,
    ).dropna()
    df = df[df["short_cusip_ytw"] > 0]

    long_bond = ticker_today_ix.df.loc[long_cusip]
    short_bond = ticker_today_ix.df.loc[short_cusip]
    d["Short Bond"].append(short_bond["BondDescription"])
    d["Price"].append(short_bond["DirtyPrice"])
    d["Holdings*(\$M)"].append(bond_holdings.get(short_cusip, 0))
    d["Long Bond"].append(long_bond["BondDescription"])
    d["Price "].append(long_bond["DirtyPrice"])
    d["Holdings*(\$M) "].append(bond_holdings.get(long_cusip, 0))

    plot_convexity(df, ticker)
    vis.savefig(f"{ticker}_convexity", path=doc.fig_dir)
    vis.close()

    plot_bond(df, "long", long_bond["BondDescription"])
    vis.savefig(f"{ticker}_long", path=doc.fig_dir)
    vis.close()

    plot_bond(df, "short", short_bond["BondDescription"])
    vis.savefig(f"{ticker}_short", path=doc.fig_dir)
    vis.close()

table = pd.DataFrame(d, index=tickers)

# %%
doc = Document("HY_convexity", path="latex/HY/2022/05", fig_dir=True)
doc.add_preamble(
    bookmarks=True,
    margin={
        "left": 0.5,
        "right": 0.5,
        "top": 0.5,
        "bottom": 0.2,
    },
)
doc.add_section("Summary")
doc.add_table(
    table,
    col_fmt="l|lrr|lrr",
    adjust=True,
    multi_row_header=True,
    caption="Tickers Identified to Exhibit Convexity",
    prec={col: "2f" for col in table.columns if not col.endswith("Bond")},
)
doc.add_pagebreak()
for ticker in tickers:
    doc.add_section(ticker)
    doc.add_figure(f"{ticker}_convexity")
    doc.add_figure([f"{ticker}_short", f"{ticker}_long"])

    doc.add_pagebreak()

doc.save()
table["Price "].mean()
table["Price"].mean()
