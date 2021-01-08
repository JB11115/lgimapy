import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import load_json, dump_json

# %%
def get_covid_sensative_tickers():
    """
    Update the top 30 tickers used for the
    BBB Top 30 Non-Fin indexes.
    """
    fid = "indexes"
    indexes = load_json(fid)

    db = Database()
    as_of_date = db.nearest_date("5/15/2020")
    prev_date = db.nearest_date("12/1/2019")
    base_kwargs = {"in_stats_index": True}
    kwargs_d = {
        "10": {**base_kwargs, "maturity": (8.25, 11)},
        "10+": {**base_kwargs, "maturity": (25, 32)},
    }

    for name, kwargs in kwargs_d.items():
        # Get current spreads.
        db.load_market_data(date=as_of_date)
        ix = db.build_market_index(**kwargs)
        ix.calc_dollar_adjusted_spreads()
        post_wides_df = ix.ticker_df

        # Get previous spreads and combine.
        db.load_market_data(prev_date)
        ix = db.build_market_index(**kwargs)
        ix.calc_dollar_adjusted_spreads()
        prev_df = ix.ticker_df
        prev_df["Prev_PX_Adj_OAS"] = prev_df["PX_Adj_OAS"]
        df = pd.concat(
            (post_wides_df, prev_df["Prev_PX_Adj_OAS"]), axis=1, join="inner"
        )

        # Find change in spreads for each ticker and get cumulative market value.
        df["spread_chg"] = df["PX_Adj_OAS"] - df["Prev_PX_Adj_OAS"]
        df.sort_values("spread_chg", inplace=True)
        df["MV%"] = df["MarketValue"] / df["MarketValue"].sum()
        df["Cum MV"] = np.cumsum(df["MV%"])
        covid_sensative_df = df[df["Cum MV"] > 0.8]
        covid_sensative_issuers = sorted(list(set(covid_sensative_df.index)))

        indexes[f"COVID_SENSITIVE_{name}"] = kwargs
        indexes[f"COVID_SENSITIVE_{name}"]["name"] = "Covid Sensitive"
        indexes[f"COVID_SENSITIVE_{name}"]["ticker"] = covid_sensative_issuers

        indexes[f"COVID_INSENSITIVE_{name}"] = kwargs
        indexes[f"COVID_INSENSITIVE_{name}"]["name"] = "Covid Insensitive"
        indexes[f"COVID_INSENSITIVE_{name}"]["ticker"] = covid_sensative_issuers
        indexes[f"COVID_INSENSITIVE_{name}"]["special_rules"] = "~Ticker"

    # Save changes.
    dump_json(indexes, fid)


if __name__ == "__main__":
    get_covid_sensative_tickers()


from lgimapy import vis

vis.style()


def update_covid_insensitive_bbb_a_ratios():
    """Update plot for BBB/A nonfin ratio for 10y and 30y bonds."""
    db = Database()
    db.load_market_data(start=db.date("5y"))
    # %%
    ix_insens_LC = db.build_market_index(
        **db.index_kwargs("COVID_INSENSITIVE_10+", financial_flag=0)
    )
    ix_insens_10 = db.build_market_index(
        **db.index_kwargs("COVID_INSENSITIVE_10", financial_flag=0)
    )

    ixs = {
        "30_A": ix_insens_LC.subset(rating=("A+", "A-")),
        "30_BBB": ix_insens_LC.subset(rating=("BBB+", "BBB-")),
        "10_A": ix_insens_10.subset(rating=("A+", "A-")),
        "10_BBB": ix_insens_10.subset(rating=("BBB+", "BBB-")),
    }
    df = pd.concat(
        [ix.market_value_median("OAS").rename(key) for key, ix in ixs.items()],
        axis=1,
        sort=True,
    ).dropna(how="any")
    df["10 yr"] = df["10_BBB"] / df["10_A"]
    df["30 yr"] = df["30_BBB"] / df["30_A"]

    # Plot
    fig, ax_left = vis.subplots(figsize=(9, 6))
    ax_right = ax_left.twinx()
    ax_right.grid(False)

    ax_left.plot(df["10 yr"], c="navy", alpha=0.9, lw=2)
    ax_left.set_ylabel("10 yr", color="navy")
    ax_left.tick_params(axis="y", colors="navy")
    ax_left.axhline(np.median(df["10 yr"]), ls=":", lw=1.5, color="navy")

    ax_right.plot(df["30 yr"], c="goldenrod", alpha=0.9, lw=2)
    ax_right.axhline(
        np.median(df["30 yr"]),
        ls=":",
        lw=1.5,
        color="goldenrod",
        label="Median",
    )
    pct = {x: np.percentile(df["30 yr"], x) for x in [5, 95]}
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    ax_right.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    ax_right.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
    ax_right.set_title(
        "Covid Insensitive Non-Fin BBB/A Ratio", fontweight="bold"
    )
    ax_right.set_ylabel("30 yr", color="goldenrod")
    ax_right.tick_params(axis="y", colors="goldenrod")
    vis.format_xaxis(ax_right, df["30 yr"], "auto")
    vis.set_percentile_limits([df["10 yr"], df["30 yr"]], [ax_left, ax_right])
    ax_right.legend()
    vis.savefig("Covid_Insensitive_Non-Fin_BBB_A_Ratio")
    vis.close()
    # %%
