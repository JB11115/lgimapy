import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%

cols = ["US_AAA", "US_AA", "US_A", "US_BBB", "US_BB", "US_B", "US_CCC"]
start = None
end = None
colors = None
beta_adjusted = True


def plot_spread_return(
    cols, fid, start=None, colors=None, end=None, beta_adjusted=False
):
    db = Database()
    if start is None:
        start = db.date("YTD")
    oas_df = db.load_bbg_data(cols, "OAS", start=start, end=end).fillna(
        method="ffill"
    )
    delta_oas_df = oas_df - oas_df.iloc[0, :]
    spread_return_df = -delta_oas_df / oas_df.iloc[0, :]
    if beta_adjusted:
        oad_df = db.load_bbg_data(cols, "OAS", start=start, end=end).fillna(
            method="ffill"
        )
        mv_df = db.load_bbg_data(cols, "MV", start=start, end=end).fillna(
            method="ffill"
        )
        dts_df = oas_df * oad_df
        mv_dts = (mv_df * dts_df).iloc[0, :]
        beta_s = mv_dts / mv_dts.mean()

        mean_delta_oas = (mv_df * delta_oas_df).sum(axis=1) / mv_df.sum(axis=1)
        beta_fcast_delta_oas_df = pd.concat(
            (
                (mean_delta_oas * beta).rename(col)
                for col, beta in beta_s.items()
            ),
            axis=1,
        )
        beta_adjusted_delta_oas_df = delta_oas_df - beta_fcast_delta_oas_df
        beta_adjusted_spread_return_df = (
            -beta_adjusted_delta_oas_df / oas_df.iloc[0, :]
        )
        df = beta_adjusted_spread_return_df.copy()
    else:
        df = spread_return_df.copy()
    df.columns = db.bbg_names(df.columns)

    fig, ax = vis.subplots(figsize=(12, 8))
    if colors is None:
        colors = sns.color_palette("dark", len(df.columns))

    last_vals = df.iloc[-1, :].sort_values(ascending=False)
    for (col, val), color in zip(last_vals.items(), colors):
        vis.plot_timeseries(
            df[col],
            color=color,
            alpha=0.7,
            label=f"{col}: {val:.0%}",
            ylabel="Spread Return",
            ytickfmt="{x:.0%}",
            ax=ax,
        )
    vis.legend(ax)
    vis.savefig(fid)


plot_spread_return(
    ["US_IG_10+", "US_IG", "US_HY", "EU_IG", "GBP_IG"],
    fid="2022_Index_Spread_Return_YTD",
)

plot_spread_return(
    ["US_AAA", "US_AA", "US_A", "US_BBB", "US_BB", "US_B", "US_CCC"],
    fid="2022_Rating_Spread_Return_YTD",
)

plot_spread_return(
    ["US_AAA", "US_AA", "US_A", "US_BBB", "US_BB", "US_B", "US_CCC"],
    beta_adjusted=True,
    fid="2022_Rating_Beta_Adjusted_Spread_Return_YTD",
)
