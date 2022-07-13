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


def plot_spread_multiple(
    cols, fid, start=None, colors=None, end=None, beta_adjusted=False
):
    db = Database()
    if start is None:
        start = db.date("YTD")
    oas_df = db.load_bbg_data(cols, "OAS", start=start, end=end).fillna(
        method="ffill"
    )
    delta_oas_df = (oas_df.iloc[-1, :] - oas_df.iloc[0, :]) / (
        oas_df.iloc[0, :]
    )

    if beta_adjusted:
        oad_df = db.load_bbg_data(cols, "OAS", start=start, end=end).fillna(
            method="ffill"
        )
        mv_df = db.load_bbg_data(cols, "MV", start=start, end=end).fillna(
            method="ffill"
        )
        dts_df = oas_df * oad_df
        mv_dts = (mv_df * dts_df).iloc[0, :]
        beta = mv_dts / mv_dts.mean()

    else:
        df = delta_oas_df.copy()
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
            label=f"{col}: {val:.2f}",
            ylabel="Spread Multiple",
            ax=ax,
        )
    vis.legend(ax)
    vis.savefig(fid)


plot_spread_multiple(
    ["US_IG_10+", "US_IG", "US_HY", "EU_IG", "GBP_IG"],
    fid="2022_Index_Spread_YTD_multiple",
)

plot_spread_multiple(
    ["US_AAA", "US_AA", "US_A", "US_BBB", "US_BB", "US_B", "US_CCC"],
    fid="2022_Rating_Spread_YTD_multiple",
)
