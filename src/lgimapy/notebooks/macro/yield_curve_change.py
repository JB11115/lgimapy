import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%

db = Database()
dates = pd.to_datetime([db.date("1m"), db.date("today")])

maturities = [2, 3, 5, 7, 10, 20, 30]
tsys = [f"UST_{mat}Y" for mat in maturities]
df = db.load_bbg_data(tsys, "YTW", start=dates[0], end=dates[1])

# %%
fig, (top_ax, bottom_ax) = vis.subplots(
    2,
    1,
    sharex=True,
    figsize=(12, 8),
    gridspec_kw={"height_ratios": [3, 1]},
)
colors = ["dodgerblue", "navy"]
for date, color in zip(dates, colors):
    top_ax.plot(
        maturities,
        df.loc[date],
        "-o",
        lw=2,
        ms=8,
        color=color,
        label=date.strftime("%b %#d"),
    )
vis.legend(top_ax, fontsize=16)
vis.format_yaxis(top_ax, "{x:.1%}")
top_ax.set_ylabel("Treasury Yield")


raw_change_df = 1e4 * (df.loc[dates[1]] - df.loc[dates[0]])
raw_change_df.index = maturities
change_df = pd.Series(0, index=range(32))
for k, v in raw_change_df.items():
    change_df.loc[k] = v

change_df.plot.bar(ax=bottom_ax, color="navy", alpha=0.9)
bottom_ax.grid(False, axis="x")
vis.format_yaxis(bottom_ax, "{x:.0f} bp")
bottom_ax.axhline(0, lw=2, c="k", alpha=0.8)

y_min = min(-1, change_df.min() - 8)
y_max = max(1, change_df.max() + 8)
bottom_ax.set_ylim(y_min, y_max)
x_labels = [mat if mat in maturities else "" for mat in change_df.index]
bottom_ax.set_xticklabels(x_labels, rotation=0)
bottom_ax.set_xticks(maturities)
vis.set_n_ticks(bottom_ax, 6)
bottom_ax.set_xlabel("Maturity")
bottom_ax.set_ylabel("Change")

n_patches = len(bottom_ax.patches)
for i, p in enumerate(bottom_ax.patches):
    if i not in maturities:
        continue
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    y_label_pos = y + 0.5 if y >= 0 else y - 0.5
    bottom_ax.annotate(
        f"{y:+.0f}",
        (x, y_label_pos),
        va="bottom" if y >= 0 else "top",
        ha="center",
        fontweight="bold",
        fontsize=10,
        color="navy",
    )

top_ax.set_title("Change in Treasury Yield Curve", fontweight="bold")
vis.savefig("tsy_curve_change")
