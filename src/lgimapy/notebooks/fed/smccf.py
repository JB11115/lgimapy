import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%
df = pd.DataFrame(
    [
        ("6/16/2020", 255),
        ("6/22/2020", 255),
        ("6/29/2020", 337),
        ("7/9/2020", 177),
        ("7/16/2020", 154),
        ("7/23/2020", 123),
        ("7/30/2020", 31),
        ("8/6/2020", 24),
        ("8/13/2020", 12),
        ("8/20/2020", 13),
        ("8/27/2020", 24),
        ("9/3/2020", 26),
        ("9/10/2020", 6),
        ("9/17/2020", 27),
        ("9/24/2020", 9),
        ("10/1/2020", 14),
        ("10/8/2020", 24),
        ("10/15/2020", 34),
        ("10/22/2020", 16),
        ("10/29/2020", 20),
        ("11/5/2020", 25),
        ("11/12/2020", 20.6),
        ("11/19/2020", 34),
    ],
    columns=["date", "amt"],
)
df["date"] = pd.to_datetime(df["date"])
s = df.set_index("date").squeeze().sort_index()


db = Database()
spreads = db.load_bbg_data("US_IG", "OAS", start=s.index[0])
# %%
lax, rax = vis.plot_double_y_axis_timeseries(
    s,
    spreads,
    ylabel_left="SMCCF Daily Purchases",
    ylabel_right="Market Credit OAS",
    ytickfmt_left="${x:.0f}M",
    lw=3,
    plot_kws_left={"color": "goldenrod"},
    plot_kws_right={"color": "darkgreen"},
    ret_axes=True,
)
lax.set_ylim(-450, 350)

vis.savefig("SMCCF_update")
