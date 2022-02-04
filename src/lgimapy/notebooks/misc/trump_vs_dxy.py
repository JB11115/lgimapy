from lgimapy import vis
from lgimapy.bloomberg import bdh

vis.style()
# %%

trump = (
    bdh("RCPPTAPP", "Index", "PX_LAST", start="11/1/2016")
    .squeeze()
    .rename("Trump Approval")
    / 100
)
dollar = (
    bdh("DXY", "Curncy", "PX_LAST", start="11/1/2016")
    .squeeze()
    .rename("DXY Spot")
)

vis.plot_double_y_axis_timeseries(
    trump,
    dollar,
    ylabel_right="DXY Spot",
    ylabel_left="Trump Appoval",
    ytickfmt_left="{x:.0%}",
    ytickfmt_right="${x:.0f}",
    plot_kws_right={"color": "k"},
    plot_kws_left={"color": "firebrick"},
)
# vis.savefig("trump_vs_dxy")
vis.show()
