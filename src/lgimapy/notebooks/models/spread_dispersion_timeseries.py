from lgimapy.data import Database
from lgimapy.models import Dispersion
from lgimapy.latex import Document

# %%
def plot_timeseries(
    self,
    maturity,
    rating,
    correction_methodology,
    sector_methodology,
    start,
):
    if sector_methodology == "inter":
        df = self._load(self._inter_sector_fid(rating, maturity))
    elif sector_methodology == "intra":
        df = self._load(self._intra_sector_fid(rating, maturity))
    col_ending = {
        "abs": "IQR_plus_MAD",
        "rel": "QCV_plus_RMAD",
    }
    df = df[
        [
            col
            for col in df.columns
            if col.endswith(col_ending[correction_methodology])
        ]
    ]
    ts = df.rank(pct=True).mean(axis=1)
    title = (
        f"{maturity}yr {rating} {correction_methodology} "
        f"{sector_methodology}"
    )
    vis.plot_timeseries(
        ts,
        color="navy",
        ytickfmt="{x:.0%}",
        start=self._db.date(start),
        title=title,
        figsize=(12, 5),
    )

    fid = title.replace(" ", "_")
    return fid

# %%
db = Database()
doc = Document(
    "Spread_Dispersion_timeseries", path="latex/IG_dispersion", fig_dir=True
)
doc.add_preamble(margin={"top": 1.5, "bottom": 0.2})
self = Dispersion("IG", db)
date = db.date("today")

for maturity in [10, 30]:
    for rating in ["A", "BBB"]:
        for correction_methodology in ["abs", "rel"]:
            for sector_methodology in ["inter", "intra"]:
                fid = plot_timeseries(
                    self,
                    maturity,
                    rating,
                    correction_methodology,
                    sector_methodology,
                    start="3y",
                )
                doc.add_figure(fid, savefig=True, dpi=200)

doc.save()
