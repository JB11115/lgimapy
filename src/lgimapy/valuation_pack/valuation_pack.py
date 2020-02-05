from datetime import datetime as dt
from shutil import copy

from lgimapy import vis
from lgimapy.latex import Document
from lgimapy.utils import root, Time

from cover_page import update_cover_page
from cross_asset import update_market_review
from macro_indicators import update_macro_indicators


# %%
def main():
    # Duplicate template file and rename to todays date.
    fid = f"{dt.today().strftime('%Y-%m-%d')}_Valuation_Pack"
    directory = root("latex/valuation_pack")
    src = directory / "template.tex"
    dst = directory / f"{fid}.tex"
    copy(src, dst)

    vis.style()
    doc = Document(fid, path="valuation_pack", fig_dir=True)
    update_cover_page(fid)
    update_market_review(fid)
    update_macro_indicators(fid)
    doc = Document(fid, path="valuation_pack", fig_dir=True, load_tex=True)
    doc.save()


if __name__ == "__main__":
    main()