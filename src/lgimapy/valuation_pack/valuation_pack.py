import os
from datetime import datetime as dt
from shutil import copy

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root, Time

from cover_page import update_cover_page
from cross_asset import update_market_review
from tail_recovery import update_lc_tail
from macro_indicators import update_macro_indicators
from equilibium_pca_model import update_equilibrium_model
from sector_vol import update_volatility_indicators

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
    db = Database()
    db.load_market_data(local=True, start=db.date("5y"))
    update_cover_page(fid, db)
    update_market_review(fid)
    update_lc_tail(fid)
    update_macro_indicators(fid)
    update_equilibrium_model(fid)
    update_volatility_indicators(fid, db)
    doc = Document(fid, path="valuation_pack", fig_dir=True, load_tex=True)
    doc.save(save_tex=True)
    os.remove(dst)


if __name__ == "__main__":
    main()
