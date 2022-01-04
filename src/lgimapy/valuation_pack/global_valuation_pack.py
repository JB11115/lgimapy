from datetime import datetime as dt

from lgimapy.latex import Document
from GBP_overview import update_GBP_credit_overview
from EUR_overview import update_EUR_credit_overview

# %%


def main():
    fid = "global_valuation_pack"
    update_GBP_credit_overview()
    update_EUR_credit_overview()
    doc = Document(
        fid,
        path="reports/global_valuation_pack",
        fig_dir=True,
        load_tex=fid,
    )
    doc.save_as(f"{dt.today():%Y-%m-%d}_Global_Valuation_Pack")
    doc.save_as("Global_Valuation_Pack", path="reports/current_reports")


# %%

if __name__ == "__main__":
    main()
