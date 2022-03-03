import pandas as pd

from lgimapy.data import Database

# %%


def update_nonfin_spreads():
    db = Database()
    ratings = db.convert_numeric_ratings(pd.Series(range(1, 11)))

    db.load_market_data(start=db.date("1y"), local=True)
    mc_oas_list = []
    lc_oas_list = []
    for rating in ratings:
        ix_mc = db.build_market_index(
            rating=rating, **db.index_kwargs("INDUSTRIALS")
        )
        ix_lc = ix_mc.subset(maturity=(10, None))
        mc_oas_list.append(ix_mc.market_value_weight("OAS").rename(rating))
        lc_oas_list.append(ix_lc.market_value_weight("OAS").rename(rating))

    df_mc = pd.concat(mc_oas_list, axis=1, sort=True)
    df_lc = pd.concat(lc_oas_list, axis=1, sort=True)

    # Write each dataframe to a different worksheet.
    fid = Database.X_drive(
        "Credit Strategy/lgimapy/data/nonfin_spreads_by_rating.xlsx"
    )
    writer = pd.ExcelWriter(fid)
    df_mc.to_excel(writer, sheet_name="Market Credit")
    df_lc.to_excel(writer, sheet_name="Long Credit")
    writer.save()
