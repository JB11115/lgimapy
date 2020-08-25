from lgimapy.data import Database

db = Database()

ixs = {}
ixs["Long Credit"] = db.load_bbg_data("US_IG_10+", "OAS", db.date("yesterday"))
ixs["Market Credit"] = db.load_bbg_data("US_IG", "OAS", db.date("yesterday"))

print(ixs["Long Credit"].index[-1].strftime("%m/%d/%Y"))

for ix, df in ixs.items():
    diff = int(df.diff()[-1])
    if diff > 0:
        print(f"  {ix}: {diff} bp wider")
    elif diff < 0:
        print(f"  {ix}: {-diff} bp tighter")
    else:
        print(f"  {ix}: unched")
