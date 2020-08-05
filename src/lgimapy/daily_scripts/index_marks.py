from lgimapy.data import Database

db = Database()
print(db.date("today").strftime("%m/%d/%Y"))

ixs = {}
ixs["Long Credit"] = db.load_bbg_data("US_IG_10+", "OAS", db.date("yesterday"))
ixs["Market Credit"] = db.load_bbg_data("US_IG", "OAS", db.date("yesterday"))

for ix, df in ixs.items():
    diff = int(df.diff()[-1])
    if diff > 0:
        print(f"  {ix}: {diff} bp wider")
    elif diff < 0:
        print(f"  {ix}: {-diff} bp tighter")
    else:
        print(f"  {ix}: unched")
