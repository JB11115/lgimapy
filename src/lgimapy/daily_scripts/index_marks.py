from lgimapy.data import Database

db = Database()

ixs = {}
ixs["Long Credit"] = db.load_bbg_data("US_IG_10+", "OAS", db.date("yesterday"))
ixs["Market Credit"] = db.load_bbg_data("US_IG", "OAS", db.date("yesterday"))

print(ixs["Long Credit"].index[-1].strftime("%m/%d/%Y"))

for ix, df in ixs.items():
    diff = df.diff().iloc[-1]
    if diff > 0:
        print(f"  {ix}: {df.iloc[-1]:.1f}, {diff:.2f} bp wider")
    elif diff < 0:
        print(f"  {ix}: {df.iloc[-1]:.1f}, {-diff:.2f} bp tighter")
    else:
        print(f"  {ix}: {df.iloc[-1]:.2f}, unched")
