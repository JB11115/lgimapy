from collections import defaultdict

from lgimapy.data import Database


# %%
year = 2021
account = "P-LD"
strategy_long_name = "Long Duration Credit"

db = Database()
start = db.date("YEAR_START", year)
end = db.date("YEAR_END", year)
monthly_dates = db.date("MONTH_STARTS", start=start, end=end)

isin_d = defaultdict(set)
bm_isins = set()
port_isins = set()
for date in monthly_dates:
    port = db.load_portfolio(account=account, date=date)
    isin_d["Benchmark"] |= set(port.bm_df["ISIN"])
    isin_d["LGIMA Composite"] |= set(port.port_df["ISIN"])

# %%
downgrade_default_d = defaultdict(dict)
for key, isins in isin_d.items():
    fallen_angels_df = db.rating_changes(
        fallen_angels=True, isins=isins, start=start, end=end
    )
    defaults_df = db.defaults(isins=isins, start=start, end=end)
    fallen_angel_tickers = sorted(fallen_angels_df["Ticker"].unique())
    fallen_angel_isins = sorted(fallen_angels_df["ISIN"].unique())
    defaulted_tickers = sorted(defaults_df["Ticker"].unique())
    defaulted_isins = sorted(defaults_df["ISIN"].unique())

    downgrade_default_d[key]["# Fallen Angel Tickers"] = len(
        fallen_angel_tickers
    )
    downgrade_default_d[key]["# Fallen Angel Bonds"] = len(fallen_angel_isins)
    downgrade_default_d[key]["Fallen Angels Tickers"] = fallen_angel_tickers
    downgrade_default_d[key]["Fallen Angels ISINs"] = fallen_angel_isins
    downgrade_default_d[key]["# Defaulted Tickers"] = len(defaulted_tickers)
    downgrade_default_d[key]["# Defaulted Bonds"] = len(defaulted_isins)
    downgrade_default_d[key]["Defaulted Tickers"] = defaulted_tickers
    downgrade_default_d[key]["Defaulted ISINs"] = defaulted_isins


# %%
print(f"{year} Downgrades and Fallen Angels in {strategy_long_name}")
for port_key, d in downgrade_default_d.items():
    print(f"\n\n{port_key}")
    for key, val in d.items():
        if isinstance(val, list) and not val:
            continue
        else:
            print(f"    {key}: {val}")
