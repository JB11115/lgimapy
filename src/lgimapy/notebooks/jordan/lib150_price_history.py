import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database

# %%
db = Database()
dates = db.trade_dates(start=db.date("YTD"))
act_name = "LIB150"
price_list = []
for date in tqdm(dates):
    acnt = db.load_portfolio(account=act_name, date=date)
    prices = acnt.df.set_index("CUSIP")["DirtyPrice"].rename(date)
    price_list.append(prices)


df = pd.concat(price_list, axis=1).dropna(how="all")
df.round(2).to_csv(f"{act_name}_YTD_security_prices.csv")
