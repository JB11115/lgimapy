from lgimapy.data import Database

#%%
db = Database()
port = db.load_portfolio(account=['P-LD', 'FEDLD'])

# %%

funcs = {
    'tickers': "account_ticker_overweights_comp",
    'sectors': "account_IG_sector_overweights_comp",
    'market_segments': "account_IG_market_segments_overweights_comp",
}
for name, func in funcs.items():
    df = eval(f"port.{func}(n=None).drop('Avg', axis=1)")
    df['diff'] = df['FEDLD'] - df['P-LD']
    df['abs_diff'] = df['diff'].abs()
    df.to_csv(f"FEDLD_vs_LD_{name}.csv")
    
