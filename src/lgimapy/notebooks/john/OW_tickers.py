from lgimapy.data import Database

# %%
account = 'P-LD'
subset = 'BBB_NON_CORP'

db = Database()
port = db.load_portfolio(account=account, universe='stats')
sub_port = port.subset(**db.index_kwargs(subset, in_stats_index=None))
full_df = sub_port.ticker_df

# %%
cols = ['Issuer', 'LGIMASector', 'P_OAD', 'BM_OAD', 'OAD_Diff']
df = full_df[cols].sort_values('OAD_Diff', ascending=False).rename_axis(None)
fid = f"{account}_{subset}_ticker_OW.csv"
df.round(3).to_csv(fid)
