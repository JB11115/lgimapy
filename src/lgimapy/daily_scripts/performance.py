from pathlib import Path

import pandas as pd

from lgimapy.data import Database

# %%

path = Path('C:/blp/data')
date = Database().date('today')
print(date.strftime('%m/%d/%Y'))
accounts = [
    'P-LD',
    'FLD',
    'GSKLC',
    'CITMC',
    'JOYLA',
    'USBGC',
]
for account in accounts:
    fids = list(path.glob(f"*Daily*{account}.{date.strftime('%Y%m%d')}.xls"))
    try:
        fid = fids[0]
    except IndexError:
        raise FileNotFoundError(f'No file found for {account}.')

    df = pd.read_excel(fid, usecols=[1, 2])
    df.columns = ['ix', 'val']
    df.set_index('ix', inplace=True)
    performance = df.loc['Outperformance (bps)'].squeeze()
    print(f"{account}: {performance:+.1f}")
