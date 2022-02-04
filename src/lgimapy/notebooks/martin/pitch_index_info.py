import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root

# %%
index = 'H4NC'

CCC = []

B = [13.7, 14.08, 8.99]
BB = [24.49, 20.921, 17.622]

fin = 0

db = Database()
df = pd.read_csv(root(f'data/HY/pitch_indexes/{index}.csv'), index_col=0)

print(f'BB: {np.sum(BB):.1f}%')
print(f'B: {np.sum(B):.1f}%')
print(f'CCC: {np.sum(CCC):.1f}%\n')

print(f"Non-Fin: {100-fin:.1f}%")
print(f"Fin: {fin:.1f}%\n")

regions = ['North America', 'Europe', 'EMEA', 'LATAM']
for region in regions:
    weight = df[df.index.isin(db.country_codes(region))]['% Weight'].sum()
    print(f"{region} {weight:.1f}%")
