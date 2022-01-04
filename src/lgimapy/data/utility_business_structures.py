import win32clipboard

import pandas as pd
import numpy as np
from lgimapy.utils import root, dump_json, load_json, to_clipboard

# %%
# raw_fid = root('data/utility_business_structure.csv')
# df = pd.read_csv(raw_fid, index_col=0).set_index('Issuer').rename_axis(None)
# d = df['BusinessStructure'].fillna(np.nan).to_dict()
# fid = 'utility_business_structure'
# dump_json(d, fid)
# load_json(fid)

# %%

win32clipboard.OpenClipboard()
raw_lines = win32clipboard.GetClipboardData()
win32clipboard.CloseClipboard()

lines = raw_lines.split("\n")
business_structure = {}
for line in lines:
    issuer = line.rstrip().rstrip("\x00").split(":")[1].strip()
    business_structure[issuer] = np.nan

tmp_fid = "utility_business_structures_tmp"
dump_json(business_structure, tmp_fid)

# %%
fid = "utility_business_structure"
current_d = load_json(fid)
tmp_d = load_json(tmp_fid)
updated_d = {**current_d, **tmp_d}
dump_json(updated_d, fid)
