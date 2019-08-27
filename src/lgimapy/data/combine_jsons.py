"""
Notebook to combine two json files.
"""

from lgimapy.utils import load_json, dump_json

# %%
# ------------------------------------------------------- #
fid = "ratings_changes"
fid_to_combine = "ratings_changes_Copy"
# ------------------------------------------------------- #

file0 = load_json(fid)
file1 = load_json(fid_to_combine)

n_fid = len(file0)

keys_to_add = []
for key in file1.keys():
    if key not in file0:
        keys_to_add.append(key)

n_add = len(keys_to_add)

print(f"# Keys in File: {n_fid:,}")
print(f"# Keys to Add: {n_add:,}")
print(f"% Keys to Add: {n_add/n_fid:.2%}")

# %%

add_dict = {k: file1[k] for k in keys_to_add}
new_json = {**file0, **add_dict}
dump_json(new_json, fid)
