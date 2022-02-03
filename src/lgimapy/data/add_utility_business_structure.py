import numpy as np

from lgimapy.utils import load_json, dump_json

# %%

def add_utility_business_structure():
    fid = "utility_business_structure"
    new_issuer = input("\nEnter Issuer Name:\n").strip()
    options = {
        0: "HOLDCO",
        1: "OPCO",
        2: np.nan,
        3: 'cancel',
    }
    print(f"\nSelect the option for the utility businss structure of")
    print(f"'{new_issuer}':\n")
    for key, val in options.items():
        print(f"  {key}) {val}")

    new_business_structure_key = int(input())
    if new_business_structure_key == 3:
        quit()

    old_d = load_json(fid)
    new_row_d = {new_issuer: options[new_business_structure_key]}
    new_d = {**old_d, **new_row_d}
    dump_json(new_d, fid)

# %%
if __name__ == "__main__":
    add_utility_business_structure()
