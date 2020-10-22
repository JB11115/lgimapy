from pathlib import Path

import pandas as pd

# %%


def get_basys_fids(market):
    """Get all files for given market."""
    basys_dir = Path(f"S:/FrontOffice/Bonds/BASys/CSVFiles/MarkIT/{market}/")
    fids = basys_dir.glob("*")
    files = {}
    for fid in fids:
        if len(fid.stem) != 21:
            # Bad file.
            continue
        date = pd.to_datetime(fid.stem[-10:])
        files[date] = fid
    return pd.Series(files).sort_index()
