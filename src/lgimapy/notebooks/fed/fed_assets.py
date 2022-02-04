import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import lgimapy.vis as vis
from lgimapy.utils import root

# %%



def read_data(fid):
    return pd.read_csv(
        fid, index_col=0, parse_dates=True, infer_datetime_format=True
    )


def main():
    # %%
    data_dir = root("data/fed")
    fids = data_dir.glob("*.csv")
    df = pd.concat(
        [read_data(fid) for fid in fids],
        axis=1,
        sort=True
    ) / 1e6
    total = df['RESPPANWW']
    df.drop('RESPPANWW', axis=1, inplace=True)
    security_map = {
        'TREAST': 'Treasuries',
        'WSHOMCB': 'MBS',
        'WORAL': 'Repo',
        'WCBLSA': 'Central Bank Liquidity Swaps',
        'WGCAL': 'Gold',
        'WLCFLL': 'Loans',
        "WTCOL": "Treasury Currency",
        'WSHOFADSL': 'Federal Agency Debt',
        'WUPSHO': 'Unamortized Premiums\non Securities',
        'TERAUCT': 'Term Auction Credit',
        'WAML1L': 'Holdings of Maiden Lane LLCs',
        'H0RESPPALDFXAWNWW': 'Commercial Paper',
    }
    df.rename(columns=security_map, inplace=True)
    df = df[list(security_map.values())].fillna(0)

    df.iloc[300:350, -2:]
    # %%


    vis.style()
    sns.set_palette('dark', len(df.columns) + 2)
    fig, ax = vis.subplots(figsize=(10, 6))
    cols = [df[col] for col in df.columns]
    ax.stackplot(
        df.index,
        *cols,
        labels=df.columns,
        alpha=0.9,
    )
    ax.plot(total, lw=1, c='k')
    vis.format_yaxis(ax, '${x:.0f}T')
    ax.set_title("Fed Balance Sheet")
    ax.legend(fontsize=12)
    vis.savefig("fed_balance_sheet")
    vis.show()
