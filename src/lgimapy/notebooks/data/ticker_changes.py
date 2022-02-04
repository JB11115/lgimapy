from lgimapy.data import Database
from lgimapy.utils import load_json, dump_json

# %%
prev_date = "6/1/2021"

all_ticker_changes = load_json("all_ticker_changes")

db = Database()
db.load_market_data()
curr_ix = db.build_market_index()

db.load_market_data(db.nearest_date(prev_date))
prev_ix = db.build_market_index()


raw_new_ticker_changes = {}
curr_tickers = curr_ix.df.set_index("ISIN")["Ticker"].to_dict()
prev_tickers = prev_ix.df.set_index("ISIN")["Ticker"].to_dict()
for isin, curr_ticker in curr_tickers.items():
    try:
        prev_ticker = prev_tickers[isin]
    except KeyError:
        continue
    if prev_ticker != curr_ticker:
        raw_new_ticker_changes[prev_ticker] = curr_ticker


new_tickers = set(raw_new_ticker_changes) - set(all_ticker_changes)
new_ticker_changes = {t: raw_new_ticker_changes[t] for t in new_tickers}

# %%
same_credit_risk_ticker_changes = {
    "ECACN": "OVV",  #  SCR - Redomiciled in the US
    # "ARNC": "HWM", # NCR - Arconic split into two companies
    # "SKYLN": "CMCSA", # NCR - Sky purchased by Comcast
    # "CVS": "CVSPAS",  # NCR - CVSPAS is structured security
    # "CTL": "LUMN", # NCR - M&A Centurytel merged with Embarq
    # "CHIWTR": "GCHWTR", #DCR, different entities
    # "EQ": "LUMN", # NCR - M&A Centurytel merged with Embarq
    # "EEP": "ENBCN", # NCR - M&A Enbridge bought in their MLP
    "ETP": "ET",  # SCR - Ticker change
    "AXASA": "EQH",  # SCR - Name change
    # "FTR": "FYBR", #  NCR - FTR went bankrupt
    # "HSECN": "CVECN", # NCR - Cenovus bought Husky
    # "GXP": "EVRG",# NCR - M&A and then name change
    "RBLN": "RKTLN",  # SCR - Name change
    # "SCG": "D", # NCR – Dominion bought Scana
    # "STI": "TFC", # NCR - Suntrust bought BBT and renamed
    # "WPZ": "WMB",  # NCR - WPZ was the MLP subsidiary of WMB
    # "UTX": "RTX", # NCR - Merger UTX merged with RTN
    # "CBS": "VIAC", # NCR - CBS bought VIAC
    # "VIA": "VIAC", # NCR - VIA bought by CBS
    "XRX": "XRXCRP",  # SCR
    "AIOAU": "PNHAU",  # SCR ticker change to match CDS
    # "ESRX": "CI", # NCR - M&A
    "HCP": "PEAK",  # SCR - Name change
    # "HRS": "LHX", # NCR - M&A L3 and Harris
    # "VR": "AIG",  # NCR - M&A Validus bought by AIG
    # "BBT": "TFC",  # NCR - BBT Bought by Suntrust
    "CNAFNL": "CNA",  # SCR
    # "CVC": "CSCHLD", # NCR - M&A
    # "CATHHE": "CATMED", # NCR - M&A
    # "D": "BRKHEC",  # NCR - Asset sale
    # "EV": "MS",  # NCR - M&A Morgan Stanley buys Eaton Vance
    "HPT": "SVC",  # SCR Name change
    # "IPCC": "KMPR",  # NCR - M&A Kemper and Infinity
    # "IR": "TT",  # NCR - Sold portion of business and renamed
    # "WR": "EVRG",  # NCR - M&A and name change
    # "MYL": "VTRS",  # NCR - M&A and then name change
    # "NFX": "OVV",  # NCR - Newfield bought by Ovintiv
    # "POL": "AVNT",  # NCR - M&A and name change
    # "PX": "LIN",  # NCR – M&A
    "RBS": "NWG",  # SCR - Name change to Natwest Group
    "SNH": "DHC",  # SCR - Name change
    # "SEP": "ENBCN",  # NCR - Spectra bought by Enbridge
    # "AMGFIN": "OMF", # NCR - M&A
    # "SYMC": "NLOK",  # NCR - M&A Symantec buys Norton
    "TMK": "GL",  # SCR  Torchmark renamed Globe Life
    # "TSS": "GPN",  # NCR - M&A
    # "WFDAU": "ULFP",  # NCR - M&A Unibail and Westfield
    # "WFTCN": "WFGCN",  # NCR Bankruptcy reorganization
    # "WYND": "TNL",  # NCR - Wyndham buys Travel & Leisure
    "ALBLLC": "ACI",  # SCR - Renamed after going public
    # "PACLIF": "ACGCAP",  # NCR - Paclife sells Aviation Capital Group
    # "SHPLN": "TACHEM",  # NCR - M&A Takeda buys Shire
    # "CRZO": "CPE",  # NCR - M&A Callon buys Carizo
    # "COVPAR": "CRK",  # NCR - M&A Comstock buys Covey Par
    "DDR": "SITC",  # SCR - Name change
    # "DPS": "KDP",  # NCR - M&A
    "FCAIM": "STLA",  # SCR - Fiat changes name to Stellantisopi
    # "GOVPIT": "OPI",  # NCR - M&A and then name change
    # "SO": "NEE",  # NCR - Asset purchase of an opco
    "HARCLA": "VERCST",  # SCR - Name change
    # "RX": "IQV", # NCR - M&A
    "JACLIF": "JXN",  # SCR
    "MTCH": "MTCHII",  # SCR, Matching equity ticker
    "MIICF": "TIGO",  # SCR, Ticker change
    "NYLD": "CWENA",  # SCR, Matching OTC ticker
    # "SOV": "SANUSA",  # NCR - M&A
    # "SHAEFF": "IHOVER",
    # "SIR": "OPI",  # NCR - M&A
    # "TRINSE": "TSE",
    # "VRXCN": "BHCCN",  # NCR - M&A
    # "VLP": "VLO",  # NCR – VLO bought in its MLP
    "WYN": "TNL",  # SCR - Ticker change
    # "APY": "CHX",  # NCR - M&A
    "WTR": "WTRG",  # SCR - Ticker change
    "BBALN": "SIGLN",  # SCR - Ticker change
    # "FGL": "FNF",  #  NCR - M&A
    # "GFLENV": "GFLCN",  # NCR - M&A
    # "HILDOC": "HLT",
    "IAASPI": "IAA",  # SCR - Name change
    # "OWLRCK": "ORCC",  # NCR
    # "NEE": "PIPFND", # NCR
    # "SPCHEM": "NRYHLD",
    "TRWH": "BALY",  # SCR - Ticker change
    # "WFT": "WFTLF" # NCR I believe WFT went bankrupt
    "TOTAL": "TTEFP",  # SCR - Ticker change
    # "ORGNON": "OGN",
    "TWC": "CHTR",  # Similar CR - Tickers not updated despite M&A by Charter
}
new_ticker_changes
# %%

dump_json(same_credit_risk_ticker_changes, "ticker_changes")


# %%
updated_all_ticker_changes = {**all_ticker_changes, **new_ticker_changes}
dump_json(updated_all_ticker_changes, "all_ticker_changes")
