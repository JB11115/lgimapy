from lgimapy.bloomberg import bdh

import warnings

import pybbg

# %%
warnings.simplefilter(action="ignore", category=UserWarning)

ovrd = {
    "Dir": "V",
    "IntrRw": "true",
    # "IntrRw": True,
    "RPSCodes": "S",
    "RPTParty": "S",
    "RPTContra": "S",
    "CondCodes": "S",
    "PCS": "TRACE",
    # "TZ": "New_York",
    # "Dts": "S",
}

ovrd
ticker = "035240AT7@TRACE Corp"
bbg = pybbg.Pybbg()
start = "2020-04-23 07:00:00"
end = "2020-04-24 15:00:00"
# df = bbg.bdh(ticker, "trade", start, end, overrides=ovrd)
df = bbg.bdh(ticker, "trade", start, end, overrides=ovrd)
bbg.session.stop()
