from lgimapy.bloomberg import bdh
securities = [
    "I30389US Index",
    "I22320US Index",
    "I05039US Index",
    "I05040US Index",
    "I12881US Index",
    "I29080US Index",
    "I30183US Index",
    "I22297US Index",
    "I01478US Index",
    "I30407US Index",
    "I04276US Index",
    "I30402US Index",
    "I29070US Index",
    "I30351US Index",
    "I22321US Index",
    "I30182US Index",
    "I12875US Index",
    "I02201EU Index",
    "I02200EU Index",
    "I05442EU Index",
    "I05443EU Index",
    "I02202EU Index",
    "I05444EU Index",
    "I02002EU Index",
    "I02500EU Index",
    "I03099EU Index",
    "I22819EU Index",
    "I02138EU Index",
    "I02135EU Index",
    "I02136EU Index",
    "I02137EU Index",
    "I09075GB Index",
    "I09074GB Index",
    "I09077GB Index",
    "I02571GB Index",
    "I05201GB Index",
    "I08055GB Index",
    "I08059GB Index",
    "I08060GB Index",
    "I08056GB Index",
    "I08057GB Index",
    "I08058GB Index",
    "I00039 Index",
    "I02766US Index",
    "I02767US Index",
    "I02768US Index",
    "I00185US Index",
    "I00182US Index",
    "I02769US Index",
    "I00188US Index",
    "I02765US Index",
    "I06728US Index",
    "I02783US Index",
    "I21098US Index",
    "I00012US Index",
    "I00011US Index",
    "I00151US Index",
    "I00643US Index",
    "I00152US Index",
    "I00648US Index"
]
yellow_keys =  ""
fields = "INDEX_MARKET_VALUE"
start= "20220315"
end= "20220405"

for security in securities:
    print(security)
    df = bdh(security, yellow_keys, fields, start, end)

# %%
df = bdh(securities, yellow_keys, fields, start, end)
df
