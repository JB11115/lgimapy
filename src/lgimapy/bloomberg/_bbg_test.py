from lgimapy.bloomberg import bdp, bdh, bds

# %%


def main():
    cusip = "097023BV6"
    yellow_key = "Corp"
    ovrd = {"USER_LOCAL_TRADE_DATE": "19501010"}

    print("\ntesting bdp()...")
    print(bdp(cusip, yellow_key, "PX_LAST"))

    print("\ntesting bdh()...")
    print(bdh(cusip, yellow_key, "PX_LAST", start="1/1/2022", end="1/10/2022"))

    print("\ntesting bds()...")
    print(bds(cusip, yellow_key, "DES_CASH_FLOW", ovrd=ovrd))


if __name__ == "__main__":
    main()
