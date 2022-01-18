from lgimapy.data import Database

# %%

def main():
    for market in ['US', 'EUR', 'GBP']:
        print(f'Testing {market} from local feather files')
        db = Database(market=market)
        db.load_market_data()
        if len(db.df):
            print('  Success\n')
        else:
            print('  No Data\n')

    for market in ['US', 'EUR', 'GBP']:
        if market == 'US':
            source = 'Datamart'
        else:
            source = 'BASys dir on S: Drive'
        print(f'Testing {market} from {source}')
        db = Database(market=market)
        db.load_market_data(local=False)
        if len(db.df):
            print('  Success\n')
        else:
            print('  No Data\n')


if __name__ == '__main__':
    main()
