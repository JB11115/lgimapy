import warnings
from collections import defaultdict
from functools import cached_property, lru_cache

import joblib
import pandas as pd
from oslo_concurrency import lockutils

from lgimapy.data import Database, get_account_strategy_map_list
from lgimapy.utils import (
    root,
    mkdir,
    dump_json,
    load_json,
    to_list,
    to_datetime,
)

# %%


class PortfolioHistory:
    def __init__(self, verbose=False):
        self._db = Database()
        self._mkdir()
        self._desired_ignored_accounts_d = self.desired_ignored_accounts_d
        self._verbose = verbose

    def _mkdir(self):
        data_dir = self._db.local("portfolios/history")
        mkdir(data_dir / "Strategy")
        mkdir(data_dir / "Account")

    @property
    def _processed_ignored_accounts_fid(self):
        return "portfolio_processed_history_ignored_accounts"

    @property
    def _desired_ignored_accounts_fid(self):
        return "portfolio_history_desired_ignored_accounts"

    @property
    def _erroneous_ignored_accounts_fid(self):
        return "portfolio_history_erroneous_ignored_accounts"

    @property
    def _erroneous_ignored_accounts_d(self):
        return load_json(
            self._erroneous_ignored_accounts_fid, empty_on_error=True
        )

    @property
    def desired_ignored_accounts_d(self):
        return load_json(
            self._desired_ignored_accounts_fid, empty_on_error=True
        )

    def desired_ignored_accounts(self, date):
        try:
            return set(self.desired_ignored_accounts_d[self._fmt_date(date)])
        except KeyError:
            return set()

    @property
    def processed_ignored_accounts_d(self):
        return load_json(
            self._processed_ignored_accounts_fid, empty_on_error=True
        )

    @cached_property
    def _cached_desired_ignored_accounts_d(self):
        return self.desired_ignored_accounts_d

    def _accounts_to_ignore(self, strategy, date):
        try:
            all_desired_ignored_accounts = set(
                self._cached_desired_ignored_accounts_d[self._fmt_date(date)]
            )
        except KeyError:
            all_desired_ignored_accounts = set()

        accounts_in_current_strategy = set(
            self._get_strategies_to_account_map_from_accounts(
                date, all_desired_ignored_accounts
            )[strategy]
        )
        return accounts_in_current_strategy & all_desired_ignored_accounts

    @property
    def strategies(self):
        return {
            "US Long Credit",
            "US Long Credit - Custom",
            "US Long Credit Plus",
            "US Long Corporate",
            "US Long Corp 2% Cap",
            "US Long Credit Ex Emerging Market",
            "US Corporate 1% Issuer Cap",
            "Global Agg USD Corp",
            "Custom RBS",
            "GM_Blend",
            "US Corporate IG",
            "US Intermediate Credit",
            "US Intermediate Credit A or better",
            "US Long GC 70/30",
            "US Long GC 75/25",
            "US Long GC 80/20",
            "US Long Government/Credit",
            "Liability Aware Long Duration Credit",
            "US Credit",
            "US Credit A or better",
            "US Credit Plus",
            "US Long A+ Credit",
            "80% US A or Better LC/20% US BBB LC",
            "Bloomberg LDI Custom - DE",
            "INKA",
            "US Long Corporate A or better",
        }

    @property
    def _sorted_strategies(self):
        """Sort strategies putting the largest ones first."""
        large_strategies = {
            "US Long Credit",
            "US Credit",
            "US Long Government/Credit",
        }
        strategies = list(self.strategies - large_strategies)
        # Put largest strategies at the beginning of the list
        # to maximize the use of parallel processing.
        for strategy in large_strategies:
            strategies.insert(0, strategy)
        return strategies

    def _fmt_date(self, date):
        return to_datetime(date).strftime("%Y-%m-%d")

    def _add_desired_ignored_account(self, account, date):
        date = self._fmt_date(date)
        try:
            ignored_accounts = set(self._desired_ignored_accounts_d[date])
        except KeyError:
            self._desired_ignored_accounts_d[date] = [account]
        else:
            ignored_accounts.add(account)
            self._desired_ignored_accounts_d[date] = sorted(
                list(ignored_accounts)
            )

    @lockutils.synchronized(
        "erroneous_ignored_accounts",
        external=True,
        lock_path=Database().local(),
    )
    def _add_erroneous_ignored_account(self, account, date):
        date = self._fmt_date(date)
        erroneous_ignored_accounts_d = self._erroneous_ignored_accounts_d
        try:
            ignored_accounts = set(erroneous_ignored_accounts_d[date])
        except KeyError:
            erroneous_ignored_accounts_d[date] = [account]
        else:
            ignored_accounts.add(account)
            erroneous_ignored_accounts_d[date] = sorted(list(ignored_accounts))
        dump_json(
            erroneous_ignored_accounts_d,
            self._erroneous_ignored_accounts_fid,
            sort_keys=True,
        )

    def _remove_desired_ignored_account(self, account, date):
        date = self._fmt_date(date)

    def save_desired_ignored_accounts(self):
        dump_json(
            self._desired_ignored_accounts_d,
            self._desired_ignored_accounts_fid,
            sort_keys=True,
        )

    @lockutils.synchronized(
        "processed_ignored_accounts",
        external=True,
        lock_path=Database().local(),
    )
    def _add_processed_ignored_account(self, account, date):
        date = self._fmt_date(date)
        processed_ignored_accounts_d = self.processed_ignored_accounts_d
        try:
            ignored_accounts = set(processed_ignored_accounts_d[date])
        except KeyError:
            processed_ignored_accounts_d[date] = [account]
        else:
            ignored_accounts.add(account)
            processed_ignored_accounts_d[date] = sorted(list(ignored_accounts))
        dump_json(
            processed_ignored_accounts_d,
            self._processed_ignored_accounts_fid,
            sort_keys=True,
        )

    @lockutils.synchronized(
        "processed_ignored_accounts",
        external=True,
        lock_path=Database().local(),
    )
    def _remove_processed_ignored_account(self, account, date):
        date = self._fmt_date(date)
        ignored_d = self.processed_ignored_accounts_d.copy()
        try:
            # Remove the current date from the dict.
            ignored_accounts = set(ignored_d.pop(date))
        except KeyError:
            return  # Account isn't in ignored list

        try:
            ignored_accounts.remove(account)
        except ValueError:
            return  # Account isn't in ignored list

        if ignored_accounts:
            # Update ignored accounts after removal
            ignored_d[date] = sorted(list(ignored_accounts))
            dump_json(
                ignored_d,
                self._processed_ignored_accounts_fid,
                sort_keys=True,
            )
        else:
            # There are no longer any ignored accounts for the
            # current date. Simply save the dict with the
            # current date removed.
            dump_json(
                ignored_d,
                self._processed_ignored_accounts_fid,
                sort_keys=True,
            )

    def add_desired_ignored_account(
        self,
        account,
        date=None,
        start=None,
        end=None,
    ):
        start = to_datetime(start)
        end = to_datetime(end)
        if date is not None:
            start = end = to_datetime(date)

        dates = self._db.trade_dates(start=start, end=end)
        for date in dates:
            self._add_desired_ignored_account(account, date)

    def _add_permanently_ignored_account(self, account):
        account_dates = self._db.account_market_values[account].dropna().index
        start = max(account_dates[0], self._db.date("PORTFOLIO_START"))
        end = account_dates[-1]
        self.add_desired_ignored_account(account, start=start, end=end)

    @cached_property
    def _account_strategy_map_list(self):
        return get_account_strategy_map_list(self._db)

    def _add_ignored_accounts_around_opening_or_closing(self):
        first_date = self._db.date("PORTFOLIO_START")
        for d in self._account_strategy_map_list:
            if d["strategy"] not in self.strategies:
                continue
            self._add_ignored_accounts_around_opening(d, first_date)
            self._add_ignored_accounts_around_closing(d, first_date)

    def _add_ignored_accounts_around_opening(self, d, first_date):
        if d["open"] <= first_date:
            return

        # Ignore dates for first month of a new account.
        try:
            month_after_open = self._db.date("+1m", d["open"])
        except IndexError:
            # Opened in past month.
            month_after_open = self._db.date("today")

        self.add_desired_ignored_account(
            d["account"], start=d["open"], end=month_after_open
        )

    def _add_ignored_accounts_around_closing(self, d, first_date):
        """Ignore dates for last month before account closed."""
        if d["close"] is None or d["close"] <= first_date:
            return
        month_before_close = max(first_date, self._db.date("1m", d["close"]))
        self.add_desired_ignored_account(
            d["account"],
            start=month_before_close,
            end=d["close"],
        )

    @cached_property
    def _all_accounts(self):
        dates = self._db.date(
            "month_starts", start=self._db.date("PORTFOLIO_START")
        )
        dates.append(self._db.date("today"))
        all_accounts = set()
        for date in dates:
            strategy_account_map = self._db._strategy_account_map(date)
            for strategy, accounts in strategy_account_map.items():
                if strategy not in self.strategies:
                    continue
                for account in accounts:
                    all_accounts.add(account)

        return sorted(list(all_accounts))

    @cached_property
    def current_accounts(self):
        all_accounts = []
        strategy_account_map = self._db._strategy_account_map()
        for strategy, accounts in strategy_account_map.items():
            if strategy not in self.strategies:
                continue
            all_accounts.extend(accounts)
        return sorted(all_accounts)

    def _add_ignored_accounts_around_large_flows(self):
        for account in self._all_accounts:
            try:
                flows = self._db.account_flows(account)
            except KeyError:
                continue

            large_flow_dates = flows[flows.abs() > 0.04].index
            for date in large_flow_dates:
                try:
                    next_date = self._db.trade_dates(start=date)[1]
                except IndexError:
                    # Current date is most recent date
                    next_date = date
                self.add_desired_ignored_account(
                    account, start=date, end=next_date
                )

    def _remove_erroneous_ignored_accounts(self):
        for date, accounts in self._erroneous_ignored_accounts_d.items():
            for account in accounts:
                self._remove_erroneous_accounts_from_desired_accounts(
                    account, date
                )

    def _remove_erroneous_accounts_from_desired_accounts(self, account, date):
        date = self._fmt_date(date)
        try:
            # Remove the current date from the dict.
            ignored_accounts = set(self._desired_ignored_accounts_d.pop(date))
        except KeyError:
            return  # Account isn't in desired ignored list

        try:
            ignored_accounts.remove(account)
        except ValueError:
            return  # Account isn't in ignored list

        if ignored_accounts:
            # Update ignored accounts after removal
            self._desired_ignored_accounts_d[date] = sorted(
                list(ignored_accounts)
            )

    def build_ignored_accounts_file(self):
        self._add_ignored_accounts_around_opening_or_closing()
        self._add_ignored_accounts_around_large_flows()
        self._add_permanently_ignored_account("SESIBNM")
        self._add_permanently_ignored_account("SEUIBNM")
        self._remove_erroneous_ignored_accounts()
        self.save_desired_ignored_accounts()

    @property
    def dates_to_fix(self):
        dates_to_fix = {}
        processed_ignored_accounts_d = self.processed_ignored_accounts_d
        for date, accounts in self.desired_ignored_accounts_d.items():
            desired_ignored_accounts = set(accounts)
            try:
                processed_ignored_accounts = set(
                    processed_ignored_accounts_d[date]
                )
            except KeyError:
                processed_ignored_accounts = set()

            accounts_to_ignore = list(
                desired_ignored_accounts - processed_ignored_accounts
            )
            if accounts_to_ignore:
                dates_to_fix[date] = accounts_to_ignore
        return dates_to_fix

    def _get_dates_to_update(fid, override_from_date, override_to_date):
        db = Database()
        # Use override date if provided.
        if override_from_date is not None:
            return self._db.trade_dates(
                start=override_from_date, end=override_to_date
            )

        # Check if any dates have been completed, otherwise use all dates.
        completed_dates = self._completed_dates
        if completed_dates is None:
            return self._db.trade_dates(start=db.date("PORTFOLIO_START"))
        else:
            last_date = completed_dates.iloc[-1]
            return self._db.trade_dates(exclusive_start=last_date)

    @property
    def _completed_dates_fid(self):
        return self._db.local("portfolios/history/completed_dates.parquet")

    @property
    def _completed_dates(self):
        try:
            df = pd.read_parquet(self._completed_dates_fid)
        except FileNotFoundError:
            return None
        else:
            return df["date"]

    def _add_completed_date(self, date):
        completed_dates = self._completed_dates
        if completed_dates is None:
            # Start date file
            pd.DataFrame({"date": [date]}).to_parquet(self._completed_dates_fid)
        else:
            # Add date to the completed date file.
            updated_completed_dates = sorted(
                list(set(completed_dates) | set([date]))
            )
            pd.DataFrame({"date": updated_completed_dates}).to_parquet(
                self._completed_dates_fid
            )

    def _strategies_to_update(self, date):
        strategies_to_update = []
        for strategy in self.strategies:
            port = self._db.load_portfolio(strategy=strategy, empty=True)
            try:
                completed_dates = set(port.stored_properties_history_df.index)
            except AttributeError:
                # Strategy has no stored property history yet.
                strategies_to_update.append(strategy)

            if date in completed_dates:
                continue
            else:
                strategies_to_update.append(strategy)
        return strategies_to_update

    def update_date(
        self,
        date,
        specific_strategies=None,
        force=True,
    ):
        date = to_datetime(date)
        if specific_strategies is not None:
            strategies = to_list(specific_strategies, dtype=str)
        elif force:
            strategies = self._sorted_strategies
        else:
            strategies = self._strategies_to_update(date)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joblib.Parallel(n_jobs=6)(
                joblib.delayed(self._update_strategy_for_date)(strategy, date)
                for strategy in strategies
            )

        # Update completed dates file.
        self._add_completed_date(date)

    def update_strategy(strategy, date=None, start=None, end=None):
        if date is not None:
            self.update_date(date, strategy)
        else:
            dates = self._db.trade_dates(start, end)
            for date in dates:
                self.update_date(date, strategy)

    def _remove_date(self, date, strategy=None, account=None):
        port = self._db.load_portfolio(
            strategy=strategy, account=account, date=date, empty=True
        )
        df = port.stored_properties_history_df
        if df is None:
            return
        df = df[df.index != port.date]
        df.to_parquet(port._stored_properties_history_fid)

    def _update_strategy_for_date(self, strategy, date):
        accounts_to_ignore = self._accounts_to_ignore(strategy, date)
        try:
            port = self._db.load_portfolio(
                strategy=strategy,
                date=date,
                ignored_accounts=accounts_to_ignore,
            )
        except ValueError:
            # No data.
            for account in accounts_to_ignore:
                self._remove_date(date, account=account)
                self._add_erroneous_ignored_account(account, date)
            return

        if self._verbose:
            print(port)

        port = port.drop_empty_accounts()
        if not len(port.df):
            self._remove_date(date, strategy=strategy)
            for account in accounts_to_ignore:
                self._remove_date(date, account=account)
                self._add_erroneous_ignored_account(account, date)
            return

        try:
            port.save_stored_properties()
            self.update_attemped_ignored_accounts(port)

        except Exception as e:
            print(f"Failure on {port}")
            raise e

    def update_attemped_ignored_accounts(self, port):
        for account in self._accounts_to_ignore(port.name, port.date):
            self._remove_date(port.date, account=account)
            if account in port._all_accounts:
                self._add_processed_ignored_account(account, port.date)
            else:
                self._add_erroneous_ignored_account(account, port.date)

    def _get_strategies_to_account_map_from_accounts(self, date, accounts):
        full_account_strategy_map = self._db._account_strategy_map(
            self._fmt_date(date)
        )
        ignored_account_strategy_map = {}
        for account in accounts:
            try:
                strategy = full_account_strategy_map[account]
            except KeyError:
                self._add_erroneous_ignored_account(account, date)
            else:
                ignored_account_strategy_map[account] = strategy

        strategy_to_account_map = defaultdict(list)
        for account, strategy in ignored_account_strategy_map.items():
            strategy_to_account_map[strategy].append(account)

        return strategy_to_account_map

    def _fill_history(self):
        all_dates = self._db.trade_dates(start=self._db.date("PORTFOLIO_START"))
        dates_to_fill = list(set(all_dates) - set(self._completed_dates))
        for date in sorted(dates_to_fill):
            self.update_date(date, force=False)
            self._add_completed_date(date)

    def _fix_history(self):
        today = self._fmt_date(self._db.date("today"))
        for date, accounts in self.dates_to_fix.items():
            if date == today:
                continue
            strategies_to_fix = (
                self._get_strategies_to_account_map_from_accounts(
                    date, accounts
                ).keys()
            )
            for strategy in strategies_to_fix:
                self._update_strategy_for_date(strategy, date)

    def update_history(self):
        self._fix_history()
        self._fill_history()


# %%
if __name__ == "__main__":
    # def main():
    # %%
    self = PortfolioHistory()
    self.build_ignored_accounts_file()
    self.update_history()
