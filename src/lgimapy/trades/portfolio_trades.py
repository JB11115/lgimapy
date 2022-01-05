from collections import defaultdict
from functools import lru_cache, cached_property

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from lgimapy.data import Database, Account, Strategy
from lgimapy.latex import Document
from lgimapy.utils import load_json, mkdir, root, to_list, to_set, Time, pprint

#%%


class TradeError(Exception):
    """
    Raised when an input trade is trying to override an
    existing one.
    """

    pass


class PortfolioTrade:
    """
    Class for constructing portfolio trades.

    Parameters
    ----------
    strategies: str or List[str], optional
        Strategies to include in portfolio trade.
    accounts: str or List[str], optional
        Accounts to include in portfolio trade.
    date: datetime, optional
        Date for portfolio trade, defaults to current date if
        not provided.
    data: Dict[str: Dict[str: :class:`Account`]], optional
        Use data from previously loaded :class:`PortfolioTrade`
        for new instance.
    """

    def __init__(
        self,
        strategies=None,
        accounts=None,
        ignored_accounts=None,
        date=None,
        data=None,
    ):
        self.strategy_names = to_list(strategies, dtype=str)
        self.account_names = to_list(accounts, dtype=str)
        self.ignored_accounts = set()
        self._sell_errors = []
        self._trades = dict()

        self.date = pd.to_datetime("today" if date is None else date).floor("D")
        self.db = Database()
        if data is not None:
            self.portfolios = data.copy()
            self.og_portfolios = data.copy()

    def __repr__(self):
        return f"PortfolioTrade({self.date:%m/%d/%Y})"

    def trades(self, account=None):
        if account is None:
            return self._trades

        if account not in self.accounts:
            raise TradeError(f"{account} does not exist in trade.")

        return self._trades.get(account, AccountTradeBuilder(account))

    def _add_trade(self, notional):
        trade_builder = self.trades(self._account)
        trade_builder.add_trade(self._cusip, notional, self._universe)
        self._trades[self._account] = trade_builder

    @cached_property
    def fid_date(self):
        return self.date.strftime("%Y-%m-%d")

    @cached_property
    def report_dir(self):
        report_dir = root(f"reports/portfolio_trades/{self.fid_date}/")
        mkdir(report_dir)
        return report_dir

    @property
    def data_dir(self):
        data_dir = root(f"data/portfolio_trades/{self.fid_date}/")
        mkdir(data_dir)
        return data_dir

    @cached_property
    def strategy_account_map(self):
        """Dict[str:str]: Strategy to list of inlcuded accounts."""
        return load_json("strategy_accounts")

    @cached_property
    def account_strategy_map(self):
        """Dict[str:str]: Account to respective strategy."""
        return load_json("account_strategy")

    def load_data(self, universe="returns"):
        """
        Load data for all specified strategies/accounts.

        Attributes
        ----------
        portfolios: Dict[str: Dict[str: :class:`Account`]]
            Strategy names mapped to included dict of account names
            to respective :class:`Account`. Meant to be modified
            by adding buys/sells.
        og_portfolios: Dict[str: Dict[str: :class:`Account`]]
            Original state of `portfolios`.

        Raises
        ------
        TradeError
            If a selected account is in :attr:`ignored_accounts`
        """
        self.portfolios = defaultdict(dict)
        for strategy in self.strategy_names:
            for account in self.strategy_account_map[strategy]:
                if account in self.ignored_accounts:
                    continue
                self.portfolios[strategy][account] = self.db.load_portfolio(
                    account=account, universe=universe
                )
            for account in self.account_names:
                if account in self.ignored_accounts:
                    raise TradeError(f"{account} is in ignored accounts.")
                strategy = self.account_strategy_map[account]
                self.portfolios[strategy][account] = self.db.load_portfolio(
                    account=account, universe=universe
                )
        self.portfolios = dict(self.portfolios)
        self.og_portfolios = self.portfolios.copy()

    @cached_property
    def all_accounts(self):
        all_accounts = set()
        for strategy, accounts in self.portfolios.items():
            all_accounts |= to_set(accounts.keys())
        return all_accounts

    @property
    def accounts(self):
        trade_accounts = set()
        for strategy, accounts in self.portfolios.items():
            trade_accounts |= to_set(accounts.keys())
        return trade_accounts - self.ignored_accounts

    def ignore_accounts(self, accounts):
        accounts = to_set(accounts, dtype=str)
        self.ignored_accounts |= accounts

    def clear_ignored_accounts(self):
        self.ignored_accounts = set()

    @lru_cache(maxsize=None)
    def notional_held(self, security_col):
        """Dict[str: float]: Notional held in all included portfolios."""
        account_notionals = []
        for strategy, accounts in self.portfolios.items():
            for account in accounts.values():
                account_notionals.append(
                    account.df[["P_Notional", security_col]]
                    .groupby(security_col)
                    .sum()
                )
        return (
            pd.concat(account_notionals, axis=1).sum(axis=1).fillna(0).to_dict()
        )

    def _invert_d(self, order_d):
        """Dict[str: float]: Dict of order to size."""
        translated_order_d = {}
        for size, securities in order_d.items():
            for security in to_list(securities, dtype=str):
                translated_order_d[security] = size
        return translated_order_d

    def _add_sell_tickers(
        self,
        account,
        by_pct=None,
        by_notional=None,
        by_OAD_change=None,
        by_OAD_target=None,
        by_DTS_change=None,
        by_DTS_target=None,
        cusips=None,
        ignored_cusips=None,
    ):
        strategy = self.account_strategy_map[account]
        df = self.og_portfolios[strategy][account].df.copy()
        p_cols = ["P_MarketValue", "P_Notional"]
        sell_df_cols = ["P_Notional", "Ticker", "CUSIP"]
        if by_pct is not None:
            # Sell specified fraction of a ticker.
            sell_pct_d = self._invert_d(by_pct)
            sell_df = df[df["Ticker"].isin(sell_pct_d)][sell_df_cols].copy()
            # sell_df['pct'] = sell_df["Ticker"].map(sell_pct_d).fillna(0)

            sell_ticker_df = sell_df.groupby("Ticker", observed=True).sum()
            sell_ticker_df["pct"] = sell_ticker_df.index.map(sell_pct_d).fillna(
                0
            )
            sell_ticker_df["sell_notional"] = (
                sell_ticker_df["pct"] * sell_ticker_df["P_Notional"]
            )
            if ignored_cusips is not None:
                sell_df = sell_df[~sell_df["CUSIP"].isin(ignored_cusips)]
            if cusips is not None:
                if isinstance(cusips, dict):
                    sell_df["cusip_weights"] = sell_df["CUSIP"]
                    cusips_weights = self._invert_d(cusips)
            for col in p_cols:
                df[p_col] *= 1 - sell_amt

        if by_notional is not None:
            # Find percent of each bond to sell based on
            # overall held notional.
            sell_notional_d = self._invert_d(by_notional)
            notional_held = self.notional_held("Ticker")
            sell_pct_d = {
                security: sell_notional * 1e6 / notional_held[security]
                for security, sell_notional in sell_notional_d.items()
            }
            sell_amt = df["Ticker"].map(sell_pct_d).fillna(0)
            for col in p_cols:
                df[p_col] *= 1 - sell_amt

        if by_OAD_change is not None:
            OAD_change_d = self._invert_d(by_OAD_change)
            ticker_df = df.set_index("Ticker", drop=False)[["CUSIP", "P_OAD"]]
            if ignored_tickers is not None:
                ticker_df = ticker_df[
                    ~ticker_df["CUSIP"].isin(ignored_cusips)
                ].copy()
            if cusips is not None:
                ticker_df = ticker_df[ticker_df["CUSIP"].isin(cusips)].copy()
                if isinstance(cusips, dict):
                    cusips_

            current_OAD = ticker_df[ticker_df.index.isin(OAD_change_d)]["P_OAD"]
            current_ticker_OAD = current_OAD.groupby("Ticker").sum()

    def _example_df(self):
        return next(
            iter(next(iter(self.og_portfolios.values())).values())
        ).df.set_index("CUSIP")

    def sell_errors(self):
        return pd.DataFrame(
            self._sell_errors,
            columns=[
                "Account",
                "Universe",
                "CUSIP",
                "Method",
                "SellAmount",
                "SellNotional",
                "HeldNotional",
            ],
        )

    def _add_sell_error(self):
        self._sell_errors.append(
            (
                self._account,
                self._universe,
                self._cusip,
                self._method,
                self._sell_value,
                self._sell_notional,
                self._held_notional,
            )
        )

    def _test_sell(self):
        if self._sell_notional > self._held_notional:
            self._add_sell_error()
        elif self._sell_notional < 0:
            self._add_sell_error()
        else:
            return "_NO_ERROR_"

    def _held(self, col):
        try:
            held_amt = self._df.at[self._cusip, col]
            return 0 if pd.isna(held_amt) else held_amt
        except KeyError:
            self._sell_notional = None
            self._held_notional = 0
            raise TradeError()

    def _sell_cusip(self, cusip, method, value):
        self._cusip = cusip
        self._method = method
        self._sell_value = value
        class_meth = getattr(self, f"_sell_cusip_{method}")

        # Get trade sizing from the portfolio.
        try:
            with np.errstate(all="raise"):
                log = class_meth(value)
        except (TradeError, FloatingPointError):
            self._add_sell_error()
            return
        else:
            if log == "_CONTINUE_":
                return  # Skip this bond.

        # Ensure the trade is possible.
        if self._test_sell() == "_NO_ERROR_":
            self._add_trade(-self._sell_notional)

    def _sell_cusip_by_pct(self, pct):
        if self._cusip not in self._held_cusips:
            return "_CONTINUE_"
        self._held_notional = self._held("P_Notional")
        self._sell_notional = pct * self._held_notional

    def _sell_cusip_by_notional(self, notional):
        self._held_notional = self._held("P_Notional")
        self._sell_notional = notional

    def _sell_cusip_by_OAD_change(self, OAD_change):
        self._held_notional = self._held("P_Notional")
        OAD_target = self._held("OAD_Diff") - OAD_change
        keep_pct = (OAD_target + self._held("BM_OAD")) / self._held("P_OAD")
        self._sell_notional = (1 - keep_pct) * self._held_notional

    def _sell_cusip_by_OAD_target(self, OAD_target):
        self._held_notional = self._held("P_Notional")
        keep_pct = (OAD_target + self._held("BM_OAD")) / self._held("P_OAD")
        self._sell_notional = (1 - keep_pct) * self._held_notional

    def _sell_cusip_by_DTS_change(self, DTS_change):
        self._held_notional = self._held("P_Notional")
        DTS_target = self._held("DTS_Diff") - DTS_change
        keep_pct = (DTS_target + self._held("BM_DTS")) / self._held("P_DTS")
        self._sell_notional = (1 - keep_pct) * self._held_notional

    def _sell_cusip_by_DTS_target(self, DTS_target):
        self._held_notional = self._held("P_Notional")
        keep_pct = (DTS_target + self._held("BM_DTS")) / self._held("P_DTS")
        self._sell_notional = (1 - keep_pct) * self._held_notional

    def _add_sell_cusips(
        self,
        account,
        by_pct=None,
        by_notional=None,
        by_OAD_change=None,
        by_OAD_target=None,
        by_DTS_change=None,
        by_DTS_target=None,
    ):
        sell_rules = {k: v for k, v in locals().items() if k.startswith("by_")}

        self._account = account
        strategy = self.account_strategy_map[account]
        self._df = self.og_portfolios[strategy][account].df.set_index("CUSIP")
        self._held_cusips = set(self._df[self._df["P_Notional"] > 0].index)

        for rule, rule_d in sell_rules.items():
            if rule_d is not None:
                for cusip, amt in self._invert_d(rule_d).items():
                    self._sell_cusip(cusip, rule, amt)

    def _kwargs(self, d):
        ignored_keys = {
            "self",
            "strategies",
            "accounts",
            "ignored_accounts",
            "ignored_strategies",
        }
        return {k: v for k, v in d.items() if k not in ignored_keys}

    def add_cusip_sells(
        self,
        by_pct=None,
        by_notional=None,
        by_OAD_change=None,
        by_OAD_target=None,
        by_DTS_change=None,
        by_DTS_target=None,
        strategies=None,
        accounts=None,
        ignored_strategies=None,
        ignored_accounts=None,
    ):
        kwargs = self._kwargs(locals())
        self._universe = "UNIVERSAL"
        for account in self.accounts:
            self._add_sell_cusips(account, **kwargs)


class AccountTradeBuilder:
    """
    Class for by combining trades on individual CUSIPs for
    a single account to build an a full set of trades.
    Error handling is incorporated to ensure iteratively
    added trades don't override existing trades.
    """

    def __init__(self, account):
        self.account = account
        self.trades = {k: dict() for k in ["UNIVERSAL", "STRATEGY", "ACCOUNT"]}

    def __repr__(self):
        return f'AccountTradeBuilder("{self.account}")'

    def add_trade(self, cusip, notional, universe):
        """
        Add single cusip and notional to current trade.
        """
        if cusip in self.trades[universe]:
            raise TradeError(f"{cusip} already in trade for {self.account}")
        self.trades[universe][cusip] = notional


strategies = [
    "US Long GC 70/30",
    "US Long GC 75/25",
]
accounts = ["P-LD"]
ignored_accounts = "SEIGC"
# self = PortfolioTrade(strategies=strategies, accounts=accounts)
# self.load_data()
self = PortfolioTrade(data=self.og_portfolios)
self.ignore_accounts("SEIGC")

by_pct = {0.5: ["037833BW9", "125523AJ9"], 0.1: "744448CJ8"}
by_OAD_change = {0.02: ["961214ET6", "665772CU1"], 0.01: "30231GBF8"}
by_OAD_target = {0.02: ["12189TAZ7", "86944BAJ2"], 0.1: "91086QAZ1"}
by_notional = {2000: ["00108WAK6", "00115AAM1"], 50000: ["037833EE6lksjdf"]}


df = self.og_portfolios["US Long Credit"]["P-LD"].df

self.add_cusip_sells(
    by_pct=by_pct,
    by_OAD_change=by_OAD_change,
    by_OAD_target=by_OAD_target,
    by_notional=by_notional,
)


pprint(self.trades("P-LD").trades)
self.sell_errors()
# %%

# list(self.accounts[strategies[0]]['AILGC'].df)


x = {"csk": (1, 2), "asd": (3, 4)}
pd.DataFrame(x, index=["Notional", "Tag"]).T
account = "P-LD"


x = {}
cusip = by_pct[0.5][0]
cusip


# %%

df = self._example_df()
cusip = df.index[78]

with Time():
    for i in range(1000):
        cusips = set(df.index)


# %%


def bond(col):
    return df.at[cusip, col]


cusip = df.index[197]
cusip = df[df["P_Notional"] > 0].index[0]
bond("P_Notional")
