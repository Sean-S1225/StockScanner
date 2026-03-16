from .Position import *
from datetime import datetime
from decimal import *

class Portfolio:
    def __init__(self, startingCash: Decimal = 1000):
        """Initialize an empty portfolio

        Args:
            startingCash (optional): The amount of money to start with. Defaults to 1000.
        """

        startingCash, = EnsureDecimal(startingCash)

        self.portfolio = {}
        self.cash = startingCash

    def Buy(self, ticker: str, sharesPurchased: Decimal, costPerShare: Decimal, reason: str, date: datetime):
        """Purchase a stock

        Args:
            ticker: The ticker to purchase
            date: The date of purchase
            sharesPurchased: The number of shares to purchase
            costPerShare: The cost per share
            reason: The reason for purchase
        """
        
        sharesPurchased, costPerShare = EnsureDecimal(sharesPurchased, costPerShare)

        if self.cash < sharesPurchased * costPerShare:
            raise ValueError(f"Not enough cash to purchase. {self.cash=} < {(sharesPurchased * costPerShare)=}.")

        if ticker in self.portfolio:
            self.portfolio[ticker].Buy(sharesPurchased, costPerShare, reason, date)
        else:
            self.portfolio[ticker] = Position(ticker, sharesPurchased, costPerShare, reason, date)

        self.cash -= sharesPurchased * costPerShare

    def Sell(self, ticker: str, sharesSold: Decimal, costPerShare: Decimal, reason: str, date: datetime):
        """Sell a stock

        Args:
            ticker: The ticker to sell
            date: The date of sell
            sharesSold: The number of shares to sell
            costPerShare: The cost per share
            reason: The reason for sell
        """
        
        sharesSold, costPerShare = EnsureDecimal(sharesSold, costPerShare)

        if ticker not in self.portfolio:
            raise ValueError(f"{ticker=} is not in the portfolio: {list(self.portfolio.keys())=}")
        if sharesSold > self.portfolio[ticker].numberOpenShares:
            raise ValueError(f"{sharesSold=} should not exceed the number of shares available: {self.portfolio[ticker].numberOpenShares=}")
        
        proceeds = self.portfolio[ticker].Sell(sharesSold, costPerShare, reason, date)
        self.cash += proceeds

    def GetPortfolioPnL(self, currentCosts: dict[str, Decimal]):
        realizedPnL = Decimal(0)
        unrealizedPnL = Decimal(0)
        realizedCostBasis = Decimal(0)
        unrealizedCostBasis = Decimal(0)

        for ticker in self.portfolio:
            realizedPnL += self.portfolio[ticker].realizedPnL
            unrealizedPnL += self.portfolio[ticker].GetUnrealizedPnL(currentCosts[ticker])[0]

            realizedCostBasis += self.portfolio[ticker].realizedCostBasis
            unrealizedCostBasis += self.portfolio[ticker].costBasis

        realizedPnL_percent = Decimal("0")
        if realizedCostBasis != 0:
            realizedPnL_percent = Decimal("100") * realizedPnL / realizedCostBasis

        unrealizedPnL_percent = Decimal("0")
        if unrealizedCostBasis != 0:
            unrealizedPnL_percent = Decimal("100") * unrealizedPnL / unrealizedCostBasis

        return PnL(realizedPnL, realizedPnL_percent, unrealizedPnL, unrealizedPnL_percent)
