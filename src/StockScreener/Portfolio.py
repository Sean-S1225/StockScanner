from dataclasses import dataclass
from .Position import *
from datetime import datetime
from decimal import *
import pandas as pd

@dataclass
class PositionSummary:
    ticker: str
    openShares: Decimal
    averageEntry: Decimal
    costBasis: Decimal
    marketValue: Decimal
    realizedPnL: Decimal
    realizedPnL_percent: Decimal
    unrealizedPnL: Decimal
    unrealizedPnL_percent: Decimal

class Portfolio:
    def __init__(self, startingCash: Decimal = Decimal("1000")):
        """Initialize an empty portfolio

        Args:
            startingCash (optional): The amount of money to start with. Defaults to 1000.
        """

        startingCash = EnsureType(startingCash, Decimal, "startingCash", condition=lambda x: x > 0)

        self.positions = {}
        self.closedPositions = []
        self.startingCash = startingCash
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
        
        ticker = EnsureType(ticker, str, "ticker")
        sharesPurchased = EnsureType(sharesPurchased, Decimal, "sharesPurchased", condition=lambda x: x > 0)
        costPerShare = EnsureType(costPerShare, Decimal, "costPerShare", condition=lambda x: x > 0)
        reason = EnsureType(reason, str, "reason")
        date = EnsureType(date, datetime, "date")

        if self.cash < sharesPurchased * costPerShare:
            raise ValueError(f"Not enough cash to purchase. {self.cash=} < {(sharesPurchased * costPerShare)=}.")

        if ticker in self.positions:
            self.positions[ticker].Buy(sharesPurchased, costPerShare, reason, date)
        else:
            self.positions[ticker] = Position(ticker, sharesPurchased, costPerShare, reason, date)

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
        
        ticker = EnsureType(ticker, str, "ticker")
        sharesSold = EnsureType(sharesSold, Decimal, "sharesSold", condition=lambda x: x > 0)
        costPerShare = EnsureType(costPerShare, Decimal, "costPerShare", condition=lambda x: x > 0)
        reason = EnsureType(reason, str, "reason")
        date = EnsureType(date, datetime, "date")

        if ticker not in self.positions:
            raise ValueError(f"{ticker=} is not in the portfolio: {list(self.positions.keys())=}")
        if sharesSold > self.positions[ticker].numberOpenShares:
            raise ValueError(f"{sharesSold=} should not exceed the number of shares available: {self.positions[ticker].numberOpenShares=}")
        
        proceeds = self.positions[ticker].Sell(sharesSold, costPerShare, reason, date)
        self.cash += proceeds

        if self.positions[ticker].numberOpenShares == 0:
            self.closedPositions.append(self.positions[ticker])
            del self.positions[ticker]

    def GetPortfolioPnL(self, currentCosts: dict[str, Decimal]):
        realizedPnL = Decimal(0)
        unrealizedPnL = Decimal(0)
        realizedCostBasis = Decimal(0)
        unrealizedCostBasis = Decimal(0)

        for position in self.closedPositions:
            realizedPnL += position.realizedPnL
            realizedCostBasis += position.realizedCostBasis

        for ticker in self.positions:
            if ticker not in currentCosts:
                raise KeyError(f"{ticker=} not found in {list(currentCosts.keys())=}")

            realizedPnL += self.positions[ticker].realizedPnL
            unrealizedPnL += self.positions[ticker].GetUnrealizedPnL(currentCosts[ticker])

            realizedCostBasis += self.positions[ticker].realizedCostBasis
            unrealizedCostBasis += self.positions[ticker].costBasis

        realizedPnL_percent = Decimal("0")
        if realizedCostBasis != 0:
            realizedPnL_percent = Decimal("100") * realizedPnL / realizedCostBasis

        unrealizedPnL_percent = Decimal("0")
        if unrealizedCostBasis != 0:
            unrealizedPnL_percent = Decimal("100") * unrealizedPnL / unrealizedCostBasis

        return PnL(realizedPnL, realizedPnL_percent, unrealizedPnL, unrealizedPnL_percent)

    def GetPortfolioMarketCap(self, currentCosts: dict[str, Decimal]):
        toReturn = Decimal(0)
        for ticker in self.positions:
            if ticker not in currentCosts:
                raise KeyError(f"{ticker=} not found in {list(currentCosts.keys())=}")
            
            toReturn += self.positions[ticker].CurrentMarketValue(currentCosts[ticker])

        return toReturn

    def GetEquity(self, currentCosts: dict[str, Decimal]):
        return self.cash + self.GetPortfolioMarketCap(currentCosts)
    
    def GetReturnPercent(self, currentCosts: dict[str, Decimal]):
        if self.startingCash == 0: return Decimal(0)
        return Decimal(100) * (self.GetEquity(currentCosts) - self.startingCash) / self.startingCash
    
    def SummarizeHoldings(self, currentCosts):
        summary = []

        for position in self.closedPositions:
            summary.append(
                PositionSummary(position.ticker, position.numberOpenShares, position.averageEntry, position.costBasis,
                                Decimal("0"), position.realizedPnL,
                                position.realizedPnL_percent, Decimal("0"),
                                Decimal("0"))
            )

        for ticker, position in self.positions.items():
            summary.append(
                PositionSummary(ticker, position.numberOpenShares, position.averageEntry, position.costBasis,
                                position.CurrentMarketValue(currentCosts[ticker]), position.realizedPnL,
                                position.realizedPnL_percent, position.GetUnrealizedPnL(currentCosts[ticker]),
                                position.GetUnrealizedPnL_percent(currentCosts[ticker]))
            )

        return pd.DataFrame(summary)