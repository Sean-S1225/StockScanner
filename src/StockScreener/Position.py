from dataclasses import dataclass
from datetime import datetime
from decimal import *

from StockScreener.enums import TransactionSide
from StockScreener.helpers import EnsureDecimal, EnsureType

@dataclass
class Transaction:
    side: TransactionSide
    shares: Decimal
    fillPrice: Decimal
    reason: str
    date: datetime

    def __post_init__(self):
        # Input validation
        self.shares, self.fillPrice = EnsureDecimal(self.shares, self.fillPrice)
        self.side = EnsureType(self.side, TransactionSide, "self.side")
        self.reason = EnsureType(self.reason, str, "self.reason")
        self.date = EnsureType(self.date, datetime, "self.date")

@dataclass
class PnL:
    realizedPnL: Decimal
    realizedPnL_percent: Decimal
    unrealizedPnL: Decimal
    unrealizedPnL_percent: Decimal

    def __post_init__(self):
        # Input validation
        self.realizedPnL, self.realizedPnL_percent, self.unrealizedPnL, self.unrealizedPnL_percent = EnsureDecimal(
            self.realizedPnL, self.realizedPnL_percent, self.unrealizedPnL, self.unrealizedPnL_percent
        )

@dataclass
class Lot:
    sharesPurchased: Decimal
    entryPrice: Decimal
    acquisitionDate: datetime

    @property
    def sharesSold(self) -> Decimal:
        return self.sharesPurchased - self.sharesRemaining
    
    @property
    def costBasis(self) -> Decimal:
        return self.sharesRemaining * self.entryPrice

    def __post_init__(self):
        self.acquisitionDate = EnsureType(self.acquisitionDate, datetime, "self.acquisitionDate")
        if not isinstance(self.acquisitionDate, datetime): 
            raise TypeError(f"acquisitionDate must be of type datetime, not {type(self.acquisitionDate)}")
        
        self.sharesPurchased, self.entryPrice = EnsureDecimal(self.sharesPurchased, self.entryPrice)

        self.sharesRemaining = self.sharesPurchased

    def SellShares(self, sharesToSell: Decimal):
        sharesToSell = EnsureType(sharesToSell, Decimal, "sharesToSell",
                                  condition=lambda x: x <= self.sharesRemaining and x > 0)
        
        self.sharesRemaining -= sharesToSell

class Position:
    def __init__(self, ticker: str, sharesPurchased: Decimal, costPerShare: Decimal, reason: str, date: datetime):
        """Begin a new position with a purchase.

        Args:
            ticker: the ticker of the stock
            date: The date the position began
            sharesPurchased: the number of positions the position began with
            costPerShare: the cost of one share
            reason: the reason for beginning the purchase
        """

        ticker = EnsureType(ticker, str, "ticker")
        self.ticker = ticker

        self.transactionHistory = []
        self.lots = []

        self.realizedPnL = Decimal("0")
        self.realizedCostBasis = Decimal("0")

        self.Buy(sharesPurchased, costPerShare, reason, date)

    @property
    def averageEntry(self) -> Decimal:
        """Use the totalCost of shares purchased/sold and the total number of shares held to
        compute the average entry
        """
        numOpenShares = self.numberOpenShares
        if numOpenShares == 0: return Decimal("0")
        return self.costBasis / numOpenShares
    
    @property
    def costBasis(self) -> Decimal:
        costBasis = Decimal("0")
        for lot in self.lots:
            costBasis += lot.costBasis

        return costBasis
    
    @property
    def numberOpenShares(self) -> Decimal:
        return sum((lot.sharesRemaining for lot in self.lots), Decimal("0"))
    
    @property
    def realizedPnL_percent(self) -> Decimal:
        if self.realizedCostBasis == 0: return Decimal("0")
        return Decimal(100) * self.realizedPnL / self.realizedCostBasis
    
    def Buy(self, sharesPurchased: Decimal, costPerShare: Decimal, reason: str, date: datetime):
        """Add to a position by purchasing more of a stock

        Args:
            date: the date the position was added to
            sharesPurchased: the number of shares added
            costPerShare: the cost for one share
            reason: the reason the position was added to
        """
        sharesPurchased = EnsureType(sharesPurchased, Decimal, "sharesPurchased", condition=lambda x: x > 0)
        costPerShare = EnsureType(costPerShare, Decimal, "costPerShare", condition=lambda x: x > 0)
        reason = EnsureType(reason, str, "reason")
        date = EnsureType(date, datetime, "date")
        
        self.transactionHistory.append(
            Transaction(TransactionSide.Buy, sharesPurchased, costPerShare, reason, date)
        )

        self.lots.append(
            Lot(sharesPurchased, costPerShare, date)
        )

    def Sell(self, sharesSold: Decimal, costPerShare: Decimal, reason: str, date: datetime):
        """Reduce a position by selling some of a stock

        Args:
            date: the date the position was reduced
            sharesSold: the number of shares reduced by
            costPerShare: the price per share
            reason: the reason the position was decreased

        Returns:
            The amount of money received upon selling
        """

        sharesSold = EnsureType(sharesSold, Decimal, "sharesSold", condition=lambda x: x > 0 and x <= self.numberOpenShares)
        costPerShare = EnsureType(costPerShare, Decimal, "costPerShare", condition=lambda x: x > 0)
        reason = EnsureType(reason, str, "reason")
        date = EnsureType(date, datetime, "date")
        
        sharesSold, costPerShare = EnsureDecimal(sharesSold, costPerShare)

        self.transactionHistory.append(
            Transaction(TransactionSide.Sell, sharesSold, costPerShare, reason, date)
        )

        i = 0
        temp = sharesSold
        while temp > 0:
            if self.lots[i].sharesRemaining > 0:
                if self.lots[i].sharesRemaining > temp:

                    self.realizedPnL += temp * (costPerShare - self.lots[i].entryPrice)
                    self.realizedCostBasis += temp * self.lots[i].entryPrice

                    self.lots[i].SellShares(temp)
                    temp = 0

                else:

                    self.realizedPnL += self.lots[i].sharesRemaining * (costPerShare - self.lots[i].entryPrice)
                    self.realizedCostBasis += self.lots[i].sharesRemaining * self.lots[i].entryPrice

                    temp -= self.lots[i].sharesRemaining
                    self.lots[i].SellShares(self.lots[i].sharesRemaining)

            i += 1

        return sharesSold * costPerShare
    
    def GetUnrealizedPnL(self, costPerShare: Decimal):
        costPerShare = EnsureType(costPerShare, Decimal, "costPerShare", condition=lambda x: x > 0)
        openCostBasis = sum(lot.sharesRemaining * lot.entryPrice for lot in self.lots)
        unrealizedPnL = self.numberOpenShares * costPerShare - openCostBasis

        return unrealizedPnL
    
    def GetUnrealizedPnL_percent(self, costPerShare: Decimal):
        costPerShare = EnsureType(costPerShare, Decimal, "costPerShare", condition=lambda x: x > 0)
        openCostBasis = sum(lot.sharesRemaining * lot.entryPrice for lot in self.lots)
        unrealizedPnL = self.numberOpenShares * costPerShare - openCostBasis

        if openCostBasis == 0:
            return Decimal("0")

        return Decimal(100) * unrealizedPnL / openCostBasis

    
    def GetPnL(self, costPerShare: Decimal):
        return PnL(self.realizedPnL, self.realizedPnL_percent, self.GetUnrealizedPnL(costPerShare), self.GetUnrealizedPnL_percent(costPerShare))

    def CurrentMarketValue(self, costPerShare: Decimal):
        """Using the cost per share, compute the current market value of the
        position

        Args:
            costPerShare: the cost for one share

        Returns:
            The total market value of the position
        """

        costPerShare = EnsureType(costPerShare, Decimal, "costPerShare", condition=lambda x: x > 0)

        return self.numberOpenShares * costPerShare