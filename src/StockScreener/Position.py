from dataclasses import dataclass, field, InitVar
from datetime import datetime
from decimal import *
from enum import Enum

class TransactionSide(str, Enum):
    Buy = "buy"
    Sell = "sell"

def EnsureDecimal(*args):
    toReturn = []
    for arg in args:
        if not isinstance(arg, Decimal):
            try:
                toReturn.append(Decimal(str(arg)))
            except:
                raise ValueError(f"Cannot convert type {type(arg)} to Decimal: {arg=}")
        else:
            toReturn.append(arg)

    return tuple(toReturn)

@dataclass
class Transaction:
    side: TransactionSide
    shares: Decimal
    fillPrice: Decimal
    reason: str
    date: datetime

    def __post_init__(self):
        self.shares = Decimal(str(self.shares))
        self.fillPrice = Decimal(str(self.fillPrice))

@dataclass
class PnL:
    realizedPnL: Decimal
    realizedPnL_percent: Decimal
    unrealizedPnL: Decimal
    unrealizedPnL_percent: Decimal

    def __post_init__(self):
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
        self.sharesPurchased, self.entryPrice = EnsureDecimal(self.sharesPurchased, self.entryPrice)

        self.sharesRemaining = self.sharesPurchased

    def SellShares(self, sharesToSell: Decimal):
        if sharesToSell > self.sharesRemaining:
            raise ValueError(f"Cannot sell more stocks than are allocated in this lot: {sharesToSell=}, {self.sharesRemaining}")
        if sharesToSell < 0:
            raise ValueError(f"Cannot sell a negative number of shares: {sharesToSell=}")
        
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

        self.ticker = ticker

        self.transactionHistory = []
        self.lots = []

        self.realizedPnL = Decimal(0)
        self.realizedCostBasis = Decimal(0)

        self.Buy(sharesPurchased, costPerShare, reason, date)

    @property
    def averageEntry(self) -> Decimal:
        """Use the totalCost of shares purchased/sold and the total number of shares held to
        compute the average entry
        """
        numOpenShares = self.numberOpenShares
        if numOpenShares == 0: return Decimal(0)
        return self.costBasis / numOpenShares
    
    @property
    def costBasis(self) -> Decimal:
        costBasis = Decimal(0)
        for lot in self.lots:
            costBasis += lot.costBasis

        return costBasis
    
    @property
    def numberOpenShares(self) -> Decimal:
        return Decimal(sum(lot.sharesRemaining for lot in self.lots))
    
    @property
    def realizedPnL_percent(self) -> Decimal:
        if self.realizedCostBasis == 0: return Decimal(0)
        return Decimal(100) * self.realizedPnL / self.realizedCostBasis
    
    def Buy(self, sharesPurchased: Decimal, costPerShare: Decimal, reason: str, date: datetime):
        """Add to a position by purchasing more of a stock

        Args:
            date: the date the position was added to
            sharesPurchased: the number of shares added
            costPerShare: the cost for one share
            reason: the reason the position was added to
        """

        sharesPurchased, costPerShare = EnsureDecimal(sharesPurchased, costPerShare)

        if sharesPurchased <= 0:
            raise ValueError(f"Shares purchased be positive, not {sharesPurchased}")
        if costPerShare is None or costPerShare <= 0:
            raise ValueError(f"Cost per share should be a valid positive integer, not {costPerShare}")
        
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
        
        sharesSold, costPerShare = EnsureDecimal(sharesSold, costPerShare)

        if sharesSold <= 0:
            raise ValueError(f"Shares purchased be positive, not {sharesSold}")
        if costPerShare is None or costPerShare <= 0:
            raise ValueError(f"Cost per share should be a valid positive integer, not {costPerShare}")
        if sharesSold >= self.numberOpenShares:
            raise ValueError(f"Cannot sell more shares than are owned: {sharesSold=}, {self.numberOpenShares=}")

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
        costPerShare, = EnsureDecimal(costPerShare)
        openCostBasis = sum(lot.sharesRemaining * lot.entryPrice for lot in self.lots)
        unrealizedPnL = self.numberOpenShares * costPerShare - openCostBasis

        if openCostBasis == 0:
            return unrealizedPnL, Decimal(0)

        return unrealizedPnL, Decimal(100) * unrealizedPnL / openCostBasis
    
    def GetPnL(self, costPerShare: Decimal):
        return PnL(self.realizedPnL, self.realizedPnL_percent, *self.GetUnrealizedPnL(costPerShare))

    def CurrentMarketValue(self, costPerShare: Decimal):
        """Using the cost per share, compute the current market value of the
        position

        Args:
            costPerShare: the cost for one share

        Returns:
            The total market value of the position
        """

        costPerShare, = EnsureDecimal(costPerShare)

        return self.numberOpenShares * costPerShare
    
    
    
    # def SaleProceeds(self):
    #     """Return the amount of money received from selling parts of the position

    #     Returns:
    #         The amount of money received from selling parts of the position
    #     """

    #     return -1 * sum(hist.dollarValueChange for hist in self.history.values() if hist.dollarValueChange < 0)