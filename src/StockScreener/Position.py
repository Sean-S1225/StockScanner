from dataclasses import dataclass, field, InitVar
from collections import OrderedDict
from numpy import isnan
from datetime import datetime
from decimal import *

@dataclass
class Transaction:
    ticker: str
    side: str # buy or sell
    shares: Decimal
    fillPrice: Decimal
    reason: str
    date: datetime

    def __post_init__(self):
        self.shares = Decimal(self.shares)
        self.fillPrice = Decimal(self.fillPrice)

@dataclass
class PnL:
    realizedPnL: Decimal
    realizedPnL_percent: Decimal
    unrealizedPnL: Decimal
    unrealizedPnL_percent: Decimal

    def __post_init__(self):
        self.realizedPnL = Decimal(self.realizedPnL)
        self.realizedPnL_percent = Decimal(self.realizedPnL_percent)
        self.unrealizedPnL = Decimal(self.unrealizedPnL)
        self.unrealizedPnL_percent = Decimal(self.unrealizedPnL_percent)

@dataclass
class Lot:
    ticker: str
    sharesPurchased: Decimal
    entryPrice: Decimal
    reason: InitVar[str]
    acquisitionDate: datetime

    @property
    def sharesSold(self) -> Decimal:
        return self.sharesPurchased - self.sharesRemaining

    def __post_init__(self, reason: str):
        self.sharesPurchased = Decimal(self.sharesPurchased)
        self.entryPrice = Decimal(self.entryPrice)

        self.sharesRemaining = self.sharesPurchased
        self.transactions = [
            Transaction(self.ticker, "buy", self.sharesPurchased, self.entryPrice, reason, self.acquisitionDate)
        ]

        self.proceeds = Decimal(0)
        self.costBasis = self.sharesPurchased * self.entryPrice
        self._pnl = PnL(0, 0, 0, 0)

    def SellShares(self, sharesToSell: Decimal, currentPrice: Decimal, date: datetime):
        if sharesToSell > self.sharesRemaining:
            raise ValueError(f"Cannot sell more stocks than are allocated in this lot: {sharesToSell=}, {self.sharesRemaining}")
        
        self.transactions.append(
            Transaction(self.ticker, "sell", sharesToSell, currentPrice, "", date)
        )

        self._pnl.realizedPnL += (sharesToSell * currentPrice) - sharesToSell * self.entryPrice
        self.sharesRemaining -= sharesToSell
        self.proceeds += sharesToSell * currentPrice
        self.costBasis -= sharesToSell * self.entryPrice

        soldCostBasis = self.sharesSold * self.entryPrice
        if soldCostBasis != 0:
            self._pnl.realizedPnL_percent = 100 * (self.proceeds - soldCostBasis) / soldCostBasis
        else:
            self._pnl.realizedPnL_percent = 0

    def GetPnL(self, currentPrice: Decimal):
        couldBeSoldCostBasis = self.sharesRemaining * self.entryPrice
        self._pnl.unrealizedPnL = ((self.sharesRemaining * currentPrice) - (couldBeSoldCostBasis))
        if couldBeSoldCostBasis != 0:
            self._pnl.unrealizedPnL_percent = 100 * self._pnl.unrealizedPnL / (self.sharesRemaining * self.entryPrice)
        else:
            self._pnl.unrealizedPnL_percent = 0

        return self._pnl


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

        self.transactionHistory = [Transaction(self.ticker, "buy", sharesPurchased, costPerShare, reason, date)]
        self.lots = [Lot(self.ticker, sharesPurchased, costPerShare, reason, date)]

        self.numberOpenShares = sharesPurchased

        self._ComputeAverageEntry()

        self.pnl = PnL(0, 0, 0, 0)

    def _ComputeAverageEntry(self):
        """Use the totalCost of shares purchased/sold and the total number of shares held to
        compute the average entry
        """
        if self.numberOpenShares == 0: return 0
        self.averageEntry = self.GetCostBasis() / self.numberOpenShares

    def Buy(self, sharesPurchased: Decimal, costPerShare: Decimal, reason: str, date: datetime):
        """Add to a position by purchasing more of a stock

        Args:
            date: the date the position was added to
            sharesPurchased: the number of shares added
            costPerShare: the cost for one share
            reason: the reason the position was added to
        """

        if sharesPurchased <= 0:
            raise ValueError(f"Shares purchased be positive, not {sharesPurchased}")
        if costPerShare is None or costPerShare <= 0:
            raise ValueError(f"Cost per share should be a valid positive integer, not {costPerShare}")

        self.transactionHistory.append(
            Transaction(self.ticker, "buy", sharesPurchased, costPerShare, reason, date)
        )

        self.lots.append(
            Lot(self.ticker, sharesPurchased, costPerShare, reason, date)
        )

        self.numberOpenShares += sharesPurchased
        self._ComputeAverageEntry()

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

        if sharesSold <= 0:
            raise ValueError(f"Shares purchased be positive, not {sharesSold}")
        if costPerShare is None or costPerShare <= 0:
            raise ValueError(f"Cost per share should be a valid positive integer, not {costPerShare}")
        if sharesSold > self.numberOpenShares:
            raise ValueError(f"Cannot sell more shares than are owned: {sharesSold=}, {self.numberOpenShares=}")
        
        self.transactionHistory.append(
            Transaction(self.ticker, "sell", sharesSold, costPerShare, reason, date)
        )

        self.numberOpenShares -= sharesSold

        i = 0
        temp = sharesSold
        while temp > 0:
            if self.lots[i].sharesRemaining > 0:
                if self.lots[i].sharesRemaining > temp:
                    self.lots[i].SellShares(temp, costPerShare, date)
                    temp = 0
                else:
                    temp -= self.lots[i].sharesRemaining
                    self.lots[i].SellShares(self.lots[i].sharesRemaining, costPerShare, date)
            i += 1

        return sharesSold * costPerShare
    
    def ComputePnL(self, costPerShare: Decimal):
        self.pnl = PnL(0, 0, 0, 0)

        soldCostBasis = 0
        unsoldCostBasis = 0

        for lot in self.lots:
            lot_pnl = lot.GetPnL(costPerShare)
            self.pnl.realizedPnL += lot_pnl.realizedPnL
            self.pnl.unrealizedPnL += lot_pnl.unrealizedPnL

            soldCostBasis += lot.entryPrice * lot.sharesSold
            unsoldCostBasis += lot.entryPrice * lot.sharesRemaining

        soldProceeds = sum(lot.proceeds for lot in self.lots)
        unsoldTheoreticalProceeds = sum(lot.sharesRemaining * costPerShare for lot in self.lots)

        if soldCostBasis != 0:
            self.pnl.realizedPnL_percent = 100 * (soldProceeds - soldCostBasis) / soldCostBasis
        if unsoldCostBasis != 0:
            self.pnl.unrealizedPnL_percent = 100 * (unsoldTheoreticalProceeds - unsoldCostBasis) / unsoldCostBasis

        return self.pnl

    def CurrentMarketValue(self, costPerShare):
        """Using the cost per share, compute the current market value of the
        position

        Args:
            costPerShare: the cost for one share

        Returns:
            The total market value of the position
        """

        return self.numberOpenShares * costPerShare
    
    def GetCostBasis(self):
        costBasis = 0
        for lot in self.lots:
            costBasis += lot.costBasis

        return costBasis
    
    # def SaleProceeds(self):
    #     """Return the amount of money received from selling parts of the position

    #     Returns:
    #         The amount of money received from selling parts of the position
    #     """

    #     return -1 * sum(hist.dollarValueChange for hist in self.history.values() if hist.dollarValueChange < 0)