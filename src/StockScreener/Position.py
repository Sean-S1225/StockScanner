from dataclasses import dataclass, field
from collections import OrderedDict
from numpy import isnan

@dataclass
class PositionChange:
    """
    Used to track buying/selling events to a position
    date (str): the date the stock was purchased/sold
    sharesChanged (float): the number of shares purchased/sold
    costPerShare (float): the cost per one whole share
    reasonForChange (str): why was the stock purchased/sold
    """
    date: str
    sharesChanged: float
    costPerShare: float
    dollarValueChange: float
    reasonForChange: str

class Position:
    def __init__(self, ticker: str, date: str, sharesPurchased: float, costPerShare: float, reason: str):
        """Begin a new position with a purchase.

        Args:
            ticker: the ticker of the stock
            date: The date the position began
            sharesPurchased: the number of positions the position began with
            costPerShare: the cost of one share
            reason: the reason for beginning the purchase
        """

        self.ticker = ticker

        self.history = [
            PositionChange(date, sharesPurchased, costPerShare, costPerShare * sharesPurchased, reason)
        ]

        self.numberOfShares = sharesPurchased
        self.costBasis = costPerShare * sharesPurchased
        self._ComputeAverageEntry()

    def _ComputeAverageEntry(self):
        """Use the totalCost of shares purchased/sold and the total number of shares held to
        compute the average entry
        """
        if self.numberOfShares == 0: return 0
        self.averageEntry = self.costBasis / self.numberOfShares

    def Buy(self, date: str, sharesPurchased: float, costPerShare: float, reason: str):
        """Add to a position by purchasing more of a stock

        Args:
            date: the date the position was added to
            sharesPurchased: the number of shares added
            costPerShare: the cost for one share
            reason: the reason the position was added to
        """

        if sharesPurchased <= 0:
            raise ValueError(f"Shares purchased be positive, not {sharesPurchased}")
        if costPerShare <= 0 or isnan(costPerShare):
            raise ValueError(f"Cost per share should be a valid positive integer, not {costPerShare}")

        cost = sharesPurchased * costPerShare
        if cost > self.cash:
            raise ValueError("Not enough cash")
        self.cash -= cost

        self.history.append(PositionChange(date, sharesPurchased, costPerShare, cost, reason))

        self.numberOfShares += sharesPurchased
        self.costBasis += cost
        self._ComputeAverageEntry()

    def Sell(self, date: str, sharesSold: float, costPerShare: float, reason: str):
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
        if costPerShare <= 0 or isnan(costPerShare):
            raise ValueError(f"Cost per share should be a valid positive integer, not {costPerShare}")

        self.history.append(PositionChange(date, -1 * sharesSold, costPerShare, -1 * costPerShare * sharesSold, reason))

        self.numberOfShares -= sharesSold
        self.costBasis -= costPerShare * sharesSold

        return sharesSold * costPerShare

    def CurrentMarketValue(self, costPerShare):
        """Using the cost per share, compute the current market value of the
        position

        Args:
            costPerShare: the cost for one share

        Returns:
            The total market value of the position
        """

        return self.numberOfShares * costPerShare
    
    def SaleProceeds(self):
        """Return the amount of money received from selling parts of the position

        Returns:
            The amount of money received from selling parts of the position
        """

        return -1 * sum(hist.dollarValueChange for hist in self.history.values() if hist.dollarValueChange < 0)