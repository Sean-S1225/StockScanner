from .Position import Position

class Portfolio:
    def __init__(self, startingCash: int = 1000):
        """Initialize an empty portfolio

        Args:
            startingCash (optional): The amount of money to start with. Defaults to 1000.
        """

        self.portfolio = {}
        self.cash = startingCash

    def Buy(self, ticker: str, date: str, sharesPurchased: float, costPerShare: float, reason: str):
        """Purchase a stock

        Args:
            ticker: The ticker to purchase
            date: The date of purchase
            sharesPurchased: The number of shares to purchase
            costPerShare: The cost per share
            reason: The reason for purchase
        """


        if ticker in self.portfolio:
            if date in self.portfolio[ticker].history:
                raise ValueError("Multiple transactions cannot occur on the same day.")
            
            self.portfolio[ticker].Buy(date, sharesPurchased, costPerShare, reason)
        else:
            self.portfolio[ticker] = Position(ticker, date, sharesPurchased, costPerShare, reason)

    def Sell(self, ticker: str, date: str, sharesSold: float, costPerShare: float, reason: str):
        """Sell a stock

        Args:
            ticker: The ticker to sell
            date: The date of sell
            sharesSold: The number of shares to sell
            costPerShare: The cost per share
            reason: The reason for sell
        """

        if ticker not in self.portfolio:
            raise ValueError(f"{ticker=} is not in the portfolio: {list(self.portfolio.keys())=}")
        if sharesSold > self.portfolio[ticker].totalShares:
            raise ValueError(f"{sharesSold=} should not exceed the number of shares available: {self.portfolio[ticker].totalShares=}")
        
        money = self.portfolio[ticker].Sell(date, sharesSold, costPerShare, reason)
        self.cash += money

    def ComputeUnrealizedPnL_ticker(self, currentPrice, history, sharesSold):
        history = [x for x in history if x[0] > 0]

        i = 0
        while sharesSold > 0:
            if sharesSold >= history[i][0]:
                sharesSold -= history[i][0]
                history.pop(0)
            else:
                history[i] = (history[i][0] - sharesSold, history[i][1])
                sharesSold = 0

        costBasis = sum(x[0] * x[1] for x in history)
        unrealizedProfits = sum(x[0] for x in history) * currentPrice

        return unrealizedProfits - costBasis, costBasis


    def ComputePnL_Ticker(self, ticker: str, currentPrice: float):
        history = []
        for (_, positionChange) in self.portfolio[ticker].history.items():
                history.append((positionChange.sharesChanged, positionChange.costPerShare))

        sharesSold = -1 * sum(x[0] for x in history if x[0] < 0)

        realizedCostBasis = 0

        i = 0
        while sharesSold > 0:
            if history[i][0] < 0:
                i+=1
                continue

            if sharesSold >= history[i][0]:
                realizedCostBasis += history[i][0] * history[i][1]
                sharesSold -= history[i][0]
            else:
                realizedCostBasis += sharesSold * history[i][1]
                sharesSold -= sharesSold

            i+=1

        sharesSold = -1 * sum(x[0] for x in history if x[0] < 0)

        realizedSaleProceeds = -1 * sum(x[0] * x[1] for x in history if x[0] < 0)

        realizedPnL = realizedSaleProceeds - realizedCostBasis
        unrealizedPnL, unrealizedCostBasis = self.ComputeUnrealizedPnL_ticker(currentPrice, history, sharesSold)

        realizedPnL_percent = 0
        if realizedCostBasis != 0:
            realizedPnL_percent = realizedPnL / realizedCostBasis * 100
        
        unrealizedPnL_percent = 0
        if unrealizedCostBasis != 0:
            unrealizedPnL_percent = unrealizedPnL / unrealizedCostBasis * 100


        return {
            "Realized PnL": realizedPnL,
            "Realized PnL (%)": realizedPnL_percent,
            "Unrealized PnL": unrealizedPnL,
            "Unrealized PnL (%)": unrealizedPnL_percent,
            "Position Cost Basis":  unrealizedCostBasis
        }

    def ComputePnL_Portfolio(self, closeData: dict[str, float]):
        """Compute PnL for all positions in the portfolio

        Args:
            closeData: A dictionary of (ticker, price) pairs

        Returns:
            A dictionary of (ticker, (raw PnL, percentage PnL)) values
        """

        toReturn = {}

        for ticker in self.portfolio:
            toReturn[ticker] = self.ComputePnL_Ticker(ticker, closeData[ticker])

        return toReturn