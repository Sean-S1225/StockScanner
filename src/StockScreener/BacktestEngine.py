from .Portfolio import Portfolio
from .Strategy import Strategy

class BacktestEngine:
    def __init__(self, buyStrategy: Strategy, sellStrategy: Strategy):
        self.portfolio = Portfolio()
        self.buyStrategy = buyStrategy
        self.sellStrategy = sellStrategy

    def RunBacktest(self):
        pass