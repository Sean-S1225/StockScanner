from enum import Enum

class PlotIndicator(str, Enum):
    EMA = "ema_200"
    FIB = "fib"
    MACD = "macd"
    VMC = "vmc"

class IndicatorPanel(str, Enum):
    PRICE = "price"
    MACD = "macd"
    VMC = "vmc"

class AxisTickMode(str, Enum):
    AUTO = "auto"
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"

class TransactionSide(str, Enum):
    Buy = "buy"
    Sell = "sell"