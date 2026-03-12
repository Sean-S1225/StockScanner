from dataclasses import dataclass, field
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

@dataclass
class ScreenConfig:
    min_num_weeks: int = 200
    min_market_cap: int = 2_000_000_000
    min_distance_from_5y_low_pct: float = 0.0
    near_ema_threshold_pct: float = 10.0
    one_year_pullback_threshold_pct: float = 40.0
    min_adtv: float = 10_000_000
    min_avg_shares: int = 300_000
    earnings_threshold: float = 5 / 8

@dataclass
class StockHistoryConfig:
    period: str = "max"
    interval: str = "1wk"

@dataclass
class EMAConfig:
    period: int
    col_name: str
    color: str = "mediumblue"
    linewidth: float = 0.75

@dataclass
class FibConfig:
    colName: str = "fib"

    bottomLineColor: str = "#000000"
    fib236LineColor: str = "#FF0000"
    fib382LineColor: str = "#008000"
    fib50LineColor: str  = "#00FF00"
    fib618LineColor: str = "#008000"
    fib764LineColor: str = "#FF0000"
    topLineColor: str = "#000000"

    bottom_fib236_fillColor: str = "#ffeb3b"
    fib236_fib382_fillColor: str = "#3399FF"
    fib382_fib50_fillColor: str = "#00FF00"
    fib50_fib618_fillColor: str = "#00FF00"
    fib618_fib764_fillColor: str = "#3399FF"
    fib764_top_fillColor: str = "#FF0000"
    fillAlpha: int = 0.1
    
    period: int = 265
    linewidth: float = 0.75

@dataclass
class MACDConfig:
    fastPeriod: int = 12
    slowPeriod: int = 26
    signalPeriod: int = 9
    macdColName: str = "MACD"
    signalColName: str = "MACD Signal"
    histColName: str = "MACD Hist"
    linewidth: float = 0.75

    macdLineColor: str = "#2962ff"
    signalLineColor: str = "#ff6d00"
    zeroLineColor: str = "#787b86"

    histPositiveDecreasingColor: str = "#b2dfdb"
    histPositiveIncreasingColor: str = "#26a69a"
    histNegativeDecreasingColor: str = "#ffcdd2"
    histNegativeIncreasingColor: str = "#ff5252"
    histBarWidth: float = 0.8

@dataclass
class VMCConfig:
    prefix: str = "VMC"

    showWaveTrend: bool = True
    showFastWT: bool = False
    showMFI: bool = False
    showRSI: bool = True
    showStoch: bool = True
    showSTC: bool = False

    showWTDivs: bool = True
    showWTHiddenDivs: bool = False
    showSecondWTDivs: bool = True
    showRSIDivs: bool = False
    showRSIHiddenDivs: bool = False
    showStochDivs: bool = False
    showStochHiddenDivs: bool = False

    showBuyDots: bool = True
    showSellDots: bool = True
    showGoldDots: bool = True

    wtChannelLen: int = 9
    wtAverageLen: int = 12
    wtMALen: int = 3

    obLevel: int = 53
    obLevel2: int = 60
    obLevel3: int = 100
    osLevel: int = -53
    osLevel2: int = -40
    osLevel3: int = -75

    wtDivOBLevel: int = 45
    wtDivOSLevel: int = -65
    wtDivOBLevelAddShow: bool = True
    wtDivOBLevelAdd: int = 15
    wtDivOSLevelAdd: int = -40
    showHiddenDivNoLimits: bool = True

    rsiMFIperiod: int = 60
    rsiMFIMultiplier: float = 150.0
    rsiMFIPosY: float = 2.5

    rsiLen: int = 14
    rsiOversold: int = 30
    rsiOverbought: int = 60
    rsiDivOBLevel: int = 60
    rsiDivOSLevel: int = 30

    stochUseLog: bool = True
    stochAvg: bool = False
    stochLen: int = 14
    stochRsiLen: int = 14
    stochKSmooth: int = 3
    stochDSmooth: int = 3

    tcLength: int = 10
    tcFastLength: int = 23
    tcSlowLength: int = 50
    tcFactor: float = 0.5

    zeroLineColor: str = "#ffffff80"
    wt1Color: str = "#4994ec"
    wt2Color: str = "#1f1559"
    wtVwapColor: str = "#ffffff80"
    rsiMFIAboveColor: str = "#3ee145"
    rsiMFIBelowColor: str = "#ff3d2e"
    rsiOverboughtColor: str = "#e13e3e"
    rsiOversoldColor: str = "#3ee145"
    rsiNeutralColor: str = "#c33ee1"
    stochKColor: str = "#21baf3"
    stochDColor: str = "#673ab7"
    stcColor: str = "#673ab7"

    wtBullDivColor: str = "#00e676"
    wtBearDivColor: str = "#e60000"
    rsiBullDivColor: str = "#38ff42"
    rsiBearDivColor: str = "#e60000"
    stochBullDivColor: str = "#38ff42"
    stochBearDivColor: str = "#e60000"

    crossUpColor: str = "#00e676"
    crossDownColor: str = "#ff5252"
    buyDotColor: str = "#3fff00"
    sellDotColor: str = "#ff0000"
    goldDotColor: str = "#e2a400"

    lineWidth: float = 1.0
    fillAlpha: float = 0.25
    dotSize: float = 20.0
    bigDotSize: float = 45.0
    markerLineWidth: float = 0.75


@dataclass
class IndicatorSpec:
    kind: PlotIndicator
    config: object | None = None

@dataclass
class PlotConfig:
    candles_to_display: int = 312
    candle_bar_width: float = 0.3
    candle_line_width: float = 0.5
    save_dpi: int = 400
    positiveColor: str = "#089981"
    negativeColor: str = "#f23645"
    
    indicators: list[IndicatorSpec] = field(
        default_factory=lambda: [
            IndicatorSpec(
                kind=PlotIndicator.EMA,
                config=EMAConfig(period=200, col_name="200 EMA")
            )
        ]
    )

    subplotRows: int = 1
    subplotCols: int = 1

    figSize: tuple[int, int] = (10, 6)
    titleFontSize: int = 15

    axisTickMode: AxisTickMode = AxisTickMode.AUTO
    axisMaxTicks: int = 8
    axisDateFormat: str | None = None
    axisDateRotation: int = 0
    axisDateHA: str = "center"