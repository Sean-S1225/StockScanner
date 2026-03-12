from pandas import MultiIndex
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.dates as mdates
from .indicators import *
import yfinance as yf
import pandas as pd
from matplotlib import figure
from matplotlib.axes import Axes
from .config import *
import numpy as np
import pandas as pd
from collections.abc import Callable
from math import ceil

@dataclass(frozen=True)
class IndicatorDefinition:
    plotter: Callable
    panel: IndicatorPanel
    heigh_ratio: float = 1.0

def GetTickerAndSymbol(name, ticker):
    if ticker != None:
        return ticker, ticker.ticker
    
    ticker = yf.Ticker(name)
    return ticker, ticker.ticker

def GetTitle(figTitle: str | None, ticker: yf.Ticker, info: dict[str, str] | None, symbol: str):
    """Generates the title of the plot. The title is chosen to be f"{displayName} (${symbol}) Stock History",
    but if display name is not available it will fall back to shortName, then longName. If longName is not
    available, it will fall back on just f"${symbol} Stock History"; symbol should always exist.

    Args:
        name: The name of the stock (ticker)
        ticker: The ticker object
        info: The info generated via ticker.get_info()

    Raises:
        ValueError: If ticker.ticker returns None

    Returns:
        Returns the title of the plot and the ticker
    """

    if figTitle != None:
        return figTitle

    # Getting ticker.get_info() is a heavy API call, so if the user
    # provides it, we don't want to get it again. If they don't, then
    # we get the info from scratch.
    if info == None:        
        info = ticker.get_info()

    # The goal is for title to be f"{displayName} (${symbol}) Stock History",
    # but if display name is not available it will fall back to shortName, then longName. If longName is not
    # available, it will fall back on just f"${symbol} Stock History"; symbol should always exist.
    displayName = None
    displayName = info.get("displayName")
    if displayName is None:
        displayName = info.get("shortName")
    if displayName is None:
        displayName = info.get("longName")

    symbol = ticker.ticker

    title = ""
    if displayName != None:
        title += displayName
        if symbol != None:
            title += f" (${symbol})"
    else:
        title = f"${symbol}"

    if symbol is None:
        raise ValueError("Symbol cannot be done. Deal with this edge case.")
    
    title = title + " Stock History"

    return title

def _get_auto_tick_mode(data: pd.DataFrame) -> AxisTickMode:
    if len(data) < 2:
        return AxisTickMode.YEAR

    span_days = (data.index[-1] - data.index[0]).days

    if span_days >= 365 * 4:
        return AxisTickMode.YEAR
    elif span_days >= 365 * 2:
        return AxisTickMode.QUARTER
    elif span_days >= 180:
        return AxisTickMode.MONTH
    elif span_days >= 60:
        return AxisTickMode.WEEK
    else:
        return AxisTickMode.DAY


def _get_tick_positions_and_labels(
    data: pd.DataFrame,
    tick_mode: AxisTickMode,
    date_format: str | None,
    max_ticks: int
) -> tuple[np.ndarray, list[str]]:
    index = data.index

    if len(index) == 0:
        return np.array([], dtype=int), []

    if tick_mode == AxisTickMode.AUTO:
        tick_mode = _get_auto_tick_mode(data)

    if tick_mode == AxisTickMode.YEAR:
        periods = index.to_period("Y")
        default_fmt = "%Y"

    elif tick_mode == AxisTickMode.QUARTER:
        periods = index.to_period("Q")
        default_fmt = None  # handled specially below

    elif tick_mode == AxisTickMode.MONTH:
        periods = index.to_period("M")
        default_fmt = "%b %Y"

    elif tick_mode == AxisTickMode.WEEK:
        periods = index.to_period("W")
        default_fmt = "%Y-%m-%d"

    elif tick_mode == AxisTickMode.DAY:
        periods = index.to_period("D")
        default_fmt = "%Y-%m-%d"

    else:
        raise ValueError(f"Unsupported tick mode: {tick_mode}")

    tick_idx = np.flatnonzero(periods != np.roll(periods, 1))
    tick_idx[0] = 0

    if len(tick_idx) > max_ticks:
        step = ceil(len(tick_idx) / max_ticks)
        tick_idx = tick_idx[::step]

    tick_dates = index[tick_idx]

    if tick_mode == AxisTickMode.QUARTER:
        if date_format is None:
            tick_labels = [f"{d.year}-Q{((d.month - 1) // 3) + 1}" for d in tick_dates]
        else:
            tick_labels = tick_dates.strftime(date_format).tolist()
    else:
        fmt = date_format if date_format is not None else default_fmt
        tick_labels = tick_dates.strftime(fmt).tolist()

    return tick_idx, tick_labels


def _format_time_ticks(ax: Axes, data: pd.DataFrame, plotConfig: PlotConfig):
    """Format compressed trading-day x-axis ticks based on semantic date boundaries."""

    tick_idx, tick_labels = _get_tick_positions_and_labels(
        data=data,
        tick_mode=plotConfig.axisTickMode,
        date_format=plotConfig.axisDateFormat,
        max_ticks=plotConfig.axisMaxTicks
    )

    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        tick_labels,
        rotation=plotConfig.axisDateRotation,
        ha=plotConfig.axisDateHA
    )

def PlotEMA(stock_history: pd.DataFrame, current_window: pd.DataFrame, ax: Axes, config: EMAConfig):
    """Plots an Exponential Moving Average onto the data

    Args:
        stock_history: The total stock history
        current_window: The data to display
        ax: The axis to plot the EMA onto
        config: A config dataclass to handle the EMA properties
    """

    if len(stock_history) < config.period: return

    if config.col_name in stock_history.columns:
        ema = stock_history[config.col_name]
    else:
        ema = ComputeExponentialMovingAvg(stock_history, period=config.period)

    x = np.arange(len(current_window))

    ax.plot(
        x,
        ema.loc[current_window.index],
        color=config.color,
        linewidth=config.linewidth
    )

def PlotFib(stock_history: pd.DataFrame, current_window: pd.DataFrame, ax: Axes, config: FibConfig):
    if len(stock_history) < config.period: return

    if config.colName in stock_history.columns:
        fib = stock_history[config.colName]
    else:
        fib = ComputeFibonacci(stock_history, period=config.period)

    x = np.arange(len(current_window))

    for col, color in zip(
        ["Bottom", "fib236", "fib382", "fib5", "fib618", "fib764", "Top"],
        [config.bottomLineColor, config.fib236LineColor, config.fib382LineColor, config.fib50LineColor, config.fib618LineColor, config.fib764LineColor, config.topLineColor]
    ):
        ax.plot(
            x,
            fib[col].loc[current_window.index],
            color=color,
            linewidth=config.linewidth, zorder=-1
        )

    for bottomCol, topCol, color in zip(
        ["Bottom", "fib236", "fib382", "fib5", "fib618", "fib764"],
        ["fib236", "fib382", "fib5", "fib618", "fib764", "Top"],
        [config.bottom_fib236_fillColor, config.fib236_fib382_fillColor, config.fib382_fib50_fillColor, config.fib50_fib618_fillColor, config.fib618_fib764_fillColor, config.fib764_top_fillColor]
    ):
        ax.fill_between(
            x,
            fib[bottomCol].loc[current_window.index],
            fib[topCol].loc[current_window.index],
            color = color,
            alpha = config.fillAlpha, zorder=-1
        )

def PlotMACD(stock_history: pd.DataFrame, current_window: pd.DataFrame, ax: Axes, config: MACDConfig):
    if len(stock_history) < config.slowPeriod:
        ax.axhline(0, linewidth=config.linewidth, color=config.zeroLineColor, zorder=0)
        return

    cols = [config.macdColName, config.signalColName, config.histColName]

    if all(col in stock_history.columns for col in cols):
        macd = stock_history[cols].copy()
    else:
        macd = ComputeMACD(stock_history, config.fastPeriod, config.slowPeriod, config.signalPeriod, config.macdColName, config.signalColName, config.histColName)

    hist = macd[config.histColName]
    color_vals = []

    if pd.isna(hist.iloc[0]):
        color_vals.append("#FFFFFF00")
    elif hist.iloc[0] > 0:
        color_vals.append(config.histPositiveIncreasingColor)
    else:
        color_vals.append(config.histNegativeIncreasingColor)

    for x in range(1, len(hist)):
        if hist.iloc[x] > 0:
            if hist.iloc[x] > hist.iloc[x - 1]:
                color_vals.append(config.histPositiveIncreasingColor)
            else:
                color_vals.append(config.histPositiveDecreasingColor)
        else:
            if hist.iloc[x] < hist.iloc[x - 1]:
                color_vals.append(config.histNegativeIncreasingColor)
            else:
                color_vals.append(config.histNegativeDecreasingColor)

    current = macd.loc[current_window.index]

    x = np.arange(len(current))

    ax.plot(x, current[config.macdColName], linewidth=config.linewidth, color=config.macdLineColor)
    ax.plot(x, current[config.signalColName], linewidth=config.linewidth, color=config.signalLineColor)
    ax.axhline(0, linewidth=config.linewidth, color=config.zeroLineColor, zorder=0)
    ax.bar(x, current[config.histColName], width=config.histBarWidth, color=color_vals[-len(current):])

def PlotVuManChu(stock_history: pd.DataFrame, current_window: pd.DataFrame, ax: Axes, config: VMCConfig):
    vmc = ComputeVuManChu(stock_history, config)
    current = vmc.loc[current_window.index]
    x = np.arange(len(current))
    prefix = config.prefix

    wt1 = current[f"{prefix} WT1"]
    wt2 = current[f"{prefix} WT2"]
    wt_vwap = current[f"{prefix} WT_VWAP"]
    rsi_mfi = current[f"{prefix} RSI_MFI"]
    rsi = current[f"{prefix} RSI"]
    stoch_k = current[f"{prefix} STOCH_K"]
    stoch_d = current[f"{prefix} STOCH_D"]
    stc = current[f"{prefix} STC"]

    ax.grid(visible=True, axis="y", zorder=-2, alpha=0.35)
    ax.axhline(0.0, linewidth=config.lineWidth, color=config.zeroLineColor, zorder=-1)
    ax.axhline(config.obLevel2, linewidth=config.lineWidth, color=config.zeroLineColor, alpha=0.35, zorder=-1)
    ax.axhline(config.obLevel3, linewidth=config.lineWidth, color=config.zeroLineColor, alpha=0.2, zorder=-1)
    ax.axhline(config.osLevel2, linewidth=config.lineWidth, color=config.zeroLineColor, alpha=0.35, zorder=-1)

    if config.showWaveTrend:
        ax.fill_between(x, 0, wt1.to_numpy(dtype=float), color=config.wt1Color, alpha=config.fillAlpha, linewidth=0)
        ax.fill_between(x, 0, wt2.to_numpy(dtype=float), color=config.wt2Color, alpha=config.fillAlpha, linewidth=0)
        ax.plot(x, wt1, color=config.wt1Color, linewidth=config.lineWidth)
        ax.plot(x, wt2, color=config.wt2Color, linewidth=config.lineWidth)

    if config.showFastWT:
        ax.fill_between(x, 0, wt_vwap.to_numpy(dtype=float), color=config.wtVwapColor, alpha=config.fillAlpha * 0.8, linewidth=0)
        ax.plot(x, wt_vwap, color=config.wtVwapColor, linewidth=config.lineWidth * 0.85, alpha=0.8)

    if config.showMFI:
        mfi_vals = rsi_mfi.to_numpy(dtype=float)
        pos_mask = np.where(mfi_vals > 0, mfi_vals, np.nan)
        neg_mask = np.where(mfi_vals <= 0, mfi_vals, np.nan)
        ax.fill_between(x, 0, pos_mask, color=config.rsiMFIAboveColor, alpha=config.fillAlpha)
        ax.fill_between(x, 0, neg_mask, color=config.rsiMFIBelowColor, alpha=config.fillAlpha)

    if config.showRSI:
        rsi_color_vals = [
            config.rsiOversoldColor if pd.notna(v) and v <= config.rsiOversold else
            config.rsiOverboughtColor if pd.notna(v) and v >= config.rsiOverbought else
            config.rsiNeutralColor
            for v in rsi
        ]
        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i - 1]) or pd.isna(rsi.iloc[i]):
                continue
            ax.plot(x[i-1:i+1], rsi.iloc[i-1:i+1], color=rsi_color_vals[i], linewidth=config.lineWidth * 0.9, alpha=0.8)

    if config.showStoch:
        ax.plot(x, stoch_k, color=config.stochKColor, linewidth=config.lineWidth)
        ax.plot(x, stoch_d, color=config.stochDColor, linewidth=config.lineWidth * 0.8)
        ax.fill_between(
            x,
            stoch_k.to_numpy(dtype=float),
            stoch_d.to_numpy(dtype=float),
            where=(stoch_k >= stoch_d).fillna(False).to_numpy(),
            interpolate=True,
            color=config.stochKColor,
            alpha=config.fillAlpha * 0.4,
        )
        ax.fill_between(
            x,
            stoch_k.to_numpy(dtype=float),
            stoch_d.to_numpy(dtype=float),
            where=(stoch_k < stoch_d).fillna(False).to_numpy(),
            interpolate=True,
            color=config.stochDColor,
            alpha=config.fillAlpha * 0.35,
        )

    if config.showSTC:
        ax.plot(x, stc, color=config.stcColor, linewidth=config.lineWidth, alpha=0.8)

    def _scatter_mask(mask_col: str, y_series: pd.Series, color: str, marker: str = "o", size: float | None = None, alpha: float = 0.9, zorder: int = 5):
        mask = current[mask_col].fillna(0).astype(bool)
        if not mask.any():
            return
        pts_x = x[mask.to_numpy()]
        pts_y = y_series[mask].to_numpy(dtype=float)
        ax.scatter(pts_x, pts_y, c=color, marker=marker, s=size or config.dotSize, alpha=alpha, linewidths=config.markerLineWidth, zorder=zorder)

    if config.showWTDivs:
        _scatter_mask(f"{prefix} WT_FRACTAL_TOP", wt2, config.wtBearDivColor, marker="v", size=config.dotSize)
        _scatter_mask(f"{prefix} WT_FRACTAL_BOT", wt2, config.wtBullDivColor, marker="^", size=config.dotSize)
        if config.showSecondWTDivs:
            _scatter_mask(f"{prefix} WT_ADD_FRACTAL_TOP", wt2, config.wtBearDivColor, marker="v", size=config.dotSize * 0.75, alpha=0.5)
            _scatter_mask(f"{prefix} WT_ADD_FRACTAL_BOT", wt2, config.wtBullDivColor, marker="^", size=config.dotSize * 0.75, alpha=0.5)

    if config.showRSIDivs:
        _scatter_mask(f"{prefix} RSI_FRACTAL_TOP", rsi, config.rsiBearDivColor, marker="v", size=config.dotSize * 0.7, alpha=0.8)
        _scatter_mask(f"{prefix} RSI_FRACTAL_BOT", rsi, config.rsiBullDivColor, marker="^", size=config.dotSize * 0.7, alpha=0.8)

    if config.showStochDivs:
        _scatter_mask(f"{prefix} STOCH_FRACTAL_TOP", stoch_k, config.stochBearDivColor, marker="v", size=config.dotSize * 0.7, alpha=0.8)
        _scatter_mask(f"{prefix} STOCH_FRACTAL_BOT", stoch_k, config.stochBullDivColor, marker="^", size=config.dotSize * 0.7, alpha=0.8)

    wt_cross_up = current[f"{prefix} WT_CROSS_UP"].fillna(0).astype(bool)
    wt_cross_down = current[f"{prefix} WT_CROSS_DOWN"].fillna(0).astype(bool)
    if wt_cross_up.any():
        ax.scatter(x[wt_cross_up.to_numpy()], wt2[wt_cross_up], c=config.crossUpColor, marker="o", s=config.dotSize, alpha=0.7, zorder=6)
    if wt_cross_down.any():
        ax.scatter(x[wt_cross_down.to_numpy()], wt2[wt_cross_down], c=config.crossDownColor, marker="o", s=config.dotSize, alpha=0.7, zorder=6)

    if config.showBuyDots:
        _scatter_mask(f"{prefix} BUY_SIGNAL", pd.Series(-107.0, index=current.index), config.buyDotColor, marker="o", size=config.bigDotSize)
        _scatter_mask(f"{prefix} BUY_SIGNAL_DIV", pd.Series(-106.0, index=current.index), config.buyDotColor, marker="o", size=config.bigDotSize * 0.9, alpha=0.8)
    if config.showSellDots:
        _scatter_mask(f"{prefix} SELL_SIGNAL", pd.Series(105.0, index=current.index), config.sellDotColor, marker="o", size=config.bigDotSize)
        _scatter_mask(f"{prefix} SELL_SIGNAL_DIV", pd.Series(106.0, index=current.index), config.sellDotColor, marker="o", size=config.bigDotSize * 0.9, alpha=0.8)
    if config.showGoldDots:
        _scatter_mask(f"{prefix} GOLD_BUY", pd.Series(-106.0, index=current.index), config.goldDotColor, marker="o", size=config.bigDotSize * 1.1, alpha=0.9)

    ax.axhline(config.osLevel2, linewidth=0.75, color = "black", ls=":")
    ax.axhline(config.obLevel2, linewidth=0.75, color = "black", ls=":")

    ax.set_ylim(-130, 130)

def _get_required_panels(plotConfig: PlotConfig, INDICATOR_DEFS: dict[PlotIndicator, IndicatorDefinition]):
    panels = [IndicatorPanel.PRICE]

    if plotConfig.indicators is not None:
        for spec in plotConfig.indicators:
            panel = INDICATOR_DEFS[spec.kind].panel
            if panel not in panels:
                panels.append(panel)

    return panels

def _get_figure_and_axis(figAx: tuple[figure.Figure, Axes] | None, plotConfig: PlotConfig, panels: list[IndicatorPanel]):
    """Returns a figure and axis for the data to be plotted on. If the user has provided a
    value for figAx, validate it, otherwise, define a new figure and axis.

    Args:
        figAx: Possible a user supplied (fig, ax) tuple, None otherwise
        plotConfig: The config dataclass for plotting

    Raises:
        ValueError: If the user passes in a value for figAx that is not a (fig, ax) tuple

    Returns:
        The figure and axis to plot data onto
    """

    if figAx is not None:
        fig, axes = figAx
        if isinstance(axes, Axes):
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = list(axes.flatten())
        else:
            axes = list(axes)

        if len(axes) != len(panels):
            raise ValueError(f"Expected {len(panels)} axes, got {len(axes)}.")
    else:
        height_ratios = [3.0 if p == IndicatorPanel.PRICE else 1.0 for p in panels]

        fig, axes = plt.subplots(
            nrows=len(panels),
            ncols=1,
            sharex=True,
            figsize=plotConfig.figSize,
            dpi=plotConfig.save_dpi,
            layout="constrained",
            gridspec_kw={"height_ratios": height_ratios}
        )

        if not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = list(axes.flatten())

    axes_by_panel = dict(zip(panels, axes))
    return fig, axes_by_panel

def _resolve_stock_history(stock_history: pd.DataFrame | None, stockHistoryConfig: StockHistoryConfig | None, ticker: yf.Ticker | None):
    """Returns the history of a stock. If the user has provided a stock history, just return it.
    Otherwise, download the stock history and return it.

    Args:
        stock_history: Possible a user-provided stock history, otherwise None
        stockHistoryConfig: The config dataclass for getting stock history
        ticker: The ticker of the stock to download

    Returns:
        The stock history of the stock
    """

    if stock_history is None:
        if stockHistoryConfig is None:
            stockHistoryConfig = StockHistoryConfig()
        stock_history = ticker.history(period=stockHistoryConfig.period, interval=stockHistoryConfig.interval)

    return stock_history

def _normalize_stock_history(stock_history: pd.DataFrame, symbol: str):
    """The stock history may be a multi-index dataframe. Return a single-index dataframe
    containing just the stock specified by symbol.

    Args:
        stock_history: The history of the stock, which may be a single- or multi-index dataframe
        symbol: The symbol to isolate

    Raises:
        ValueError: If the symbol does not appear in the dataframe

    Returns:
        A single-index dataframe containing only the stock of interest
    """

    if isinstance(stock_history.columns, MultiIndex):
        if symbol not in stock_history.columns.get_level_values(0):
            raise ValueError(f"{symbol} not found in stock_history MultiIndex columns.")
        stock_history = stock_history[symbol].copy()
    else:
        stock_history = stock_history.copy()

    return stock_history

def _draw_candles(ax: Axes, data: pd.DataFrame, plotConfig: PlotConfig):
    """Given an axis, draw stock data as candles on it

    Args:
        ax: The axis to draw the candles onto
        data: The stock data, given as a dataframe containing columns 'Open', 'Close', 'High', and 'Low'
        plotConfig: The config dataclass for plotting
    """

    x_vals = np.arange(len(data), dtype=float)

    # Invisible line so matplotlib picks up the axis scaling cleanly
    ax.plot(x_vals, data["Close"], alpha=0)

    rects_pos, rects_neg = [], []
    lines_pos, lines_neg = [], []

    for x, (_, row) in zip(x_vals, data.iterrows()):
        y = min(row["Open"], row["Close"])
        height = abs(row["Close"] - row["Open"])
        bar_width = plotConfig.candle_bar_width

        if row["Close"] >= row["Open"]:
            rects_pos.append(Rectangle((x - bar_width/2, y), bar_width, height))
            lines_pos.append([(x, row["Low"]), (x, row["High"])])
        else:
            rects_neg.append(Rectangle((x - bar_width/2, y), bar_width, height))
            lines_neg.append([(x, row["Low"]), (x, row["High"])])

    ax.add_collection(LineCollection(lines_pos, colors=plotConfig.positiveColor, linewidth=plotConfig.candle_line_width))
    ax.add_collection(LineCollection(lines_neg, colors=plotConfig.negativeColor, linewidth=plotConfig.candle_line_width))
    ax.add_collection(PatchCollection(rects_pos, facecolor=plotConfig.positiveColor, edgecolor=plotConfig.positiveColor))
    ax.add_collection(PatchCollection(rects_neg, facecolor=plotConfig.negativeColor, edgecolor=plotConfig.negativeColor))

def _draw_indicators(stock_history: pd.DataFrame, data: pd.DataFrame, axes_by_panel: dict[IndicatorPanel, Axes], plotConfig: PlotConfig | None, INDICATOR_DEFS: dict[PlotIndicator, IndicatorDefinition]):
    """Draw the specified indicators on the plot

    Args:
        stock_history: The entire stock history, used to compute indicators as needed
        data: The window of time being plotted to the screen
        ax: The axis on which to draw the indicator
        plotConfig: The config dataclass for plotting
        INDICATOR_PLOTTERS: A dictionary containing indicator-function pairs to handle calculating and drawing the indicator
    """

    if plotConfig.indicators is None:
        return
    
    for spec in plotConfig.indicators:
        indicator_def = INDICATOR_DEFS[spec.kind]
        ax = axes_by_panel[indicator_def.panel]
        indicator_def.plotter(stock_history, data, ax, spec.config)

def _PlotStockHistoryFromData(title: str, INDICATOR_DEFS: dict[PlotIndicator, IndicatorDefinition], stock_history: pd.DataFrame,
                     savefile: str | None, figAx: tuple[figure.Figure, Axes],
                     plotConfig: PlotConfig | None, useDefaultFidTitleFontSize: bool):
    """Assuming that the data has already been generated, now actually plot it

    Args:
        title: The title of the plot
        INDICATOR_PLOTTERS: A dictionary containing indicator-function pairs to handle calculating and drawing the indicator 
        stock_history: The entire stock history
        savefile: If not None, save to the specified file
        figAx: Possibly a user-provided (figure, axis) tuple
        plotConfig: The config dataclass for plotting

    Returns:
        The (figure, axis) on which the data was plotted
    """
    
    if plotConfig is None: plotConfig = PlotConfig()
    
    requiredPanels = _get_required_panels(plotConfig, INDICATOR_DEFS)
    fig, axes_by_panel = _get_figure_and_axis(figAx, plotConfig, requiredPanels)
    
    price_ax = axes_by_panel[IndicatorPanel.PRICE]
    price_ax.grid(visible=True, axis="y", zorder=-1, alpha=0.5)

    if useDefaultFidTitleFontSize:
        fig.suptitle(title)
    else:
        fig.suptitle(title, fontsize=plotConfig.titleFontSize)

    data = stock_history[-plotConfig.candles_to_display:]

    _draw_candles(price_ax, data, plotConfig)
    _draw_indicators(stock_history, data, axes_by_panel, plotConfig, INDICATOR_DEFS)
    
    axes = list(axes_by_panel.values())
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    _format_time_ticks(axes[-1], data, plotConfig)

    if savefile is not None:
        plt.savefig(savefile, dpi=plotConfig.save_dpi, bbox_inches="tight")

    return fig, axes_by_panel

def PlotStockHistory(name: str, stock_history: pd.DataFrame = None, info: dict[str, str] = None,
                     ticker: yf.Ticker = None, savefile=None, figAx: tuple[figure.Figure, Axes] = None,
                     plotConfig: PlotConfig | None = None, stockHistoryConfig: StockHistoryConfig | None = None,
                     figTitle = None):
    """Plots the weekly history of a stock.

    Args:
        name: The stock ticker to plot
        stock_history (optional): If a history is provided, use this data to plot. Otherwise, fetch data using yfinance. Defaults to None.
        info (optional): If a info is provided, use this. Otherwise, fetch info using yfinance. Defaults to None.
        ticker (optional): If info is not provided, fall back on a ticker, use this data to plot. Otherwise, use name to fetch ticker using yfinance. Defaults to None.
        weeklyCandlesToDisplay (optional): The number of weekly candles to plot. Defaults to 364.
        savefile (optional): If provided, save the figure to a file. Defaults to None.
        figAx (optional): If a fig, ax tuple, add the stock data. Otherwise, generate a new figure, axis pair. Defaults to None.
        indicators (optional): The indicators to include on the plot. Includes ["200 EMA"]. Defaults to None.

    Raises:
        ValueError: If symbol not in the stock_history
    """

    INDICATOR_DEFS = {
        PlotIndicator.EMA : IndicatorDefinition(
            plotter=PlotEMA,
            panel=IndicatorPanel.PRICE,
            heigh_ratio=1.0
        ),
        PlotIndicator.FIB : IndicatorDefinition(
            plotter=PlotFib,
            panel=IndicatorPanel.PRICE,
            heigh_ratio=1.0
        ),
        PlotIndicator.MACD : IndicatorDefinition(
            plotter=PlotMACD,
            panel=IndicatorPanel.MACD,
            heigh_ratio=1.0
        ),
        PlotIndicator.VMC : IndicatorDefinition(
            plotter=PlotVuManChu,
            panel=IndicatorPanel.VMC,
            heigh_ratio=1.0
        ),
    }

    ticker, symbol = GetTickerAndSymbol(name, ticker)
    title = GetTitle(figTitle, ticker, info, symbol)
    stock_history = _resolve_stock_history(stock_history, stockHistoryConfig, ticker)
    stock_history = _normalize_stock_history(stock_history, symbol)
    return _PlotStockHistoryFromData(title, INDICATOR_DEFS, stock_history, savefile, figAx, plotConfig, figAx != None)
