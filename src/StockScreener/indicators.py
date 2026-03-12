import pandas as pd
import numpy as np

def MyEWA(data: np.ndarray, period: int):
    alpha = 2 / (period + 1)
    ema_vals = np.full(data.shape, np.nan)
    ema_vals[period - 1] = np.mean(data[:period])

    for x in range(period, data.shape[0]):
        ema_vals[x] = alpha * data[x] + (1 - alpha) * ema_vals[x - 1]

    return ema_vals

def ComputeExponentialMovingAvg(df, period=200):
    """Given some data, compute an exponential moving average:

    EMA(period) := The average closing value of the `period` previous candles
    EMA(period + t + 1) = (alpha * CloseValue[t + 1]) + ((1 - alpha) * EMA(period + t))

    Where alpha = 2 / (period + 1)

    Args:
        df: The data to compute the EMA of
        period (optional): The period to use to compute alpha. Defaults to 200.

    Returns:
        The EMA of the dataset
    """
    close = df["Close"].dropna()

    ema = pd.Series(np.nan, index=df.index, dtype=float)

    if len(close) < period:
        return ema

    close_vals = close.to_numpy()

    # ema_vals = np.full(close_vals.shape, np.nan)
    # ema_vals[period - 1] = np.mean(close_vals[:period])

    # for x in range(period, close_vals.shape[0]):
    #     ema_vals[x] = alpha * close_vals[x] + (1 - alpha) * ema_vals[x - 1]

    ema.loc[close.index] = MyEWA(close_vals, period)
    return ema

def ComputeFibonacci(df, period=265):
    close = df["Close"].dropna()

    fib = pd.DataFrame(index=df.index, columns=["Bottom", "fib236", "fib382", "fib5", "fib618", "fib764", "Top"])

    if len(close) < period:
        return fib
    
    close_vals = close.to_numpy()
    bottom = np.full(len(fib), np.nan)
    top = np.full(len(fib), np.nan)
    rng = np.full(len(fib), np.nan)

    for x in range(period, close_vals.shape[0]):
        bottom[x] = close_vals[x - period : x].min()
        top[x] = close_vals[x - period : x].max()
        rng[x] = top[x] - bottom[x]

    fib["Bottom"] = bottom
    fib["fib236"] = bottom + 0.236 * rng
    fib["fib382"] = bottom + 0.382 * rng
    fib["fib5"] = top - 0.5 * rng
    fib["fib618"] = top - 0.382 * rng
    fib["fib764"] = top - 0.236 * rng
    fib["Top"] = top

    return fib

def ComputeMACD(df, fast_period, slow_period, signal_period, macdColName, signalColName, histColName):
    close = df["Close"].dropna()
    close_vals = close.to_numpy()
    macd_df = pd.DataFrame(index=df.index, columns=[macdColName, signalColName, histColName], dtype=float)

    fastEMA = MyEWA(close_vals, fast_period)
    slowEMA = MyEWA(close_vals, slow_period)
    macd = fastEMA - slowEMA

    macd_vals = macd[~np.isnan(macd)]

    signal = MyEWA(macd_vals, signal_period)
    hist = macd_vals - signal

    macd_df.loc[close.index, macdColName] = macd
    macd_df.loc[close.index[-signal.shape[0]:], signalColName] = signal
    macd_df.loc[close.index[-signal.shape[0]:], histColName] = hist

    return macd_df

def _series_ema(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be positive")

    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()

    if len(valid) < period:
        return out

    out.loc[valid.index] = MyEWA(valid.to_numpy(dtype=float), period)
    return out


def _series_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _compute_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)
    return rsi


def _compute_stochrsi(src: pd.Series, stoch_len: int, rsi_len: int, smooth_k: int, smooth_d: int, use_log: bool, use_avg: bool):
    src = src.astype(float)
    if use_log:
        src = src.where(src > 0)
        src = np.log(src)

    rsi = _compute_rsi(src, rsi_len)
    lowest = rsi.rolling(stoch_len, min_periods=stoch_len).min()
    highest = rsi.rolling(stoch_len, min_periods=stoch_len).max()
    denom = (highest - lowest).replace(0.0, np.nan)
    raw = 100.0 * (rsi - lowest) / denom
    k = _series_sma(raw, smooth_k)
    d = _series_sma(k, smooth_d)
    if use_avg:
        k = (k + d) / 2.0
    return k, d


def _compute_schaff_tc(src: pd.Series, length: int, fast_length: int, slow_length: int, factor: float) -> pd.Series:
    ema1 = _series_ema(src, fast_length)
    ema2 = _series_ema(src, slow_length)
    macd_val = ema1 - ema2

    alpha = macd_val.rolling(length, min_periods=length).min()
    beta = macd_val.rolling(length, min_periods=length).max() - alpha
    gamma = 100.0 * (macd_val - alpha) / beta.replace(0.0, np.nan)

    delta = pd.Series(np.nan, index=src.index, dtype=float)
    for i in range(len(src)):
        g = gamma.iloc[i]
        if np.isnan(g):
            continue
        if i == 0 or np.isnan(delta.iloc[i - 1]):
            delta.iloc[i] = g
        else:
            delta.iloc[i] = delta.iloc[i - 1] + factor * (g - delta.iloc[i - 1])

    epsilon = delta.rolling(length, min_periods=length).min()
    zeta = delta.rolling(length, min_periods=length).max() - epsilon
    eta = 100.0 * (delta - epsilon) / zeta.replace(0.0, np.nan)

    stc = pd.Series(np.nan, index=src.index, dtype=float)
    for i in range(len(src)):
        e = eta.iloc[i]
        if np.isnan(e):
            continue
        if i == 0 or np.isnan(stc.iloc[i - 1]):
            stc.iloc[i] = e
        else:
            stc.iloc[i] = stc.iloc[i - 1] + factor * (e - stc.iloc[i - 1])

    return stc


def _find_divergences(src: pd.Series, high: pd.Series, low: pd.Series, top_limit: float, bot_limit: float, use_limits: bool) -> pd.DataFrame:
    n = len(src)
    src_vals = src.to_numpy(dtype=float)
    high_vals = high.to_numpy(dtype=float)
    low_vals = low.to_numpy(dtype=float)

    fractal_top = np.zeros(n, dtype=bool)
    fractal_bot = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)
    bull_div = np.zeros(n, dtype=bool)
    bear_hidden = np.zeros(n, dtype=bool)
    bull_hidden = np.zeros(n, dtype=bool)

    prev_top_idx = None
    prev_bot_idx = None

    for i in range(4, n):
        window = src_vals[i - 4 : i + 1]
        if np.isnan(window).any():
            continue

        center = i - 2

        is_top = window[0] < window[2] and window[1] < window[2] and window[2] > window[3] and window[2] > window[4]
        is_bot = window[0] > window[2] and window[1] > window[2] and window[2] < window[3] and window[2] < window[4]

        if is_top and (not use_limits or src_vals[center] >= top_limit):
            fractal_top[center] = True
            if prev_top_idx is not None:
                bear_div[center] = high_vals[center] > high_vals[prev_top_idx] and src_vals[center] < src_vals[prev_top_idx]
                bear_hidden[center] = high_vals[center] < high_vals[prev_top_idx] and src_vals[center] > src_vals[prev_top_idx]
            prev_top_idx = center

        if is_bot and (not use_limits or src_vals[center] <= bot_limit):
            fractal_bot[center] = True
            if prev_bot_idx is not None:
                bull_div[center] = low_vals[center] < low_vals[prev_bot_idx] and src_vals[center] > src_vals[prev_bot_idx]
                bull_hidden[center] = low_vals[center] > low_vals[prev_bot_idx] and src_vals[center] < src_vals[prev_bot_idx]
            prev_bot_idx = center

    return pd.DataFrame(
        {
            "fractal_top": fractal_top,
            "fractal_bot": fractal_bot,
            "bear_div": bear_div,
            "bull_div": bull_div,
            "bear_hidden_div": bear_hidden,
            "bull_hidden_div": bull_hidden,
        },
        index=src.index,
    )


def ComputeVuManChu(df: pd.DataFrame, config) -> pd.DataFrame:
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    prefix = config.prefix
    out = pd.DataFrame(index=df.index, dtype=float)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    hlc3 = (high + low + close) / 3.0

    rsi = _compute_rsi(close, config.rsiLen)
    out[f"{prefix} RSI"] = rsi

    rsi_mfi_base = ((close - open_) / (high - low).replace(0.0, np.nan)) * config.rsiMFIMultiplier
    rsi_mfi = _series_sma(rsi_mfi_base, config.rsiMFIperiod) - config.rsiMFIPosY
    out[f"{prefix} RSI_MFI"] = rsi_mfi

    esa = _series_ema(hlc3, config.wtChannelLen)
    de = _series_ema((hlc3 - esa).abs(), config.wtChannelLen)
    ci = (hlc3 - esa) / (0.015 * de.replace(0.0, np.nan))
    wt1 = _series_ema(ci, config.wtAverageLen)
    wt2 = _series_sma(wt1, config.wtMALen)
    wt_vwap = wt1 - wt2

    out[f"{prefix} WT1"] = wt1
    out[f"{prefix} WT2"] = wt2
    out[f"{prefix} WT_VWAP"] = wt_vwap
    out[f"{prefix} WT_OVERSOLD"] = (wt2 <= config.osLevel).astype(float)
    out[f"{prefix} WT_OVERBOUGHT"] = (wt2 >= config.obLevel).astype(float)

    wt_cross_up = (wt1 >= wt2) & (wt1.shift(1) < wt2.shift(1))
    wt_cross_down = (wt1 <= wt2) & (wt1.shift(1) > wt2.shift(1))
    wt_cross = wt_cross_up | wt_cross_down

    out[f"{prefix} WT_CROSS"] = wt_cross.astype(float)
    out[f"{prefix} WT_CROSS_UP"] = wt_cross_up.astype(float)
    out[f"{prefix} WT_CROSS_DOWN"] = wt_cross_down.astype(float)

    stoch_k, stoch_d = _compute_stochrsi(close, config.stochLen, config.stochRsiLen, config.stochKSmooth, config.stochDSmooth, config.stochUseLog, config.stochAvg)
    out[f"{prefix} STOCH_K"] = stoch_k
    out[f"{prefix} STOCH_D"] = stoch_d

    stc = _compute_schaff_tc(close, config.tcLength, config.tcFastLength, config.tcSlowLength, config.tcFactor)
    out[f"{prefix} STC"] = stc

    wt_div = _find_divergences(wt2, high, low, config.wtDivOBLevel, config.wtDivOSLevel, True)
    wt_div_add = _find_divergences(wt2, high, low, config.wtDivOBLevelAdd, config.wtDivOSLevelAdd, True)
    wt_div_nl = _find_divergences(wt2, high, low, 0.0, 0.0, False)
    rsi_div = _find_divergences(rsi, high, low, config.rsiDivOBLevel, config.rsiDivOSLevel, True)
    rsi_div_nl = _find_divergences(rsi, high, low, 0.0, 0.0, False)
    stoch_div = _find_divergences(stoch_k, high, low, 0.0, 0.0, False)

    for name, div_df in {
        "WT": wt_div,
        "WT_ADD": wt_div_add,
        "WT_NL": wt_div_nl,
        "RSI": rsi_div,
        "RSI_NL": rsi_div_nl,
        "STOCH": stoch_div,
    }.items():
        for col in div_df.columns:
            out[f"{prefix} {name}_{col.upper()}"] = div_df[col].astype(float)

    wt_bull_div = wt_div["bull_div"]
    wt_bear_div = wt_div["bear_div"]
    wt_bull_div_add = wt_div_add["bull_div"]
    wt_bear_div_add = wt_div_add["bear_div"]
    rsi_bull_div = rsi_div["bull_div"]
    rsi_bear_div = rsi_div["bear_div"]
    stoch_bull_div = stoch_div["bull_div"]
    stoch_bear_div = stoch_div["bear_div"]

    buy_signal = wt_cross_up & (wt2 <= config.osLevel)
    sell_signal = wt_cross_down & (wt2 >= config.obLevel)

    buy_signal_div = wt_bull_div | wt_bull_div_add | stoch_bull_div | rsi_bull_div
    sell_signal_div = wt_bear_div | wt_bear_div_add | stoch_bear_div | rsi_bear_div

    wt_fractal_bot = wt_div["fractal_bot"]
    last_rsi = rsi.where(wt_fractal_bot).ffill().shift(1)
    wt_low_prev = wt2.where(wt_fractal_bot).ffill().shift(1)

    wt_gold_buy = (wt_bull_div | rsi_bull_div) & (wt_low_prev <= config.osLevel3) & (wt2 > config.osLevel3) & ((wt_low_prev - wt2) <= -5.0) & (last_rsi < 30)

    out[f"{prefix} BUY_SIGNAL"] = buy_signal.astype(float)
    out[f"{prefix} SELL_SIGNAL"] = sell_signal.astype(float)
    out[f"{prefix} BUY_SIGNAL_DIV"] = buy_signal_div.astype(float)
    out[f"{prefix} SELL_SIGNAL_DIV"] = sell_signal_div.astype(float)
    out[f"{prefix} GOLD_BUY"] = wt_gold_buy.astype(float)

    return out
