import pandas as pd
import numpy as np
import yfinance as yf
from .indicators import ComputeExponentialMovingAvg
from .data import call_with_backoff

def EnsureMinimumNumWeeks(close, minNumWeeks = 200):
    #filter out companies that don't have at least minimum weeks worth of data
    if len(close) < minNumWeeks:
        return False, len(close)

    return True, len(close)

def EnsureMinimumMarketCap(marketCap, info, minimumMarketCap = 2_000_000_000):
    if marketCap is not None and not pd.isna(marketCap) and marketCap > 0:
        return marketCap >= minimumMarketCap, marketCap

    if info.get("currentPrice") is None or info.get("sharesOutstanding") is None:
        return False, None

    currentPrice = info.get("currentPrice")
    sharesOutstanding = info.get("sharesOutstanding")
    marketCap = currentPrice * sharesOutstanding
    
    #filter out companies whose market cap is less than minimum
    if marketCap < minimumMarketCap:
        return False, marketCap

    return True, marketCap

def EnsureCurrentPriceGreaterThan5YearLow(df, current_close, lookback_weeks=52*5, minDistancePercent=0):
    prior_low = df["Low"].iloc[-lookback_weeks-1:-2].min()

    if pd.isna(prior_low) or prior_low == 0:
        return False, None
    
    distance = (current_close - prior_low) / prior_low * 100
    return distance > minDistancePercent, prior_low

def EnsurePositiveTTM_EPS(earnings):
    ttm_eps = earnings["Reported EPS"].head(4).sum()
    isProfitable = ttm_eps > 0

    return isProfitable, ttm_eps

def EnsureEarningsGrowing(earnings):
    eps = earnings["Reported EPS"].to_numpy()
    if len(eps) < 2:
        return False, None

    m, _ = np.polyfit(range(len(eps)), eps[::-1], 1)
    return m > 0, m

def EnsureEarningsMeetingThreshold(earnings, threshold = 5/8):
    surprise = earnings["Surprise(%)"].dropna()
    if surprise.shape[0] == 0:
        return False, None, None

    num_beats = int((surprise > 0).sum())
    return num_beats / surprise.shape[0] >= threshold, num_beats, surprise.shape[0]

def EnsureQuarterlyRevenueIncreasing(growth, q):
    if growth is None:
        if q is None:
            return False, None
        
        if "Total Revenue" not in q.index or q.shape[1] < 5:
            return False, None
        revenue = q.loc["Total Revenue"]
        if revenue.iloc[4] == 0:
            return False, None
        growth = (revenue.iloc[0] - revenue.iloc[4]) / revenue.iloc[4]

    if growth < 0:
        return False, growth

    return True, growth

def EnsurePriceNearOrBelowLongTermAverage(df, currentClose, threshold=10):
    current_ema = df["200 EMA"].iloc[-2]

    if pd.isna(current_ema) or pd.isna(currentClose) or current_ema == 0:
        # print("here")
        return False, None

    percent_off_mean = (currentClose - current_ema) / current_ema * 100

    if percent_off_mean >= threshold:
        return False, percent_off_mean
    
    return True, percent_off_mean

def Ensure1YearPullback(df, currentClose, threshold = -40, lookback_weeks=52*1):
    maxPrice = df["High"][-lookback_weeks:].max()

    pullbackPercent = (currentClose - maxPrice) / maxPrice * 100

    return pullbackPercent <= threshold, pullbackPercent

def EnsureTradingVolume(hist_daily):
    if len(hist_daily) < 40:
        return False, None, None

    adtv = (hist_daily["Close"] * hist_daily["Volume"]).tail(63).mean()  # ~3 months
    if adtv < 10_000_000:
        return False, None, None

    avg_shares = hist_daily["Volume"].tail(63).mean()
    if avg_shares < 300_000:
        return False, None, None
        
    return True, adtv, avg_shares

def FilterNames(names):
    stocks_trading_data = yf.download(
        names,
        period="max",
        interval="1wk",
        group_by="ticker",
        threads=True,
        auto_adjust=False
    )

    stocks_tickers = yf.Tickers(names)

    results = []
    ema_cols = {}

    for name in names:
        # try:
        if name not in stocks_trading_data.columns.get_level_values(0):
            continue

        df = stocks_trading_data[name].dropna(subset=["Close"]).copy()
        if df.empty:
            continue

        ticker = stocks_tickers.tickers[name]

        # ----------------------------------
        # --------- MINIMUM LENGTH ---------
        # ----------------------------------

        if "Close" not in stocks_trading_data[name]:
            continue

        close = df["Close"].dropna()

        passed_minNumWeeks, numWeeks = EnsureMinimumNumWeeks(close)

        if not passed_minNumWeeks:
            print(f"{name} has failed minNumWeeks; {numWeeks}")
            continue

        # ----------------------------------------------
        # --------- PRICE GREATER THAN 5Y LOW  ---------
        # ----------------------------------------------

        currentClose = close.iloc[-2]

        passed_minPrice, previousLow = EnsureCurrentPriceGreaterThan5YearLow(df, currentClose)

        if not passed_minPrice:
            print(f"{name} has failed minPrice; {previousLow}")
            continue

        # -----------------------------------------
        # --------- PRICE DIP or PULLBACK ---------
        # -----------------------------------------

        # print(nyse_data[name])
        df["200 EMA"] = ComputeExponentialMovingAvg(df)
        ema_cols[(name, "200 EMA")] = df["200 EMA"]

        passed_PercentOffMean, percentOffMean = EnsurePriceNearOrBelowLongTermAverage(df, currentClose)
        passedPullback, pullbackPercent = Ensure1YearPullback(df, currentClose)

        pullback_score = (
            int(passed_PercentOffMean)
            + int(passedPullback)
        )

        if pullback_score < 1:
            print(f"{name} has failed price pullback; {percentOffMean=}, {pullbackPercent}, {pullback_score=}")
            continue

        # ------------------------------
        # --------- MARKET CAP ---------
        # ------------------------------

        info = None
        marketCap = ticker.fast_info.get("marketCap")
        if marketCap is None:
            info = call_with_backoff(ticker.get_info)

        passed_minMarketCap, marketCap = EnsureMinimumMarketCap(marketCap, info)

        if not passed_minMarketCap:
            print(f"{name} has failed minMarketCap; {marketCap}")
            continue

        # ----------------------------------
        # --------- REVENUE GROWTH ---------
        # ----------------------------------
        
        growth = None
        q = None

        if info is None:
            info = call_with_backoff(ticker.get_info)
        if info is None:
            continue

        growth = info.get("revenueGrowth")

        if growth is None:
            q = call_with_backoff(lambda: ticker.quarterly_income_stmt)

        passed_RevenueGrowth, revenueGrowth = EnsureQuarterlyRevenueIncreasing(growth, q)

        if not passed_RevenueGrowth:
            print(f"{name} has failed revenue growth; {revenueGrowth}")
            continue

        # ---------------------------------------
        # --------- INCREASING EARNINGS ---------
        # ---------------------------------------

        # Verify the earnings seem to be increasing quarter-over-quarter
        try:
            earnings = call_with_backoff(ticker.get_earnings_dates)
        except KeyError as e:
            continue

        if earnings is None:
            continue

        # Keep only actual reported quarters
        earnings = earnings[earnings["Reported EPS"].notna()].head(8)

        if len(earnings) < 6:
            continue

        passed_TTM_EPS, ttm_eps = EnsurePositiveTTM_EPS(earnings)

        passed_EarningGrowth, slope = EnsureEarningsGrowing(earnings)

        passed_EarningMeetingThreshold, numPassed, total = EnsureEarningsMeetingThreshold(earnings)

        earnings_score = (
            int(passed_TTM_EPS)
            + int(passed_EarningGrowth)
            + int(passed_EarningMeetingThreshold)
        )

        if earnings_score < 2:
            print(f"{name} has failed on earnings; {ttm_eps=}, {slope=}, {numPassed}/{total}, {earnings_score=}")
            continue

        # -----------------------------------------------
        # --------- AVERAGE DAILY TRADING VALUE ---------
        # -----------------------------------------------

        hist_daily = call_with_backoff(
            ticker.history,
            period="6mo",
            interval="1d",
            auto_adjust=False
        )

        passed_TradingVolume, adtv, avg_shares = EnsureTradingVolume(hist_daily)

        if not passed_TradingVolume:
            print(f"{name} has failed trading volume; {adtv=}, {avg_shares=}")
            continue

        print(f"{name} has passed")

        # stocks_trading_data.loc[:, (name, "200 EMA")] = df["200 EMA"]

        results.append({
            "Symbol": name,
            "Market Cap": marketCap,
            "Percent off Mean": percentOffMean,
            "Pullback Percent": pullbackPercent,
            "Greater Pullback": min(percentOffMean, pullbackPercent),
            "Num Weeks": numWeeks,
            "Current Close": currentClose,
            "TTM_EPS": ttm_eps,
            "Earnings Slope Fit": slope,
            "Earnings Met": numPassed,
            "Total Number of Earnings": total,
            "Revenue Growth": revenueGrowth,
            "ADTV": adtv,
            "Average Shares Traded": avg_shares,
            "Previous Low": previousLow
        })

        # except Exception as e:
            # print(f"{name}: {e}")

    ema_df = pd.concat(ema_cols, axis=1)
    stocks_trading_data = pd.concat([stocks_trading_data, ema_df], axis=1)

    return pd.DataFrame(results), stocks_trading_data