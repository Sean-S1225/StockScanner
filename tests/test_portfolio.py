import pytest
from StockScreener.Portfolio import Portfolio, PnL
from StockScreener.Position import Position
from copy import deepcopy
from decimal import Decimal
from datetime import datetime
import pandas as pd

def test_portfolio_initialization():
    p = Portfolio(Decimal("1000"))

    assert p.positions == {}
    assert p.closedPositions == []
    assert p.startingCash == Decimal("1000")
    assert p.cash == Decimal("1000")

    p = Portfolio(Decimal("1"))

    assert p.positions == {}
    assert p.closedPositions == []
    assert p.startingCash == Decimal("1")
    assert p.cash == Decimal("1")

@pytest.mark.parametrize("invalidCash", [1000, Decimal("0"), Decimal("-1"), None])
def test_invalid_portfolio_initialization_raises(invalidCash):
    with pytest.raises(ValueError):
        Portfolio(invalidCash)

@pytest.mark.parametrize("buyOrder", [
    ("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1)),
    ("AAPL", Decimal("10"), Decimal("100"), "", datetime(2026, 1, 1))
])
def test_portfolio_buy(buyOrder):
    p = Portfolio(Decimal("1000"))
    p.Buy(*buyOrder)

    assert buyOrder[0] in p.positions
    assert isinstance(p.positions[buyOrder[0]], Position)
    assert p.cash == Decimal("1000") - (buyOrder[1] * buyOrder[2])

def test_portfolio_multibuy():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Buy("MSFT", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    assert len(p.positions) == 2
    assert p.closedPositions == []
    assert p.cash == (Decimal("1000")
                      - (Decimal("5") * Decimal("10"))
                      - (Decimal("10") * Decimal("5")))
    assert p.positions["AAPL"].numberOpenShares == Decimal("5")
    assert p.positions["MSFT"].numberOpenShares == Decimal("10")

    p.Buy("AAPL", Decimal("7.5"), Decimal("6"), "", datetime(2026, 1, 2))
    p.Buy("MSFT", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 2))

    assert len(p.positions) == 2
    assert p.closedPositions == []
    assert p.cash == (
        Decimal("1000")
        - (Decimal("5") * Decimal("10"))
        - (Decimal("10") * Decimal("5"))
        - (Decimal("7.5") * Decimal("6"))
        - (Decimal("10") * Decimal("5"))
    )

    assert p.positions["AAPL"].numberOpenShares == Decimal("12.5")
    assert p.positions["MSFT"].numberOpenShares == Decimal("20")

@pytest.mark.parametrize("buyOrder", [
    (None, Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1)),
    ("AAPL", None, Decimal("10"), "", datetime(2026, 1, 1)),
    ("AAPL", 10, Decimal("10"), "", datetime(2026, 1, 1)),
    ("AAPL", Decimal("5"), None, "", datetime(2026, 1, 1)),
    ("AAPL", Decimal("5"), 10, "", datetime(2026, 1, 1)),
    ("AAPL", Decimal("5"), Decimal("10"), None, datetime(2026, 1, 1)),
    ("AAPL", Decimal("5"), Decimal("10"), "", None),
    (123, Decimal("10"), Decimal("100"), "", datetime(2026, 1, 1)),
    ("AAPL", Decimal("11"), Decimal("100"), "", datetime(2026, 1, 1))
])
def test_invalid_portfolio_buy_raises(buyOrder):
    p = Portfolio(Decimal("1000"))
    with pytest.raises(ValueError):
        p.Buy(*buyOrder)

    assert p.closedPositions == []
    assert p.positions == {}
    assert p.startingCash == Decimal("1000")
    assert p.cash == Decimal("1000")

    p.Buy("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 2))

    assert p.closedPositions == []
    assert p.positions == {"AAPL": Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 2))}
    assert p.startingCash == Decimal("1000")
    assert p.cash == Decimal("1000") - (Decimal("10") * Decimal("5"))

    with pytest.raises(ValueError):
        p.Buy(*buyOrder)

    assert p.closedPositions == []
    assert p.positions == {"AAPL": Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 2))}
    assert p.startingCash == Decimal("1000")
    assert p.cash == Decimal("1000") - (Decimal("10") * Decimal("5"))

    p.Sell("AAPL", Decimal("10"), Decimal("10"), "", datetime(2026, 1, 3))

    temp = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 2))
    temp.Sell(Decimal("10"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.closedPositions == [temp]
    assert p.positions == {}
    assert p.startingCash == Decimal("1000")
    assert p.cash == Decimal("1000") - (Decimal("10") * Decimal("5")) + (Decimal("10") * Decimal("10"))

    with pytest.raises(ValueError):
        p.Buy(*buyOrder)

    assert p.closedPositions == [temp]
    assert p.positions == {}
    assert p.startingCash == Decimal("1000")
    assert p.cash == Decimal("1000") - (Decimal("10") * Decimal("5")) + (Decimal("10") * Decimal("10"))

def test_portfolio_sell():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    p.Sell("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 2))
    assert p.cash == (
        Decimal("1000")
        - (Decimal("10") * Decimal("5"))
        + (Decimal("5") * Decimal("10")))
    assert p.closedPositions == []
    assert "AAPL" in p.positions

    pos = deepcopy(p.positions["AAPL"])
    pos.Sell(Decimal("5"), Decimal("15"), "", datetime(2026, 1, 2))

    p.Sell("AAPL", Decimal("5"), Decimal("15"), "", datetime(2026, 1, 2))
    assert p.cash == (Decimal("1000")
                      - (Decimal("10") * Decimal("5"))
                      + (Decimal("5") * Decimal("10"))
                      + (Decimal("5") * Decimal("15")))
    assert len(p.closedPositions) == 1
    assert p.closedPositions[0].transactionHistory == pos.transactionHistory
    assert p.closedPositions[0].lots == pos.lots
    assert "AAPL" not in p.positions

@pytest.mark.parametrize("sellOrder", [
    (None, Decimal("5"), Decimal("10"), "", datetime(2026, 1, 2)),
    ("AAPL", None, Decimal("10"), "", datetime(2026, 1, 2)),
    ("AAPL", Decimal("5"), None, "", datetime(2026, 1, 2)),
    ("AAPL", Decimal("5"), Decimal("10"), None, datetime(2026, 1, 2)),
    ("AAPL", Decimal("5"), Decimal("10"), "", None),
    ("AAPL", 5, Decimal("10"), "", datetime(2026, 1, 2)),
    ("AAPL", Decimal("5"), 10, "", datetime(2026, 1, 2)),
    ("AAPL", Decimal("15"), Decimal("10"), "", datetime(2026, 1, 2)),
    ("MSFT", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 2)),
])
def test_portfolio_invalid_sell_raises(sellOrder):
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    with pytest.raises(ValueError):
        p.Sell(*sellOrder)

    assert p.cash == (Decimal("1000")
                      - (Decimal("10") * Decimal("5")))
    assert p.closedPositions == []
    assert "AAPL" in p.positions

def test_portfolio_buy_sell_buy():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))

    pos = deepcopy(p.positions["AAPL"])
    pos.Sell(Decimal("5"), Decimal("20"), "", datetime(2026, 1, 2))

    p.Sell("AAPL", Decimal("5"), Decimal("20"), "", datetime(2026, 1, 2))
    p.Buy("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 3))

    assert len(p.positions) == 1
    assert len(p.closedPositions) == 1
    assert p.closedPositions[0].transactionHistory == pos.transactionHistory
    assert p.closedPositions[0].lots == pos.lots
    assert p.cash == (Decimal("1000")
                      - (Decimal("5") * Decimal("10"))
                      + (Decimal("5") * Decimal("20"))
                      - (Decimal("10") * Decimal("5")))
    
# mostly random values
@pytest.mark.parametrize("vals", [
    (Decimal("5"), Decimal("10"), Decimal("10"), Decimal("5"), Decimal("12"), Decimal("2.5"), Decimal("8"), Decimal("10"), Decimal("18"), Decimal("4")),
	(Decimal("89.9"), Decimal("87.2"), Decimal("63.3"), Decimal("37.4"), Decimal("86.1"), Decimal("87.9"), Decimal("23.1"), Decimal("24.0"), Decimal("20.7"), Decimal("89.1")),
	(Decimal("51.0"), Decimal("96.0"), Decimal("90.9"), Decimal("66.9"), Decimal("48.4"), Decimal("62.7"), Decimal("52.5"), Decimal("38.5"), Decimal("5.2"), Decimal("91.5")),
	(Decimal("26.7"), Decimal("18.4"), Decimal("67.8"), Decimal("19.8"), Decimal("53.2"), Decimal("0.8"), Decimal("55.4"), Decimal("96.1"), Decimal("23.2"), Decimal("33.4")),
	(Decimal("52.9"), Decimal("53.0"), Decimal("56.9"), Decimal("90.1"), Decimal("15.0"), Decimal("17.6"), Decimal("0.9"), Decimal("93.4"), Decimal("60.5"), Decimal("85.9")),
	(Decimal("18.4"), Decimal("8.6"), Decimal("68.3"), Decimal("98.9"), Decimal("73.0"), Decimal("22.1"), Decimal("81.4"), Decimal("70.6"), Decimal("42.0"), Decimal("14.0")),
])
def test_get_portfolio_pnl(vals):
    p = Portfolio(Decimal("10000000"))

    aaplShares, aaplPurchasePrice, msftShares, msftPurchasePrice, aaplCurrPrice1, msftCurrPrice1, aaplSellPrice, msftSellPrice, aaplCurrPrice2, msftCurrPrice2 = vals

    p.Buy("AAPL", aaplShares, aaplPurchasePrice, "", datetime(2026, 1, 1))
    p.Buy("MSFT", msftShares, msftPurchasePrice, "", datetime(2026, 1, 1))

    pnl = p.GetPortfolioPnL(
        {"AAPL": aaplCurrPrice1, "MSFT": msftCurrPrice1}
    )

    assert pnl.realizedPnL == Decimal("0")
    assert pnl.realizedPnL_percent == Decimal("0")

    assert pnl.unrealizedPnL == (
        (aaplShares * aaplCurrPrice1) - (aaplShares * aaplPurchasePrice)
        + (msftShares * msftCurrPrice1) - (msftShares * msftPurchasePrice)
    )

    assert pnl.unrealizedPnL_percent == Decimal("100") * (
        (aaplShares * aaplCurrPrice1) - (aaplShares * aaplPurchasePrice)
        + (msftShares * msftCurrPrice1) - (msftShares * msftPurchasePrice)
    ) / ((aaplShares * aaplPurchasePrice) + (msftShares * msftPurchasePrice))

    aaplSellShares = Decimal("0.66") * aaplShares
    msftSellShares = Decimal("0.5") * msftShares

    p.Sell("AAPL", aaplSellShares, aaplSellPrice, "", datetime(2026, 1, 2))
    p.Sell("MSFT", msftSellShares, msftSellPrice, "", datetime(2026, 1, 2))

    pnl = p.GetPortfolioPnL(
        {"AAPL": aaplCurrPrice2, "MSFT": msftCurrPrice2}
    )

    assert pnl.realizedPnL == (
        (aaplSellShares * aaplSellPrice) - (aaplSellShares * aaplPurchasePrice)
        + (msftSellShares * msftSellPrice) - (msftSellShares * msftPurchasePrice)
    )

    assert pnl.realizedPnL_percent == Decimal("100") * (
        (aaplSellShares * aaplSellPrice) - (aaplSellShares * aaplPurchasePrice)
        + (msftSellShares * msftSellPrice) - (msftSellShares * msftPurchasePrice)
    ) / ((aaplSellShares * aaplPurchasePrice) + (msftSellShares * msftPurchasePrice))

    assert pnl.unrealizedPnL == (
        ((aaplShares - aaplSellShares) * aaplCurrPrice2) - ((aaplShares - aaplSellShares) * aaplPurchasePrice)
        + ((msftShares - msftSellShares) * msftCurrPrice2) - ((msftShares - msftSellShares) * msftPurchasePrice)
    )

    assert pnl.unrealizedPnL_percent == Decimal("100") * (
        ((aaplShares - aaplSellShares) * aaplCurrPrice2) - ((aaplShares - aaplSellShares) * aaplPurchasePrice)
        + ((msftShares - msftSellShares) * msftCurrPrice2) - ((msftShares - msftSellShares) * msftPurchasePrice)
    ) / (((aaplShares - aaplSellShares) * aaplPurchasePrice) + ((msftShares - msftSellShares) * msftPurchasePrice))

def test_portfolio_pnl_fully_sold():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    p.Buy("MSFT", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))

    p.Sell("AAPL", Decimal("10"), Decimal("10"), "", datetime(2026, 1, 2))
    p.Sell("MSFT", Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))

    assert p.positions == {}
    assert len(p.closedPositions) == 2

    assert p.GetPortfolioPnL({"AAPL": Decimal("500"), "MSFT": Decimal("1")}) == PnL(
        ((Decimal("10") * Decimal("10")) + (Decimal("5") * Decimal("7.5")) - 
        ((Decimal("10") * Decimal("5")) + (Decimal("5") * Decimal("10")))),
        Decimal("100") * ((Decimal("10") * Decimal("10")) + (Decimal("5") * Decimal("7.5")) - 
        ((Decimal("10") * Decimal("5")) + (Decimal("5") * Decimal("10")))) / ((Decimal("10") * Decimal("5")) + (Decimal("5") * Decimal("10"))),
        Decimal("0"),
        Decimal("0"))
    
    assert p.GetPortfolioMarketCap({"AAPL": Decimal("500"), "MSFT": Decimal("1")}) == Decimal("0")
    assert p.GetEquity({"AAPL": Decimal("500"), "MSFT": Decimal("1")}) == (
        Decimal("1000") +
        ((Decimal("10") * Decimal("10")) + (Decimal("5") * Decimal("7.5")) - 
        ((Decimal("10") * Decimal("5")) + (Decimal("5") * Decimal("10"))))
    )

def test_portfolio_market_cap():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Buy("MSFT", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    mc = p.GetPortfolioMarketCap({"AAPL": Decimal("15"), "MSFT": Decimal("4.5")})

    assert mc == Decimal("5") * Decimal("15") + Decimal("10") * Decimal("4.5")

    p.Sell("AAPL", Decimal("2"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Sell("MSFT", Decimal("8"), Decimal("5"), "", datetime(2026, 1, 1))

    mc = p.GetPortfolioMarketCap({"AAPL": Decimal("10"), "MSFT": Decimal("7")})

    assert mc == Decimal("3") * Decimal("10") + Decimal("2") * Decimal("7")

def test_portfolio_get_equity():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Buy("MSFT", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    equity = p.GetEquity({"AAPL": Decimal("15"), "MSFT": Decimal("4.5")})

    assert equity == (
        Decimal("1000")
        - Decimal("5") * Decimal("10")
        - Decimal("10") * Decimal("5")
        + Decimal("5") * Decimal("15")
        + Decimal("10") * Decimal("4.5")
    )

    p.Sell("AAPL", Decimal("2"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Sell("MSFT", Decimal("8"), Decimal("5"), "", datetime(2026, 1, 1))

    equity = p.GetEquity({"AAPL": Decimal("10"), "MSFT": Decimal("7")})

    assert equity == (
        Decimal("1000")
        - Decimal("5") * Decimal("10")
        - Decimal("10") * Decimal("5")
        + Decimal("2") * Decimal("10")
        + Decimal("8") * Decimal("5")
        + Decimal("3") * Decimal("10")
        + Decimal("2") * Decimal("7")
    )

def test_portfolio_return_percent():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Buy("MSFT", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    returnPercent = p.GetReturnPercent({"AAPL": Decimal("15"), "MSFT": Decimal("4.5")})

    assert returnPercent == Decimal("100") * (
        - Decimal("5") * Decimal("10")
        - Decimal("10") * Decimal("5")
        + Decimal("5") * Decimal("15")
        + Decimal("10") * Decimal("4.5")
    ) / Decimal("1000")

    p.Sell("AAPL", Decimal("2"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Sell("MSFT", Decimal("8"), Decimal("5"), "", datetime(2026, 1, 1))

    returnPercent = p.GetReturnPercent({"AAPL": Decimal("10"), "MSFT": Decimal("7")})

    assert returnPercent == Decimal("100") * (
        - Decimal("5") * Decimal("10")
        - Decimal("10") * Decimal("5")
        + Decimal("2") * Decimal("10")
        + Decimal("8") * Decimal("5")
        + Decimal("3") * Decimal("10")
        + Decimal("2") * Decimal("7")
    ) / Decimal("1000")

def test_summarize_holdings():
    p = Portfolio(Decimal("1000"))
    p.Buy("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Buy("MSFT", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    p.Sell("AAPL", Decimal("5"), Decimal("12"), "", datetime(2026, 1, 2))

    summary = p.SummarizeHoldings({"MSFT": Decimal("7")})

    assert isinstance(summary, pd.DataFrame)
    assert all(x in summary for x in ["ticker", "openShares", "averageEntry", "costBasis", "marketValue", "realizedPnL", "realizedPnL_percent", "unrealizedPnL", "unrealizedPnL_percent"])

    assert (summary[summary["ticker"] == "AAPL"]["openShares"] == Decimal("0")).item()
    assert (summary[summary["ticker"] == "MSFT"]["openShares"] == Decimal("10")).item()

    assert len(summary) == 2
    assert (summary[summary["ticker"] == "AAPL"]["marketValue"] == Decimal("0")).item()
    assert (summary[summary["ticker"] == "MSFT"]["marketValue"] != Decimal("0")).item()

def test_portfolio_keys_not_found():
    p = Portfolio(Decimal("1000"))

    p.Buy("AAPL", Decimal("5"), Decimal("10"), "", datetime(2026, 1, 1))
    p.Buy("MSFT", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))

    p.Sell("AAPL", Decimal("5"), Decimal("12"), "", datetime(2026, 1, 2))

    with pytest.raises(KeyError):
        p.GetPortfolioMarketCap({"AAPL": Decimal("10")})

    with pytest.raises(ValueError):
        p.GetPortfolioMarketCap({"MSFT": 10})

    with pytest.raises(KeyError):
        p.GetPortfolioPnL({"AAPL": Decimal("10")})

    with pytest.raises(ValueError):
        p.GetPortfolioPnL({"MSFT": 10})

    # Test these do not raise errors
    p.GetPortfolioMarketCap({"MSFT": Decimal(10)})
    p.GetPortfolioPnL({"MSFT": Decimal(10)})
