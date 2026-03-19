import pytest
from decimal import Decimal
from datetime import datetime
from StockScreener.Position import Position, Transaction, Lot, PnL
from StockScreener.enums import TransactionSide

@pytest.mark.parametrize("init_vals", [
    ("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
])
def test_position_initialization(init_vals):
    ticker, sharesPurchased, costPerShare, reason, date = init_vals
    p = Position(ticker, sharesPurchased, costPerShare, reason, date)

    assert isinstance(p.ticker, str)
    assert p.ticker == ticker

    assert p.realizedPnL == Decimal("0")
    assert p.realizedCostBasis == Decimal("0")

    assert len(p.transactionHistory) == 1
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Buy
    assert t.shares == Decimal("10")
    assert t.fillPrice == Decimal("5")
    assert t.reason == ""
    assert t.date == datetime(2026, 1, 1)

    assert len(p.lots) == 1
    assert isinstance(p.lots[-1], Lot)
    l = p.lots[-1]
    assert l.sharesPurchased == Decimal("10")
    assert l.entryPrice == Decimal("5")
    assert l.acquisitionDate == datetime(2026, 1, 1)

    assert p.numberOpenShares == Decimal("10")
    assert p.realizedCostBasis == Decimal("0")

@pytest.mark.parametrize("init_vals", [
    (None, Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1)),
    (123, Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1)),
    ("AAPL", None, Decimal("5"), "", datetime(2026, 1, 1)),
    ("AAPL", Decimal("10"), None, "", datetime(2026, 1, 1)),
    ("AAPL", Decimal("10"), Decimal("5"), None, datetime(2026, 1, 1)),
    ("AAPL", Decimal("10"), Decimal("5"), "", None),
])
def test_invalid_position_initialization_raises(init_vals):
    with pytest.raises(ValueError):
        Position(*init_vals)

def test_position_buy():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert len(p.transactionHistory) == 1
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Buy
    assert t.shares == Decimal("10")
    assert t.fillPrice == Decimal("5")
    assert t.reason == ""
    assert t.date == datetime(2026, 1, 1)

    assert len(p.lots) == 1
    assert isinstance(p.lots[-1], Lot)
    l = p.lots[-1]
    assert l.sharesPurchased == Decimal("10")
    assert l.entryPrice == Decimal("5")
    assert l.acquisitionDate == datetime(2026, 1, 1)

    p.Buy(Decimal("15"), Decimal("10"), "hi", datetime(2026, 1, 2))
    assert len(p.transactionHistory) == 2
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Buy
    assert t.shares == Decimal("15")
    assert t.fillPrice == Decimal("10")
    assert t.reason == "hi"
    assert t.date == datetime(2026, 1, 2)

    assert len(p.lots) == 2
    assert isinstance(p.lots[-1], Lot)
    l = p.lots[-1]
    assert l.sharesPurchased == Decimal("15")
    assert l.entryPrice == Decimal("10")
    assert l.acquisitionDate == datetime(2026, 1, 2)

    p.Buy(Decimal("15"), Decimal("10"), "hi", datetime(2026, 1, 3))
    assert len(p.transactionHistory) == 3
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Buy
    assert t.shares == Decimal("15")
    assert t.fillPrice == Decimal("10")
    assert t.reason == "hi"
    assert t.date == datetime(2026, 1, 3)

    assert len(p.lots) == 3
    assert isinstance(p.lots[-1], Lot)
    l = p.lots[-1]
    assert l.sharesPurchased == Decimal("15")
    assert l.entryPrice == Decimal("10")
    assert l.acquisitionDate == datetime(2026, 1, 3)

@pytest.mark.parametrize("invalidBuy", [
    (Decimal("0"), Decimal("5"), "", datetime(2026, 1, 1)),
    (None, Decimal("5"), "", datetime(2026, 1, 1)),
    (Decimal("10"), None, "", datetime(2026, 1, 1)),
    (Decimal("10"), Decimal("5"), None, datetime(2026, 1, 1)),
    (Decimal("10"), Decimal("5"), "", None),
    (Decimal("-1"), Decimal("5"), "", datetime(2026, 1, 1)),
    (Decimal("10"), Decimal("-5"), "", datetime(2026, 1, 1)),
])
def test_invalid_position_buy_raises(invalidBuy):
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        p.Buy(*invalidBuy)

    assert len(p.transactionHistory) == 1
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Buy
    assert t.shares == Decimal("10")
    assert t.fillPrice == Decimal("5")
    assert t.reason == ""
    assert t.date == datetime(2026, 1, 1)

    assert len(p.lots) == 1
    assert isinstance(p.lots[-1], Lot)
    l = p.lots[-1]
    assert l.sharesPurchased == Decimal("10")
    assert l.entryPrice == Decimal("5")
    assert l.acquisitionDate == datetime(2026, 1, 1)

def test_position_sell():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    
    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 2))
    assert len(p.transactionHistory) == 2
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Sell
    assert t.shares == Decimal("5")
    assert t.fillPrice == Decimal("10")
    assert t.reason == ""
    assert t.date == datetime(2026, 1, 2)

    assert len(p.lots) == 1
    assert isinstance(p.lots[-1], Lot)
    l = p.lots[-1]
    assert l.acquisitionDate == datetime(2026, 1, 1)
    assert l.entryPrice == Decimal("5")
    assert l.sharesPurchased == Decimal("10")
    assert l.sharesRemaining == Decimal("5")
    assert l.sharesSold == Decimal("5")

    p.Buy(Decimal("12"), Decimal("7.5"), "", datetime(2026, 1, 3))
    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026, 1, 4))
    assert len(p.transactionHistory) == 4
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Sell
    assert t.shares == Decimal("10")
    assert t.fillPrice == Decimal("12")
    assert t.reason == ""
    assert t.date == datetime(2026, 1, 4)

    assert len(p.lots) == 2
    assert isinstance(p.lots[-1], Lot)
    l1, l2 = p.lots
    assert l1.acquisitionDate == datetime(2026, 1, 1)
    assert l2.acquisitionDate == datetime(2026, 1, 3)
    assert l1.entryPrice == Decimal("5")
    assert l2.entryPrice == Decimal("7.5")
    assert l1.sharesPurchased == Decimal("10")
    assert l2.sharesPurchased == Decimal("12")
    assert l1.sharesRemaining == Decimal("0")
    assert l2.sharesRemaining == Decimal("7")
    assert l1.sharesSold == Decimal("10")
    assert l2.sharesSold == Decimal("5")

def test_sell_returns_proceeds():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))

    proceeds = p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert proceeds == Decimal("5") * Decimal("10")

    proceeds = p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert proceeds == Decimal("10") * Decimal("12")

@pytest.mark.parametrize("invalidSell", [
    (Decimal("0"), Decimal("5"), "", datetime(2026, 1, 1)),
    (None, Decimal("5"), "", datetime(2026, 1, 1)),
    (Decimal("10"), None, "", datetime(2026, 1, 1)),
    (Decimal("10"), Decimal("5"), None, datetime(2026, 1, 1)),
    (Decimal("10"), Decimal("5"), "", None),
    (Decimal("11"), Decimal("5"), "", datetime(2026, 1, 1)),
    (Decimal("-1"), Decimal("5"), "", datetime(2026, 1, 1)),
    (Decimal("10"), Decimal("-5"), "", datetime(2026, 1, 1)),
])
def test_invalid_position_sell_raises(invalidSell):
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        p.Sell(*invalidSell)

    assert len(p.transactionHistory) == 1
    assert isinstance(p.transactionHistory[-1], Transaction)
    t = p.transactionHistory[-1]
    assert t.side == TransactionSide.Buy
    assert t.shares == Decimal("10")
    assert t.fillPrice == Decimal("5")
    assert t.reason == ""
    assert t.date == datetime(2026, 1, 1)

    assert len(p.lots) == 1
    assert isinstance(p.lots[-1], Lot)
    l = p.lots[-1]
    assert l.sharesPurchased == Decimal("10")
    assert l.entryPrice == Decimal("5")
    assert l.acquisitionDate == datetime(2026, 1, 1)

def test_position_cost_basis():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.costBasis == (Decimal("10") * Decimal("5"))

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.costBasis == (
        Decimal("10") * Decimal("5")
         + Decimal("5") * Decimal("7.5")
    )

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.costBasis == (
        Decimal("5") * Decimal("5")
        + Decimal("5") * Decimal("7.5")
    )

def test_position_average_entry():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.averageEntry == (Decimal("10") * Decimal("5")) / Decimal("10")

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.averageEntry == (
        Decimal("10") * Decimal("5")
        + Decimal("5") * Decimal("7.5")
    ) / (Decimal("10") + Decimal("5"))

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.averageEntry == (
        Decimal("5") * Decimal("5")
        + Decimal("5") * Decimal("7.5")
    ) / (Decimal("10"))

def test_position_number_open_shares():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.numberOpenShares == Decimal("10")

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.numberOpenShares == Decimal("15")

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.numberOpenShares == Decimal("10")

def test_position_realized_cost_basis():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.realizedCostBasis == Decimal("0")

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.realizedCostBasis == Decimal("0")

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.realizedCostBasis == Decimal("5") * Decimal("5")

    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert p.realizedCostBasis == Decimal("5") * Decimal("10") + Decimal("5") * Decimal("7.5")

def test_position_realized_pnl():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.realizedPnL == Decimal("0")

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.realizedPnL == Decimal("0")

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.realizedPnL == (Decimal("5") * Decimal("10")) - (Decimal("5") * Decimal("5"))

    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert p.realizedPnL == ((
        Decimal("5") * Decimal("10")
        + Decimal("5") * Decimal("12")
        + Decimal("5") * Decimal("12"))
    - (
        Decimal("5") * Decimal("5")
        + Decimal("5") * Decimal("5")
        + Decimal("5") * Decimal("7.5"))
    )

def test_poasition_realizedPnL_percent():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.realizedPnL_percent == Decimal("0")

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.realizedPnL_percent == Decimal("0")

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.realizedPnL_percent == Decimal("100") * (
        (Decimal("5") * Decimal("10"))
        - (Decimal("5") * Decimal("5"))
    ) / (Decimal("5") * Decimal("5"))

    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert p.realizedPnL_percent == Decimal("100") * (
        (Decimal("5") * Decimal("10")
         + Decimal("5") * Decimal("12")
         + Decimal("5") * Decimal("12"))
        - (Decimal("5") * Decimal("5")
           + Decimal("5") * Decimal("5")
           + Decimal("5") * Decimal("7.5"))
    ) / (Decimal("5") * Decimal("5")
         + Decimal("5") * Decimal("5")
         + Decimal("5") * Decimal("7.5"))


def test_position_unrealized_pnl():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.GetUnrealizedPnL(Decimal("5")) == Decimal("0")
    assert p.GetUnrealizedPnL(Decimal("10")) == (
        (Decimal("10") * Decimal("10"))
        - (Decimal("10") * Decimal("5"))
    )

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.GetUnrealizedPnL(Decimal("10")) == (
        (Decimal("10") * Decimal("10")
         + Decimal("10") * Decimal("5"))
        - (Decimal("10") * Decimal("5")
           + Decimal("5") * Decimal("7.5"))
    )
    assert p.GetUnrealizedPnL(Decimal("5")) == (
        (Decimal("5") * Decimal("10")
         + Decimal("5") * Decimal("5"))
         - (Decimal("10") * Decimal("5")
            + Decimal("5") * Decimal("7.5"))
    )

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.GetUnrealizedPnL(Decimal("10")) == (
        (Decimal("5") * Decimal("10")
         + Decimal("5") * Decimal("10"))
        - (Decimal("5") * Decimal("5")
           + Decimal("5") * Decimal("7.5"))
    )

    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert p.GetUnrealizedPnL(Decimal("10")) == Decimal("0")

@pytest.mark.parametrize("invalidNum", [None, 0, -1, 2.2, Decimal("0"), Decimal("-1")])
def test_position_invalid_unrealized_pnl_raises(invalidNum):
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        p.GetUnrealizedPnL(invalidNum)

def test_position_unrealized_pnl_percent():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.GetUnrealizedPnL_percent(Decimal("5")) == Decimal("100") * Decimal("0")
    assert p.GetUnrealizedPnL_percent(Decimal("10")) == Decimal("100") * (
        (Decimal("10") * Decimal("10"))
        - (Decimal("10") * Decimal("5"))
    ) / (Decimal("10") * Decimal("5"))

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.GetUnrealizedPnL_percent(Decimal("10")) == Decimal("100") * (
        (Decimal("10") * Decimal("10")
         + Decimal("10") * Decimal("5"))
         - (Decimal("10") * Decimal("5")
            + Decimal("5") * Decimal("7.5"))
    ) / (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5"))
    assert p.GetUnrealizedPnL_percent(Decimal("5")) == Decimal("100") * (
        (Decimal("5") * Decimal("10")
         + Decimal("5") * Decimal("5"))
         - (Decimal("10") * Decimal("5")
            + Decimal("5") * Decimal("7.5"))
    ) / (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5"))

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.GetUnrealizedPnL_percent(Decimal("10")) == Decimal("100") * (
        (Decimal("5") * Decimal("10")
         + Decimal("5") * Decimal("10"))
         - (Decimal("5") * Decimal("5")
            + Decimal("5") * Decimal("7.5"))
    ) / (Decimal("5") * Decimal("5") + Decimal("5") * Decimal("7.5"))

    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert p.GetUnrealizedPnL_percent(Decimal("10")) == Decimal("100") * Decimal("0")

@pytest.mark.parametrize("invalidNum", [None, 0, -1, 2.2, Decimal("0"), Decimal("-1")])
def test_invalid_position_unrealized_pnl_percent_raises(invalidNum):
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        p.GetUnrealizedPnL_percent(invalidNum)

def test_position_get_pnl():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.GetPnL(Decimal("5")) == PnL(
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0")
    )
    assert p.GetPnL(Decimal("10")) == PnL(
        Decimal("0"),
        Decimal("0"),
        (Decimal("10") * Decimal("10")) - (Decimal("10") * Decimal("5")),
        Decimal("100") * ((Decimal("10") * Decimal("10")) - (Decimal("10") * Decimal("5"))) /(Decimal("10") * Decimal("5"))
    )

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.GetPnL(Decimal("10")) == PnL(
        Decimal("0"),
        Decimal("0"),
        (Decimal("10") * Decimal("10") + Decimal("10") * Decimal("5")) - (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5")),
        Decimal("100") * ((Decimal("10") * Decimal("10") + Decimal("10") * Decimal("5")) - (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5"))) / (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5"))
    )
    assert p.GetPnL(Decimal("5")) == PnL(
        Decimal("0"),
        Decimal("0"),
        (Decimal("5") * Decimal("10") + Decimal("5") * Decimal("5")) - (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5")),
        Decimal("100") * ((Decimal("5") * Decimal("10") + Decimal("5") * Decimal("5")) - (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5"))) / (Decimal("10") * Decimal("5") + Decimal("5") * Decimal("7.5"))
    )

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.GetPnL(Decimal("10")) == PnL(
        (Decimal("5") * Decimal("10")) - (Decimal("5") * Decimal("5")),
        Decimal("100") * ((Decimal("5") * Decimal("10")) - (Decimal("5") * Decimal("5"))) / (Decimal("5") * Decimal("5")),
        (Decimal("5") * Decimal("10") + Decimal("5") * Decimal("10")) - (Decimal("5") * Decimal("5") + Decimal("5") * Decimal("7.5")),
        Decimal("100") * ((Decimal("5") * Decimal("10") + Decimal("5") * Decimal("10")) - (Decimal("5") * Decimal("5") + Decimal("5") * Decimal("7.5"))) / (Decimal("5") * Decimal("5") + Decimal("5") * Decimal("7.5"))
    )

    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert p.GetPnL(Decimal("10")) == PnL(
        (Decimal("5") * Decimal("10") + Decimal("5") * Decimal("12") + Decimal("5") * Decimal("12")) - (Decimal("5") * Decimal("5") + Decimal("5") * Decimal("5") + Decimal("5") * Decimal("7.5")),
        Decimal("100") * ((Decimal("5") * Decimal("10") + Decimal("5") * Decimal("12") + Decimal("5") * Decimal("12")) - (Decimal("5") * Decimal("5") + Decimal("5") * Decimal("5") + Decimal("5") * Decimal("7.5"))) / (Decimal("5") * Decimal("5") + Decimal("5") * Decimal("5") + Decimal("5") * Decimal("7.5")),
        Decimal("0"),
        Decimal("100") * Decimal("0")
    )

@pytest.mark.parametrize("invalidNum", [None, 0, -1, 2.2, Decimal("0"), Decimal("-1")])
def test_invalid_position_get_pnl_raises(invalidNum):
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        p.GetPnL(invalidNum)

def test_position_market_value():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    assert p.CurrentMarketValue(Decimal("5")) == Decimal("5") * Decimal("10")
    assert p.CurrentMarketValue(Decimal("10")) == Decimal("10") * Decimal("10")

    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    assert p.CurrentMarketValue(Decimal("10")) == Decimal("10") * Decimal("15")
    assert p.CurrentMarketValue(Decimal("5")) == Decimal("5") * Decimal("15")

    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    assert p.CurrentMarketValue(Decimal("10")) == Decimal("10") * Decimal("10")

    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))
    assert p.CurrentMarketValue(Decimal("10")) == Decimal("10") * Decimal("0")

@pytest.mark.parametrize("invalidNum", [None, 0, -1, 2.2, Decimal("0"), Decimal("-1")])
def test_invalid_position_market_value_raises(invalidNum):
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        p.CurrentMarketValue(invalidNum)

def test_state_invariance():
    p = Position("AAPL", Decimal("10"), Decimal("5"), "", datetime(2026, 1, 1))
    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))
    p.Sell(Decimal("10"), Decimal("12"), "", datetime(2026,1, 4))

    assert p.numberOpenShares == sum(l.sharesRemaining for l in p.lots)
    assert p.costBasis == sum(l.costBasis for l in p.lots)
    assert p.averageEntry == Decimal("0") if p.numberOpenShares == 0 else p.costBasis / p.numberOpenShares

    p.Buy(Decimal("5"), Decimal("5"), "", datetime(2026, 1, 2))
    p.Buy(Decimal("5"), Decimal("7.5"), "", datetime(2026, 1, 2))
    p.Sell(Decimal("5"), Decimal("10"), "", datetime(2026, 1, 3))

    assert p.numberOpenShares == sum(l.sharesRemaining for l in p.lots)
    assert p.costBasis == sum(l.costBasis for l in p.lots)
    assert p.averageEntry == Decimal("0") if p.numberOpenShares == 0 else p.costBasis / p.numberOpenShares
