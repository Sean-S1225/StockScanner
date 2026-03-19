from StockScreener.Position import Lot
from datetime import datetime
import pytest
from decimal import Decimal

def test_valid_initialization():
    lot = Lot(Decimal("10"), Decimal("15"), datetime(2026, 1, 1))
    assert isinstance(lot.acquisitionDate, datetime)
    assert isinstance(lot.sharesPurchased, Decimal)
    assert isinstance(lot.entryPrice, Decimal)
    assert isinstance(lot.sharesRemaining, Decimal)

    assert lot.acquisitionDate == datetime(2026, 1, 1)
    assert lot.sharesPurchased == Decimal("10")
    assert lot.entryPrice == Decimal("15")
    assert lot.sharesRemaining == Decimal("10")

@pytest.mark.parametrize("bad_shares", ["hi", None])
def test_invalid_shares_initialization_raises(bad_shares):
    with pytest.raises(ValueError):
        Lot(bad_shares, Decimal("10"), datetime(2026, 1, 1))

@pytest.mark.parametrize("bad_entry", ["hi", None])
def test_invalid_entry_price_initialization_raises(bad_entry):
    with pytest.raises(ValueError):
        Lot(Decimal("10"), bad_entry, datetime(2026, 1, 1))

@pytest.mark.parametrize("bad_date", ["hi", None])
def test_invalid_acquisition_date_raises(bad_date):
    with pytest.raises(ValueError):
        Lot(Decimal("10"), Decimal("10"), bad_date)

def test_shares_remaining():
    lot = Lot(Decimal("10"), Decimal("10"), datetime(2026, 1, 1))
    assert lot.sharesRemaining == Decimal("10")
    lot.SellShares(Decimal("5"))
    assert lot.sharesRemaining == Decimal("5")
    lot.SellShares(Decimal("4"))
    assert lot.sharesRemaining == Decimal("1")

    with pytest.raises(ValueError):
        lot.SellShares(Decimal("2"))
    
    lot.SellShares(Decimal("1"))
    assert lot.sharesRemaining == Decimal("0")

def test_nonpositive_sells_raises():
    lot = Lot(Decimal("10"), Decimal("10"), datetime(2026, 1, 1))

    with pytest.raises(ValueError):
        lot.SellShares(Decimal("-1"))

    with pytest.raises(ValueError):
        lot.SellShares(Decimal("0"))

    with pytest.raises(ValueError):
        lot.SellShares(None)

def test_shares_sold():
    lot = Lot(Decimal("10"), Decimal("10"), datetime(2026, 1, 1))
    assert lot.sharesSold == Decimal("0")
    lot.SellShares(Decimal("5"))
    assert lot.sharesSold == Decimal("5")
    lot.SellShares(Decimal("4"))
    assert lot.sharesSold == Decimal("9")    
    lot.SellShares(Decimal("1"))
    assert lot.sharesSold == Decimal("10")

def test_cost_basis():
    lot = Lot(Decimal("10"), Decimal("10"), datetime(2026, 1, 1))
    assert lot.costBasis == Decimal("100")
    lot.SellShares(Decimal("5"))
    assert lot.costBasis == Decimal("50")
    lot.SellShares(Decimal("4"))
    assert lot.costBasis == Decimal("10")
    lot.SellShares(Decimal("1"))
    assert lot.costBasis == Decimal("0")