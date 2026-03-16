from StockScreener.Position import Lot
from datetime import datetime
import pytest
from decimal import Decimal

def test_valid_initialization():
    lot = Lot(10, 15, datetime(2026, 1, 1))
    assert isinstance(lot.acquisitionDate, datetime)
    assert isinstance(lot.sharesPurchased, Decimal)
    assert isinstance(lot.entryPrice, Decimal)
    assert isinstance(lot.sharesRemaining, Decimal)

    assert lot.acquisitionDate == datetime(2026, 1, 1)
    assert lot.sharesPurchased == Decimal(10)
    assert lot.entryPrice == Decimal(15)
    assert lot.sharesRemaining == Decimal(10)

@pytest.mark.parametrize("bad_shares", ["hi", None])
def test_invalid_shares_initialization_raises(bad_shares):
    with pytest.raises(ValueError) as exc_info:
        Lot(bad_shares, 10, datetime(2026, 1, 1))
    assert "Cannot convert type" in str(exc_info.value)

@pytest.mark.parametrize("bad_entry", ["hi", None])
def test_invalid_entry_price_initialization_raises(bad_entry):
    with pytest.raises(ValueError) as exc_info:
        Lot(10, bad_entry, datetime(2026, 1, 1))
    assert "Cannot convert type" in str(exc_info.value)

@pytest.mark.parametrize("bad_date", ["hi", None])
def test_invalid_acquisition_date_raises(bad_date):
    with pytest.raises(TypeError) as exc_info:
        Lot(10, 10, bad_date)
    assert "acquisitionDate must be of type datetime" in str(exc_info.value)

def test_decimal_conversion():
    lot = Lot(1.1, 2.2, datetime(2026, 1, 1))
    assert lot.sharesRemaining == Decimal("1.1")
    assert lot.entryPrice == Decimal("2.2")
    assert lot.costBasis == Decimal("1.1") * Decimal("2.2")

def test_shares_remaining():
    lot = Lot(10, 10, datetime(2026, 1, 1))
    assert lot.sharesRemaining == Decimal(10)
    lot.SellShares(5)
    assert lot.sharesRemaining == Decimal(5)
    lot.SellShares(4)
    assert lot.sharesRemaining == Decimal(1)

    with pytest.raises(ValueError) as exc_info:
        lot.SellShares(2)
    assert "Cannot sell more stocks than are allocated in this lot" in str(exc_info.value)
    
    lot.SellShares(1)
    assert lot.sharesRemaining == Decimal(0)

def test_nonpositive_sells_raises():
    lot = Lot(10, 10, datetime(2026, 1, 1))

    with pytest.raises(ValueError) as exc_info:
        lot.SellShares(-1)
    assert "Cannot sell a non-positive number of shares" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        lot.SellShares(0)
    assert "Cannot sell a non-positive number of shares" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        lot.SellShares(None)
    assert "sharesToSell must be castable to type Decimal, not of type" in str(exc_info.value)

def test_shares_sold():
    lot = Lot(10, 10, datetime(2026, 1, 1))
    assert lot.sharesSold == Decimal(0)
    lot.SellShares(5)
    assert lot.sharesSold == Decimal(5)
    lot.SellShares(4)
    assert lot.sharesSold == Decimal(9)    
    lot.SellShares(1)
    assert lot.sharesSold == Decimal(10)

def test_cost_basis():
    lot = Lot(10, 10, datetime(2026, 1, 1))
    assert lot.costBasis == Decimal(100)
    lot.SellShares(5)
    assert lot.costBasis == Decimal(50)
    lot.SellShares(4)
    assert lot.costBasis == Decimal(10)
    lot.SellShares(1)
    assert lot.costBasis == Decimal(0)