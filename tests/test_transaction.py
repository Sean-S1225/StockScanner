import pytest
from StockScreener.Portfolio import Transaction
from StockScreener.enums import TransactionSide
from decimal import Decimal
from datetime import datetime

@pytest.mark.parametrize("transaction_input", [
    (TransactionSide.Buy, Decimal("10"), Decimal("5"), "hi", datetime(2026, 1, 1)),
    (TransactionSide.Sell, Decimal("10"), Decimal("5.5"), "hi", datetime(2026, 1, 1))
])
def test_transaction_valid_initialization(transaction_input):
    side, shares, fillPrice, reason, date = transaction_input
    transaction = Transaction(*transaction_input)
    assert isinstance(transaction.side, TransactionSide)
    assert isinstance(transaction.shares, Decimal)
    assert isinstance(transaction.fillPrice, Decimal)
    assert isinstance(transaction.reason, str)
    assert isinstance(transaction.date, datetime)

    assert transaction.side == side
    assert transaction.shares == Decimal(str(shares))
    assert transaction.fillPrice == Decimal(str(fillPrice))
    assert transaction.reason == reason
    assert transaction.date == date

@pytest.mark.parametrize("transaction_input", [
    (TransactionSide.Buy, None, Decimal("5"), "hi", datetime(2026, 1, 1)),
    (TransactionSide.Buy, Decimal("10"), None, "hi", datetime(2026, 1, 1)),
])
def test_transaction_invalid_decimal_initialization_raises(transaction_input):
    with pytest.raises(ValueError):
        Transaction(*transaction_input)


@pytest.mark.parametrize("transaction_input", [
    ("buy", Decimal("10"), Decimal("5"), "hi", datetime(2026, 1, 1)),
    (TransactionSide.Buy, Decimal("10"), Decimal("5"), 20, datetime(2026, 1, 1)),
    (TransactionSide.Buy, Decimal("10"), Decimal("5"), "hi", "hi"),
])
def test_transaction_invalid_typed_fields_raise(transaction_input):
    with pytest.raises(ValueError):
        Transaction(*transaction_input)

