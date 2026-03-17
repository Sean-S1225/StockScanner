import pytest
from decimal import Decimal
from StockScreener.Position import PnL

@pytest.mark.parametrize("pnl_input", [
    (Decimal("10"), Decimal("5"), Decimal("7"), Decimal("14")),
    (10, 5, 7, 14),
    (10.1, 10.2, 11.3, 11.4),
    (-1, -2.2, Decimal("-3"), Decimal("0"))
])
def test_pnl_initialization(pnl_input):
    pnl = PnL(*pnl_input)
    assert isinstance(pnl.realizedPnL, Decimal)
    assert isinstance(pnl.realizedPnL_percent, Decimal)
    assert isinstance(pnl.unrealizedPnL, Decimal)
    assert isinstance(pnl.unrealizedPnL_percent, Decimal)

    assert pnl.realizedPnL == Decimal(str(pnl_input[0]))
    assert pnl.realizedPnL_percent == Decimal(str(pnl_input[1]))
    assert pnl.unrealizedPnL == Decimal(str(pnl_input[2]))
    assert pnl.unrealizedPnL_percent == Decimal(str(pnl_input[3]))

@pytest.mark.parametrize("pnl_input", [
    (None, Decimal("5"), Decimal("7"), Decimal("14")),
    (10, None, 7, 14),
    (10.1, 10.2, None, 11.4),
    (-1, -2.2, Decimal("-3"), None)
])
def test_invalid_pnl_initialization_raises(pnl_input):
    with pytest.raises(ValueError):
        PnL(*pnl_input)