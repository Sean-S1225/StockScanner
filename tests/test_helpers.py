import pytest
from decimal import Decimal
from StockScreener.helpers import *

def test_ensure_decimal():
    a, b, c = EnsureDecimal(Decimal("1.1"), 2, "3.3")
    assert isinstance(a, Decimal)
    assert isinstance(b, Decimal)
    assert isinstance(c, Decimal)

    assert a == Decimal("1.1")
    assert b == Decimal("2")
    assert c == Decimal("3.3")

@pytest.mark.parametrize("bad_decimal", [None, "hi", [1, 2, 3]])
def test_invalid_ensure_decimal_raises(bad_decimal):
    with pytest.raises(ValueError):
        EnsureDecimal(bad_decimal)

@pytest.mark.parametrize("ensureType", [(1, int), ("hi", str), (1.1, float)])
def test_ensure_type_no_change(ensureType):
    x, typeShouldBe = ensureType
    newX = EnsureType(x, typeShouldBe)
    assert type(newX) == typeShouldBe
    assert newX == x

@pytest.mark.parametrize("ensureType", [
    (1, int, 1), ("hi", str, "hi"), (1.1, float, 1.1), ("1", int, 1), ("2.2", Decimal, Decimal("2.2"))
])
def test_ensure_type_cast(ensureType):
    x, typeShouldBe, output = ensureType
    newX = EnsureType(x, typeShouldBe, tryCasting=True)
    assert type(newX) == typeShouldBe
    assert newX == output

@pytest.mark.parametrize("ensureType", [(1, str), ("hi", float), (1.1, int)])
def test_invalid_ensure_type_no_change_raises(ensureType):
    with pytest.raises(ValueError):
        x, typeShouldBe = ensureType
        EnsureType(x, typeShouldBe)

@pytest.mark.parametrize("ensureType", [
    (None, int), ([1, 2], float)
])
def test_invalid_ensure_type_cast_raises(ensureType):
    with pytest.raises(ValueError):
        x, typeShouldBe = ensureType
        EnsureType(x, typeShouldBe, tryCasting=True)
