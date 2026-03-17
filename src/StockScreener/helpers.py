from decimal import Decimal

def EnsureDecimal(*args):
    toReturn = []
    for arg in args:
        if not isinstance(arg, Decimal):
            try:
                toReturn.append(Decimal(str(arg)))
            except:
                raise ValueError(f"Cannot convert type {type(arg)} to Decimal: {arg=}")
        else:
            toReturn.append(arg)

    return tuple(toReturn)

def EnsureType(x, typeShouldBe: type, name: str, tryCasting: bool = False):
    if isinstance(x, typeShouldBe):
        return x
    else:
        if tryCasting:
            try:
                x = typeShouldBe(x)
                return x
            except:
                raise ValueError(f"Cannot cast type type({name})={type(x)} to {typeShouldBe}")
        raise ValueError(f"type({name})={type(x)} is not {typeShouldBe=}")
