from decimal import Decimal
from collections.abc import Callable

def EnsureDecimal(*args, tryCasting: bool = False):
    def RaiseError():
        raise ValueError(f"Cannot convert type {type(arg)} to Decimal: {arg=}")
    
    toReturn = []
    for arg in args:
        if not isinstance(arg, Decimal):
            if tryCasting:
                try:
                    toReturn.append(Decimal(str(arg)))
                except:
                    RaiseError()
            else:
                RaiseError()
                
        else:
            toReturn.append(arg)

    return tuple(toReturn)

def EnsureType(x, typeShouldBe: type, name: str, tryCasting: bool = False, condition: Callable | None = None):
    def EnsureCondition(x, condition: Callable):
        if condition(x):
            return x
        else:
            raise ValueError("The condition did not pass.")
        
    if condition is None:
        condition = lambda x: True
    
    if isinstance(x, typeShouldBe):
        return EnsureCondition(x, condition)
    else:
        if tryCasting:
            try:
                x = typeShouldBe(x)
                return EnsureCondition(x, condition)
            except:
                raise ValueError(f"Cannot cast type type({name})={type(x)} to {typeShouldBe}")
        raise ValueError(f"type({name})={type(x)} is not {typeShouldBe=}")
