from typing import Callable, Any, Tuple, Dict
from datetime import datetime as dt
from inspect import isgeneratorfunction

Seconds = float


def benchmark_function(function: Callable, *args, **kwargs) -> Tuple[Any, Seconds]:
    start = dt.now()
    if isgeneratorfunction(function):
        output = [x for x in function(*args, **kwargs)]
    else:
        output = function(*args, **kwargs)
    seconds = (dt.now() - start).total_seconds()
    return output, seconds
