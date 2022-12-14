import numpy as np
from typing import Callable

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize the vector.

    Args:
        x: a vector
    
    Return:
        norm: normalized vector.
    """

    return x / np.linalg.norm(x)


def partition(lst: list, first: int, last: int, fun: Callable) -> int:
    """Python version of std::partition.
    """

    if first == last:
        return first

    for i in range(first, last):
        if fun(lst[i]):
            lst[first], lst[i] = lst[i], lst[first]
            first += 1
    # print(lst)
    return first