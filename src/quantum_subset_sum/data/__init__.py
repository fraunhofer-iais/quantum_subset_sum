from typing import Union, List, Tuple
import numpy as np

# from .results import Result, Results
# from .qubo import QUBO
# from hopfield import HopfieldNetwork

Numbers = Union[List[float], np.ndarray]
Target = Union[float, np.ndarray]
Table = Union[List[float], np.ndarray]
# Solution is a tuple of indices, not actual numbers, to account for duplicate numbers.
# E.g. numbers   = [1, 2, 2, 3]
#      target    = 3
#      solutions = [(0, 1), (0, 2)]
Solution = Union[Tuple[int], List[int]]
Mask = Union[List[bool], np.ndarray]

