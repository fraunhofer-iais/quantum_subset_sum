from typing import Tuple, List
import numpy as np
from collections import Counter
from functools import partial

from tqdm import tqdm as tqdm_std

from quantum_subset_sum.data import Numbers, Mask, Target


tqdm = partial(tqdm_std, dynamic_ncols=True)


def sample_numbers_and_target(n: int, k: int, min_value: int, max_value: int, seed: int = 1) -> Tuple[Numbers, Mask, Target]:
    """
    Sample numbers, binary masks and target sums to test the algorithms.
    List of numbers is returned sorted.
    By default, duplicate numbers are not allowed.
    """
    np.random.seed(seed)
    numbers = np.sort(_sample_numbers(n, min_value, max_value))
    mask_indices = np.random.choice(list(range(n)), size=k, replace=False)
    mask = np.zeros(n)
    mask[mask_indices] = 1
    mask = mask.astype(bool)
    target_sum = numbers[mask].sum()
    return numbers, mask, target_sum


def _sample_numbers(n, min_value, max_value):
    interval_size = max_value - min_value
    numbers = np.random.rand(n)
    numbers = numbers * interval_size
    numbers = numbers.astype(int) + min_value
    return numbers

def unique_list_of_lists(data: List[List]) -> List[List]:
    return [list(x) for x in set(tuple(x) for x in data)]


def mask_to_indices(mask: Mask) -> List[int]:
    return np.where(mask)[0].tolist()


def count_solutions(solutions: List) -> Counter:
    return Counter(solutions)
