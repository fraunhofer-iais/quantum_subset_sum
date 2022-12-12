from typing import List, Tuple, Counter, Union, Optional, Iterator

import multiprocessing as mp
from functools import partial

from quantum_subset_sum.data import Numbers, Target, Mask, Solution
from quantum_subset_sum.utils.benchmark import benchmark_function, Seconds
from quantum_subset_sum.utils.helper import count_solutions


class BaseAlgorithm:

    def run(self, numbers: Numbers, target: Target, *args, **kwargs) -> Iterator[Solution]:
        raise NotImplementedError()

    def benchmark(self, *args, **kwargs) -> Tuple[List[Solution], Seconds]:
        solutions, seconds = benchmark_function(self.run, *args, **kwargs)
        return solutions, seconds

    def run_multiprocessing(self, num_workers, *args, **kwargs):
        solution_queue = mp.Queue()
        iolock = mp.Lock()
        partial_function = partial(self.run_partial, *args, **kwargs)
        pool = mp.Pool(num_workers, initializer=partial_function, initargs=(solution_queue, iolock))
        while True:
            solution = solution_queue.get()
            yield solution
            if self.break_condition(solution):
                break

    def run_partial(self, solution_queue, iolock, *args, **kwargs):
        raise NotImplementedError()

    def break_condition(self, solution):
        return False

    def __str__(self):
        raise NotImplementedError
