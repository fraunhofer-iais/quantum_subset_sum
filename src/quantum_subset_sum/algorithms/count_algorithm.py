from typing import List, Iterator
import numpy as np

import multiprocessing as mp
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool
from math import log2
from functools import partial
from multiprocessing import Queue

from quantum_subset_sum.data import Numbers, Target, Solution
from quantum_subset_sum.utils.hopfield_helper import yield_binary_masks
from quantum_subset_sum.utils.helper import mask_to_indices
from quantum_subset_sum.algorithms.base_algorithm import BaseAlgorithm


class CountAlgorithm(BaseAlgorithm):
    verbose = False

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __str__(self):
        return 'count'

    def run(self, numbers: Numbers, target: Target, num_workers: int = 1,
            break_at_first_solution: bool = False,
            max_iterator: int = None,
            ground_truth_solution: Solution = None) -> Iterator[Solution]:
        if num_workers == 1:
            for solution in self._run_single_worker(numbers, target, max_iterator):
                yield solution
        else:
            for solution in self._run_multiprocessing(numbers, target, num_workers,
                                                      break_at_first_solution,
                                                      max_iterator, ground_truth_solution):
                yield solution

    def _run_single_worker(self, numbers: Numbers, target: Target, max_iterator: int = None) -> Iterator[Solution]:
        n = len(numbers)

        total = 2 ** n
        if self.verbose:
            print(f"Checking {total:,} combinations!")

        for idx, mask in tqdm(enumerate(yield_binary_masks(n)), total=total, disable=not self.verbose):
            subset = numbers[mask]
            if subset.sum() == target:
                solution = tuple(mask_to_indices(mask))
                yield solution, idx
                if self.verbose:
                    tqdm.write(f"Combination {idx:,} ({idx * 100 / total:.2f}%): {solution}")
            if max_iterator is not None and idx > max_iterator:
                break

    def _run_multiprocessing(self, numbers: Numbers, target: Target, num_workers: int,
                             break_at_first_solution: bool = False,
                             max_iterator: int = None,
                             ground_truth_solution: Solution = None) -> Iterator[Solution]:
        # At the moment calculates all solutions and then yields them all at once.
        assert num_workers in [2 ** i for i in range(10)]
        preface_n = int(log2(num_workers))
        if self.verbose:
            print(f'Running with {num_workers} workers and a preface length of {preface_n}.')

        n = len(numbers)
        if self.verbose:
            print(f'Checking {2 ** n:,} combinations in total.')
            print(f'Each worker checks {2 ** (n - preface_n):,} combinations.')

        manager = mp.Manager()
        solutions = manager.Queue()
        pool = Pool()
        for worker_index, preface_mask in enumerate(yield_binary_masks(preface_n)):
            if self.verbose:
                print(f'Starting pool {worker_index} of {num_workers} with preface mask {preface_mask}.')
            pool.apply_async(self._run_partial_algorithm,
                             args=(numbers, target, preface_mask, n - preface_n,
                                   solutions,
                                   worker_index,
                                   break_at_first_solution,
                                   max_iterator//num_workers,
                                   ground_truth_solution))
        pool.close()
        pool.join()
        while True:
            if solutions.empty():
                break
            solution, idx = solutions.get(timeout=1)
            yield solution, idx * num_workers

    def _run_partial_algorithm(self, numbers: Numbers, target: Target, preface_mask: np.ndarray, binary_digits: int,
                               solutions: Queue,
                               worker_index: int = 0,
                               break_at_first_solution: bool = False,
                               max_iterator: int = None,
                               ground_truth_solution: Solution = None) -> None:
        total = 2 ** binary_digits
        for idx, mask in tqdm(enumerate(yield_binary_masks(binary_digits)),
                              desc=f'Worker {worker_index} progress',
                              disable=(worker_index != 0), miniters=10_000, total=total):
            # check queue every 10_000 steps to decrease processing time
            if break_at_first_solution and idx % 10_000 == 0:
                if not solutions.empty():
                    break

            if max_iterator is not None and idx > max_iterator:
                break

            mask = np.concatenate([preface_mask, mask])
            subset = numbers[mask]
            if subset.sum() == target:
                solution = tuple(mask_to_indices(mask))
                if ground_truth_solution is not None:
                    if solution != ground_truth_solution:
                        continue
                solutions.put((solution, idx + 1))
                if self.verbose:
                    tqdm.write(f"Worker {worker_index} - Combination {idx} ({idx * 100 / total:.2f}%): {solution}")
                if break_at_first_solution:
                    if self.verbose:
                        print(f'Worker {worker_index}: found solution {solution}, breaking!')
                    break
