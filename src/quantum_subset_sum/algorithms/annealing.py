from typing import List, Tuple, Dict, Iterator

import neal

from collections import Counter

from quantum_subset_sum.data.qubo import QUBO, binary_to_index
from quantum_subset_sum.data import Numbers, Mask, Target, Solution
from quantum_subset_sum.utils.helper import unique_list_of_lists
from quantum_subset_sum.algorithms.base_algorithm import BaseAlgorithm


class AnnealingAlgorithm(BaseAlgorithm):

    def __init__(self, verbose: bool = True, assert_correctness: bool = True):
        self.verbose = verbose
        self.assert_correctness = assert_correctness

    def __str__(self):
        return 'annealing'

    def run(self, numbers: Numbers, target: Target, samples: int = 100, **kwargs) -> Iterator[Solution]:
        qubo = QUBO.from_numbers_and_target_sum(numbers, target)
        model = qubo.qubo_model
        bqm = model.to_bqm()

        sampler = neal.SimulatedAnnealingSampler()
        if self.verbose:
            print(f'Sampling {samples} samples...')
        sampleset = sampler.sample(bqm, num_reads=samples)
        if self.verbose:
            print(f'Decoding...')
        decoded_samples = model.decode_sampleset(sampleset)
        if self.verbose:
            print(f'Calculate minimal energy...')
        min_energy = min([sample.energy for sample in decoded_samples])
        if self.verbose:
            print(f'Getting best samples...')
        best_samples: List[Dict] = [sample.sample for sample in decoded_samples if sample.energy == min_energy]
        if self.verbose:
            print(f'Extracting solutions...')
        solutions = [self.solution_from_sample(sample) for sample in best_samples]
        # if len(solutions) == 0:
        #     return []
        if self.assert_correctness:
            if not self.assert_solution_correctness(numbers, target, solutions):
                return []
        for solution in solutions:
            yield solution

    def assert_solution_correctness(self, numbers: Numbers, target: Target, solutions: List[Solution]) -> bool:
        found_sum = numbers[list(solutions[0])].sum()
        if found_sum != target:
            if self.verbose:
                print(f'Best solution {found_sum} does not match target {target}. Returning empty counter.')
            return False
        return True

    @staticmethod
    def solution_from_sample(sample) -> Tuple[int]:
        solution = tuple(sorted([binary_to_index(key) for key, value in sample.items() if value == 1]))
        return solution
