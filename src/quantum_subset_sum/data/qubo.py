from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

import pyqubo

from quantum_subset_sum.data import Numbers, Mask, Target, Solution

Weights = np.ndarray
Biases = np.ndarray


@dataclass
class QUBO:
    weights: Weights
    biases: Biases
    qubo_model: pyqubo.Model = None

    def __post_init__(self):
        n, k = self.weights.shape
        assert n == k
        assert len(self.biases) == n

    @classmethod
    def from_numbers_and_target_sum(cls, numbers: Numbers, target: Target) -> QUBO:
        weights, biases = qubo_weights_from_numbers_and_target(numbers, target)
        qubo_model = qubo_model_from_numbers_and_target(numbers, target)
        return cls(weights, biases, qubo_model)

    def __len__(self):
        return len(self.biases)

    def to_hopfield(self) -> Tuple[Weights, Biases]:
        return hopfield_weights_from_qubo_weights(self.weights, self.biases)

    def to_upper_triangular(self) -> Weights:
        # matrix must be symmetric
        assert np.all(self.weights == self.weights.T)
        # matrix must have 0 diagonal
        assert np.all(np.diag(self.weights)) == 0

        # set lower triangular values to 0
        upper_triangular = np.triu(self.weights)
        # double upper triangular weights (i.e. add lower weights to upper weights)
        upper_triangular *= 2
        # fill diagonal with biases
        np.fill_diagonal(upper_triangular, self.biases)
        return upper_triangular


def qubo_model_from_numbers_and_target(numbers: Numbers, target: Target) -> pyqubo.Model:
    binaries = [pyqubo.Binary(index_to_binary(i)) for i in range(len(numbers))]
    expression = (sum([numbers[i] * binaries[i] for i in range(len(numbers))]) - target) ** 2
    model = expression.compile()
    return model


def binary_to_index(binary: str) -> int:
    return int(binary.replace('x', ''))


def index_to_binary(index: int) -> str:
    return f'x{str(index).zfill(3)}'


def qubo_weights_from_numbers_and_target(numbers, target, move_squares_into_biases: bool = True):
    numbers = numbers.astype(np.float)
    target = target.astype(np.float)
    weights = np.outer(numbers, numbers)
    biases = - 2 * target * numbers
    if move_squares_into_biases:
        biases = biases + np.diag(weights)
        np.fill_diagonal(weights, 0)
    return weights, biases


def hopfield_weights_from_numbers_and_target(numbers, target):
    qubo_weights, qubo_biases = qubo_weights_from_numbers_and_target(numbers, target, move_squares_into_biases=False)
    return hopfield_weights_from_qubo_weights(weights=qubo_weights, biases=qubo_biases)


def hopfield_weights_from_qubo_weights(weights, biases):
    _weights = -2 * 0.25 * weights
    np.fill_diagonal(_weights, 0)
    _biases = 0.50 * (weights @ np.ones_like(biases) + biases)
    return _weights, _biases
