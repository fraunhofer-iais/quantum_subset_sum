from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from quantum_subset_sum.data import Numbers, Mask, Target, Solution
from quantum_subset_sum.data.qubo import Weights, Biases, QUBO
from quantum_subset_sum.utils.hopfield_helper import signum

State = np.ndarray


@dataclass
class HopfieldNetwork:
    state: State
    weights: Weights
    biases: Biases

    @classmethod
    def from_qubo(cls, qubo: QUBO, verbose: bool = True) -> HopfieldNetwork:
        n = len(qubo)
        state = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
        weights, biases = qubo.to_hopfield()
        return cls(state, weights, biases)

    @property
    def energy(self) -> float:
        return hopfield_energy(self.state, self.weights, self.biases)

    def print_timestep(self, time) -> None:
        state = ' '.join(['+' if x >= 0 else '-' for x in self.state])
        energy = self.energy
        print('{:4d}  {}  {:+.1f}'.format(time, state, energy))

    def __len__(self) -> int:
        return len(self.state)

    def step_random(self) -> None:
        idx = np.random.randint(0, len(self))
        self.state[idx] = signum(self.weights[idx] @ self.state - self.biases[idx])

    @property
    def gradient(self) -> None:
        return self.weights @ self.state - self.biases

    def step_gradient(self) -> None:
        gradient = self.gradient
        delta_h = self.state * gradient
        update_index = np.argmin(delta_h)
        self.state[update_index] = signum(gradient[update_index])

    def step_synchronous(self) -> None:
        self.state = signum(self.weights @ self.state - self.biases)

    @property
    def mask(self) -> Mask:
        return np.where(self.state > 0, True, False)


def hopfield_energy(state, weights, bias) -> float:
    return -0.5 * state @ weights @ state + bias @ state
