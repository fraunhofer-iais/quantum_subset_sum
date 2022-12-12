from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterator
import numpy as np

from tqdm import tqdm
import torch

from quantum_subset_sum.data import Numbers, Target, Solution
from quantum_subset_sum.data.hopfield import HopfieldNetwork
from quantum_subset_sum.utils.hopfield_helper import signum
from quantum_subset_sum.data.qubo import QUBO
from quantum_subset_sum.utils.helper import mask_to_indices
from quantum_subset_sum.algorithms.base_algorithm import BaseAlgorithm


class HopfieldAlgorithmTorch(BaseAlgorithm):

    def run_partial(self, solution_queue, iolock, *args, **kwargs):
        raise NotImplementedError

    def __init__(self, verbose: bool = True, device: str = 'cuda'):
        self.verbose = verbose
        self.device = torch.device(device)

    def __str__(self):
        return 'hopfield_torch'

    def _init_net(self, numbers, target):
        return HopfieldNetwork.from_qubo(QUBO.from_numbers_and_target_sum(numbers, target))

    def run(self, numbers, target, steps, runs, batch_size, max_error: float = 0., device: str = 'cuda'):
        device = torch.device(device)
        batches = runs // batch_size
        net = self._init_net(numbers, target)

        weights = torch.tensor(net.weights).to(device).view(len(net), len(net)).type(torch.float)
        biases = torch.tensor(net.biases).to(device).view(len(net), 1).type(weights.type())
        target = torch.tensor(target).type(weights.type())

        for batch_index in tqdm(range(batches), dynamic_ncols=True):
            states = 2 * np.random.binomial(n=1, p=0.5, size=batch_size * len(net)) - 1
            states = torch.tensor(states).to(device).view(batch_size, len(net), 1).type(weights.type())
            for _ in tqdm(range(steps), disable=not self.verbose):
                states = self.step_gradient(weights, states, biases)
            masks = states > 0
            subsets = torch.stack([torch.tensor(numbers)] * batch_size).unsqueeze(2).to(device) * masks
            sums = subsets.sum(dim=1)
            solution_indices = torch.where(torch.abs(sums - target) <= max_error)[0]
            for mask in masks[solution_indices]:
                solution = torch.where(mask)[0]
                # output solution and count of runs so far
                yield tuple(solution.tolist()), (batch_index + 1) * batch_size

    def gradient(self, weights, state, biases):
        return torch.matmul(weights, state) - biases

    def step_gradient(self, weights, state, biases):
        grad = self.gradient(weights, state, biases)
        delta_h = state * grad
        update_indices = torch.argmin(delta_h, dim=1)[:, 0]
        state[:, update_indices, 0] = self.signum(grad[:, update_indices, 0]).type(state.type())
        return state

    def signum(self, x):
        return torch.where(x >= 0, 1., -1.)


class HopfieldAlgorithm(BaseAlgorithm):

    def __init__(self, verbose: bool = True, update: str = 'gradient'):
        self.verbose = verbose
        assert update in ['gradient', 'random', 'synchronous']
        self.update = update

    def __str__(self):
        return 'hopfield'

    def run(self, numbers: Numbers, target: Target, runs: int, steps: int = None) -> Iterator[Solution]:
        steps = steps or len(numbers) // 2
        for run in tqdm(range(runs)):
            net = self._init_net(numbers, target)
            self._run_net(net, numbers=numbers, target=target, time_max=steps)
            subset_sum = np.sum(numbers[net.mask])
            if self.verbose:
                print(f"Run {str(run).zfill(2): <3}: Target Value {target: <5}, Subset Sum {subset_sum: <5}")
            if subset_sum == target:
                solution = tuple(mask_to_indices(net.mask))
                yield solution
                if self.verbose:
                    print(f"Run {str(run).zfill(2): <3}: Found solution {solution}")

    def _run_net(self, net: HopfieldNetwork, numbers, target, time_max):
        if self.update == 'gradient':
            self.run_gradient(net, time_max, numbers, target)
        elif self.update == 'random':
            self.run_random(net, time_max)
        elif self.update == 'synchronous':
            self.run_synchronous(net, time_max)
        else:
            raise NotImplementedError

    def _init_net(self, numbers, target):
        return HopfieldNetwork.from_qubo(QUBO.from_numbers_and_target_sum(numbers, target), self.verbose)

    def run_gradient(self, net, time_max, numbers: Numbers = None, target_sum: Target = None):
        for time in range(time_max):
            net.step_gradient()
            if numbers is not None:
                subset_sum = numbers[net.mask].sum()
                if subset_sum == target_sum:
                    return
            if self.verbose:
                net.print_timestep(time)

    def run_synchronous(self, net, time_max):
        for time in range(time_max):
            net.step_synchronous()
            if self.verbose:
                net.print_timestep(time)

    def run_random(self, net, time_max):
        for time in range(time_max):
            net.step_random()
            if self.verbose:
                net.print_timestep(time)


if __name__ == '__main__':
    pass
