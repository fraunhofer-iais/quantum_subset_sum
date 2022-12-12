from typing import List, Dict

from dataclasses import dataclass
from dataclasses_json import dataclass_json

import numpy as np

from quantum_subset_sum.data import Numbers, Solution, Target


SumStructure = Dict[int, List[int]]


@dataclass_json
@dataclass
class SubsetSumProblem:
    name: str
    group: str
    numbers: Numbers
    target: Target
    ground_truth_solution: Solution
    config: dict = None


@dataclass_json
@dataclass
class TableSumStructure:
    original_vector: List
    numeric_vector: List
    sum_structures_rows: SumStructure
    sum_structures_indices: SumStructure
    subset_sum_problems: List[SubsetSumProblem]
