from typing import List, Tuple, Dict
import datetime
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dataclasses_json import dataclass_json
from quantum_subset_sum.data import Numbers, Solution, Target
from quantum_subset_sum.data.subset_sum import SubsetSumProblem


@dataclass_json
@dataclass
class Result:
    solution: Solution
    problem: SubsetSumProblem
    total_seconds: float
    start_time: str
    end_time: str
    steps_per_run: int
    runs: int
    algo: str
    args: dict


Results = List[Result]
