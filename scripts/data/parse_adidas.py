from typing import List, Dict, Tuple
import json
import os
import logging
from collections import Counter

from decimal import Decimal
import numpy as np
from quantum_subset_sum import project_path
from quantum_subset_sum.data.subset_sum import TableSumStructure, SubsetSumProblem
from quantum_subset_sum.data.qubo import QUBO
from table_check.table import Table
from table_check.vector_tree_parser import VectorTreeParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

numbers_path = os.path.join(project_path, "data/financial_documents/adidas_2020_small.csv")
structure_path = os.path.join(project_path, "data/financial_documents/adidas2020.json")
output_file = os.path.join(project_path, 'data/financial_documents/parsed_adidas.jsonl')
qubos_file = os.path.join(project_path, 'data/financial_documents/qubos_adidas.jsonl')

factor = 1
with open(numbers_path, 'r') as f:
    numbers = f.read().splitlines()
numbers = [Decimal(float(x) * factor) for x in numbers]

# rows of the sheet that contain sums, with corresponding solution
sum_indices_excel = {
    5: [1, 3],
    6: [2, 4],
    13: [7, 8, 9, 11],
    14: [7, 8, 10, 12],
    16: [13, 15],
    17: [14, 15]
}
# excel indexing starts at 1, here at 0
sum_indices = {(i - 1): [j - 1 for j in values] for i, values in sum_indices_excel.items()}

message = ''
for sum_index, children_indices in sum_indices.items():
    message += f'\t{sum_index:<2} = {" + ".join([str(i) for i in children_indices])}\n'
logger.info(f'Starting parsing. Looking for this sum structure:\n' + message)

def convert_to_subset_sum(numeric_vector, sum_index, children_indices, name, group) -> SubsetSumProblem:
    children_indices = [index if index < sum_index else index - 1 for index in children_indices]
    sum_value = numeric_vector[sum_index]
    output_vector = numeric_vector[:sum_index] + numeric_vector[sum_index + 1:]
    if not sum([output_vector[i] for i in children_indices]) == sum_value:
        sum_value = sum([output_vector[i] for i in children_indices])
    return SubsetSumProblem(name=name, group=group,
                            numbers=output_vector, target=sum_value,
                            ground_truth_solution=children_indices)


output: List[SubsetSumProblem] = []
correct_counter = Counter()
incorrect_counter = Counter()
for i, (sum_index, children_indices) in enumerate(sum_indices.items()):
    sum_value = numbers[sum_index]
    children_values = [numbers[index] for index in children_indices]
    subset_sum_problem = convert_to_subset_sum(numbers, sum_index, children_indices,
                                               name=f'adidas_{str(i).zfill(2)}',
                                               group='adidas')
    if subset_sum_problem is not None:
        output.append(subset_sum_problem)

logger.info(f'Writing into output file {output_file}')
with open(output_file, 'w') as f:
    for subset_sum_problem in output:
        f.write(subset_sum_problem.to_json() + '\n')

with open(qubos_file, 'w') as f:
    for subset_sum_problem in output:
        numbers = np.array([float(x) for x in subset_sum_problem.numbers])
        target = np.array(float(subset_sum_problem.target))
        qubo = QUBO.from_numbers_and_target_sum(numbers, target)
        matrix = qubo.to_upper_triangular()
        output = {'matrix': matrix.tolist(),
                  'solution': subset_sum_problem.ground_truth_solution}
        f.write(json.dumps(output) + '\n')

