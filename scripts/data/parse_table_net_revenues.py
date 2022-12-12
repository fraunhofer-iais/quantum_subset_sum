from typing import List, Dict, Tuple
import json
import os
import logging
from collections import Counter

from quantum_subset_sum import project_path
from quantum_subset_sum.data.subset_sum import TableSumStructure, SubsetSumProblem
from table_check.table import Table
from table_check.vector_tree_parser import VectorTreeParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_path = os.path.join(project_path, "data/financial_documents/4Q20_FDS.xlsx")
output_file = os.path.join(project_path, 'data/financial_documents/parsed_net_revenues.jsonl')

factor = 100_000_000
tab = Table()
tab.from_excel(file_path, sheetname="NetRevenues")

# rows of the sheet that contain sums, with corresponding solution
sum_indices_excel = [
    (8, [6, 7]),
    (8, [10, 11, 12]),
    (18, [15, 16, 17]),
    (25, [21, 22]),
    (22, [23, 24]),
    (25, [27, 28, 29]),
    (35, [32, 33, 34]),
    (39, [8, 18, 25, 35, 37]),
    (43, [39, 41])
]
# excel indexing starts at 1, here at 0
sum_indices = [((i - 1), [j - 1 for j in values]) for i, values in sum_indices_excel]

message = ''
for sum_index, children_indices in sum_indices:
    message += f'\t{sum_index:<2} = {" + ".join([str(i) for i in children_indices])}\n'
logger.info(f'Starting parsing. Looking for this sum structure:\n' + message)


def is_row_vector(vector):
    cols = set([x['col'] for x in vector['index_mapping']])
    return len(cols) == 1


newlinetab = '\n\t'
numeric_vectors = tab.get_numeric_columns_rows()
numeric_vectors = [v for v in numeric_vectors if is_row_vector(v)]
logger.info(f'Found {len(numeric_vectors)} numeric row vectors.\n'
            f'Removing last 3 rows because they only contain percentages.\n'
            f'Printing first row:'
            f'{newlinetab + newlinetab.join([f"{float(value):>18}" for value in numeric_vectors[0]["numeric_vector"]])}'
            f'\n')
for vector in numeric_vectors:
    vector['numeric_vector'] = [int(factor * x) for x in vector['numeric_vector']]
logger.info(f'Multiplying all numbers by factor {factor:,} to convert from float to int.\n'
            f'Printing first row as ints:'
            f'{newlinetab + newlinetab.join([f"{value:>18,}" for value in numeric_vectors[0]["numeric_vector"]])}'
            f'\n')


def get_numeric_vector_index_to_row(index_mapping):
    return [x['row'] for x in index_mapping]


def get_row_to_numeric_vector_index(index_mapping, length=None):
    numeric_row_to_index = {x['row']: i for i, x in enumerate(index_mapping)}
    if length is None:
        length = max(numeric_row_to_index.keys()) + 1
    row_to_index = {row: numeric_row_to_index.get(row, None) for row in range(length)}
    return row_to_index


def convert_to_subset_sum(numeric_vector, sum_index, children_indices, name, group) -> SubsetSumProblem:
    children_indices = [index if index < sum_index else index - 1 for index in children_indices]
    sum_value = numeric_vector[sum_index]
    output_vector = numeric_vector[:sum_index] + numeric_vector[sum_index + 1:]
    assert sum([output_vector[i] for i in children_indices]) == sum_value
    return SubsetSumProblem(name=name, group=group,
                            numbers=output_vector, target=sum_value,
                            ground_truth_solution=children_indices)


total_output: List[TableSumStructure] = []
correct_counter = Counter()
incorrect_counter = Counter()
for i, vector in enumerate(numeric_vectors[:11]):
    message = f'\n= = = = = = = = = = = = = = = = = = = = = = = =\n'\
                     f'Vector #{i}:\n'\
                     f'Length of original vector                = {len(vector["original_vector"])}\n'\
                     f'Length of numerical vector without zeros = {len(vector["numeric_vector"])}\n'
    numeric_vector = vector['numeric_vector']
    row_to_numeric_vector_index = get_row_to_numeric_vector_index(vector['index_mapping'])
    numeric_vector_index_to_row = get_numeric_vector_index_to_row(vector['index_mapping'])

    vector_output = TableSumStructure(
        original_vector=vector['original_vector'],
        numeric_vector=vector['numeric_vector'],
        sum_structures_rows={},
        sum_structures_indices={},
        subset_sum_problems=[]
    )
    for sum_row, children_rows in sum_indices:
        sum_index = row_to_numeric_vector_index[sum_row]
        sum_value = numeric_vector[sum_index]
        children_indices = [row_to_numeric_vector_index[row] for row in children_rows]
        children_indices = [index for index in children_indices if index is not None]
        children_values = [numeric_vector[index] for index in children_indices]
        # if sum_value is None:

        if sum(children_values) == sum_value:
            message += f'\tSum is correct for {sum_row} = {" + ".join([str(i) for i in children_rows])}\n'
            correct_counter[sum_row] += 1
            vector_output.sum_structures_rows[sum_row] = children_rows
            vector_output.sum_structures_indices[sum_index] = children_indices
            vector_output.subset_sum_problems.append(
                convert_to_subset_sum(numeric_vector, sum_index, children_indices,
                                      name=f'column_{i}_row_{sum_row}',
                                      group='net_revenues'))
        else:
            message += f'\tSum is incorrect for {sum_row} = {" + ".join([str(i) for i in children_rows])},\n' \
                       f'\t    with difference {abs(sum_value - sum(children_values))} :(\n'
            incorrect_counter[sum_row] += 1

    total_output.append(vector_output)
    logger.info(message)

message = f'Reminder: Looking for this sum structure:\n'
for sum_index, children_indices in sum_indices:
    message += f'\t{sum_index:<2} = {" + ".join([str(i) for i in children_indices])}\n'

for sum_row, _ in sum_indices:
    correct = correct_counter[sum_row]
    incorrect = incorrect_counter[sum_row]
    message += f'Sum in row {sum_row:>2}:   {correct:>2} correct, {incorrect:>2} incorrect, ratio {correct / (correct + incorrect):.1%}\n'
logger.info(message)

total_problems: List[SubsetSumProblem] = [problem for vector in total_output for problem in vector.subset_sum_problems]
logger.info(f'Found a total of {len(total_problems)} problems in the parsed file.')

logger.info(f'Writing into output file {output_file}')
with open(output_file, 'w') as f:
    for vector_output in total_output:
        f.write(vector_output.to_json() + '\n')
