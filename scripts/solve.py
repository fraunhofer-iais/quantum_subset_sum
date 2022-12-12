from typing import List, Dict, Tuple
import numpy as np
import argparse
import os
import glob
from datetime import datetime as dt
import json
from tqdm import tqdm

from quantum_subset_sum.data.results import Result, Results
from quantum_subset_sum import project_path
from quantum_subset_sum.algorithms import RoundingAlgorithm, HopfieldAlgorithm, HopfieldAlgorithmTorch, CountAlgorithm
from quantum_subset_sum.data.subset_sum import TableSumStructure, SubsetSumProblem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-type', type=str, choices=['financial', 'artificial'])
    parser.add_argument('-o', '--output-file', required=True)
    parser.add_argument('-a', '--algorithm', type=str, choices=['hopfield', 'hopfield_rounded', 'count'])
    parser.add_argument('--ground-truth-solution-only', action='store_true')

    parser.add_argument('-r', '--runs', type=int, default=1_000_000)
    parser.add_argument('-b', '--batch-size', type=int, default=10_000)
    parser.add_argument('-s', '--steps', type=int, default=100)
    parser.add_argument('-rf', '--rounding-factor', default=1_000_000, type=int)
    parser.add_argument('-me', '--max-error', type=float, default=None)
    parser.add_argument('-nw', '--num-workers', type=int, default=1)

    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args


def read_data(args: argparse.Namespace) -> List[SubsetSumProblem]:
    if args.data_type == 'financial':
        data_path = os.path.join(project_path, 'data', 'financial_documents')
        files = glob.glob(os.path.join(data_path, 'parsed_*.jsonl'))
        data = []
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    table_sum_structure = TableSumStructure.from_json(line)
                    data.extend(table_sum_structure.subset_sum_problems)
    elif args.data_type == 'artificial':
        data_path = os.path.join(project_path, 'data', 'artificial_data')
        files = glob.glob(os.path.join(data_path, '*.jsonl'))
        data = []
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    data.append(SubsetSumProblem.from_json(line))
    else:
        raise NotImplementedError
    print(f'Found {len(data)} individual subset sum problems')
    return data


def init_algorithm(args):
    if args.algorithm == 'hopfield_rounded':
        hopfield_algo = HopfieldAlgorithmTorch(verbose=False)
        algo = RoundingAlgorithm(algorithm=hopfield_algo,
                                 verbose=args.verbose,
                                 rounding_factor=args.rounding_factor,
                                 assert_correctness=True)
    elif args.algorithm == 'hopfield':
        algo = HopfieldAlgorithmTorch(verbose=args.verbose)
    elif args.algorithm == 'count':
        algo = CountAlgorithm(verbose=args.verbose)
    else:
        raise NotImplementedError
    return algo


def init_run_args(args, algo, ground_truth_solution):
    if isinstance(algo, CountAlgorithm):
        run_args = {'max_iterator': args.runs,
                    'break_at_first_solution': True,
                    'num_workers': args.num_workers,
                    }
        if args.ground_truth_solution_only:
            run_args['ground_truth_solution'] = ground_truth_solution
        return run_args
    run_args = {
        'runs': args.runs,
        'steps': args.steps,
        'batch_size': args.batch_size
    }
    if args.max_error is not None:
        run_args['max_error'] = args.max_error
    if isinstance(algo, RoundingAlgorithm):
        run_args['ground_truth_solution'] = ground_truth_solution
    return run_args


def solve(args, data, algo):
    results: List[Result] = []
    for i, problem in tqdm(enumerate(data), desc='Running algorithms on problem configs:', total=len(data), dynamic_ncols=True):
        tqdm.write(
            f'\n* * * * * * * * * * * * * * * * * * * * * * * * * *\n'
            f'New Problem {i} or {len(data)}:\n'
            f'{problem.group} -- {problem.name}\n'
            f'* * * * * * * * * * * * * * * * * * * * * * * * * *\n'
        )

        numbers = np.array(problem.numbers)
        target = np.array(problem.target)
        ground_truth_solution = problem.ground_truth_solution

        run_args = init_run_args(args, algo, ground_truth_solution)

        if args.verbose:
            print(numbers, target, numbers[ground_truth_solution].sum())

        start_time = dt.now()
        for solution, run_index in algo.run(numbers, target, **run_args):
            # found some solution!
            if args.ground_truth_solution_only:
                # solution found that is not the ground truth?
                if tuple(solution) != tuple(problem.ground_truth_solution):
                    if args.verbose:
                        print(f'Found solution that is not the ground_truth: {solution}')
                    # continue search
                    continue
            # found solution
            break
        else:
            # entire algorithm ran, no solution found
            solution, run_index = None, None
        end_time = dt.now()

        result = Result(
            solution=solution,
            problem=problem,
            start_time=start_time,
            end_time=end_time,
            total_seconds=(end_time - start_time).total_seconds(),
            steps_per_run=args.steps,
            runs=run_index,
            algo=str(algo),
            args=args.__dict__
        )

        if solution is not None:
            tqdm.write(
                f'\tFound solution {solution}, ground_truth solution {problem.ground_truth_solution}'
            )
        else:
            tqdm.write(
                f'\tDid not find solution :('
            )
        if args.verbose:
            print(result)
        with open(args.output_file, 'a') as f:
            f.write(result.to_json() + '\n')
        results.append(result)
    return results


def print_stats(results: List[Result]) -> None:

    start_time = results[0].start_time
    end_time = results[-1].end_time
    total_time = (end_time - start_time).total_seconds()

    total_results = len(results)
    found_results = len([x for x in results if x.solution is not None])
    ground_truth_results = len([x for x in results if x.solution == x.problem.ground_truth_solution])

    print(
        f'Algorithm ran for {total_time} seconds / {np.round(total_time/60)} minutes / {np.round(total_time / 3600)} hours.\n'
        f'Of {total_results} problems, we solved {found_results} ({found_results / total_results:.1%}) with some solution (including ground truth solutions).\n'
        f'Of {total_results} problems, we solved {ground_truth_results} ({ground_truth_results / total_results:.1%}) with the ground truth solution.\n'
    )


def main():
    args = parse_args()
    data = read_data(args)
    algo = init_algorithm(args)
    results = solve(args=args, data=data, algo=algo)
    print_stats(results)


if __name__ == '__main__':
    main()
