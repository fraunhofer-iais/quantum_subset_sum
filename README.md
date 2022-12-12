# Quantum Subset Sum

This repository contains code for the Smart Data Innovation Challenges (SDI-C) Project "Solving Accounting Optimization Problems in the Cloud", which was funded by BMBF, in collaboration with PWC.

The research was published at IEEE Conference on Machine Learning and Applications:  
Biesner et al., "Solving Subset Sum Problems using Quantum Inspired Optimization Algorithms with Applications in Auditing and Financial Data Analysis", ICMLA 2022  
and is available on arxiv:
https://arxiv.org/abs/2211.02653

## Overview

The goal is the analysis tables in financial documents for numerical errors.
Given a table which contains sums, we want to check if the sums are calculated correctly.

We do not want to implement rule-based algorithms for extracting a sum-structure in the table,
since this is very inflexible and hard to generalize to all possible table formats.

Instead, we consider the entire table (or a single column) as a set of numbers without inherent structure.
We consider each number and check whether a subset of the remaining numbers adds up to the target number.
This way we find sum-structures in the table which can be used for consistency checks.

This is an application of the subset-sum problem, which in general is NP-hard.
A QUBO (quadratic unconstrained binary optimization) formulation of the subset-sum problem can be efficiently 
solved using adiabatic quantum computers.

If no working quantum computer is available, the quantum annealing process can be simulated by using Hopfield networks on GPUs.

This repository contains code to
- parse financial tables into subset-sum problems
- solve subset-sum problems with Hopfield networks
- solve subset-sum problems with Hopfield networks efficiently on GPUs
- solve subset-sum problems with classical solving algorithms as baseline


## Installation

In order to set up the necessary environment:

1. create a new environment `qss` with the help of conda:
   ```bash
   conda create -n qss python=3.8
   conda activate qss
   ```

2. install the required packages and the `quantum_subset_sum` package
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
## Using the algorithms

Main entry point to the code is the script `scripts/evaluation/solve.py`, which takes the following arguments:

``` yaml
--data-type                   -> "financial" or "artificial" - type of data to run algorithms on
--output-file                 -> path to output.json file with results
--algorithm                   -> "hopfield" or "count" - type of algorithm to use
--ground-truth-solution-only  -> only stop when groundtruth solution is found or stop if any solution is found
--runs                        -> maximum number of runs for each problem
--batch-size                  -> maximum number of concurrent runs on one GPU
--steps                       -> maximum number of steps per run
--rouding-factor              -> round each value in the table by this factor to increase numerical stability
--max-error                   -> maximum error due to rounding allowed for a solution to be returned
--num-workers                 -> number of processes for count algorithm
--verbose                     -> verbose logging
```

### Input data

Dependent on the given `data-type` argument the script parses all files in `data/artificial_data/*.jsonl` or all files `data/financial_data/parsed_*.jsonl`.
Each `jsonl` file is expected to have the following structure:
```json
{
    "name": "sample_00",                       # name of the sample
    "group": "n_016_1B",                       # name of the sample group
    "numbers": [1, 2, 5, 6, ...],              # numbers of the subset sum problem
    "target": 21,                              # target value
    "ground_truth_solution": [4, 7, 9, 15],    # groundtruth (for evaluation, optional)
    "config": {"n": 16,                        # config for reproduction (optional)
               "k": 4, 
               "min_value": -100, "max_value": 100, 
               "seed": 15}
}
```
see `data/artificial_data` for some example files.

### Parsing data

We provide scripts to parse the financial data described in the paper into the format above in `scripts/data/parse_*.py`.
We provide a script to generate artificial data as `scripts/data/generate_artificial_data.py`.

## Evaluation

We provide scripts to evaluate the output of `scripts/solve.py` into the statistics and plots shown in the paper in `scripts/evaluation/`

## Acknowledgments
