from typing import Tuple, List, Dict
import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from quantum_subset_sum.data.results import Result
from quantum_subset_sum.utils.plots import box_strip_plot


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, type=str)
parser.add_argument('-o', '--output', required=True, type=str)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

data = []
with open(args.input, 'r') as f:
    for line in f:
        data.append(Result.from_json(line))
# skip first datapoint because GPU warmup
data = data[1:]

# map min/max values to a string-descriptor
minmax_map = {
    10000: '10k',
    100000: '100k',
    1000000: '1M',
    10000000: '10M',
    100000000: '100M',
    1000000000: '1B',
    10000000000: '10B',
    100000000000: '100B',
    1000000000000: '1T'
}

sep = '\n' + ' = '*10 + '\n'

print(sep)
# n values to keep in the data
keep_n = [16, 32, 64, 128, 256]
_len = len(data)
data = [x for x in data if x.problem.config['n'] in keep_n]
print(
    f"Filtered n-values: Keeping {keep_n}.\n"
    f"Before {_len} results, now {len(data)} remaining."
)

keep_minmax = ['10k', '100k', '1M']
_len = len(data)
data = [x for x in data if minmax_map[x.problem.config['max_value']] in keep_minmax]
print(
    f"Filtered minmax-values: Keeping {keep_minmax}.\n"
    f"Before {_len} results, now {len(data)} remaining."
)

max_runs = data[0].args['runs']
df = pd.DataFrame(
    {
        'name':           [x.problem.group for x in data],
        'solution_found': [x.solution is not None for x in data],
        'ground_truth_solution_found': [x.solution is not None and x.solution == x.problem.ground_truth_solution for x in data],
        'total_seconds': [x.total_seconds for x in data],
        'runs': [x.runs if x.solution is not None else max_runs for x in data],
        'minmax': [minmax_map[x.problem.config['max_value']] for x in data],
        'n': [x.problem.config['n'] for x in data],
    }
)


def get_runs_per_second(df):
    total_total_seconds = df[df['solution_found']]['total_seconds'].sum()
    total_runs = df[df['solution_found']]['runs'].sum()
    return total_runs / total_total_seconds


runs_per_second = get_runs_per_second(df)
df['time'] = df['runs'] / runs_per_second

print(sep)
_df = df.groupby(['n', 'minmax', 'solution_found']).size()
path_solution_found_stats = os.path.join(args.output, 'solution_found_stats.txt')
path_solution_found_stats_latex = os.path.join(args.output, 'solution_found_stats.tex')
print(f"Stats on found solutions:\n"
      f"{str(_df)}\n"
      f"Saving into {path_solution_found_stats} and\n"
      f"            {path_solution_found_stats_latex}")
with open(path_solution_found_stats, 'w') as f:
    f.write(str(_df))
with open(path_solution_found_stats_latex, 'w') as f:
    f.write(_df.to_latex())

print(sep)
_df = df[df['solution_found']]
_df = _df.groupby(['name', 'n']).agg({'time': 'mean', 'runs': 'mean', 'total_seconds': 'mean'})
_df.runs = _df.runs.apply(lambda x: f'{x:.2e}')
path_times_stats = os.path.join(args.output, 'times_stats.txt')
path_times_stats_latex = os.path.join(args.output, 'times_stats.tex')
print(f"Stats on found solutions:\n"
      f"{str(_df)}\n"
      f"Saving into {path_times_stats} and\n"
      f"            {path_times_stats_latex}")
with open(path_times_stats, 'w') as f:
    f.write(str(_df))
with open(path_times_stats_latex, 'w') as f:
    f.write(_df.to_latex())

print(sep)
limit_run_time = df['time'].max()
max_run_time = df[df['time'] != limit_run_time]['time'].max()
print(
    f"Limit run time for\n"
    f"\tmax_runs: {max_runs}\n"
    f"\truns_per_second: {runs_per_second:.2f}\n"
    f"\tlimit_run_time: {limit_run_time:.2f}.\n"
    f"Runs are aborted after {limit_run_time:.2f} seconds.\n"
    f"Longest run time under limit_run_time:\n"
    f"\tmax_run_time: {max_run_time:.2f} for runs:"
    f"\t{df[df['time'] == max_run_time]['name'].tolist()}"
)

print(sep)
_df = df[~df['solution_found']]
print(f"Run times for runs were the solution is not found:")
print(_df['total_seconds'])
print(f"Mean total_seconds if solution is not found: {_df['total_seconds'].mean()}")

min_total_seconds_if_not_found = _df['total_seconds'].min()
max_total_second_if_found = df[df['solution_found']]['total_seconds'].max()
assert min_total_seconds_if_not_found > max_total_second_if_found

# _df['total_seconds'] = _df['total_seconds'].mean()

print(sep)
print(f"Saving plots into:\n"
      f"\t{os.path.join(args.output, 'plot_box_hopfield_artificial_total_seconds.png')}\n",
      f"\t{os.path.join(args.output, 'plot_box_hopfield_artificial_runs.png')}\n",
      f"\t{os.path.join(args.output, 'plot_box_hopfield_artificial_time.png')}\n",
      f"\t{os.path.join(args.output, 'plot_box_strip_hopfield_artificial_total_seconds.png')}\n",
      f"\t{os.path.join(args.output, 'plot_box_strip_hopfield_artificial_runs.png')}\n",
      f"\t{os.path.join(args.output, 'plot_box_strip_hopfield_artificial_time.png')}",
      )
f = box_strip_plot(df, x='n', y='total_seconds', hue='minmax', max_line=_df['total_seconds'].mean(),
                   title='hopfield algorithm on artificial data',
                   hue_order=keep_minmax,
                   y_lim=(0, None))
f.savefig(os.path.join(args.output, 'plot_box_hopfield_artificial_total_seconds.png'), dpi=200, bbox_inches='tight')

f = box_strip_plot(df, x='n', y='runs', hue='minmax', max_line=max_runs,
                   title='hopfield algorithm on artificial data',
                   hue_order=keep_minmax,
                   y_lim=(0, None))
f.savefig(os.path.join(args.output, 'plot_box_hopfield_artificial_runs.png'), dpi=200, bbox_inches='tight')

f = box_strip_plot(df, x='n', y='time', hue='minmax', max_line=limit_run_time,
                   title='hopfield algorithm on artificial data',
                   hue_order=keep_minmax,
                   y_lim=(0, None))
f.savefig(os.path.join(args.output, 'plot_box_hopfield_artificial_time.png'), dpi=200, bbox_inches='tight')

f = box_strip_plot(df, x='n', y='total_seconds', hue='minmax', max_line=_df['total_seconds'].mean(),
                   title='hopfield algorithm on artificial data',
                   hue_order=keep_minmax,
                   add_strip_plot=True,
                   y_lim=(0, 10 ** 3))
f.savefig(os.path.join(args.output, 'plot_box_strip_hopfield_artificial_total_seconds.png'), dpi=200, bbox_inches='tight')

f = box_strip_plot(df, x='n', y='runs', hue='minmax', max_line=max_runs,
                   title='hopfield algorithm on artificial data',
                   hue_order=keep_minmax,
                   add_strip_plot=True,
                   y_lim=(0, None))
f.savefig(os.path.join(args.output, 'plot_box_strip_hopfield_artificial_runs.png'), dpi=200, bbox_inches='tight')

f = box_strip_plot(df, x='n', y='time', hue='minmax', max_line=limit_run_time,
                   title='hopfield algorithm on artificial data',
                   hue_order=keep_minmax,
                   add_strip_plot=True,
                   y_lim=(0, None))
f.savefig(os.path.join(args.output, 'plot_box_strip_hopfield_artificial_time.png'), dpi=200, bbox_inches='tight')