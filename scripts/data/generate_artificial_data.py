import json
import os
from itertools import product


from tqdm import tqdm
from quantum_subset_sum.utils.helper import sample_numbers_and_target, mask_to_indices
from quantum_subset_sum.data.subset_sum import SubsetSumProblem
from quantum_subset_sum import project_path

output_path = os.path.join(project_path, 'data', 'artificial_data')
os.makedirs(output_path, exist_ok=True)

samples_per_config = 5
n_configs = [
    # {'n': 8, 'k': 2},
    {'n': 16, 'k': 4},
    # {'n': 24, 'k': 8},
    {'n': 32, 'k': 8},
    {'n': 64, 'k': 8},
    {'n': 128, 'k': 8},
    {'n': 256, 'k': 8}
]
min_max_configs = [
    {'name': '10k',
     'min_value': -10_000,
     'max_value': 10_000, },
    {'name': '100k',
     'min_value': -100_000,
     'max_value': 100_000, },
    {'name': '1M',
     'min_value': -1_000_000,
     'max_value': 1_000_000},
    # {'name': '100M',
    #  'min_value': -100_000_000,
    #  'max_value': 100_000_000, },
    {'name': '1B',
     'min_value': -1000_000_000,
     'max_value': 1000_000_000, },
    # {'name': '100B',
    #  'min_value': -100_000_000_000,
    #  'max_value': 100_000_000_000, },
    {'name': '1T',
     'min_value': -1_000_000_000_000,
     'max_value': 1_000_000_000_000, }
]

seed_counter = 0
for n_config in tqdm(n_configs, desc='Iterating over n'):
    for min_max_config in tqdm(min_max_configs, desc='Iterating over min_max'):
        min_max_config = min_max_config.copy()
        name = f'n_{str(n_config["n"]).zfill(3)}_' + min_max_config.pop('name')
        tqdm.write(name)
        samples = []
        for i in tqdm(range(samples_per_config), desc='Multiple samples'):
            numbers, mask, target = sample_numbers_and_target(seed=seed_counter, **n_config, **min_max_config)
            solution = mask_to_indices(mask)
            sample = SubsetSumProblem(
                name=f'sample_{str(i).zfill(2)}',
                group=name,
                numbers=numbers.astype(float).tolist(),
                target=int(target),
                ground_truth_solution=solution,
                config={** n_config, **min_max_config, 'seed': seed_counter}
            )
            samples.append(sample)
            seed_counter += 1
        with open(os.path.join(output_path, name + '.jsonl'), 'w') as f:
            for sample in samples:
                f.write(sample.to_json() + '\n')
