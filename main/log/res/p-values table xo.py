from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
import os
import csv


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def load_run_by_run_data(path):
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=',')
        data = [[parse_float(num) for num in row if num] for row in reader]
        data = [row for row in data if all(num is not None for num in row) and row]
    return [row[-1] for row in data if row]


def get_slim_data(name_ds):
    # Load data from the CSV file
    data = pd.read_csv('../xo_20250312.csv', names=["algo", "experiment_id", "dataset", "seed", "generation",
                                                    "training_fitness", "timing", "pop_node_count",
                                                    "test_fitness", "elite_size", "log_level"])

    data = data[data['dataset'] == name_ds]
    data.replace({'algo': {'SlimGSGP_1_mul_False': 'SLIM*ABS',
                           'SlimGSGP_1_mul_True': 'SLIM*SIG1',
                           'SlimGSGP_1_sum_False': 'SLIM+ABS',
                           'SlimGSGP_1_sum_True': 'SLIM+SIG1',
                           'SlimGSGP_2_mul_False': 'SLIM*SIG2',
                           'SlimGSGP_2_sum_False': 'SLIM+SIG2',
                           # 'SlimGSGP_1_mul_False': 'SLIM*ABS',
                           'SLIM*1SIG': 'SLIM*SIG1',
                           # 'SlimGSGP_1_sum_False': 'SLIM+ABS',
                           'SLIM+1SIG': 'SLIM+SIG1',
                           'SLIM*2SIG': 'SLIM*SIG2',
                           'SLIM+2SIG': 'SLIM+SIG2',
                           'SLIM+*2SIG': 'SLIM*SIG2'
                           }},
                 inplace=True)
    return data


def get_last_generation_data(data, algo_name):
    return data[(data['algo'] == algo_name) & (data['generation'] == 2000)]


def pairwise_pval_table(name_ds, metric='test_fitness'):
    slim_data = get_slim_data(name_ds)

    gsgp_path = f'RUN_BY_RUN/TEST_RMSE/RESULT_Fitness_On_Test_Run_By_Run_Config_GSGP_random_MS_04_09_2023_{name_ds.upper()}.txt'
    gp_path = f'RUN_BY_RUN/TEST_RMSE/RESULT_Fitness_On_Test_Run_By_Run_Config_STDGP_04_09_2023_{name_ds.upper()}.txt'

    gsgp_values = load_run_by_run_data(gsgp_path)
    gp_values = load_run_by_run_data(gp_path)

    algos = [
        'SLIM+SIG2',
        'SLIM*SIG2',
        'SLIM+ABS',
        'SLIM*ABS',
        'SLIM+SIG1',
        'SLIM*SIG1'

    ]
    ref_algos = {'gp': gp_values, 'gsgp': gsgp_values}

    results_df = pd.DataFrame(np.nan, index=['gp', 'gsgp'], columns=algos)

    for ref in ref_algos:
        for slim_algo in algos:
            slim_values = get_last_generation_data(slim_data, slim_algo)[metric].values
            if len(slim_values) == 0 or len(ref_algos[ref]) == 0:
                continue
            pval = mannwhitneyu(slim_values, ref_algos[ref])[1]
            # sign = '*' if np.median(slim_values) < np.median(ref_algos[ref]) else ''
            results_df.loc[ref, slim_algo] = pval

    print(results_df.fillna('--').to_latex(caption=f'Wilcoxon p-values on {metric} for dataset {name_ds}'))


# Usage
for dataset in ["toxicity", "instanbul", "energy", "ppb", "concrete", "resid_build_sale_price"]:
    pairwise_pval_table(dataset, metric='test_fitness')
