import os
import csv
import random

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def get_last_generation_data(data, algo_name):
    # Filter data by the specified algorithm and the last generation (2000)
    last_gen_data = data[(data['algo'] == algo_name) & (data['generation'] == 2000)]

    # Ensure to select the data from the 30 different runs (based on unique seed)
    unique_seeds = last_gen_data['seed'].unique()

    # Only select rows with unique seeds to ensure data from 30 different runs
    last_gen_unique_runs = last_gen_data[last_gen_data['seed'].isin(unique_seeds)]

    return last_gen_unique_runs


def compute_pval(name_ds, argument):
    print(name_ds)
    # Set paths and caption based on the argument
    if argument == 'elite_size':
        gsgp_path = os.getcwd() + f'/RUN_BY_RUN/SIZE/RESULT_Size_Of_Best_Run_By_Run_Config_GSGP_random_MS_04_09_2023_{name_ds.upper()}.txt'
        gp_path = os.getcwd() + f'/RUN_BY_RUN/SIZE/RESULT_Size_Of_Best_Run_By_Run_Config_STDGP_04_09_2023_{name_ds.upper()}.txt'
    elif argument == 'test_fitness':
        gsgp_path = os.getcwd() + f'/RUN_BY_RUN/TEST_RMSE/RESULT_Fitness_On_Test_Run_By_Run_Config_GSGP_random_MS_04_09_2023_{name_ds.upper()}.txt'
        gp_path = os.getcwd() + f'/RUN_BY_RUN/TEST_RMSE/RESULT_Fitness_On_Test_Run_By_Run_Config_STDGP_04_09_2023_{name_ds.upper()}.txt'

    # Read the file
    with open(gp_path, "r") as file:
        reader = csv.reader(file, delimiter=',')
        data_gp = [[parse_float(num) for num in row if num] for row in reader]
        # Filter out rows with None values or empty rows
        data_gp = [row for row in data_gp if all(num is not None for num in row) and row]

    # Now `data` is a 2D list (list of lists) containing the numbers
    gp_last_generation = [row[-1] for row in data_gp if row]
    # print(gp_last_generation)

    # Read the file
    with open(gsgp_path, "r") as file:
        reader = csv.reader(file, delimiter=',')
        data_gsgp = [[parse_float(num) for num in row if num] for row in reader]
        # Filter out rows with None values or empty rows
        data_gsgp = [row for row in data_gsgp if all(num is not None for num in row) and row]

    # Now `data` is a 2D list (list of lists) containing the numbers
    gsgp_last_generation = [row[-1] for row in data_gsgp if row]
    # print(gsgp_last_generation)

    csv_path = os.getcwd() + f'/../res/slim_{name_ds}.csv'
    # Load data from the CSV file
    data = pd.read_csv(csv_path, names=["algo", "experiment_id", "dataset", "seed", "generation", "training_fitness",
                                        "timing", "pop_node_count", "test_fitness", "elite_size", "log_level"])

    # Example replacements
    data.replace({'algo': {'SlimGSGP_1_mul_False': 'SLIM-NORM*1',
                           'SlimGSGP_1_mul_True': 'SLIM-SIG*1',
                           'SlimGSGP_1_sum_False': 'SLIM-NORM+1',
                           'SlimGSGP_1_sum_True': 'SLIM-SIG+1',
                           'SlimGSGP_2_mul_False': 'SLIM*2',
                           'SlimGSGP_2_sum_False': 'SLIM+2'
                           }},
                 inplace=True)

    slim_algo = ['SLIM-NORM*1', 'SLIM-SIG*1', 'SLIM-NORM+1', 'SLIM-SIG+1', 'SLIM*2', 'SLIM+2']
    for alg in slim_algo:
        result = get_last_generation_data(data, alg)
        result_slim = list(result[argument])

        # print(alg)
        # print("slim " + str(np.mean(result_slim)))
        # print("gp " + str(np.mean(gp_last_generation)))

        stat, p = wilcoxon(x=result_slim, y=gsgp_last_generation)

        print("%-15s %-10s" % (alg, p))


for name_ds in ["concrete", "ppb", "instanbul", "toxicity", "resid_build_sale_price", "energy"]:
    compute_pval(name_ds, "test_fitness")
