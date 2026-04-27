import os
import csv
import random

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata


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


def compute_pval(alg1, alg2, name_ds, argument):
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

    csv_path = os.getcwd() + f'/results_def/slim_{name_ds}.csv'
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

    if alg1 == "gp":
        result_1 = gp_last_generation
    if alg1 == "gsgp":
        result_1 = gsgp_last_generation

    result_slim = get_last_generation_data(data, alg2)
    result_2 = list(result_slim[argument])

    stat, p = wilcoxon(x=result_1, y=result_2)

    if np.median(result_1) > np.median(result_2):
        better = '*'
    else:
        better = ''

    print("%-15s %-15s %-10s %-10s" % (alg1, alg2, p, better))


slim_algs = ['SLIM-NORM*1', 'SLIM-SIG*1', 'SLIM-NORM+1', 'SLIM-SIG+1', 'SLIM*2', 'SLIM+2']
for name_ds in ["toxicity", "instanbul", "energy", "ppb", "concrete", "resid_build_sale_price"]:
    print(name_ds)
    alg2 = "SLIM-SIG+1"
    compute_pval("gp", alg2, name_ds, "test_fitness")
    compute_pval("gsgp", alg2, name_ds, "test_fitness")


def compute_median_ranking(name_ds, argument):
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

    csv_path = os.getcwd() + f'/results_def/slim_{name_ds}.csv'
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

    algos = [
        'SLIM*2',
        'SLIM+2',
        'SLIM-NORM*1',
        'SLIM-NORM+1',
        'SLIM-SIG*1',
        'SLIM-SIG+1',
    ]

    result_1 = gp_last_generation
    gsgp_last_generation

    all_results = [*[list(get_last_generation_data(data, algo)[argument]) for algo in algos], gp_last_generation,
                   gsgp_last_generation]

    medians = [np.median(res) for res in all_results]

    return rankdata(medians, method='min')


for metric in ["test_fitness", "elite_size"]:
    print(metric)
    rank_list = []
    for name_ds in ["concrete", "ppb", "instanbul", "toxicity", "resid_build_sale_price", "energy"]:
        rank = compute_median_ranking(name_ds, metric)
        rank_list.append(rank)
    rank_list = np.array(rank_list)
    median = np.median(rank_list, axis=0)
    print(f"median {median}")
