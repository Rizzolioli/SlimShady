from algorithms.GP.representations.tree_utils import create_grow_random_tree, create_full_random_tree
from parametrization import FUNCTIONS, generate_random_uniform
from random_utils import *
from algorithms.GP.representations.tree_utils import flatten
import numpy as np
import torch
import random
import csv
import datetime

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

for study in ['', 'complex']:

    path = f'log/random_study_{study}_{day}.csv'
    s_it = 0
    for seed in range(1000):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if study == 'complex':

            t1 = create_full_random_tree(depth = 3,
                                         FUNCTIONS = FUNCTIONS,
                                         TERMINALS = {f"x{i}": i for i in range(20)},
                                         CONSTANTS = None,
                                         p_c = 0)
            t2 = create_full_random_tree(depth = 3,
                                         FUNCTIONS = FUNCTIONS,
                                         TERMINALS = {f"x{i}": i for i in range(20)},
                                         CONSTANTS = None,
                                         p_c = 0)
        else:
            t1 = create_grow_random_tree(depth = 6,
                                         FUNCTIONS = FUNCTIONS,
                                         TERMINALS = {f"x{i}": i for i in range(20)},
                                         CONSTANTS = None,
                                         p_c = 0)
            t2 = create_grow_random_tree(depth = 6,
                                         FUNCTIONS = FUNCTIONS,
                                         TERMINALS = {f"x{i}": i for i in range(20)},
                                         CONSTANTS = None,
                                         p_c = 0)

        try:
            info_t1 = get_info(t1)
            info_t2 = get_info(t2)

            ms = generate_random_uniform(0,1)()

            s1 = mutate_tree(t1, mutation_type='sigmoid', ms=ms)
            abs = mutate_tree(t1, mutation_type='abs', ms=ms)
            s2 = mutate_tree(t1, mutation_type='sigmoid2', ms=ms, t2=t2)

            info_s1 = get_info(s1)
            info_abs = get_info(abs)
            info_s2 = get_info(s2)

            with open(path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, 'S1'] + info_t1 + (['NaN'] * 5) + info_s1)
                writer.writerow([seed, 'ABS'] + info_t1 + (['NaN'] * 5) + info_abs)
                writer.writerow([seed, 'S2'] + info_t1 + info_t2 + info_s2)

        except:
    #
            print('skipped iteration')
            s_it += 1

    print(f'skipped  {s_it} iterations')