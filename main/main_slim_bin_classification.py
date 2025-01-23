import time
import uuid

import numpy as np
import torch

from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from utils.utils import get_terminals, train_test_split, protected_div
from utils.logger import log_settings
from algorithms.SLIM_GSGP.operators.mutators import *
from utils.utils import generate_random_uniform
from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
import os
from utils.utils import get_best_min, binary_sign_transformer, minmax_binarizer, modified_sigmoid
from evaluators.fitness_functions import binarized_rmse, rmse, bin_ce
from algorithms.GP.operators.initializers import rhh
from datasets.data_loader import *
import csv

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split as tts_sklearn
from sklearn.decomposition import PCA
from algorithms.SSHC.sshc import SSHC


import datetime


now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]

path = '../../../GAMETES dataset/data'
data_loaders = os.listdir(path)

########################################################################################################################

# RUNNING THE ALGORITHM & DEFINING
#    DATA-DEPENDANT PARAMETERS

########################################################################################################################

# saving the elites looks:

elites = {}

# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

#n_runs = 10
settings_dict = {"p_test": 0.2}

FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2}
}

binarizer = modified_sigmoid(1)
final_binarizer = binary_sign_transformer

# class_metric = accuracy_score
class_metric = matthews_corrcoef

# CONSTANTS = {
#     'constant_2': lambda x: torch.tensor(2).float(),
#     'constant_3': lambda x: torch.tensor(3).float(),
#     'constant_4': lambda x: torch.tensor(4).float(),
#     'constant_5': lambda x: torch.tensor(5).float(),
#     'constant__1': lambda x: torch.tensor(-1).float()
# }

CONSTANTS = {f'constant_{int}' : lambda x: torch.tensor(int).float() for int in range(-10, 10, 1)}

# CONSTANTS  = lambda : torch.tensor(-20 * random.random() + 10).float()



slim_gsgp_solve_parameters = {"elitism": True,
                              "log": 0,
                              "verbose": 1,
                              "test_elite": True,
                              "log_path": os.path.join(os.getcwd(), "log", f"slim_gametes_{day}.csv"),
                              "run_info": None,
                              "ffunction": binarized_rmse(binarizer),
                              # "ffunction": bin_ce(binarizer),
                              "n_iter": 1000,
                              "max_depth": None,
                              "n_elites": 1,
                              "reconstruct" : True,
                              "gp_imputing_missing_values" : False
                              }

slim_GSGP_parameters = {"initializer": rhh,
                        "selector": tournament_selection_min_slim(2),
                        "crossover": None,
                        "ms": None,
                        "inflate_mutator": None,
                        "deflate_mutator": deflate_mutation,
                        "p_xo": 0,
                        "pop_size": 100,
                        "settings_dict": settings_dict,
                        "find_elit_func": get_best_min,
                        "p_inflate": None,
                        "copy_parent": None,
                        "operator": None
                        }

mutation_parameters ={
"sig": None,
"two_trees": None
}

inflate_mutator = inflate_mutation

slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

slim_gsgp_pi_init = {'init_pop_size': slim_GSGP_parameters["pop_size"],
                     'init_depth': 3,
                     'FUNCTIONS': FUNCTIONS,
                     'CONSTANTS': CONSTANTS,
                     "p_c": 0}

all_params = {"SLIM_GSGP": ["slim_gsgp_solve_parameters", "slim_GSGP_parameters", "slim_gsgp_pi_init", "settings_dict"],
              "GSGP": ["gsgp_solve_parameters", "GSGP_parameters", "gsgp_pi_init", "settings_dict"],
              "GP": ["gp_solve_parameters", "GP_parameters", "gp_pi_init", "settings_dict"]}

slim_dataset_params = {"toxicity": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                        "ld50": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                       "concrete_strength": {"p_inflate": 0.5, "ms": generate_random_uniform(0, 0.3)},
                       "other": {"p_inflate": 0.5, "ms": generate_random_uniform(0, 1)}}


for loader in ['2w_10a_0.4her.txt']:
    # for each dataset, run all the planned algorithms
    for algo_name in algos:

        for (sig, ttress, op, gsgp) in [
                                        # (True, False, "mul", False),
                                        # (False, False, "mul", False),
                                        (True, True, "sum", False)
                                        ]:  # (True, True, "sum"), (True, True, 'std') (True, False, "mul", False), (False, False, "mul", False), (True, True, "sum", False)

            # getting the log file name according to the used parameters:

            if (sig, ttress, op, gsgp) == (True, False, "mul", True):
                algo = 'GSGP*1SIG'
            elif (sig, ttress, op, gsgp) == (False, False, "mul", True):
                algo = 'GSGP*ABS'
            if (sig, ttress, op, gsgp) == (True, False, "mul", False):
                algo = 'SLIM*1SIG'
            elif (sig, ttress, op, gsgp) == (False, False, "mul", False):
                algo = 'SLIM*ABS'
            elif (sig, ttress, op, gsgp) == (True, True, "sum", False):
                algo = 'SLIM+2SIG'
            elif (sig, ttress, op, gsgp) == (True, True, "sum", True):
                algo = 'GSGP'

            if op == 'std':
                op = 'sum'

            slim_GSGP_parameters["two_trees"] = ttress
            slim_GSGP_parameters["operator"] = op

            print(algo)
            # running each dataset + algo configuration n_runs times

            # getting the name of the dataset:
            curr_dataset = loader[:-4]

            for seed in range(1):
                start = time.time()

                # Loads the data via the dataset loader
                data = pd.read_csv(path + '/' + loader, sep = '\t')

                X = data.values[:, :-1]
                y = data.values[:, -1]

                X_train, X_test, y_train, y_test = tts_sklearn(X, y,
                                                               stratify= y,
                                                               test_size=settings_dict['p_test'],
                                                               shuffle = True,
                                                               random_state=seed)
                # pca = PCA()
                # X_train = pca.fit_transform(X_train)
                # X_test = pca.transform(X_test)

                X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), \
                                                    torch.tensor(y_train), torch.tensor(y_test)

                # getting the name of the dataset
                dataset = loader[:-4]

                # getting the terminals and defining the terminal-dependant parameters
                TERMINALS = {f"x{i}": i for i in range(X.shape[1])}

                # slim_GSGP_parameters["ms"] = generate_random_uniform(0, torch.median(y_train).item())

                slim_GSGP_parameters["ms"] = generate_random_uniform(0, 1 )

                # setting up the dataset related slim parameters:
                if dataset in slim_dataset_params.keys():
                    # slim_GSGP_parameters["ms"] = slim_dataset_params[dataset]["ms"]
                    slim_GSGP_parameters['p_inflate'] = slim_dataset_params[dataset]["p_inflate"]

                else:
                    # slim_GSGP_parameters["ms"] = slim_dataset_params["other"]["ms"]

                    slim_GSGP_parameters['p_inflate'] = slim_dataset_params["other"]["p_inflate"]

                slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']

                if gsgp:
                    slim_GSGP_parameters['p_inflate'] = 1
                    slim_GSGP_parameters['p_deflate'] = 0


                # setting up the dataset related parameters:
                slim_gsgp_pi_init["TERMINALS"] = TERMINALS

                slim_GSGP_parameters["inflate_mutator"] = inflate_mutator(FUNCTIONS=FUNCTIONS,
                                                                          TERMINALS=TERMINALS,
                                                                          CONSTANTS=CONSTANTS,
                                                                          two_trees=slim_GSGP_parameters[
                                                                              'two_trees'],
                                                                          operator=slim_GSGP_parameters[
                                                                              'operator'],
                                                                          sig=sig)


                # adding the dataset name and algorithm name to the run info for the logger
                slim_gsgp_solve_parameters['run_info'] = [algo, dataset]

                optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=seed)

                optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                curr_dataset=curr_dataset,
                                **slim_gsgp_solve_parameters)

                if '*' in algo:
                    corr = class_metric(y_test, final_binarizer(torch.prod(optimizer.elite.test_semantics, dim = 0)))

                elif '+' in algo or algo == 'GSGP':
                    corr = class_metric(y_test, final_binarizer(torch.sum(optimizer.elite.test_semantics, dim = 0)))

                else:
                    print('Dont know what algorithm to use')

                print(corr)

                sorted_idxs = torch.argsort(torch.tensor(optimizer.population.fit))[:5]
                inds = []
                fits = []

                for i in range(5):

                    print(f'STARTING {i}th LOCAL SEARCH')

                    local_search = SSHC(X_train=X_train,
                                        y_train=y_train,
                                        ffunction=slim_gsgp_solve_parameters['ffunction'],
                                        eval_operator=slim_GSGP_parameters["operator"],
                                        neigh_operator=slim_GSGP_parameters["deflate_mutator"],
                                        FUNCTIONS=FUNCTIONS,
                                        TERMINALS=TERMINALS,
                                        CONSTANTS=CONSTANTS,
                                        reconstruct=slim_gsgp_solve_parameters['reconstruct'],
                                        curr_dataset=curr_dataset,
                                        X_test=X_test,
                                        y_test=y_test,
                                        log=0,
                                        log_path=slim_gsgp_solve_parameters['log_path'],
                                        run_info=slim_gsgp_solve_parameters['run_info'],
                                        verbose=1,
                                        initial_depth=None,
                                        individual=optimizer.population[sorted_idxs[i].item()])

                    local_search.solve(neighborhood_size=slim_GSGP_parameters['pop_size'],
                                       generations=1000,
                                       start_gen=1000,
                                       early_stopping=10)

                    inds.append(local_search.individual)
                    fits.append(local_search.individual.fitness)

                # optimizer = local_search
                optimizer.elite = inds[np.argmin(fits)]

                if '*' in algo:
                    corr = class_metric(y_test, final_binarizer(torch.prod(optimizer.elite.test_semantics, dim = 0)))

                elif '+' in algo or algo == 'GSGP':
                    corr = class_metric(y_test, final_binarizer(torch.sum(optimizer.elite.test_semantics, dim = 0)))

                else:
                    print('Dont know what algorithm to use')

                print(corr)

                optimizer.elite.print_tree_representation()

                if slim_gsgp_solve_parameters['log'] > 0:
                    with open(os.path.join(os.getcwd(), "log", f"elite_looks_gametes_{day}.csv"), 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [algo, seed, unique_run_id, dataset, corr, optimizer.elite.get_tree_representation()])


                print(time.time() - start)
                print("THE USED SEED WAS", seed)



# elite_saving_path = os.path.join(os.getcwd(), "log", f"elite_looks_gametes_{day}.txt")
# with open(elite_saving_path, 'w+') as file:
#     file.write(str(elites))
