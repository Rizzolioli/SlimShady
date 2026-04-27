import os
import time
import uuid

import torch

from utils.utils import protected_div, get_best_min, get_terminals, train_test_split, generate_random_uniform
from evaluators.fitness_functions import rmse
from algorithms.GP.operators.initializers import rhh
from algorithms.GSGP.operators.crossover_operators import geometric_crossover
from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation, deflate_mutation
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from datasets.data_loader import load_preloaded
from utils.logger import log_settings

########################################################################################################################

# TREE PARAMETERS

########################################################################################################################

FUNCTIONS = {
    'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
}

CONSTANTS = {
    'constant_2':  lambda x: torch.tensor(2).float(),
    'constant_3':  lambda x: torch.tensor(3).float(),
    'constant_4':  lambda x: torch.tensor(4).float(),
    'constant_5':  lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float(),
}

########################################################################################################################

# RUN PARAMETERS

########################################################################################################################

n_runs = 30
settings_dict = {"p_test": 0.2}

########################################################################################################################

# SLIM-GSGP PARAMETERS

########################################################################################################################

slim_gsgp_solve_parameters = {
    "elitism":    True,
    "log":        0,        # 0 = disabled, 1 = enabled
    "verbose":    1,
    "test_elite": True,
    "log_path":   os.path.join(os.getcwd(), "log", "results.csv"),
    "run_info":   None,
    "ffunction":  rmse,
    "n_iter":     2000,
    "max_depth":  None,
    "n_elites":   1,
    "reconstruct": True,
}

slim_GSGP_parameters = {
    "initializer":     rhh,
    "selector":        tournament_selection_min_slim(2),
    "crossover":       geometric_crossover,
    "ms":              None,    # set per-dataset via slim_dataset_params
    "inflate_mutator": None,    # built per-run below
    "deflate_mutator": deflate_mutation,
    "p_xo":            0,
    "pop_size":        100,
    "settings_dict":   settings_dict,
    "find_elit_func":  get_best_min,
    "p_inflate":       None,    # set per-dataset via slim_dataset_params
    "p_deflate":       None,    # derived as 1 - p_inflate
    "copy_parent":     None,
    "operator":        None,    # set per-run: "sum" or "mul"
    "two_trees":       False,
}

slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

slim_gsgp_pi_init = {
    'init_pop_size': slim_GSGP_parameters["pop_size"],
    'init_depth':    6,
    'FUNCTIONS':     FUNCTIONS,
    'CONSTANTS':     CONSTANTS,
    "p_c":           0,
}

# Dataset-specific inflate probability and mutation step size.
# Datasets not listed here fall back to "other".
slim_dataset_params = {
    "toxicity": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
    "concrete": {"p_inflate": 0.5, "ms": generate_random_uniform(0, 0.3)},
    "other":    {"p_inflate": 0.3, "ms": generate_random_uniform(0, 1)},
}

all_params = {
    "SLIM_GSGP": ["slim_gsgp_solve_parameters", "slim_GSGP_parameters", "slim_gsgp_pi_init", "settings_dict"],
}

########################################################################################################################

# DATASETS & SWEEP CONFIGURATION

########################################################################################################################

algo_name = "SlimGSGP"

# Datasets to run — string keys use pre-loaded splits; pass a loader function for custom data.
data_loaders = ["ppb"]
# data_loaders = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]

operators        = ["sum", "mul"]  # how blocks are combined
sig_values       = [True]          # signed inflate mutation
two_trees_values = [False]         # one tree (False) or two trees (True) per mutation

########################################################################################################################

# RUN

########################################################################################################################

unique_run_id = uuid.uuid1()

for loader in data_loaders:
    for sig in sig_values:
        for two_trees in two_trees_values:
            slim_GSGP_parameters["two_trees"] = two_trees

            for op in operators:
                slim_GSGP_parameters["operator"] = op

                algo = f'{algo_name}_{1 + two_trees}_{op}_{sig}'

                for seed in range(n_runs):
                    start = time.time()

                    if isinstance(loader, str):
                        dataset = loader
                        curr_dataset = f"load_{dataset}"
                        TERMINALS = get_terminals(loader, seed + 1)
                        X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True,  X_y=True)
                        X_test,  y_test  = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)
                    else:
                        X, y = loader(X_y=True)
                        curr_dataset = loader.__name__
                        dataset = loader.__name__.split("load_")[-1]
                        TERMINALS = get_terminals(loader)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X=X, y=y, p_test=settings_dict['p_test'], seed=seed)

                    # apply dataset-specific mutation parameters
                    params = slim_dataset_params.get(dataset, slim_dataset_params["other"])
                    slim_GSGP_parameters["ms"]        = params["ms"]
                    slim_GSGP_parameters["p_inflate"] = params["p_inflate"]
                    slim_GSGP_parameters["p_deflate"] = 1 - params["p_inflate"]

                    slim_gsgp_pi_init["TERMINALS"] = TERMINALS
                    slim_GSGP_parameters["inflate_mutator"] = inflate_mutation(
                        FUNCTIONS=FUNCTIONS,
                        TERMINALS=TERMINALS,
                        CONSTANTS=CONSTANTS,
                        two_trees=two_trees,
                        operator=op,
                        sig=sig,
                    )

                    slim_gsgp_solve_parameters["run_info"] = [algo, unique_run_id, dataset]

                    optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=seed)
                    optimizer.solve(
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                        curr_dataset=curr_dataset,
                        **slim_gsgp_solve_parameters,
                    )

                    print(f"[{dataset}] seed={seed} op={op} sig={sig} time={time.time()-start:.1f}s")
                    optimizer.elite.print_tree_representation()

log_settings(
    path=os.path.join(os.getcwd(), "log", "settings.csv"),
    settings_dict=[globals()[d] for d in all_params["SLIM_GSGP"]],
    unique_run_id=unique_run_id,
)
