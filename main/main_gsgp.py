import uuid
from parametrization import *
from algorithms.GSGP.gsgp import GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from utils.logger import log_settings

from utils.utils import protected_div, mean_, get_best_min, get_best_max
from evaluators.fitness_functions import rmse
from algorithms.GP.operators.initializers import rhh
from algorithms.GP.operators.crossover_operators import crossover_trees
from algorithms.GSGP.operators.crossover_operators import geometric_crossover, combined_geometric_crossover
from algorithms.GSGP.operators.mutators import *
from algorithms.GP.operators.selection_algorithms import tournament_selection_min
from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
from datasets.data_loader import *
from algorithms.SLIM_GSGP.operators.mutators import *
from algorithms.GP.representations.tree_utils import tree_pruning
from utils.utils import generate_random_uniform

import datetime

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")


########################################################################################################################

                                            # PARAMETERS

########################################################################################################################

gsgp_solve_parameters = {"elitism": True,
                         "log": 5,
                         "verbose": 1,
                         "test_elite": True,
                         "log_path": os.path.join(os.getcwd(), "log", f"GSGP_CGXO_{day}.csv"),
                         "run_info": None,
                         "ffunction": rmse,
                         "n_iter": 1000,
                         "reconstruct": False,
                         "n_elites": 1,
                         }

GSGP_parameters = {"initializer": rhh,
                   "selector": tournament_selection_min(2),
                   "crossover": None,
                   "ms": None,
                   "mutator": standard_geometric_mutation,
                   "p_xo": None,
                   "pop_size": 100,
                   "settings_dict": settings_dict,
                   "find_elit_func": get_best_min
                   }


gsgp_pi_init = {'init_pop_size': GSGP_parameters["pop_size"],
                'init_depth': 8,
                'FUNCTIONS': FUNCTIONS,
                'CONSTANTS': CONSTANTS,
                "p_c": 0}



########################################################################################################################

                                            # DATASETS & ALGORITHMS

########################################################################################################################

algos = ["GSGP"]

# data_loaders = [ "airfoil", "concrete_slump", "concrete_strength", "ppb", "ld50", "bioavalability", "yatch"]
data_loaders = [load_yatch, load_airfoil, load_concrete_slump, load_concrete_strength, load_ppb,
                load_bioav, load_ld50]

########################################################################################################################

                                            # RUNNING THE ALGORITHM & DEFINING
                                            #    DATA-DEPENDANT PARAMETERS

########################################################################################################################
# attibuting a unique id to the run
unique_run_id = uuid.uuid1()


# for each dataset
for loader in data_loaders:

    # getting the name of the dataset
    dataset = loader.__name__.split("load_")[-1]

    for experiment in ['only_gsm', 'only_gxo', 'only_cgxo']:
        # adding the dataset name and algorithm name to the run info for the logger
        gsgp_solve_parameters['run_info'] = [algos[0], experiment, unique_run_id ,dataset]

        # running each dataset + algo configuration n_runs times
        for seed in range(n_runs):

            # Loads the data via the dataset loader
            X, y = loader(X_y=True)



            # getting the terminals and defining the terminal-dependant parameters
            TERMINALS = {f"x{i}": i for i in range(X.shape[1])}
            # Performs train/test split
            X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                                p_test=settings_dict['p_test'],
                                                                seed=seed)

            GSGP_parameters["ms"] = generate_random_uniform(0, torch.median(y_train).item())

            if experiment == 'only_gsm':

                GSGP_parameters["p_xo"] = 0
                GSGP_parameters["crossover"] = None

            elif experiment == 'only_gxo':

                GSGP_parameters["p_xo"] = 1
                GSGP_parameters["crossover"] = geometric_crossover


            elif experiment == 'only_cgxo':

                GSGP_parameters["p_xo"] = 1
                GSGP_parameters["crossover"] = combined_geometric_crossover

            else:

                raise Exception("invalid experiment")

            GSGP_parameters["p_m"] = 1 - GSGP_parameters["p_xo"]

            gsgp_pi_init["TERMINALS"] = TERMINALS

            optimizer = GSGP(pi_init=gsgp_pi_init, **GSGP_parameters, seed=seed)

            optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, curr_dataset=dataset, **gsgp_solve_parameters)

log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"), settings_dict=[globals()[d] for d in all_params["GSGP"]], unique_run_id=unique_run_id)

