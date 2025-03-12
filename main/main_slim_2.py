import time
import uuid

from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from algorithms.SLIM_GSGP.operators.mutators import *
from algorithms.SLIM_GSGP.operators.standard_geometric_crossover import *
from algorithms.SLIM_GSGP.operators.our_geometric_crossover import *
from algorithms.SLIM_GSGP.operators.our_geometric_crossover import *
from algorithms.SLIM_GSGP.operators.crossover_operators import improved_donor_xo, best_donor_xo
from utils.logger import log_settings
from utils.utils import show_individual
import datetime

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

data_loaders = [ "toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]


########################################################################################################################

# RUNNING THE ALGORITHM & DEFINING
#    DATA-DEPENDANT PARAMETERS

########################################################################################################################

# saving the elites looks:

elites = {}

# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

n_runs = 30
settings_dict = {"p_test": 0.2}

FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2}
}

CONSTANTS = {
    'constant_2': lambda x: torch.tensor(2).float(),
    'constant_3': lambda x: torch.tensor(3).float(),
    'constant_4': lambda x: torch.tensor(4).float(),
    'constant_5': lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float()
}

slim_gsgp_solve_parameters = {"elitism": True,
                              "log": 1,
                              "verbose": 1,
                              "test_elite": True,
                              "log_path": os.path.join(os.getcwd(), "log", f"xo_{day}.csv"),
                              "run_info": None,
                              "ffunction": rmse,
                              "n_iter": 2000,
                              "max_depth": None,
                              "n_elites": 1,
                              "reconstruct" : False
                              }

slim_GSGP_parameters = {"initializer": rhh,
                        "selector": tournament_selection_min_slim(2),
                        "crossover": None,
                        "ms": None,
                        "inflate_mutator": None,
                        "deflate_mutator": deflate_mutation,
                        "p_xo": 0.5,
                        "pop_size": 100,
                        "settings_dict": settings_dict,
                        "find_elit_func": get_best_min,
                        "p_inflate": None,
                        "copy_parent": None,
                        "operator": None
                        }
slim_GSGP_parameters["crossover"] = best_donor_xo()

mutation_parameters ={
"sig": None,
"two_trees": None
}


slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

slim_gsgp_pi_init = {'init_pop_size': slim_GSGP_parameters["pop_size"],
                     'init_depth': 6,
                     'FUNCTIONS': FUNCTIONS,
                     'CONSTANTS': CONSTANTS,
                     "p_c": 0}

all_params = {"SLIM_GSGP": ["slim_gsgp_solve_parameters", "slim_GSGP_parameters", "slim_gsgp_pi_init", "settings_dict"],
              "GSGP": ["gsgp_solve_parameters", "GSGP_parameters", "gsgp_pi_init", "settings_dict"],
              "GP": ["gp_solve_parameters", "GP_parameters", "gp_pi_init", "settings_dict"]}

slim_dataset_params = {"toxicity": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                        "ld50": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                       "concrete_strength": {"p_inflate": 0.5, "ms": generate_random_uniform(0, 0.3)},
                       "other": {"p_inflate": 0.3, "ms": generate_random_uniform(0, 1)}}


slim_GSGP_parameters['p_deflate'] = 0
slim_GSGP_parameters['p_inflate'] = 1


# RUNNING THE ALGORITHM & DEFINING
#    DATA-DEPENDANT PARAMETERS

########################################################################################################################

# saving the elites looks:

elites = {}

# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

# for each dataset
for loader in data_loaders:
    for (sig, ttress, op, gsgp) in [
        (True, True, "mul", False),  # SLIM*2SIG
        (True, True, "sum", False), #SLIM+2SIG
        (False, False, "mul", False),  # SLIM*ABS
        (False, False, "sum", False),  #SLIM+ABS
        (True, False, "mul", False), #SLIM*1SIG
        (True, False, "sum", False),  # SLIM+1SIG

    ]:

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
        elif (sig, ttress, op, gsgp) == (True, True, "mul", False):
            algo = 'SLIM+*2SIG'
        elif (sig, ttress, op, gsgp) == (True, False, "sum", False):
            algo = 'SLIM+1SIG'
        elif (sig, ttress, op, gsgp) == (False, False, "sum", False):
            algo = 'SLIM+ABS'
        elif (sig, ttress, op, gsgp) == (True, True, "sum", True):
            algo = 'GSGP'
        else:
            raise Exception('invalid variant')

        if op == 'std':
            op = 'sum'

        slim_GSGP_parameters["two_trees"] = ttress
        slim_GSGP_parameters["operator"] = op
        slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

        # running each dataset + algo configuration n_runs times
        for seed in range(n_runs):

            start = time.time()

            if isinstance(loader, str):
                dataset = loader
                curr_dataset = f"load_{dataset}"
                TERMINALS = get_terminals(loader, seed + 1)
                X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True, X_y=True)
                X_test, y_test = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

            else:
                X, y = loader(X_y=True)
                curr_dataset = loader.__name__
                dataset = loader.__name__.split("load_")[-1]
                TERMINALS = get_terminals(loader)
                X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                                    p_test=settings_dict['p_test'],
                                                                    seed=seed)

            # setting up the dataset related slim parameters:
            if dataset in slim_dataset_params.keys():
                slim_GSGP_parameters["ms"] = slim_dataset_params[dataset]["ms"]
                slim_GSGP_parameters['p_inflate'] = slim_dataset_params[dataset]["p_inflate"]

            else:
                slim_GSGP_parameters["ms"] = slim_dataset_params["other"]["ms"]
                slim_GSGP_parameters['p_inflate'] = slim_dataset_params["other"]["p_inflate"]

            # if cross != "no_xo":
            #     slim_GSGP_parameters['p_inflate'] = 1
            slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']

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
            slim_gsgp_solve_parameters['run_info'] = [algo, unique_run_id, dataset]

            optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=seed)

            optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                            curr_dataset=curr_dataset,
                            **slim_gsgp_solve_parameters)

            print(time.time() - start)
            print("THE USED SEED WAS", seed)
