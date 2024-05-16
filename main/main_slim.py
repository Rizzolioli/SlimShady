import random
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
from utils.logger import log_settings
from utils.utils import show_individual

########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]

data_loaders = ["resid_build_sale_price", "toxicity", "concrete", "instanbul", "ppb", "energy"]

########################################################################################################################

# RUNNING THE ALGORITHM & DEFINING
#    DATA-DEPENDANT PARAMETERS

########################################################################################################################

# saving the elites looks:

elites = {}

# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

# for each dataset
for loader in data_loaders:
    for algo_name in algos:
        for (sig, ttress, op) in [(True, False, "mul"), (False, False, "mul"), (True, True, "sum")]:
            if op == "sum":
                list_crossover = ["c", "sc", "adc-0.3", "adc-0.7", "sdc-0.3", "sdc-0.7"]
            else:
                list_crossover = ["sc", "sdc-0.3", "sdc-0.7"]
            for cross in list_crossover:
                if cross == "c":
                    list_prob = [0.2]
                else:
                    list_prob = [0.2, 0.5, 0.8]
                for cross_prob in list_prob:

                    slim_GSGP_parameters["two_trees"] = ttress
                    slim_GSGP_parameters["operator"] = op

                    slim_GSGP_parameters["p_xo"] = cross_prob
                    slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

                    # getting the log file name according to the used parameters:
                    algo = f'{algo_name}_{1 + slim_GSGP_parameters["two_trees"] * 1}_{slim_GSGP_parameters["operator"]}' \
                           f'_{sig}_{cross}_{cross_prob}'

                    print('SIG:', sig)
                    print('2T:', ttress)
                    print('OPERATOR', op)

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

                        if cross == "c":
                            slim_GSGP_parameters["crossover"] = slim_geometric_crossover(FUNCTIONS=FUNCTIONS,
                                                                                         TERMINALS=TERMINALS,
                                                                                         CONSTANTS=CONSTANTS,
                                                                                         operator=slim_GSGP_parameters[
                                                                                             'operator'])
                        elif cross == "ac":
                            slim_GSGP_parameters["crossover"] = slim_alpha_geometric_crossover(
                                operator=slim_GSGP_parameters[
                                    'operator'])

                        elif cross == "sc":
                            slim_GSGP_parameters["crossover"] = slim_swap_geometric_crossover

                        elif cross == "sda-0.3":
                            slim_GSGP_parameters["crossover"] = slim_alpha_deflate_geometric_crossover(
                                operator=slim_GSGP_parameters['operator'], perc_off_blocks=0.3)

                        elif cross == "sda-0.7":
                            slim_GSGP_parameters["crossover"] = slim_alpha_deflate_geometric_crossover(
                                operator=slim_GSGP_parameters['operator'], perc_off_blocks=0.7)

                        elif cross == "sdc-0.3":
                            slim_GSGP_parameters["crossover"] = slim_swap_deflate_geometric_crossover(perc_off_blocks=0.3)

                        elif cross == "sdc-0.7":
                            slim_GSGP_parameters["crossover"] = slim_swap_deflate_geometric_crossover(perc_off_blocks=0.7)

                        else:
                            break

                        # adding the dataset name and algorithm name to the run info for the logger
                        slim_gsgp_solve_parameters['run_info'] = [algo, unique_run_id, dataset]

                        optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=seed)

                        optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                        curr_dataset=curr_dataset,
                                        **slim_gsgp_solve_parameters)

                        print(time.time() - start)
                        print("THE USED SEED WAS", seed)

    log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"),
                 settings_dict=[globals()[d] for d in all_params["SLIM_GSGP"]], unique_run_id=unique_run_id)
