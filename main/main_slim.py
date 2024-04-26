import time
import uuid

from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from algorithms.SLIM_GSGP.operators.mutators import *
from utils.logger import log_settings
from utils.utils import show_individual
########################################################################################################################

                                            # DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP_pls_work"]

#data_loaders = [ "toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price"]

data_loaders = ["ppb"]


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

    # for each dataset, run all the planned algorithms
    for algo_name in algos:

        for ttress in [False]:

            slim_GSGP_parameters["two_trees"] = ttress

            for op in ["mul"]:

                slim_GSGP_parameters["operator"] = op

                # getting the log file name according to the used parameters:
                algo = f'{algo_name}_{1 + slim_GSGP_parameters["two_trees"] * 1}_{slim_GSGP_parameters["operator"]}'

                # running each dataset + algo configuration n_runs times
                for seed in range(n_runs):
                    start = time.time()

                    if isinstance(loader, str):
                        # getting the name of the dataset
                        dataset = loader

                        curr_dataset = f"load_{dataset}"

                        TERMINALS = get_terminals(loader, seed + 1)

                        X_train, y_train = load_preloaded(loader, seed= seed + 1, training=True, X_y=True)

                        X_test, y_test = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

                        seed = seed + 74
                    else:

                        # Loads the data via the dataset loader
                        X, y = loader(X_y=True)

                        # getting the name of the dataset:
                        curr_dataset = loader.__name__

                        # getting the name of the dataset
                        dataset = loader.__name__.split("load_")[-1]

                        # getting the terminals and defining the terminal-dependant parameters
                        TERMINALS = get_terminals(loader)

                        # Performs train/test split
                        X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, p_test=settings_dict['p_test'], seed=seed)


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
                                                                              TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
                                                                              two_trees=slim_GSGP_parameters['two_trees'],
                                                                              operator=slim_GSGP_parameters['operator'],
                                                                              new=True)


                    # adding the dataset name and algorithm name to the run info for the logger
                    slim_gsgp_solve_parameters['run_info'] = [algo, unique_run_id, dataset]

                    optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=seed)

                    optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, curr_dataset=curr_dataset,
                                    **slim_gsgp_solve_parameters)

                    print(time.time() - start)

                    elites[seed] = {"structure":optimizer.elite.structure, "looks": show_individual(optimizer.elite, operator=slim_GSGP_parameters['operator']),
                                    "collection": optimizer.elite.collection}
                    print("THE USED SEED WAS", seed)

log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"), settings_dict=[globals()[d] for d in all_params["SLIM_GSGP"]], unique_run_id=unique_run_id)

elite_saving_path = os.path.join(os.getcwd(), "log", "elite_looks.txt")
with open(elite_saving_path, 'w+') as file:
    file.write(str(elites))

