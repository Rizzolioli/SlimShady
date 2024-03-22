import time
import uuid

from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from algorithms.SLIM_GSGP.operators.mutators import *
from utils.logger import log_settings

########################################################################################################################

                                            # DATASETS & ALGORITHMS

########################################################################################################################

# creating a list with the datasets that are to be benchmarked

datas = ["ld50"]

# datas = ["ppb"]

# obtaining the data loading functions using the dataset name
# data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

# defining the names of the algorithms to be run

algos = ["SlimGSGP"]

#data_loaders = [ "concrete", "instanbul", "ppb", "resid_build_sale_price"]

data_loaders = ["toxicity"]

########################################################################################################################

                                            # RUNNING THE ALGORITHM & DEFINING
                                            #    DATA-DEPENDANT PARAMETERS

########################################################################################################################
# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

# for each dataset
for loader in data_loaders:


    # for each dataset, run all the planned algorithms
    for algo in algos:

        # running each dataset + algo configuration n_runs times
        for seed in range(n_runs):
            start = time.time()

            if isinstance(loader, str):
                # getting the name of the dataset
                dataset = loader

                curr_dataset = f"load_{dataset}"

                TERMINALS = get_terminals(loader, seed+1)

                X_train, y_train = load_preloaded(loader, seed=seed+1, training=True, X_y=True)

                X_test, y_test = load_preloaded(loader, seed=seed+1, training=False, X_y=True)

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

            slim_gsgp_pi_init["TERMINALS"] = TERMINALS

            slim_GSGP_parameters["inflate_mutator"] = inflate_mutator(FUNCTIONS=FUNCTIONS,
                                                                      TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
                                                                      two_trees=True,
                                                                      operator=slim_GSGP_parameters['operator'])

            # getting the log file name according to the used parameters:
            log_file_name = f'{dataset}_{1 + slim_GSGP_parameters["inflate_mutator"].__closure__[4].cell_contents * 1}_{slim_GSGP_parameters["operator"]}.csv'

            # changing the logger path to the full file name
            slim_gsgp_solve_parameters["log_path"] = os.path.join(slim_gsgp_solve_parameters["log_path"], log_file_name)

            # adding the dataset name and algorithm name to the run info for the logger
            slim_gsgp_solve_parameters['run_info'] = [algo, unique_run_id, dataset]

            optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=seed)

            optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, curr_dataset=curr_dataset,
                            **slim_gsgp_solve_parameters)

            print(time.time() - start)

log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"), settings_dict=[globals()[d] for d in all_params["SLIM_GSGP"]], unique_run_id=unique_run_id)
