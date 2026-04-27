import time
import uuid

from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from utils.logger import log_settings

########################################################################################################################

# CONFIGURATION

########################################################################################################################

algo_name = "SlimGSGP"

# Datasets to benchmark — pass a string (uses pre-loaded splits) or a loader function.
data_loaders = ["ppb"]
# data_loaders = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]

# Operator variants and mutation flags to sweep over.
operators  = ["sum", "mul"]   # combine blocks with sum or product
sig_values = [True]           # signed inflate mutation
two_trees_values = [False]    # use one tree (False) or two trees (True) for mutations

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

                    # dataset-specific mutation parameters
                    params = slim_dataset_params.get(dataset, slim_dataset_params["other"])
                    slim_GSGP_parameters["ms"]        = params["ms"]
                    slim_GSGP_parameters["p_inflate"] = params["p_inflate"]
                    slim_GSGP_parameters["p_deflate"] = 1 - params["p_inflate"]

                    slim_gsgp_pi_init["TERMINALS"] = TERMINALS
                    slim_GSGP_parameters["inflate_mutator"] = inflate_mutator(
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
