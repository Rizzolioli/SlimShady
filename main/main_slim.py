"""
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import time
import uuid

from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from parametrization import *
from utils.logger import log_settings
from utils.utils import get_terminals, train_test_split

ALGOS = ["SlimGSGP"]
DATA_LOADERS = [
    "toxicity", "concrete", "instanbul", "ppb",
    "resid_build_sale_price", "energy"
]
N_RUNS = 10
ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()


def main():
    """
    Main function to execute the SLIM_GSGP algorithm on specified datasets with various configurations.
    """
    for loader in DATA_LOADERS:
        for algo_name in ALGOS:
            for sig in [True]:
                for ttress in [False]:
                    slim_GSGP_parameters["two_trees"] = ttress

                    for op in ["mul", "sum"]:
                        slim_GSGP_parameters["operator"] = op

                        if ttress and sig:
                            continue

                        algo = f'{algo_name}_{1 + slim_GSGP_parameters["two_trees"] * 1}_{slim_GSGP_parameters["operator"]}_{sig}'

                        for seed in range(N_RUNS):
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

                            if dataset in slim_dataset_params.keys():
                                slim_GSGP_parameters["ms"] = slim_dataset_params[dataset]["ms"]
                                slim_GSGP_parameters['p_inflate'] = slim_dataset_params[dataset]["p_inflate"]
                            else:
                                slim_GSGP_parameters["ms"] = slim_dataset_params["other"]["ms"]
                                slim_GSGP_parameters['p_inflate'] = slim_dataset_params["other"]["p_inflate"]

                            slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']
                            slim_gsgp_pi_init["TERMINALS"] = TERMINALS

                            slim_GSGP_parameters["inflate_mutator"] = inflate_mutation(
                                FUNCTIONS=FUNCTIONS,
                                TERMINALS=TERMINALS,
                                CONSTANTS=CONSTANTS,
                                two_trees=slim_GSGP_parameters['two_trees'],
                                operator=slim_GSGP_parameters['operator'],
                                sig=sig
                            )

                            slim_gsgp_solve_parameters['run_info'] = [algo, UNIQUE_RUN_ID, dataset]

                            optimizer = SLIM_GSGP(
                                pi_init=slim_gsgp_pi_init,
                                **slim_GSGP_parameters,
                                seed=seed
                            )

                            optimizer.solve(
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                                curr_dataset=curr_dataset,
                                **slim_gsgp_solve_parameters
                            )

                            print(time.time() - start)
                            print("THE USED SEED WAS", seed)

    log_settings(
        path=os.path.join(os.getcwd(), "log", "settings.csv"),
        settings_dict=[globals()[d] for d in all_params["SLIM_GSGP"]],
        unique_run_id=UNIQUE_RUN_ID
    )


if __name__ == "__main__":
    main()
