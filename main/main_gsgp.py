"""
This script runs the StandardGSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid

from algorithms.GSGP.gsgp import GSGP
from parametrization import *
from utils.logger import log_settings
from utils.utils import get_terminals

DATA_LOADERS = [
    "toxicity", "concrete", "instanbul", "ppb",
    "resid_build_sale_price", "energy"
]
ALGOS = ["StandardGSGP"]
N_RUNS = 10


def main():
    """
    Main function to execute the StandardGSGP algorithm on specified datasets with various configurations.

    This function iterates over different datasets and algorithm configurations,
    sets the necessary parameters, performs training and testing, and logs the results.
    """
    unique_run_id = uuid.uuid1()

    for loader in DATA_LOADERS:
        dataset = loader

        for algo_name in ALGOS:
            gsgp_solve_parameters["run_info"] = [algo_name, unique_run_id, dataset]
            algo_name = f"{algo_name}"

            for seed in range(N_RUNS):
                if isinstance(loader, str):
                    dataset = loader
                    curr_dataset = f"load_{dataset}"
                    TERMINALS = get_terminals(loader, seed + 1)

                    X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True, X_y=True)
                    X_test, y_test = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

                gsgp_pi_init["TERMINALS"] = TERMINALS
                optimizer = GSGP(pi_init=gsgp_pi_init, **GSGP_parameters, seed=seed)

                optimizer.solve(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    curr_dataset=curr_dataset,
                    **gsgp_solve_parameters
                )

    log_settings(
        path=os.path.join(os.getcwd(), "log", "settings.csv"),
        settings_dict=[globals()[d] for d in all_params["GSGP"]],
        unique_run_id=unique_run_id,
    )


if __name__ == "__main__":
    main()
