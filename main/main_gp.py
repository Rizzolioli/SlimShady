"""
This script runs the StandardGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import time
import uuid

from algorithms.GP.gp import GP
from algorithms.GP.operators.mutators import mutate_tree_subtree
from algorithms.GP.representations.tree_utils import tree_depth
from parametrization import *
from utils.logger import log_settings
from utils.utils import get_terminals

DATA_LOADERS = [
    "toxicity", "concrete", "instanbul", "ppb",
    "resid_build_sale_price", "energy"
]
ALGOS = ["StandardGP"]
N_RUNS = 10


def main():
    """
    Main function to execute the StandardGP algorithm on specified datasets with various configurations.

    This function iterates over different datasets and algorithm configurations,
    sets the necessary parameters, performs training and testing, and logs the results.
    """
    unique_run_id = uuid.uuid1()

    for loader in DATA_LOADERS:
        for algo in ALGOS:
            gp_solve_parameters['run_info'] = [algo, unique_run_id, loader]

            for seed in range(N_RUNS):
                start = time.time()

                if isinstance(loader, str):
                    dataset = loader
                    curr_dataset = f"load_{dataset}"
                    TERMINALS = get_terminals(loader, seed + 1)

                    X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True, X_y=True)
                    X_test, y_test = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

                gp_pi_init["TERMINALS"] = TERMINALS
                GP_parameters["mutator"] = mutate_tree_subtree(
                    gp_pi_init['init_depth'], TERMINALS, CONSTANTS, FUNCTIONS, p_c=gp_pi_init['p_c']
                )
                gp_solve_parameters["tree_pruner"] = tree_pruning(
                    TERMINALS=TERMINALS, CONSTANTS=CONSTANTS, FUNCTIONS=FUNCTIONS, p_c=gp_pi_init["p_c"]
                )
                gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=FUNCTIONS)

                optimizer = GP(pi_init=gp_pi_init, **GP_parameters, seed=seed)
                optimizer.solve(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    curr_dataset=curr_dataset,
                    **gp_solve_parameters
                )

                print(time.time() - start)

    log_settings(
        path=os.path.join(os.getcwd(), "log", "settings.csv"),
        settings_dict=[globals()[d] for d in all_params["GP"]],
        unique_run_id=unique_run_id,
    )


if __name__ == "__main__":
    main()
