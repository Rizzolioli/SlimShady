"""
This script runs the StandardGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import time
import uuid

from slim.algorithms.GP.gp import GP
from slim.algorithms.GP.operators.mutators import mutate_tree_subtree
from slim.algorithms.GP.representations.tree_utils import tree_depth, tree_pruning
from slim.config.gp_config import *
from slim.utils.logger import log_settings
from slim.utils.utils import get_terminals, validate_inputs


# todo: would not be better to first log the settings and then perform the algorithm?
def gp(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
       dataset_name : str = None, n_runs: int = 30, pop_size: int = 100, n_iter: int = 1000, p_xo: float = 0.8,
       elitism: bool = True, n_elites: int = 1, max_depth: int = 17, init_depth: int = 6,
       log_path: str = os.path.join(os.getcwd(), "log", "gp.csv")):
    """
    Main function to execute the StandardGP algorithm on specified datasets

    Parameters
    ----------
    X_train: (torch.Tensor)
        Training input data.
    y_train: (torch.Tensor)
        Training output data.
    X_test: (torch.Tensor), optional
        Testing input data.
    y_test: (torch.Tensor), optional
        Testing output data.
    dataset_name : str, optional
        Dataset name, for logging purposes
    n_runs : int, optional
        The number of runs to execute for each dataset and algorithm combination (default is 30).
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    p_xo : float, optional
        The probability of crossover in the genetic programming algorithm. Must be a number between 0 and 1 (default is 0.8).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    n_elites : int, optional
        The number of elites.
    max_depth : int, optional
        The maximum depth for the GP trees.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved.

    Returns
    -------
    None
        This function does not return any values. It performs the execution of the StandardGP algorithm and logs the results.
    """

    validate_inputs(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, n_runs=n_runs,
                    pop_size=pop_size, n_iter=n_iter, elitism=elitism, n_elites=n_elites, init_depth=init_depth,
                    log_path=log_path)
    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"
    assert isinstance(max_depth, int), "Input must be a int"

    if not elitism:
        n_elites = 0

    # todo: here should be created in authomatic way the folder path defined by users
    # if not os.path.exists(log_path):
    #     os.mkdir(log_path)

    unique_run_id = uuid.uuid1()

    algo  = "StandardGP"
    gp_solve_parameters['run_info'] = [algo, unique_run_id, dataset_name]

    for seed in range(n_runs):

        TERMINALS = get_terminals(X_train)


        gp_pi_init["TERMINALS"] = TERMINALS
        gp_pi_init["init_pop_size"] = pop_size
        gp_pi_init["init_depth"] = init_depth

        gp_parameters["p_xo"] = p_xo
        gp_parameters["p_m"] = 1 - gp_parameters["p_xo"]
        gp_parameters["pop_size"] = pop_size
        gp_parameters["mutator"] = mutate_tree_subtree(
            gp_pi_init['init_depth'], TERMINALS, CONSTANTS, FUNCTIONS, p_c=gp_pi_init['p_c']
        )

        gp_solve_parameters["log_path"] = log_path
        gp_solve_parameters["elitism"] = elitism
        gp_solve_parameters["n_elites"] = n_elites
        gp_solve_parameters["max_depth"] = max_depth
        gp_solve_parameters["n_iter"] = n_iter
        gp_solve_parameters["tree_pruner"] = tree_pruning(
            TERMINALS=TERMINALS, CONSTANTS=CONSTANTS, FUNCTIONS=FUNCTIONS, p_c=gp_pi_init["p_c"]
        )
        gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=FUNCTIONS)
        if X_test is not None and y_test is not None:
            gp_solve_parameters["test_elite"] = True
        else:
            gp_solve_parameters["test_elite"] = False

        optimizer = GP(pi_init=gp_pi_init, **gp_parameters, seed=seed)
        optimizer.solve(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            curr_dataset = dataset_name,
            **gp_solve_parameters
        )

    log_settings(
        path=log_path[:-4] + "_settings.csv",
        settings_dict=[gp_solve_parameters,
                       gp_parameters,
                       gp_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
)


if __name__ == "__main__":
    data = 'instanbul'

    X_train, y_train = load_preloaded(data, seed= 1, training=True, X_y=True)
    X_test, y_test = load_preloaded(data, seed= 1, training=False, X_y=True)
    n_runs = 1

    gp(X_train = X_train, y_train = y_train,
         X_test = X_test, y_test = y_test,
         dataset_name=data,
         n_runs=1, pop_size=100, n_iter=10)
