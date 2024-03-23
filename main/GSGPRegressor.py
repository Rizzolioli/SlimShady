import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import time
import uuid
from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from algorithms.SLIM_GSGP.operators.mutators import *
from utils.logger import log_settings

class GSGPRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, random_state=0, algo="SlimGSGP", **params):
        self.algo = algo
        self.random_state = random_state
        for key, value in params.items():
            if key in slim_gsgp_pi_init.keys():
                slim_gsgp_pi_init[key] = value
            elif key in slim_GSGP_parameters.keys():
                slim_GSGP_parameters[key] = value
            elif key in slim_gsgp_solve_parameters.keys():
                slim_gsgp_solve_parameters[key] = value
            setattr(self, key, value)

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)

        if len(X) < 2:
            raise ValueError("Estimator requires more than 1 sample to function")

        self.n_features_in_ = len(X[0])
        TERMINALS = {f"x{i}": i for i in range(len(X[0]))}

        self.X_train, self.y_train = torch.from_numpy(X), torch.from_numpy(y)

        slim_gsgp_pi_init["TERMINALS"] = TERMINALS
        slim_GSGP_parameters["inflate_mutator"] = inflate_mutator(FUNCTIONS=FUNCTIONS,
                                                                  TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
                                                                  two_trees=slim_GSGP_parameters['two_trees'],
                                                                  operator=slim_GSGP_parameters['operator'])
        # getting the log file name according to the used parameters:
        algo_name = f'{self.algo}_{1 + slim_GSGP_parameters["inflate_mutator"].__closure__[4].cell_contents * 1}_{slim_GSGP_parameters["operator"]}.csv'
        slim_gsgp_solve_parameters['run_info'] = [algo_name, 1, "test"]
        self.optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=self.random_state)


        self.optimizer.solve(X_train=self.X_train, X_test=None, y_train=self.y_train, y_test=None, curr_dataset="test",
                **slim_gsgp_solve_parameters)

    def score(self, X, y):
        # Implementation of score method
        return root_mean_squared_error(torch.from_numpy(y), self.predict(X))

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)  # Input validation
        if len(X[0])!= self.n_features_in_:
            raise ValueError("The number of features present in the data to predict is different from the number used in fit.")
        result = self.optimizer.elite.apply_individual(torch.from_numpy(X))
        return result


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "..", "datasets/pre_loaded_data/TRAINING_1_TOXICITY.txt")
    df = pd.read_csv(path, sep = " ", header=None).iloc[:, :-1]
    print(df.shape)
    print({f"x{i}": i for i in range(len(df.iloc[0]))})
    gsgp_reg = GSGPRegressor(random_state=0, test_elite = False, n_iter = 100)
    X, y = df.values[:, :-1], df.values[:, -1]
    gsgp_reg.fit(X, y)
    result = gsgp_reg.predict(X)
    score = gsgp_reg.score(X, y)
    print(result)
    print(score)
    print("Finish")


