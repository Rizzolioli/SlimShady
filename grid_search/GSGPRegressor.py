from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from main.parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from algorithms.SLIM_GSGP.operators.mutators import *
from sklearn.model_selection import train_test_split, cross_validate
from utils.logger import log_settings
from sklearn.metrics import make_scorer


from algorithms.SLIM_GSGP.representations.individual import apply_individual_fixed
class GSGPRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, random_state=0, algo="SlimGSGP", **params):
        self.algo = algo
        self.random_state = random_state
        print(params.items())
        for key, value in params.items():
            if key in slim_gsgp_pi_init.keys():
                slim_gsgp_pi_init[key] = value
            elif key in slim_GSGP_parameters.keys():
                slim_GSGP_parameters[key] = value
            elif key in slim_gsgp_solve_parameters.keys():
                slim_gsgp_solve_parameters[key] = value
            elif parameter in mutation_parameters.keys():
                mutation_parameters[parameter] = value
            setattr(self, key, value)


    def set_params(self, **params):

        for parameter, value in params.items():

            if parameter in slim_gsgp_pi_init.keys():
                slim_gsgp_pi_init[parameter] = value
            elif parameter in slim_GSGP_parameters.keys():
                slim_GSGP_parameters[parameter] = value
            elif parameter in slim_gsgp_solve_parameters.keys():
                slim_gsgp_solve_parameters[parameter] = value
            elif parameter in mutation_parameters.keys():
                mutation_parameters[parameter] = value
            setattr(self, parameter, value)

        # setting up probability of deflate in accordance to probability of inflating
        slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']

        setattr(self, 'p_deflate', slim_GSGP_parameters['p_deflate'])

        return self

    def fit(self, X, y=None):

        X, y = check_X_y(X, y)

        if len(X) < 2:
            raise ValueError("Estimator requires more than 1 sample to function")

        self.n_features_in_ = len(X[0])

        TERMINALS = {f"x{i}": i for i in range(self.n_features_in_)}

        self.X_train, self.y_train = torch.from_numpy(X), torch.from_numpy(y)

        slim_gsgp_pi_init["TERMINALS"] = TERMINALS

        slim_GSGP_parameters["inflate_mutator"] = inflate_mutator(FUNCTIONS=FUNCTIONS,
                                                                  TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
                                                                  two_trees=mutation_parameters['two_trees'],
                                                                  operator=slim_GSGP_parameters['operator'],
                                                                  sig = mutation_parameters['sig'])

        # getting the log file name according to the used parameters:
        #algo_name = f'{self.algo}_{1 + slim_GSGP_parameters["inflate_mutator"].__closure__[4].cell_contents * 1}_{slim_GSGP_parameters["operator"]}.csv'

        algo_name = f'{self.algo}_{1 + mutation_parameters["two_trees"] * 1}_{slim_GSGP_parameters["operator"]}' \
               f'_{mutation_parameters["sig"]}'

        slim_gsgp_solve_parameters['run_info'] = [algo_name, 1, "test"]

        slim_GSGP_parameters["ms"] = self.ms

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

        #result = self.optimizer.elite.apply_individual(torch.from_numpy(X))

        result = apply_individual_fixed(self.optimizer.elite, data=torch.from_numpy(X), operator=slim_GSGP_parameters['operator'])

        return result , self.optimizer.elite.nodes_count