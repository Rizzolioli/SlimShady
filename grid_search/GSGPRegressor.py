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
class GSGPRegressor(BaseEstimator, RegressorMixin): # TODO: remove algo as a given parameter. It is always slim

    def __init__(self, random_state=0, algo="SlimGSGP", **params):

        self.algo = algo
        self.random_state = random_state

        # setting up the given model creation parameters to the parameterization dictionaries
        for key, value in params.items():
            if key in slim_gsgp_pi_init.keys():
                slim_gsgp_pi_init[key] = value
            elif key in slim_GSGP_parameters.keys():
                slim_GSGP_parameters[key] = value
            elif key in slim_gsgp_solve_parameters.keys():
                slim_gsgp_solve_parameters[key] = value
            elif key in mutation_parameters.keys():
                mutation_parameters[key] = value

            # setting up the given model creation parameters as GSGPRegressor class attributes
            setattr(self, key, value)


    def set_params(self, **params):

        # setting up the grid-searchable parameters to the parameterization dictionaries
        for parameter, value in params.items():
            if parameter in slim_gsgp_pi_init.keys():
                slim_gsgp_pi_init[parameter] = value
            elif parameter in slim_GSGP_parameters.keys():
                slim_GSGP_parameters[parameter] = value
            elif parameter in slim_gsgp_solve_parameters.keys():
                slim_gsgp_solve_parameters[parameter] = value
            elif parameter in mutation_parameters.keys():
                mutation_parameters[parameter] = value

        # setting up the grid-searchable parameters as GSGPRegressor class attributes
            setattr(self, parameter, value)

        # setting up probability of deflate in accordance to probability of inflating
        slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']

        # setting up the aforementioned probability as a GSGPRegressor attribute
        setattr(self, 'p_deflate', slim_GSGP_parameters['p_deflate'])

        return self # TODO: diogo why do we need to return self here?

    def fit(self, X, y=None):

        # checking that X and y are of consistent length while assuring X is be 2D and y 1D.
        X, y = check_X_y(X, y)

        # assuring that at least two samples are given for the estimation
        if len(X) < 2:
            raise ValueError("Estimator requires more than 1 sample to function") # TODO: why do we need at least 2?

        # obtaining the number of features in the dataset
        self.n_features_in_ = len(X[0])

        # creating a terminals dictionary based on the number of features
        TERMINALS = {f"x{i}": i for i in range(self.n_features_in_)}

        # saving the terminals to the parameterization dictionaries
        slim_gsgp_pi_init["TERMINALS"] = TERMINALS

        # creating X_train and y_train attributes, turning the data into torch Tensors.
        self.X_train, self.y_train = torch.from_numpy(X), torch.from_numpy(y)

        # setting up the inflate mutator based on the previously established/obtained parameterization variables
        slim_GSGP_parameters["inflate_mutator"] = inflate_mutator(FUNCTIONS=FUNCTIONS,
                                                                  TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
                                                                  two_trees=mutation_parameters['two_trees'],
                                                                  operator=slim_GSGP_parameters['operator'],
                                                                  sig = mutation_parameters['sig'])

        # getting the log file name according to the used parameters:
        #algo_name = f'{self.algo}_{1 + slim_GSGP_parameters["inflate_mutator"].__closure__[4].cell_contents * 1}_{slim_GSGP_parameters["operator"]}.csv'

        # getting the log file name according to the used parameters, saving 2 in the name if two_trees is true and 1 otherwise
        algo_name = f'{self.algo}_{1 + mutation_parameters["two_trees"] * 1}_{slim_GSGP_parameters["operator"]}' \
               f'_{mutation_parameters["sig"]}'

        # setting up the run_info # TODO: why algo_name, 1, test?
        slim_gsgp_solve_parameters['run_info'] = [algo_name, 1, "test"]

        # updating the dictionary ms based on the previously given ms in set_params
        slim_GSGP_parameters["ms"] = self.ms # TODO: does this need to be here? isnt this done previously indeed?

        # setting up the SLIM_GSGP model
        self.optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=self.random_state)

        # training our optimizer in order to obtain a final model
        self.optimizer.solve(X_train=self.X_train, X_test=None, y_train=self.y_train, y_test=None, curr_dataset="test",
                **slim_gsgp_solve_parameters) # todo: do we need to change curr_dataset?


    def score(self, X, y):
        # Implementation of score method
        return root_mean_squared_error(torch.from_numpy(y), self.predict(X))

    def predict(self, X):

        # making sure the regressor has been fitted before trying to perform predictions
        check_is_fitted(self)

        # validating the given input, assuring its a non-empty 2D array containing only finite values.
        X = check_array(X)

        # making sure the given input contains the same number of features as the ones used for training
        if len(X[0]) != self.n_features_in_:
            raise ValueError("The number of features present in the data to predict is different from the number used in fit.")

        # result = self.optimizer.elite.apply_individual(torch.from_numpy(X)) TODO: remove this ok?

        # reconstructing the fitted model's elite on the given prediction data, obtaining the final predictions.
        result = apply_individual_fixed(self.optimizer.elite, data=torch.from_numpy(X),
                                                            operator=slim_GSGP_parameters['operator'],
                                        sig=mutation_parameters['sig'])


        # returning both the final predicitions of the model (for future rmse calculation)
        # as well as the node count of the individual
        return result, self.optimizer.elite.nodes_count

    #TODO: DOCUMENT FILE