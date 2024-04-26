from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from algorithms.SLIM_GSGP.operators.mutators import *
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
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

            setattr(self, key, value)


    def set_params(self, **params):

        for parameter, value in params.items():

            if parameter in slim_gsgp_pi_init.keys():
                slim_gsgp_pi_init[parameter] = value
            elif parameter in slim_GSGP_parameters.keys():
                slim_GSGP_parameters[parameter] = value
            elif parameter in slim_gsgp_solve_parameters.keys():
                slim_gsgp_solve_parameters[parameter] = value

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
                                                                  two_trees=slim_GSGP_parameters['two_trees'],
                                                                  operator=slim_GSGP_parameters['operator'],
                                                                  single_tree_sigmoid=self.single_tree_sigmoid)

        # getting the log file name according to the used parameters:
        algo_name = f'{self.algo}_{1 + slim_GSGP_parameters["inflate_mutator"].__closure__[4].cell_contents * 1}_{slim_GSGP_parameters["operator"]}.csv'

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

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "..", "datasets/pre_loaded_data/TRAINING_1_TOXICITY.txt")
    df = pd.read_csv(path, sep=" ", header=None).iloc[:, :-1]
    # gsgp_reg = GSGPRegressor(random_state=0, test_elite = False, n_iter = 50)
    X, y = df.values[:, :-1], df.values[:, -1]

    params = {
        'ms': [generate_random_uniform(0, 0.01), generate_random_uniform(0, 0.1), generate_random_uniform(0, 1)
            , generate_random_uniform(0, 3), generate_random_uniform(0, 10)],
        'p_inflate': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_depth': [None, 17, 50, 100],
        'copy_parent': [True, False],
        'single_tree_sigmoid': [True, False]

    }


    def rmse(y_true, y_pred):
        return root_mean_squared_error(torch.from_numpy(y_true), y_pred[0])

    def size (y_true, y_pred):
        return y_pred[1]

    scorers = {"rmse": make_scorer(rmse, greater_is_better=False),
            "size": make_scorer(size, greater_is_better=False)}

    model = GSGPRegressor(random_state=0, test_elite=False, n_iter=100, verbose=0, pop_size = 200)

    search = GridSearchCV(model, params, verbose=3, scoring=scorers, refit=False) # todo: remove none, do this

    search.fit(X, y)

    print(f"Best mean score found was {search.best_score_}")
    print(search.best_params_)
    # print([sc.cell_contents for sc in search.best_params_['ms'].__closure__]) # to find out what random mutation step params where best


    # printed 2204.756630546981
# during evolution 2105.099418297513
