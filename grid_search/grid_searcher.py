import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from utils.utils import gs_rmse, gs_size
import datasets.data_loader as ds
from utils.utils import generate_random_uniform
from GSGPRegressor import GSGPRegressor

# TODO: memory requirements code.

# setting up the datasets
datas = ["toxicity"]

# obtaining the data looading functions from the dataset names
#data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]


# creating a list of parameter dictionaries where sigmoid is not used when ttres is true.
params = [{
    'ms': [generate_random_uniform(0, 0.01), generate_random_uniform(0, 0.1),
           generate_random_uniform(0, 1), generate_random_uniform(0, 3),
           generate_random_uniform(0, 10)],

    'p_inflate': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_depth': [None, 17, 50, 100],
    'copy_parent': [True, False],
    'operator': ['mul','sum'],
    'sig': [False],
    'two_trees': [True]},

{
    'ms': [generate_random_uniform(0, 0.01), generate_random_uniform(0, 0.1),
           generate_random_uniform(0, 1), generate_random_uniform(0, 3),
           generate_random_uniform(0, 10)],
    'p_inflate': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_depth': [None, 17, 50, 100],
    'copy_parent': [True, False],
    'operator': ['mul','sum'],
    'sig': [True, False],
    'two_trees': [False]}]

# setting up both the rmse and the individual size as fitness parameters
scorers = {"rmse": make_scorer(gs_rmse, greater_is_better=False),
               "size": make_scorer(gs_size, greater_is_better=False)}

# running the grid search for all the intended datasets
for dataset in datas:

    # obtaining the data from the loader
    X, y = ds.load_merged_data(dataset, X_y=True)

    # creating the gsgp regressor model
    model = GSGPRegressor(random_state=74, test_elite=False, n_iter=500, verbose=0, pop_size=200, reconstruct=True) # reconstruct must be true
                                                                                                 # in order to evaluate the individual on new data

    # setting up the grid search
    search = GridSearchCV(model, params, verbose=3, scoring=scorers, refit=False, cv=5)

    # fitting the data
    search.fit(X, y)

    # print(f"Best mean score found was {search.best_score_}")
    # print(search.best_params_)

    # saving the cross validation results
    results =  pd.DataFrame(search.cv_results_)

    # saving the lower and upper bounds of the mutation step function
    results["param_ms"] = results["param_ms"].map(lambda x: (x.lower, x.upper))

    # logging the cross validation results
    results.to_csv(f"log/{dataset}_grid_search.csv")

    # todo: add to change the parametrization logger for slim results to yield --> slim grid full results --> name :)