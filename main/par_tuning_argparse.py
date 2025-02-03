import argparse
import os
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split as tts_sklearn, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import json


def main(args):
    # Parse the models and param grids from JSON
    models = json.loads(args.models)
    param_grids = json.loads(args.param_grids)

    results = {}
    path = args.data_path
    data_loaders = os.listdir(path)

    for model_name in models.keys():
        print(model_name)
        data_results = {}

        for loader in data_loaders:
            data = pd.read_csv(os.path.join(path, loader), sep='\t')

            data_name = loader[:-4]
            X = data.values[:, :-1]
            y = data.values[:, -1]

            X_train, X_test, y_train, y_test = tts_sklearn(X, y, test_size=0.4, stratify=y)

            estimator = eval(models[model_name])()  # Initialize the model dynamically

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grids[model_name],
                scoring='f1_weighted',
                n_jobs=10,
                cv=3,
                verbose=True
            )

            grid_search.fit(X_train, y_train)

            print(data_name)
            print(grid_search.best_params_)

            data_results[data_name] = grid_search.best_params_

        results[model_name] = data_results

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GridSearch with specified models and parameters.")
    parser.add_argument('--models', required=True, help="JSON string of models with their import names.")
    parser.add_argument('--param_grids', required=True, help="JSON string of parameter grids for each model.")
    parser.add_argument('--data_path', required=True, help="Path to the dataset directory.")
    args = parser.parse_args()

    main(args)
