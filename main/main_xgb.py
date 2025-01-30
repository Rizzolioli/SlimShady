import time

from datasets.data_loader import *
import csv

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split as tts_sklearn
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV



import datetime


now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

parameters = {
                 # 'gamma': [0, 0.1, 0.4, 1.6, 3.2,  12.8,  51.2, 200],
                  'learning_rate': [0.01, 0.025, 0.1,  0.25,  0.5],
                  'max_depth': [1, 3, 4, 6],
                  'n_estimators': [ 500, 750, 1000],
                  'reg_alpha': [0, 0.1, 1, 10, 50],
                  'reg_lambda': [0, 1],
                  'subsample' : [0.1, 0.25, 0.5, 0.7]
              }

########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]

path = '../../GAMETES dataset/data'
data_loaders = os.listdir(path)

results = {}

for loader in data_loaders:

        # Loads the data via the dataset loader
        data = pd.read_csv(path + '/' + loader, sep='\t')

        X = data.values[:, :-1]
        y = data.values[:, -1]

        X_train, X_test, y_train, y_test = tts_sklearn(X, y,
                                                       stratify=y,
                                                       test_size=0.2,
                                                       shuffle=True,
                                                       random_state=1)
#
#         # getting the name of the dataset
        dataset = loader[:-4]
#
#         clf = XGBClassifier(seed=42,
#                             nthread=4)
#
#         grid_search = GridSearchCV(
#             estimator=clf,
#             param_grid=parameters,
#             scoring='f1_weighted',
#             n_jobs=10,
#             cv=3,
#             verbose=True
#         )
#
#         grid_search.fit(X_train, y_train)
#
#         print(dataset)
#         print(grid_search.best_params_)
#
#         results[dataset] = grid_search.best_params_
#
# print(results)

        for seed in range(30):
            start = time.time()

            clf = XGBClassifier(seed=seed, n_estimators = 500, learning_rate = 0.0001, max_depth = 10
                                )



            X_train, X_val, y_train, y_val = tts_sklearn(X_test, y_test,
                                                           stratify= y_test,
                                                           test_size=0.25,
                                                           shuffle = True,
                                                           random_state=seed)


            clf.fit(X_train, y_train, eval_set = [(X_val, y_val)])

            train_pred = clf.predict(X_train)
            train_corr = matthews_corrcoef(y_train, train_pred)

            val_pred = clf.predict(X_val)
            val_corr = matthews_corrcoef(y_val, val_pred)

            test_pred = clf.predict(X_test)
            test_corr = matthews_corrcoef(y_test, test_pred)


            with open(os.path.join(os.getcwd(), "log", f"tuned_xgb_gametes_{day}.csv"), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['XGBClassifier', seed, dataset, train_corr, val_corr, test_corr])