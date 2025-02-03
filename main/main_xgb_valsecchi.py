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


data = pd.read_csv('../../../Bicocca/Valsecchi/data/icuyn_dataset.csv')

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = tts_sklearn(X, y,
                                               stratify=y,
                                               test_size=0.2,
                                               shuffle=True,
                                               random_state=1)
#
# # getting the name of the dataset
# dataset = loader[:-4]
#
# clf = XGBClassifier(nthread=4,
#                     seed=42)
#
# grid_search = GridSearchCV(
#     estimator=clf,
#     param_grid=parameters,
#     scoring='f1_weighted',
#     n_jobs=10,
#     cv=3,
#     verbose=True
# )
#
# grid_search.fit(X_train, y_train)

for seed in range(1):
    start = time.time()

    clf = XGBClassifier(nthread=4,
                        seed=seed,
                        ) #**grid_search.best_params_



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

    print(train_corr)
    print(val_corr)
    print(test_corr)


    # with open(os.path.join(os.getcwd(), "log", f"not_tuned_xgb_valsecchi_{day}.csv"), 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(
    #         ['XGBClassifier', seed, 'valsecchi', train_corr, val_corr, test_corr, ]) #grid_search.best_params_