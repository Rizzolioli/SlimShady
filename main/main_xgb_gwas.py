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



########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]



results = {}
data = pd.read_csv('../../../Bicocca/GWAS/data/gwas_cleaned_ordered.csv')

X = data.values[:, :-1]
y = data.values[:, -1]

#
#         # getting the name of the dataset
dataset = 'GWAS'


for seed in range(30):
    start = time.time()

    clf = XGBClassifier(seed=seed, n_estimators = 500, learning_rate = 0.0001, max_depth = 10
                        )

    X_train, X_test, y_train, y_test = tts_sklearn(X, y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   shuffle=True,
                                                   random_state=seed)

    # X_train, X_val, y_train, y_val = tts_sklearn(X_test, y_test,
    #                                                stratify= y_test,
    #                                                test_size=0.25,
    #                                                shuffle = True,
    #                                                random_state=seed)


    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    train_corr = matthews_corrcoef(y_train, train_pred)

    # val_pred = clf.predict(X_val)
    # val_corr = matthews_corrcoef(y_val, val_pred)

    test_pred = clf.predict(X_test)
    test_corr = matthews_corrcoef(y_test, test_pred)


    with open(os.path.join(os.getcwd(), "log", f"tuned_xgb_gwas_{day}.csv"), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['XGBClassifier', seed, dataset, train_corr,  test_corr])