import time

from datasets.data_loader import *
import csv

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split as tts_sklearn
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from mdr import MDR



import datetime


now = datetime.datetime.now()
day = now.strftime("%Y%m%d")


########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################



results = {}

# Loads the data via the dataset loader
data = pd.read_csv('../../../Bicocca/GWAS/data/gwas_cleaned_ordered.csv')

# getting the name of the dataset
dataset = 'GWAS'
curr_dataset = 'GWAS'

X = data.values[:, :-1]
y = data.values[:, -1]



for seed in range(30):

    start = time.time()


    X_train, X_test, y_train, y_test = tts_sklearn(X, y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   shuffle=True,
                                                   random_state=seed)



    my_mdr = MDR()


    train_pred = my_mdr.fit_transform(X_train, y_train)
    test_pred = my_mdr.transform(X_test)


    # train_pred = clf.predict(X_train)
    train_corr = matthews_corrcoef(y_train, train_pred)


    # test_pred = clf.predict(X_test)
    test_corr = matthews_corrcoef(y_test, test_pred)


    with open(os.path.join(os.getcwd(), "log", f"fixed_mdr_gwas_{day}.csv"), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['MDR', seed, dataset, train_corr,  test_corr])