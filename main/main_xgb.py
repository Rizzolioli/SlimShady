import time

from datasets.data_loader import *
import csv

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split as tts_sklearn
from sklearn.decomposition import PCA
from xgboost import XGBClassifier



import datetime


now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]

path = '../../../GAMETES dataset/data'
data_loaders = os.listdir(path)

for loader in data_loaders:

        for seed in range(30):
            start = time.time()

            # Loads the data via the dataset loader
            data = pd.read_csv(path + '/' + loader, sep = '\t')

            X = data.values[:, :-1]
            y = data.values[:, -1]

            X_train, X_test, y_train, y_test = tts_sklearn(X, y,
                                                           stratify= y,
                                                           test_size=0.4,
                                                           shuffle = True,
                                                           random_state=seed)

            X_val, X_test, y_val, y_test = tts_sklearn(X_test, y_test,
                                                           stratify= y_test,
                                                           test_size=0.5,
                                                           shuffle = True,
                                                           random_state=seed)
            # getting the name of the dataset
            dataset = loader[:-4]

            clf = XGBClassifier()
            clf.fit(X_train, y_train, eval_set = [(X_val, y_val)])

            train_pred = clf.predict(X_train)
            train_corr = matthews_corrcoef(y_train, train_pred)

            val_pred = clf.predict(X_val)
            val_corr = matthews_corrcoef(y_val, val_pred)

            test_pred = clf.predict(X_test)
            test_corr = matthews_corrcoef(y_test, test_pred)


            with open(os.path.join(os.getcwd(), "log", f"xgb_gametes_{day}.csv"), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['XGBClassifier', seed, dataset, train_corr, val_corr, test_corr])