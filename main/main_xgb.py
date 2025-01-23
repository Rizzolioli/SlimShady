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

for loader in ['2w_10a_0.4her.txt']:

        for seed in range(1):
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

            pred = clf.predict(X_test)
            print(matthews_corrcoef(y_test, pred))