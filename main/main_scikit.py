import time

from datasets.data_loader import *
import csv

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split as tts_sklearn
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import datetime




models = {'DecisonTree' : DecisionTreeClassifier(),
          'SupportVectorMachine' : SVC(),
          'NaiveBayes' : GaussianNB(),
          'LogisticRegression' : LogisticRegression()}






now = datetime.datetime.now()
day = now.strftime("%Y%m%d")


########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]

path = '../../../GAMETES dataset/data'
data_loaders = os.listdir(path)

results = {}

for loader in data_loaders:

        # Loads the data via the dataset loader
        data = pd.read_csv(path + '/' + loader, sep='\t')

        X = data.values[:, :-1]
        y = data.values[:, -1]

#
#         # getting the name of the dataset
        dataset = loader[:-4]

        for model in models.keys():


            for seed in range(30):
                start = time.time()

                clf = models[model]

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


                with open(os.path.join(os.getcwd(), "log", f"models_gametes_{day}.csv"), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [model, seed, dataset, train_corr,  test_corr])