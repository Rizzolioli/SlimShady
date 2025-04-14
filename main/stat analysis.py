from scipy.stats import chisquare, fisher_exact
from statsmodels.stats.contingency_tables import Table
import pandas as pd
import csv
import os
import datetime
import time


now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

def stat_analysis(input_vars, target, epistatic_loci, test = 'chisquare'):

    if test == 'chisquare':
        stest = chisquare #pvalue 1 index
        pvalue_idx = 1
    elif test == 'fisher':
        stest = fisher_exact #pvalue 1 index
        pvalue_idx = 1
    elif test == 'trend':
        stest = Table.test_ordinal_association #pvalue 4 index
        pvalue_idx = 4
    else:
        raise Exception("Invalid test, choose between chi2, fisher or trend")

    found_loci = []
    for i in range(input_vars.shape[1]):
        pvalue = stest(input_vars[:,i], target)[pvalue_idx]
        if pvalue < (0.05/input_vars.shape[1]):
            found_loci.append(i)


    return epistatic_loci[0] in found_loci, epistatic_loci[1] in found_loci


path = '../../../GAMETES dataset/data'
data_loaders = os.listdir(path)



for loader in data_loaders:

    # Loads the data via the dataset loader
    data = pd.read_csv(path + '/' + loader, sep='\t')

    X = data.values[:, :-1]
    y = data.values[:, -1]

    for test in ['chisquare', 'fisher', 'trend']:
        start = time.time()

        locus1, locus2 = stat_analysis(X, y, epistatic_loci=[X.shape[1]-2, X.shape[1]-1], test = test)



        with open(os.path.join(os.getcwd(), "log", f"stat_test_gametes_{day}.csv"), 'a', newline='') as file:

            time_passed = time.time() - start
            print([test, loader[:-4], locus1, locus2, time_passed])

            writer = csv.writer(file)
            writer.writerow(
            [test, loader[:-4], locus1, locus2, time_passed]
            )