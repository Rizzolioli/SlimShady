from wtest import w_test
import pandas as pd
import numpy as np
import datetime
import time
import os
import csv

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

def stat_analysis(input_vars, target):

    found_loci = []
    found_loci_nb = []
    for i in range(input_vars.shape[1]):
        for j in range(i, input_vars.shape[1]):

            if i != j:

                _, pvalue, _, _ = w_test(input_vars[:,i], input_vars[:,j], target)

                print((i,j))
                print(pvalue)
                if pvalue < (0.05/((input_vars.shape[1]*(input_vars.shape[1]-1))/2)): #bonferroni /input_vars.shape[1]
                    found_loci.append((i,j))
                elif pvalue < 0.05:
                    found_loci_nb.append((i, j))
                else:
                    print('invalid combination')

    # print(found_loci)


    return found_loci, found_loci_nb


path = '../../../GAMETES dataset/data'
data_loaders = os.listdir(path)


# data_loaders = list(filter(lambda x: ('100a' in x ) , data_loaders) )

for loader in data_loaders:

    # Loads the data via the dataset loader
    data = pd.read_csv(path + '/' + loader, sep='\t')

    X = data.values[:, :-1]
    y = data.values[:, -1]

    start = time.time()
    print(loader[:-4])

    # try:
    found_loci, found_loci_nb = stat_analysis(X, y)

    if True:
        with open(os.path.join(os.getcwd(), "log", f"stat_wtest_gametes_{day}.csv"), 'a', newline='') as file:

            time_passed = time.time() - start

            writer = csv.writer(file)
            writer.writerow(
            [ loader[:-4], found_loci, found_loci_nb, time_passed]
            )
