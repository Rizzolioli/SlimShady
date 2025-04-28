from scipy.stats import chisquare, fisher_exact, mannwhitneyu, chi2_contingency
from statsmodels.stats.contingency_tables import Table
import pandas as pd
import csv
import os
import datetime
import time
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

def stat_analysis(input_vars, target, epistatic_loci, test = 'chisquare'):

    if test == 'chisquare':
        stest = chi2_contingency #pvalue 1 index
        pvalue_idx = 1
    elif test == 'fisher':
        stest = fisher_exact #pvalue 1 index
        pvalue_idx = 1
    elif test == 'trend':
        pass
        # stest = Table.test_ordinal_association #pvalue 4 index
        # pvalue_idx = 4
    # elif test == 'mannwu':
    #     stest = mannwhitneyu
    #     pvalue_idx = 1
    else:
        raise Exception("Invalid test, choose between chi2, fisher, trend or mannwu")

    found_loci = []
    for i in range(input_vars.shape[1]):
        if test == 'chisquare':
            table = pd.crosstab(input_vars[:,i], target)
            pvalue = stest(table)[pvalue_idx]

        elif test == 'fisher':
            mask = np.isin(input_vars[:,i], [0, 1])
            sub_group = input_vars[:,i][mask]
            sub_outcome = target[mask]
            table = pd.crosstab(sub_group, sub_outcome)
            pvalue = stest(table)[pvalue_idx]
        elif test == 'trend':
            df = pd.DataFrame({
                'group': input_vars[:,i],
                'outcome': target
            })

            # Cochran-Armitage is a test for trend, modeled via logistic regression
            model = smf.glm("outcome ~ group", data=df, family=sm.families.Binomial()).fit()
            # print(model.summary())

            # You can check the p-value for the trend via the coefficient for 'group'
            pvalue = model.pvalues['group']

        if pvalue < (0.05/input_vars.shape[1]): #bonferroni /input_vars.shape[1]
            found_loci.append(i)

    print(found_loci)


    return epistatic_loci[0] in found_loci, epistatic_loci[1] in found_loci, found_loci


path = '../../../GAMETES dataset/data'
data_loaders = os.listdir(path)



for loader in data_loaders:

    # Loads the data via the dataset loader
    data = pd.read_csv(path + '/' + loader, sep='\t')

    X = data.values[:, :-1]
    y = data.values[:, -1]

    for test in ['chisquare', 'trend']: #'fisher',
        start = time.time()

        # try:
        locus1, locus2, found_loci = stat_analysis(X, y, epistatic_loci=[X.shape[1]-2, X.shape[1]-1], test = test)

        # except:
        #     print('failed test')
        #     locus1, locus2 = False, False

        if True:
            with open(os.path.join(os.getcwd(), "log", f"stat_test_gametes_{day}.csv"), 'a', newline='') as file:

                time_passed = time.time() - start
                print([test, loader[:-4], locus1, locus2, time_passed])

                writer = csv.writer(file)
                writer.writerow(
                [test, loader[:-4], locus1, locus2, found_loci, time_passed]
                )