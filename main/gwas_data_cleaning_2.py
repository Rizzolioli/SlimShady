import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

significant = pd.read_excel('../../../Bicocca/GWAS/data/MG-PerMed database_Italy_shared 31224.xlsx', sheet_name='SIGNIFICANT NGS VARIANTS')
all = pd.read_excel('../../../Bicocca/GWAS/data/MG-PerMed database_Italy_shared 31224.xlsx', sheet_name='ALL NGS DATA-GENETIC VARIANTS')

complete = pd.merge(significant, all, how = 'inner', left_on = 'MG-PerMed', right_on = 'Cod. MG-PerMed')

complete.drop(columns = ['MG-PerMed', 'codice INNCB_x',  'codice INNCB_y', 'Unnamed: 11', 'Cod. MG-PerMed',
                         'familyID in ngs FILES ', 'individualID', 'Cod. RF-2016'], inplace=True)
# complete.drop(index = [0, 3], inplace = True)

print(np.all((complete['status (R/NR to IS drugs)_y'] == complete['status (R/NR to IS drugs)_y']).values))
# complete['status (R/NR to IS drugs)'] = complete['status (R/NR to IS drugs)'].replace({'NR?': 'NR'})

columns = list(complete.columns)
columns.remove('status (R/NR to IS drugs)_y')
columns.remove('status (R/NR to IS drugs)_x')
columns.append('status (R/NR to IS drugs)_x')

complete = complete[columns]

complete.dropna(axis = 0, how = 'any', inplace = True)

enc = OrdinalEncoder()

complete = enc.fit_transform(complete)

pd.DataFrame(complete).to_csv('../../../Bicocca/GWAS/data/gwas_FINAL_cleaned_ordered.csv', index = None)
