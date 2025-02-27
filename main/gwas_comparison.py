import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data_all = pd.read_excel('../../../Bicocca/GWAS/data/MG-PerMed database_Italy_shared 31224.xlsx', sheet_name='ALL NGS DATA-GENETIC VARIANTS')

data_syn = pd.read_excel('../../../Bicocca/GWAS/data/MG-PerMed database_Italy_shared 31224.xlsx', sheet_name='SIGNIFICANT NGS VARIANTS')

common_indexes = [index for index in data_all[ 'codice INNCB'].values if index in data_syn[ 'codice INNCB'].values]

print(len(common_indexes))