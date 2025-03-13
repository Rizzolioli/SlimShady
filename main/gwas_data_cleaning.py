import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_excel('../../../Bicocca/GWAS/data/MG-PerMed database_Italy_shared 31224.xlsx', sheet_name='ALL NGS DATA-GENETIC VARIANTS')

data.drop(columns = ['Cod. MG-PerMed', 'codice INNCB', 'familyID in ngs FILES ', 'individualID', 'Cod. RF-2016'], inplace=True)

columns = list(data.columns)
columns.pop(1)
columns.append('status (R/NR to IS drugs)')

data = data[columns]

enc = OrdinalEncoder()

data = enc.fit_transform(data)

pd.DataFrame(data).to_csv('../../../Bicocca/GWAS/data/gwas_cleaned_ordered.csv', index = None)