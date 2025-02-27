import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_excel('../../../Bicocca/GWAS/data/MG-PerMed database_Italy_shared 31224.xlsx', sheet_name='SIGNIFICANT NGS VARIANTS')

data.drop(columns = ['MG-PerMed', 'codice INNCB', 'Unnamed: 11'], inplace=True)
data.drop(index = [0, 3], inplace = True)

data['status (R/NR to IS drugs)'] = data['status (R/NR to IS drugs)'].replace({'NR?': 'NR'})

columns = list(data.columns)
columns.pop(0)
columns.append('status (R/NR to IS drugs)')

data = data[columns]

enc = OrdinalEncoder()

data = enc.fit_transform(data)

pd.DataFrame(data).to_csv('../../../Bicocca/GWAS/data/gwas_FINAL_cleaned_ordered.csv')