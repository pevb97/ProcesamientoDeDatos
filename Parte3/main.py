import pandas as pd

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
filtroDeads = dataset['DEATH_EVENT'] == 1
deadsDataset = dataset[filtroDeads]
lifesDataset = dataset[~filtroDeads]
promAgeDeads = deadsDataset['age'].mean()
promAgeLifes = lifesDataset['age'].mean()
print(dataset.info())
dataset[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']] = dataset[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']].astype('bool')
print(dataset.info())
dataset.to_csv('heart_failure_clinical_records_dataset.csv', index=False)
generoFumadores = dataset.groupby(['sex', 'smoking']).size()
print(generoFumadores)


