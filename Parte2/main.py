import pandas as pd

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
filtroDeads = dataset['DEATH_EVENT'] == 1
deadsDataset = dataset[filtroDeads]
lifesDataset = dataset[~filtroDeads]
promAgeDeads = deadsDataset['age'].mean()
promAgeLifes = lifesDataset['age'].mean()
print(promAgeDeads, promAgeLifes)
