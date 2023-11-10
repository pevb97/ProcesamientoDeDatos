#%%
import pandas as pd
import numpy as np
import requests
import concurrent.futures

#%%
url = "https://public.opendatasoft.com/explore/dataset/us-cities-demographics/download/?format=csv&timezone=Europe/Berlin&lang=en&use_labels_for_header=true&csv_separator=%3B"
data = pd.read_csv(url, sep=';')

#%%
data.to_csv('data.csv')
# %%
def getRequest(info):
    response = requests.get(info['url'], headers={'X-Api-Key': 'irde+0wFc7WF7URoAEXvig==MmdeUCNxHk6fnE2q'})
    if response.status_code == 200:
        return {'city': info['nameCity'], 'data': response.json()}
    return None

# %%
cities = data['City'].unique()
infoCitites = [{"nameCity": city,"url": f'https://api.api-ninjas.com/v1/airquality?city={city}'} for city in cities]
#cities = map(lambda city: {city: getRequest(f'https://api.api-ninjas.com/v1/airquality?city={city}')}, cities)
#print(cities_dict)
#print(urls)
#%%
data_list = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(getRequest, infoCitites)
    for res in results:
        if res != None:
            data_dict = {'City': res['city']}
            for key in res['data'].keys():
                if not isinstance(res['data'][key], (int, float)):
                    data_dict[key] = res['data'][key]['concentration']
            data_list.append(data_dict)

#%%
dataCityContamination = pd.DataFrame(data_list)
#dataCityContamination.to_csv('dataCityContamination.csv')
# %%
merged_df = data.merge(dataCityContamination, on='City', how='left')
print(merged_df.info())
# %%
dataMerge = merged_df.drop(['Race', 'Count', 'Number of Veterans'], axis=1)
#dataMerge.to_csv('dataMerge.csv')
print(dataMerge.info())
# %%
filas_duplicadas = dataMerge.duplicated(keep='first')

# Filtrar el DataFrame original para obtener las filas duplicadas
filas_duplicadas = dataMerge[filas_duplicadas]
#filas_duplicadas.to_csv('duplicados.csv')
dataMergeSinDuplicados = dataMerge.drop_duplicates()
dataMergeSinDuplicados.to_csv('dataMergeSinDuplicados.csv')
print(dataMerge.info())
print(filas_duplicadas.head())
# %%
