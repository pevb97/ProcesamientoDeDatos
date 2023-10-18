import requests
import pandas as pd

def getRequestCSV(url: str):
    response = requests.get(url)
    if(response.status_code == requests.codes.ok):
        with open('datos.csv', 'wb') as archivo_local:
            archivo_local.write(response.content)
        return "Archivo guardado con exito"
    else:
        return "Peticion no exitosa"
    
#print(getRequestCSV('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv'))

data = pd.read_csv('datos.csv')
def processData(Dataframe: pd.DataFrame):
    dataSinFaltantes = data.dropna()
    dataSinFaltantes.drop_duplicates(inplace=True)
    dataSinAtipicos = dataSinFaltantes.copy()
    for columna in dataSinAtipicos.columns:
        if dataSinAtipicos[columna].dtype in ['int64', 'float64']:
            Q1 = dataSinAtipicos[columna].quantile(0.25)
            Q3 = dataSinAtipicos[columna].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            dataSinAtipicos = dataSinAtipicos[(dataSinAtipicos[columna] >= limite_inferior) & (dataSinAtipicos[columna] <= limite_superior)]
    dataSinAtipicos['ageCategory'] = pd.cut(dataSinAtipicos['age'], bins=[0, 12, 19, 39, 59, float('inf')], labels=['Niño', 'Adolescente', 'Joven adulto', 'Adulto', 'Adulto mayor'])
    dataSinAtipicos.to_csv('datosProcesados.csv', index=False)

processData(data)