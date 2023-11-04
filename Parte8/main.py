import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt

def getRequestCSV(url: str):
    try:
        response = requests.get(url)
        if(response.status_code == requests.codes.ok):
            with open('datos.csv', 'wb') as archivo_local:
                archivo_local.write(response.content)
            return True
        else:
            return False
    except:
        return False

# LINK DATOS: https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv

def processData(dataframe: pd.DataFrame):
    dataSinFaltantes = dataframe.dropna()
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

#PARTE6
'''if len(sys.argv) > 1:
    if getRequestCSV(sys.argv[1]):
        data = pd.read_csv('datos.csv')
        processData(data)
    else:
        print('Fallo el proceso')'''
#PARTE 7
'''data = pd.read_csv('datosProcesados.csv')
data['age'].plot(title='Distribucion de Edades', kind='hist', edgecolor='black', xlabel='Edades', ylabel='Frecuencia')
plt.show()
dataAnemiaBySex = data[data['anaemia'] == 1].groupby('sex').size()
dataDiabetesBySex = data[data['diabetes'] == 1].groupby('sex').size()
dataFumadorBySex = data[data['smoking'] == 1].groupby('sex').size()
dataMuertoBySeX = data[data['DEATH_EVENT'] == 1].groupby('sex').size()
dataGroupedBySex = pd.DataFrame({'Anemia': dataAnemiaBySex, 'Diabetes': dataDiabetesBySex,
                            'Fumadores': dataFumadorBySex, 'Muertos': dataMuertoBySeX})
dataGroupedBySex.index = dataGroupedBySex.index.map({0: 'Mujeres', 1: 'Hombres'})
dataGroupedBySex = dataGroupedBySex.T
dataGroupedBySex.plot(title='Histograma Agrupado por Sexo' ,kind='bar', rot=0, ylabel='Cantidad', xlabel='Categorias')
plt.show()'''

#PARTE8
data = pd.read_csv('datosProcesados.csv')
newColumnsData = ['Anemia', 'Diabetes', 'Fumador', 'Muerto']
data = data.rename(columns={'anaemia': newColumnsData[0], 'diabetes': newColumnsData[1],
                            'smoking': newColumnsData[2], 'DEATH_EVENT': newColumnsData[3]})
fig, axes = plt.subplots(nrows=1, ncols=len(newColumnsData))
for index, column in enumerate(newColumnsData):
    columnCounts = data[column].map({1: 'Sí', 0: 'No'}).value_counts()
    axes[index].pie(columnCounts, labels=columnCounts.index, autopct='%1.1f%%')
    axes[index].set_title(column)
plt.tight_layout()
plt.show()