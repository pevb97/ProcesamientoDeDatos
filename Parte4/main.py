import requests
def getRequestCSV(url: str):
    response = requests.get(url)
    if(response.status_code == requests.codes.ok):
        with open('datos.csv', 'wb') as archivo_local:
            archivo_local.write(response.content)
        return "Archivo guardado con exito"
    else:
        return "Peticion no exitosa"

print(getRequestCSV('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv'))