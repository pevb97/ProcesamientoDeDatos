import pandas as pd
from typing import Set


def ej_1_cargar_datos_demograficos() -> pd.DataFrame:
    url = "https://public.opendatasoft.com/explore/dataset/us-cities-demographics/download/?format=csv&timezone=Europe/Berlin&lang=en&use_labels_for_header=true&csv_separator=%3B"
    data = pd.read_csv(url, sep=';')
    return data

def ej_2_cargar_calidad_aire(ciudades: Set[str]) -> None:
    pass
