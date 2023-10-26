import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('Notas Estudiantes.csv', delimiter=';')
data['aprobado'] = data['aprobado'].map({'Sí': True, 'No': False})
#print(data['materia'].unique())
data.boxplot('nota', by='materia')
plt.title('Distribución de Notas')
plt.show()
data['aprobado'].value_counts().plot.pie(
    y='aprobado',
    labels=['Aprobado', 'No Aprobados'],
    autopct='%.1f%%',
    title='Distribucion de aprobados')
plt.show()
