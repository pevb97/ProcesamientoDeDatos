import pandas as pd
import plotly.graph_objects as go

data = pd.read_csv('Notas Estudiantes.csv', delimiter=';')

graphBoxPlot = go.Figure()

for materia in data['materia'].unique():
    materiaData = data[data['materia'] == materia]
    graphBoxPlot.add_trace(go.Box(x=materiaData['materia'], y=materiaData['nota'], name=materia))
graphBoxPlot.update_layout(title='Distribución de notas')

aprobadosData = data['aprobado'].value_counts()
graphPie = go.Figure()
graphPie.add_trace(go.Pie(labels=aprobadosData.index, values=aprobadosData))

graphPie.update_layout(title="Distribucion de Aprobados")

graphBoxPlot.write_html('graficaBoxPlot.html')
graphPie.write_html('graficaPie.html')
