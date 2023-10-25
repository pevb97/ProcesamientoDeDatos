import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Datos de calificaciones de los estudiantes
matematicas = rng.integers(50, 100, 20)
ciencias = rng.integers(40, 95, 20)
literatura = rng.integers(60, 100, 20)

# Datos de errores para el gráfico de barras de error
errores_matematicas = rng.uniform(2, 5, 2)
errores_matematicas = [min(errores_matematicas), max(errores_matematicas)]

errores_ciencias = rng.uniform(1, 4, 2)
errores_ciencias = [min(errores_ciencias), max(errores_ciencias)]

errores_literatura = rng.uniform(3, 6, 2)
errores_literatura = [min(errores_literatura), max(errores_literatura)]

fig, axes = plt.subplots(3, 1, figsize=(6, 10))

axes[0].scatter(matematicas, ciencias)
axes[0].set_xlabel('Calificaciones de Matemáticas')
axes[0].set_ylabel('Calificaciones de Ciencias')
axes[0].set_title('Relación entre las calificaciones de Matematicas y Ciencias')

axes[1].errorbar(['Matematicas', 'Ciencias', 'Literatura'],[matematicas.mean(), ciencias.mean(), literatura.mean()],
            yerr=tuple(zip(errores_matematicas, errores_ciencias, errores_literatura)), fmt='o', capsize=5)
axes[1].set_ylabel('Califica<iones Promedio')
axes[1].set_title('Calificaciones promedio con barras de error')

axes[2].hist(matematicas, edgecolor='k')
axes[2].set_xlabel('Calificaciones de Matemáticas')
axes[2].set_ylabel('Frecuencia')
axes[2].set_title('Distribucion de las calificaciones de Matematicas')
fig.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
