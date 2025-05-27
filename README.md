

### **A. Seleccionar un Sector y Definir una Necesidad Real**

*Fallas en vehículos: por no realizarle mantenimiento oportuno*

Sector: **Empresas con vehículos**

**Problema Identificado:**

Fallas de vehículos: vehículos varados por daños de motor, dirección, transmisión, culata, caja de dirección, 

Costos muy altos por mano de obra y repuestos:



### **Solución propuesta**

Construir un **modelo de machine learning con `KNeighborsRegressor`** para predecir la variable **`Km_Pend`** (kilómetros pendientes,   en un contexto de mantenimiento vehicular) a partir de un conjunto de variables numéricas y categóricas.

Brinda una **solución de regresión predictiva** para estimar **`Km_Pend`**   usando  un enfoque especial a la variable categórica **"Tipo_Vehiculo"**. 

`KNeighborsRegressor` es un modelo de **machine learning supervisado** que se utiliza para resolver **problemas de regresión**. Pertenece a la familia de los algoritmos **K-Nearest Neighbors (KNN)**, que se basan en la similitud entre ejemplos.



#### 1.Importar librerías necesarias

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
```



#### 2.Definir columnas, variables , establecer el procesamiento con el modelo de  pipeline , realizar el entrenamiento 

```python
# Renombrar columnas
df.rename(columns={
    'Ult Km/Hr': 'Ult_Km',
    'Km/ Hr Plan': 'Km_Plan',
    'Km/Hr Cambio': 'Km_Cambio',
    'Dias Pend': 'Dias_Pend',
    'Tipo de Vehiculo': 'Tipo_Vehiculo',
    'Km/Hr Pend': 'Km_Pend'
}, inplace=True)

# Variables
numeric_features = ['Ult_Km', 'Km_Plan', 'Km_Cambio', 'Dias_Pend', 'Progreso']
categorical_features = ['Tipo_Vehiculo']

# Preprocesamiento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Modelo con pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=3))
])

# Variables para entrenamiento
X = df[numeric_features + categorical_features]
y = df['Km_Pend']

# Entrenamiento
model.fit(X, y)
```



![Logo](https://i.pinimg.com/1200x/33/8d/0c/338d0c1e53cd64340d37460bbf80ce37.jpg)





El preprocesador definido con `ColumnTransformer` aplica transformaciones distintas dependiendo del tipo de variable. A las columnas numéricas, definidas en la lista `numeric_features`, se les aplica la transformación `StandardScaler()`. Esta técnica estandariza los datos numéricos para que tengan una media de 0 y una desviación estándar de 1. Este paso es fundamental cuando se utiliza un modelo como `KNeighborsRegressor`, ya que este se basa en la distancia entre los puntos, y si las variables están en diferentes escalas, el modelo puede dar más importancia a unas variables que a otras sin intención.

Por otro lado, a las columnas categóricas listadas en `categorical_features`, se les aplica `OneHotEncoder()`. Esta transformación convierte cada valor categórico en una nueva columna binaria (0 o 1), representando la presencia o ausencia de cada categoría. Además, el argumento `handle_unknown='ignore'` permite que el modelo maneje de forma segura categorías nuevas que no fueron vistas durante el entrenamiento, evitando errores si en los datos de prueba aparecen valores desconocidos.

#### 3.Interacción con el usuario y visualización  de resultados

```python
resultados = []

print("\n--- SISTEMA DE PREDICCIÓN DE KM PENDIENTES PARA REALIZAR EL PRÓXIMO MANTENIMIENTO ---")

while True:
    tipos_vehiculo = df['Tipo_Vehiculo'].unique().tolist()
    print("\nElige el tipo de vehículo:")
    for i, tipo in enumerate(tipos_vehiculo, start=1):
        print(f"{i}. {tipo}")
    while True:
        opcion = input(f"Tu opción (1-{len(tipos_vehiculo)}): ")
        if opcion.isdigit() and 1 <= int(opcion) <= len(tipos_vehiculo):
            tipo_vehiculo = tipos_vehiculo[int(opcion) - 1]
            break
        else:
            print("Opción inválida.")

    while True:
        try:
            km_recorridos = float(input("Ingrese los kilómetros recorridos (Ult_Km): "))
            break
        except ValueError:
            print("Ingresa un número válido.")

    input_data = pd.DataFrame([{
        'Ult_Km': km_recorridos,
        'Km_Plan': df['Km_Plan'].mean(),
        'Km_Cambio': df['Km_Cambio'].mean(),
        'Dias_Pend': df['Dias_Pend'].mean(),
        'Progreso': df['Progreso'].mean(),
        'Tipo_Vehiculo': tipo_vehiculo
    }])

    # Predicción
    pred = model.predict(input_data)[0]
    print(f"\n Predicción de KNN: {pred:.2f}")

    resultados.append({
        'Ult_Km': km_recorridos,
        'Tipo_Vehiculo': tipo_vehiculo,
        'KNN': pred
    })

    continuar = input("\n¿Deseas hacer otra predicción? (s/n): ").strip().lower()
    if continuar != 's':
        break

# Mostrar resultados
df_resultados = pd.DataFrame(resultados)

print("\n--- RESUMEN DE PREDICCIONES ---")
print(df_resultados)

# Gráfica
plt.figure(figsize=(10, 6))
for i, row in df_resultados.iterrows():
    etiqueta = f"{row['Tipo_Vehiculo']}, {i+1}"
    valor = row['KNN']
    plt.bar(etiqueta, valor, color='skyblue')
    plt.text(etiqueta, valor + 0.5, f"{valor:.2f}", ha='center', va='bottom', fontsize=10)
plt.title('Predicciones de Km Pendientes (KNN)')
plt.ylabel('Kilómetros')
plt.xlabel('Casos de Predicción')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
**Se ingresó la misma cantidad de kilómetros recorridos para todos los tipos de vehículo disponibles en el conjunto de datos, con el fin de analizar las diferencias en los kilómetros pendientes estimados según el tipo de vehículo**


![Logo](https://i.pinimg.com/1200x/a7/d7/ed/a7d7edc9177235b9d851417856bf540f.jpg)


#### A.Resultados esperados

Visualizar cuántos kilómetros le restan a cada tipo de vehículo antes de necesitar mantenimiento, con el fin de prevenir fallos costosos o accidentes.

El algoritmo no solo predice un valor, sino que **convierte datos históricos y operativos en decisiones prácticas**, ayudando a tomar medidas preventivas **basadas en evidencia** y **personalizadas por tipo de vehículo**.

A.Reflexión emprendedora
Este sistema es ideal para empresas que gestionan flotas de vehículos. Muchas de estas organizaciones invierten grandes sumas en reparaciones que podrían evitarse con un mantenimiento predictivo. Al anticipar posibles fallas mediante datos, se
reduce el tiempo de inactividad, se optimiza el presupuesto y se alarga la vida útil de los vehículos. Esta solución, convertida en una herramienta digital, podría integrarse fácilmente con los sistemas internos de logística y gestión de flotas.



#### **Modelo de negocio:**

Crear una **plataforma web y móvil** que use inteligencia artificial para predecir el mantenimiento de vehículos según su uso real, tipo, y condición operativa. Esta plataforma alertaría a las empresas cuando sus vehículos estén cerca de requerir mantenimiento, **evitando daños costosos y aumentando la seguridad**.



### **Clientes objetivo:**

- Empresas de transporte y logística
- Compañías de mensajería y entregas (tipo Rappi, Servientrega)
- Servicios públicos con flotas (basuras, buses, ambulancias)
- Rentadoras de vehículos
- Gobiernos municipales con vehículos oficiales
- Mercado de compra y venta de vehículos nuevos y usados