Este proyecto utiliza **machine learning** para predecir la calidad del vino basándose en diversas características fisicoquímicas del producto. El dataset contiene variables como la acidez, el azúcar residual, los niveles de dióxido de azufre, y otras, que se utilizan para entrenar diferentes modelos de clasificación y regresión.

## Descripción del Proyecto

El objetivo del proyecto es predecir la calidad del vino mediante la construcción de modelos de aprendizaje automático. Se aplicaron varias técnicas de clasificación y regresión, tales como:

- **Regresión Lineal**
- **Regresión Logística**
- **Random Forest**
- **Support Vector Machines (SVM)**

El flujo del proyecto sigue los pasos típicos de un pipeline de machine learning, que incluye:
1. Carga y exploración del dataset.
2. Preprocesamiento de los datos.
3. Entrenamiento y evaluación de modelos.
4. Comparación de resultados entre diferentes algoritmos.

## Dataset

El dataset utilizado contiene información sobre las características químicas de los vinos y su calidad. Las columnas incluyen:

- `fixed acidity`: Acidez fija
- `volatile acidity`: Acidez volátil
- `citric acid`: Ácido cítrico
- `residual sugar`: Azúcar residual
- `chlorides`: Cloruros
- `free sulfur dioxide`: Dióxido de azufre libre
- `total sulfur dioxide`: Dióxido de azufre total
- `density`: Densidad
- `pH`: pH
- `sulphates`: Sulfatos
- `alcohol`: Contenido alcohólico
- `quality`: Calidad del vino (variable objetivo)
- Otras columnas relevantes

### Fuente del Dataset
El dataset fue tomado de [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) disponible en UCI Machine Learning Repository.

## Estructura del Proyecto

El proyecto se organiza de la siguiente manera:

Modelos Aplicados
1. Regresión Lineal
Predice la calidad del vino como una variable continua utilizando un modelo de regresión lineal.

2. Regresión Logística
Clasifica la calidad del vino en tres categorías (baja, media, alta) y usa la regresión logística para la predicción.

3. Random Forest
Utiliza un conjunto de árboles de decisión (Random Forest) para mejorar la precisión en la clasificación de la calidad.

4. Support Vector Machines (SVM)
Implementa un modelo SVM para clasificar la calidad del vino con un enfoque basado en hiperplanos de decisión.

Visualización de Resultados
Se han incluido diversas visualizaciones que ayudan a entender mejor los datos, tales como:

Matrices de correlación entre las variables.
Histogramas de la distribución de las características del vino.
Matrices de confusión para los modelos de clasificación.
Evaluación de Modelos
Cada modelo fue evaluado utilizando métricas como:

Mean Squared Error (MSE) para modelos de regresión.
Accuracy para modelos de clasificación.
Confusion Matrix para visualización del rendimiento en clasificación.
Los resultados mostraron que el Random Forest tuvo el mejor rendimiento en términos de precisión.
