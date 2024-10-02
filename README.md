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

from google.colab import files

# Subir archivo
uploaded = files.upload()

import pandas as pd

file_name = list(uploaded.keys())[0]

# Leer el archivo CSV
data = pd.read_csv(file_name)

# Mostrar las primeras filas del dataset
data.head()

![image](https://github.com/user-attachments/assets/adac62d5-a65e-41a0-b784-1e01351f3d65)


# Listar los archivos subidos
!ls

drive  sample_data  WineQT.csv

import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación entre las variables')
plt.show()

![image](https://github.com/user-attachments/assets/b6b61b55-7dc4-4250-a891-9e752261379d)
![image](https://github.com/user-attachments/assets/5a577360-047e-4af2-96c8-dbab8b358c05)
![image](https://github.com/user-attachments/assets/9b265d7d-370d-4719-8a9f-340540d6f43f)


# Distribución de la calidad
plt.figure(figsize=(6, 4))
sns.countplot(data['quality'])
plt.title('Distribución de la calidad del vino')
plt.show()

# Distribución de una variable (ejemplo: alcohol)
plt.figure(figsize=(6, 4))
sns.histplot(data['alcohol'], kde=True)
plt.title('Distribución de la variable alcohol')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separar las variables predictoras (X)y  objetivo (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Convertir calidad en categorías
y_train_class = y_train.apply(lambda x: 0 if x <= 5 else (1 if x == 6 else 2))
y_test_class = y_test.apply(lambda x: 0 if x <= 5 else (1 if x == 6 else 2))

# Crear y entrenar el modelo de regresión logística
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train_class)

# Hacer predicciones
y_pred_class = log_reg.predict(X_test_scaled)

# Evaluar el modelo
accuracy = accuracy_score(y_test_class, y_pred_class)
conf_matrix = confusion_matrix(y_test_class, y_pred_class)

print(f"Accuracy: {accuracy}")
print("Matriz de confusión:")
print(conf_matrix)

Accuracy: 0.6637554585152838
Matriz de confusión:
[[77 23  2]
 [30 60  9]
 [ 2 11 15]]

from sklearn.ensemble import RandomForestClassifier

# Crear y entrenar el modelo de Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train_class)

# Hacer predicciones
y_pred_rf = rf_clf.predict(X_test_scaled)

# Evaluar el modelo
accuracy_rf = accuracy_score(y_test_class, y_pred_rf)

print(f"Accuracy (Random Forest): {accuracy_rf}")

Accuracy (Random Forest): 0.6986899563318777

from sklearn.svm import SVC

# Crear y entrenar el modelo de SVM
svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train_scaled, y_train_class)

# Hacer predicciones
y_pred_svm = svm_clf.predict(X_test_scaled)

# Evaluar el modelo
accuracy_svm = accuracy_score(y_test_class, y_pred_svm)

print(f"Accuracy (SVM): {accuracy_svm}")

Accuracy (SVM): 0.6681222707423581

