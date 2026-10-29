from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#Cargamos el dataset que contiene las características y el target
df_modelo=pd.read_csv('data/df_modelo.csv')

#Separamos las variables que fungirán como características (x) y la variable que fungirá como target (y)
X=df_modelo.drop(columns=['cpu_milli'])
y=df_modelo['cpu_milli']

#Realizamos la división de los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##Fase 1: Entrenamiento de la regresión lineal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Creamos el modelo de regresión lineal
model_lr = LinearRegression()
#Entrenamos el modelo con los datos de entrenamiento
model_lr.fit(X_train, y_train)
#Realizamos predicciones con los datos de prueba
y_pred_lr = model_lr.predict(X_test)
#Finalmente, evaluamos el modelo utilizando métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R^2)
r2_lr = r2_score(y_test, y_pred_lr)

print(".:Resultados del modelo de regresión lineal:.\n")
print(f"Precisión del modelo (R^2): {r2_lr:.4f}")