from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import numpy as np

#Cargamos los dataframes que nos serviran para el entrenamiento del modelo de regresión lineal
X_train=pd.read_csv('data/processed/X_train.csv')
X_test=pd.read_csv('data/processed/X_test.csv')
y_train=pd.read_csv('data/processed/y_train.csv')
y_test=pd.read_csv('data/processed/y_test.csv')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##Entrenamiento del modelo de regresión lineal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Creamos el modelo de regresión lineal
model_lr = LinearRegression()
#Entrenamos el modelo con los datos de entrenamiento
model_lr.fit(X_train, y_train)
#Realizamos predicciones con los datos de prueba
y_pred_lr = model_lr.predict(X_test)
#Finalmente, evaluamos el modelo utilizando métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R^2)
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr=mean_squared_error(y_test, y_pred_lr)
print(".:Resultados del modelo de regresión lineal:.\n")
print(f"Precisión del modelo (R^2): {r2_lr:.4f}")
print(f"Error cuadrático medio (MSE): {np.sqrt(mse_lr):.4f}")

#Guardamo el modelo entrenado
joblib.dump(model_lr, 'models/model_lr.joblib')