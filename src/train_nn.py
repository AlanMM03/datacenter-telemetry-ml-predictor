from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#Cargamos el dataset que contiene las características y el target
df_modelo=pd.read_csv('data/df_modelo.csv')

#Seleccionamos las variables que fungirán como características (x) y la variable que fungirá como target (y)
X=df_modelo.drop(columns=['cpu_milli'])
y=df_modelo['cpu_milli']

#Realizamos la división de los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenamos la red neuronal con los datos de entrenamiento
#Primero, tenemos que escalar los datos para que la red neuronal funcione correctamente
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Creamos el modelo con una arquitectura de 2 capas ocultas con 64 y 32 neuronas respectivamente
modelo_nn = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
#Entrenamos el modelo con los datos de entrenamiento escalados
modelo_nn.fit(X_train_scaled, y_train)
#Una vez entrenado, procedemos a realizar la predicción con los datos de prueba escalados
y_pred_nn = modelo_nn.predict(X_test_scaled)

#Finalmente, evaluamos la precisión del modelo con métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R^2)
r2_nn = r2_score(y_test, y_pred_nn)
print(".:Resultados del modelo de red neuronal:.\n")
print(f"Precisión del modelo (R^2): {r2_nn:.4f}")