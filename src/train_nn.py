from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib

#Cargamos los dataframes que nos serviran para el entrenamiento del modelo de red neuronal
X_train=pd.read_csv('data/processed/X_train.csv')
X_test=pd.read_csv('data/processed/X_test.csv')
y_train=pd.read_csv('data/processed/y_train.csv')
y_test=pd.read_csv('data/processed/y_test.csv')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##Entrenamiento del modelo de red neuronal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Entrenamos la red neuronal con los datos de entrenamiento
#Primero, tenemos que escalar los datos para que la red neuronal funcione correctamente
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Creamos el modelo con una arquitectura de 2 capas ocultas con 64 y 32 neuronas respectivamente
modelo_nn = MLPRegressor(hidden_layer_sizes=(64,32),solver='adam',verbose=True, max_iter=2000, batch_size=50, random_state=42)
#Entrenamos el modelo con los datos de entrenamiento escalados
modelo_nn.fit(X_train_scaled, y_train)
#Una vez entrenado, procedemos a realizar la predicción con los datos de prueba escalados
y_pred_nn = modelo_nn.predict(X_test_scaled)

#Finalmente, evaluamos la precisión del modelo con métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R^2)
r2_nn = r2_score(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(".:Resultados del modelo de red neuronal:.\n")
print(f"Precisión del modelo (R^2): {r2_nn:.4f}")
print(f"Error cuadrático medio (MSE): {mse_nn:.4f}")

#Guardamos el modelo entrenado y el escalador para poder usarlo en la predicción de nuevos datos
joblib.dump(modelo_nn, 'models/model_nn.joblib')
joblib.dump(scaler, 'models/scaler_nn.joblib')