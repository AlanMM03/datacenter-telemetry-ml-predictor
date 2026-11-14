from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

#Creamos el objeto de tipo FastAPI
app=FastAPI(
    title= "MLOps CPU Predictor",
    description="API que predice el consumo de CPU de contenedores",
    version="1.0.0"
)

#Cargamos los modelos que permitirán la predicción de los datos.
try:
    modelo_nn=joblib.load("models/model_nn.joblib")
    scaler=joblib.load("models/scaler_nn.joblib")
except Exception as ex:
    raise RuntimeError(f"Error al cargar los modelos: {ex}")

#Creamos la validación de datos que utilizará la API.
class PodFeatures(BaseModel):
    momory_mib: int
    num_gpu: int
    gpu_milli: int
    duration_time: int
    qos_Burstable: int
    qos_Guaranteed: int
    qos_LS: int

#Creamos el método que servirá como predictor
@app.post("/predict")
def predict_cpu(features: PodFeatures):
    try:
        input_data=pd.DataFrame([features.model_dump()])
        
        #Realizamos el escalado
        data_scaled=scaler.transform(input_data)
        
        #Realizamos la predicción
        prediction=modelo_nn.predict(data_scaled)

        #Devolvemos la respuesta en formato JSON
        return {
            "status": "success",
            "cpu_milli_prediction": float(np.round(prediction[0], 2))    
                }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex)) 