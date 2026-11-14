from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import io

#Creamos el objeto de tipo FastAPI
app=FastAPI(
    title= "MLOps CPU Predictor",
    description="API que predice el consumo de CPU de contenedores de manera individual y por lotes",
    version="1.1.0"
)

#Cargamos los modelos que permitirán la predicción de los datos.
try:
    modelo_nn=joblib.load("models/model_nn.joblib")
    scaler=joblib.load("models/scaler_nn.joblib")
except Exception as ex:
    raise RuntimeError(f"Error al cargar los modelos: {ex}")

#Creamos la validación de datos que utilizará la API.
class PodFeatures(BaseModel):
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    duration_time: int
    qos_Burstable: int
    qos_Guaranteed: int
    qos_LS: int

#función para clasificar el consumo operativo de manera legible
def clasificador_carga(vcpus: float)->str:
    if vcpus < 2.0:
        return "Carga Ligera (Microservicio estándar)"
    elif vcpus < 8.0:
        return "Carga Moderada (Procesamiento medio)"
    else:
        return "Alta Demanda (Cómputo intensivo)"


#Creamos el método que servirá como predictor
@app.post("/predict", summary="Predicción Individual")
def predict_cpu(features: PodFeatures):
    try:
        #Convertimos los datos validados con pydantic en un DataFrame de Pandas para hacer el escalado
        input_data=pd.DataFrame([features.model_dump()])
        #Realizamos el escalado de dichos datos para la predicción
        data_scaled=scaler.transform(input_data)
        
        #Realizamos la predicción
        predict_milli=float(modelo_nn.predict(data_scaled)[0])
        #Obtenemos la cantidad vcpus que consumirá
        vcpus= round(predict_milli/1000.0, 2)

        #Devolvemos la respuesta en formato JSON
        return {
            "status": "success",
            "cpu_milli_prediction": round(predict_milli, 2),
            "vcpus_estimados": vcpus,
            "clasificador_consumo": clasificador_carga(vcpus),
            "interpretacion": f"Este contenedor requerirá aproximadamente {vcpus} nucleos virtuales."    
                }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex)) 