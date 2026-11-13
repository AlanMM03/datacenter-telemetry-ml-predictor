#Creamos la imagen de python para nuestra aplicación
FROM python:3.11-slim

#Establecemos el directorio de trabajo dentro del contenedor de Docker
WORKDIR /app

#Copiamos el archivo de dependencias para aprovechar el cache de Docker
COPY requirements.txt .

#Instalamos las dependencias dentro del contenedor de Docker
RUN pip install --no-cache-dir -r requirements.txt

#Copimos los modelos y el código fuente de python.
COPY /src ./src/
COPY /models ./models/

#Exponemos el purto donde correra FastAPI
EXPOSE 8000

#El comando que iniciara el contenedor.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]