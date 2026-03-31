# Utilizar una imagen oficial y liviana de Python
FROM python:3.11-slim

# Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la API y el modelo preentrenado
COPY app.py .
COPY modelo_iris.joblib .

# Exponer el puerto en el que corre Uvicorn
EXPOSE 8000

# Comando para levantar la API al arrancar el contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
