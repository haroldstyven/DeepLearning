from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np

# Cargar el modelo preentrenado
modelo = joblib.load('modelo_iris.joblib')

# Inicializar API
app = FastAPI(title='Iris Classifier API Pro', version='2.0')

# ==========================================
# EJERCICIO 2: Validación de la entrada
# ==========================================
class InputData(BaseModel):
    # Obligamos a que sea una lista de floats
    features: list[float]

    @validator('features')
    def validate_features_length(cls, v):
        if len(v) != 4:
            raise ValueError('¡Error! Se esperan exactamente 4 características numéricas.')
        return v

# ==========================================
# EJERCICIO 1: Diccionario de Probabilidades
# ==========================================
class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict  # Ahora es un diccionario en lugar de una lista plana

@app.post('/predict', response_model=PredictionResponse)
def predict(data: InputData):
    try:
        # Preparar los datos
        X = np.array(data.features).reshape(1, -1)

        # Predecir
        pred = modelo.predict(X)
        proba = modelo.predict_proba(X)[0] # Probabilidades planas

        # Clases de la flor
        class_names = ['setosa', 'versicolor', 'virginica']

        # (EJERCICIO 1) Crear diccionario dinámico de probabilidades
        prob_dict = {class_names[i]: float(proba[i]) for i in range(len(class_names))}

        return PredictionResponse(
            prediction=int(pred[0]),
            class_name=class_names[int(pred[0])],
            probabilities=prob_dict
        )
    except Exception as e:
        # Manejo de cualquier otro error para que no caiga el servidor
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/health')
def health():
    return {'status': 'ok', 'message': 'API funcionando correctamente'}
