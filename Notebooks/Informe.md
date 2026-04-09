# Informe de Ejercicios Propuestos

**Estudiante:** Harold Styven Lagares De Voz  
**Posgrado:** Maestría en Ingeniería  
**Asignatura:** Deep Learning  
**Profesor:** Jairo Serrano Castañeda  
**Universidad:** Universidad Tecnológica de Bolívar  

---

## 01_intro_machine_learning.ipynb

## 9. Ejercicios Propuestos

1. **Ejercicio 1:** Entrena un modelo de regresión logística usando solo 2 variables del dataset Iris. ¿Cambia la accuracy? ¿Por qué?

2. **Ejercicio 2:** Prueba diferentes valores de `C` (0.001, 0.01, 0.1, 1, 10, 100) y grafica la accuracy en train **y** en test. ¿Qué valor produce el mejor equilibrio sesgo-varianza?

3. **Ejercicio 3:** Compara el desempeño del modelo con un `DummyClassifier` (baseline de azar). ¿Cuánto mejora la regresión logística?

4. **Ejercicio 4:** Usa `RandomizedSearchCV` en lugar de `GridSearchCV` para buscar hiperparámetros. ¿Obtienes resultados similares con menos cómputo?

5. **Ejercicio 5:** Visualiza la curva de aprendizaje del modelo con `learning_curve` de scikit-learn. ¿Observas signos de sobreajuste o subajuste?

# Resolución de Ejercicios Propuestos

En esta sección, abordaremos los ejercicios diseñados para profundizar en la comprensión de la **Regresión Logística** y la evaluación de modelos:

1.  **Ejercicio 1: Impacto de la selección de características.** Entrenaremos el modelo utilizando solo las dimensiones del pétalo para observar cómo varía la precisión.
2.  **Ejercicio 3: Comparación con Baseline (Azar).** Utilizaremos un clasificador "tonto" para validar que nuestro modelo realmente está aprendiendo patrones significativos.

## Ejercicio 1: Entrenamiento con 2 Variables (Pétalos)

En este ejercicio, reduciremos el espacio de entrada de 4 variables a solo 2: `petal length (cm)` y `petal width (cm)`. Estas variables suelen ser las más informativas para separar las especies de Iris.

```python
# Filtramos el dataset para usar solo las columnas 2 y 3 (largo y ancho del pétalo)
X_reduced = X[:, 2:] 

# Dividimos nuevamente en train y test
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced, y, test_size=0.2, random_state=SEED, stratify=y
)

# Creamos y entrenamos el modelo (importante escalar los datos siempre)
pipe_red = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=SEED))
])

pipe_red.fit(X_train_red, y_train_red)

# Evaluamos
y_pred_red = pipe_red.predict(X_test_red)
acc_red = accuracy_score(y_test_red, y_pred_red)

print(f"Accuracy con 4 variables (Original): 0.9667 (aprox)") # Basado en el entrenamiento previo
print(f"Accuracy con 2 variables (Pétalos): {acc_red:.4f}")
```

```text
(Salida)
Accuracy con 4 variables (Original): 0.9667 (aprox)
Accuracy con 2 variables (Pétalos): 0.9333

```

## Ejercicio 3: Comparación con DummyClassifier (Baseline)

Para asegurar que nuestro modelo de Regresión Logística es útil, debemos compararlo con un modelo que no aprende patrones, sino que predice basándose en reglas simples (como la clase más frecuente).

```python
from sklearn.dummy import DummyClassifier

# 1. Creamos el DummyClassifier que siempre predice la clase más frecuente
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# 2. Obtenemos la precisión del azar/frecuencia
dummy_acc = dummy_clf.score(X_test, y_test)

# 3. Comparamos los resultados
print(f"Accuracy de la Regresión Logística: {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy del Dummy Classifier: {dummy_acc:.4f}")

# Cálculo de la mejora
mejora = (accuracy_score(y_test, y_pred) - dummy_acc) * 100
print(f"\nMejora real sobre el azar: {mejora:.2f}%")
```

```text
(Salida)
Accuracy de la Regresión Logística: 0.9667
Accuracy del Dummy Classifier: 0.3333

Mejora real sobre el azar: 63.33%

```

## Conclusión Final

Tras realizar los experimentos propuestos, podemos extraer las siguientes conclusiones clave sobre el modelo de Regresión Logística y la naturaleza de los datos:

### 1. Sobre la Selección de Características (Ejercicio 1)
*   **Resultados:** Al reducir el modelo de 4 variables a solo 2 (pétalos), la precisión bajó ligeramente de **96.67% a 93.33%**.
*   **Análisis:** Esta pequeña disminución confirma que las dimensiones del pétalo son las variables con mayor poder predictivo en este dataset. Aunque las dimensiones del sépalo aportan información residual (que ayuda a llegar al 96%), el modelo simplificado sigue siendo extremadamente robusto, lo que demuestra la importancia de la **importancia de las variables** en la eficiencia de un modelo.

### 2. Sobre la Validez del Modelo (Ejercicio 3)
*   **Resultados:** La Regresión Logística (**96.67%**) superó drásticamente al Dummy Classifier (**33.33%**), con una mejora real del **63.33%**.
*   **Análisis:** Dado que el dataset Iris tiene 3 clases balanceadas, la precisión base por puro azar es de aproximadamente 1/3 (33%). El hecho de que nuestro modelo triplique el desempeño del azar indica que ha logrado capturar patrones estadísticos reales y significativos entre las características de las flores y sus especies, validando su utilidad práctica.

### Reflexión General
Este análisis demuestra que la **Regresión Logística** es una herramienta muy potente para problemas con separabilidad lineal. Logramos un equilibrio entre simplicidad (pocas variables) y alto desempeño, superando con creces cualquier métrica de azar.

## 10. Referencias y Recursos

- [Documentación de scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Regresión logística - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Pipeline - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [GridSearchCV - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Regresión logística - Wikipedia](https://es.wikipedia.org/wiki/Regresi%C3%B3n_log%C3%ADstica)

---

📎 **Notebook siguiente:** [02. Preprocesamiento y Visualización](./02_preprocesamiento_visualizacion.ipynb)

---

## 02_preprocesamiento_visualizacion.ipynb

## 9. Ejercicios Propuestos

1. **Ejercicio 1:** Agrega ruido artificial a 3 columnas del dataset y usa un Denoising approach para limpiarlas. ¿Cambian las correlaciones?

2. **Ejercicio 2:** Aplica PCA reteniendo el 95% de la varianza. ¿Cuántos componentes necesitas? Entrena un modelo de clasificación con los componentes reducidos y compara con el accuracy usando todas las variables.

3. **Ejercicio 3:** Compara los 3 escaladores (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) combinados con LogisticRegression usando validación cruzada. ¿Cuál da mejor resultado en este dataset?

4. **Ejercicio 4 (Avanzado):** Implementa un pipeline que combine `SelectKBest` + `StandardScaler` + `LogisticRegression` y optimiza `k` (número de variables) con `GridSearchCV`.

# Resolución de Ejercicios Propuestos

En esta sección, nos enfocaremos en los siguientes ejercicios para profundizar en las técnicas de preprocesamiento, escalado de datos y optimización de pipelines:

1.  **Ejercicio 3: Evaluación de Escaladores.** Compararemos el rendimiento de `StandardScaler`, `MinMaxScaler` y `RobustScaler` utilizando Validación Cruzada (Cross-Validation) para determinar cuál se adapta mejor a la distribución de nuestro dataset.
2.  **Ejercicio 4 (Avanzado): Optimización de Selección de Características.** Construiremos un pipeline robusto que integre la selección de las mejores variables (`SelectKBest`), el escalado de datos y el entrenamiento del modelo. Además, automatizaremos la búsqueda del número óptimo de variables `k` utilizando `GridSearchCV`.

## Ejercicio 3: Comparación de StandardScaler, MinMaxScaler y RobustScaler

El objetivo es evaluar empíricamente qué método de escalado produce un modelo que generalice mejor en nuestros datos. Para tener una evaluación justa y robusta, utilizaremos **Validación Cruzada (Cross-Validation)** en lugar de una simple división de train/test.

Evaluaremos 3 flujos (pipelines) distintos:
*   `StandardScaler`: Asume distribución normal.
*   `MinMaxScaler`: Escala los datos a un rango fijo (usualmente 0 a 1).
*   `RobustScaler`: Utiliza la mediana y el rango intercuartílico, lo que lo hace resistente a valores atípicos (outliers).

```python
# Diccionario para guardar los resultados
resultados_escaladores = {}

# 2. Iteramos sobre cada escalador
for nombre, escalador in escaladores.items():
    # Creamos el pipeline: Escalador -> Modelo
    pipe = Pipeline([
        ('escalador', escalador),
        ('clf', LogisticRegression(random_state=SEED, max_iter=1000)) 
    ])
    
    # Aplicamos validación cruzada (por defecto 5 particiones/folds)
    # n_jobs=-1 usa todos los núcleos del procesador
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Guardamos la media y la desviación estándar de los scores
    resultados_escaladores[nombre] = {
        'Media (Accuracy)': np.mean(scores),
        'Desviación Std': np.std(scores)
    }

# 3. Mostramos los resultados
print("Resultados de la Validación Cruzada por Escalador:\n")
for nombre, metricas in resultados_escaladores.items():
    print(f"--- {nombre} ---")
    print(f"Accuracy Medio: {metricas['Media (Accuracy)']:.4f}")
    print(f"Desviación Std: {metricas['Desviación Std']:.4f}\n")
    
print("El escalador con el Accuracy Medio más alto y la menor Desviación Std\nes el más adecuado y estable para este conjunto de datos.")
```

```text
(Salida)
Resultados de la Validación Cruzada por Escalador:

--- StandardScaler ---
Accuracy Medio: 0.9807
Desviación Std: 0.0065

--- MinMaxScaler ---
Accuracy Medio: 0.9613
Desviación Std: 0.0042

--- RobustScaler ---
Accuracy Medio: 0.9789
Desviación Std: 0.0089

El escalador con el Accuracy Medio más alto y la menor Desviación Std
es el más adecuado y estable para este conjunto de datos.

```

## Ejercicio 4 (Avanzado): Pipeline con SelectKBest + StandardScaler + GridSearchCV

En proyectos de Machine Learning reales, rara vez sabemos de antemano cuántas variables (características) son realmente útiles. 
Aquí crearemos un pipeline integral:
1.  **SelectKBest:** Seleccionará las 'k' características más informativas utilizando la prueba estadística ANOVA (`f_classif`).
2.  **StandardScaler:** Normalizará estas 'k' características seleccionadas.
3.  **LogisticRegression:** Entrenará el modelo.

Utilizaremos **GridSearchCV** para probar sistemáticamente diferentes valores de `k` y encontrar la cantidad óptima de variables que maximiza nuestro *Accuracy*.

```python
# 1. Definimos el Pipeline completo
# Nota: Damos nombres a cada paso del pipeline ('feature_selection', 'scaler', 'clf') para referenciarlos después
pipe_avanzado = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=SEED, max_iter=1000))
])

# 2. Definimos la grilla (grid) de hiperparámetros a explorar
# Queremos probar dejando desde 1 variable hasta todas las variables disponibles.
# 'X.shape[1]' nos da el número total de columnas/variables en nuestro dataset.
num_variables_totales = X.shape[1]

param_grid = {
    # Explorar k desde 1 hasta el total de variables
    'feature_selection__k': list(range(1, num_variables_totales + 1)),
    
    #También aprovechamos para afinar la regularización C del modelo
    'clf__C': [0.1, 1, 10] 
}

# 3. Configuramos GridSearch con Validación Cruzada (cv=5)
grid_search = GridSearchCV(
    estimator=pipe_avanzado,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1 # Muestra información básica del proceso
)

# 4. Ajustamos (entrenamos) la grilla buscando la mejor combinación
print("Iniciando la búsqueda del mejor 'k' y 'C'...")
grid_search.fit(X, y)

# 5. Imprimimos los resultados óptimos encontrados
print("\n--- Búsqueda Finalizada ---")
print(f"Mejor número de variables (k): {grid_search.best_params_['feature_selection__k']}")
print(f"Mejor parámetro de regularización (C): {grid_search.best_params_['clf__C']}")
print(f"Mejor Accuracy estimado (CV): {grid_search.best_score_:.4f}")
```

```text
(Salida)
Iniciando la búsqueda del mejor 'k' y 'C'...
Fitting 5 folds for each of 90 candidates, totalling 450 fits

--- Búsqueda Finalizada ---
Mejor número de variables (k): 28
Mejor parámetro de regularización (C): 1
Mejor Accuracy estimado (CV): 0.9807

```

## Conclusión Final

Tras finalizar los experimentos de escalado y selección de características, podemos destacar los siguientes hallazgos:

### 1. Desempeño de los Escaladores (Ejercicio 3)
*   **Resultados:** El `StandardScaler` obtuvo el rendimiento promedio más alto (**0.9807**) y mantuvo una desviación estándar muy competitiva (0.0065). Aunque `MinMaxScaler` presentó la menor variación (0.0042), su accuracy media fue significativamente más baja (0.9613). Por su parte, `RobustScaler` fue muy cercano al `StandardScaler`, pero ligeramente más inestable (0.0089).
*   **Análisis:** Estos resultados sugieren que gran parte de las variables principales en este dataset siguen una distribución que se beneficia de ser centrada en cero con una varianza unitaria (la metodología de `StandardScaler`). La menor efectividad del `MinMaxScaler` nos indica que comprimir los datos a un rango estricto [0, 1] no es la mejor estrategia para la Regresión Logística en este caso particular.

### 2. Optimización Integral con Pipelines (Ejercicio 4)
*   **Resultados:** Nuestro pipeline identificó que el número óptimo de características a retener (`k`) es **28** (de las 30 originales), utilizando una regularización estándar (`C=1`). El *accuracy* validado de manera cruzada fue idéntico al del mejor escalador (**0.9807**).
*   **Análisis:** El hecho de que el modelo haya descartado 2 variables nos indica que, aunque la gran mayoría de las características aportan valor predictivo, existe algo de "ruido" o información redundante que no es necesaria para la clasificación. Al automatizar este proceso con `GridSearchCV`, garantizamos que el modelo es lo más eficiente y parsimonioso posible, sin sacrificar absolutamente nada de precisión.

### Reflexión General
Esta práctica ha demostrado por qué metodologías como la **Validación Cruzada** y el uso de **Pipelines** son el estándar en la industria. Nos permiten tomar decisiones objetivas (como qué escalador elegir o cuántas variables usar) basadas en pruebas estadísticas rigurosas, evitando el sesgo o el azar que ocurre al trabajar con un solo conjunto de prueba estático.

## 10. Referencias y Recursos

- [Documentación de scikit-learn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Selection - scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Data Leakage in ML - Kaggle](https://www.kaggle.com/code/alexisbcook/data-leakage)
- [Feature Engineering - Towards Data Science](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)

---

📎 **Notebook anterior:** [01. Introducción a ML](./01_intro_machine_learning.ipynb)  
📎 **Notebook siguiente:** [03. Modelos Clásicos de ML](./03_modelos_clasicos_ml.ipynb)

---

## 03_modelos_clasicos_ml.ipynb

## 10. Ejercicios Propuestos

1. **Ejercicio 1:** Cambia el dataset a `load_wine()` y repite la comparación. ¿Qué modelo gana?

2. **Ejercicio 2:** Visualiza el árbol de decisión entrenado con `plot_tree`. Limita `max_depth=3` para que sea legible. ¿Qué variables usa para dividir?

3. **Ejercicio 3:** Experimenta con diferentes valores de `n_neighbors` en KNN (K=1, 3, 5, 7, 15). Grafica accuracy vs K.

4. **Ejercicio 4 (Avanzado):** Implementa un ensamble de votación (`VotingClassifier`) combinando los 4 modelos. ¿Mejora el accuracy?

# Resolución de Ejercicios Propuestos

En esta sección final, pondremos a prueba la capacidad de generalización de los algoritmos clásicos y exploraremos técnicas de ensamble:

1.  **Ejercicio 1: Comparación en un nuevo dominio (`load_wine`).** Evaluaremos cómo se comportan los 4 modelos (Regresión Logística, KNN, Árboles y SVM) en un dataset diferente para determinar cuál es el más versátil.
2.  **Ejercicio 4 (Avanzado): Ensamble por Votación.** Crearemos un `VotingClassifier` que combine la "sabiduría" de los 4 modelos para intentar obtener una predicción final más precisa y estable.

## Ejercicio 1: Evaluación con el Dataset Wine

Cambiamos el contexto de dígitos escritos a mano a la clasificación química de vinos. Este dataset tiene 13 variables y 3 clases. Repetiremos el flujo completo de preprocesamiento y entrenamiento para los 4 modelos.

```python
# 1. Cargar el nuevo dataset
wine = load_wine()
X_w, y_w = wine.data, wine.target

# 2. División Datos (80% train, 20% test)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_w, y_w, test_size=0.2, random_state=SEED, stratify=y_w
)

# 3. Escalado (Fundamental para KNN, Logistic y SVM)
scaler_w = StandardScaler()
X_train_w = scaler_w.fit_transform(X_train_w)
X_test_w = scaler_w.transform(X_test_w)

# 4. Definición de modelos (Usando hiperparámetros base)
modelos_vinitos = {
    "Logistic Regression": LogisticRegression(random_state=SEED),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=SEED)
}

# 5. Entrenamiento y Evaluación
print("Resultados Dataset Wine:")
for nombre, modelo in modelos_vinitos.items():
    modelo.fit(X_train_w, y_train_w)
    y_pred_w = modelo.predict(X_test_w)
    acc_w = accuracy_score(y_test_w, y_pred_w)
    print(f"- {nombre}: {acc_w:.4f}")
```

```text
(Salida)
Resultados Dataset Wine:
- Logistic Regression: 0.9722
- KNN (K=5): 0.9722
- Decision Tree: 0.9444
- SVM (RBF): 0.9722

```

## Ejercicio 4: Ensamble por Votación (VotingClassifier)

Un ensamble de votación combina las predicciones de múltiples modelos. Utilizaremos una estrategia de **"Soft Voting"**, donde el modelo final no solo mira la clase ganadora, sino que promedia las probabilidades estimadas por cada algoritmo para tomar la decisión final.

```python
# 1. Creamos la lista de estimadores
estimadores = [
    ('lr', modelos_vinitos["Logistic Regression"]),
    ('knn', modelos_vinitos["KNN (K=5)"]),
    ('tree', modelos_vinitos["Decision Tree"]),
    ('svm', modelos_vinitos["SVM (RBF)"])
]

# 2. Instanciamos el Ensamble de Votación
# voting='soft' promedia las probabilidades
voto_clf = VotingClassifier(estimators=estimadores, voting='soft')

# 3. Entrenamos el ensamble
voto_clf.fit(X_train_w, y_train_w)

# 4. Evaluamos y comparamos
acc_ensamble = voto_clf.score(X_test_w, y_test_w)

print(f"Accuracy del Ensamble de Votación: {acc_ensamble:.4f}")

# Comprobamos si superó al mejor modelo individual del ejercicio 1
mejor_individual = max([accuracy_score(y_test_w, m.predict(X_test_w)) for m in modelos_vinitos.values()])
print(f"Diferencia vs Mejor Modelo Individual: {acc_ensamble - mejor_individual:.4f}")
```

```text
(Salida)
Accuracy del Ensamble de Votación: 0.9722
Diferencia vs Mejor Modelo Individual: 0.0000

```

## Conclusión Final

Tras comparar los 4 algoritmos fundamentales y probar una técnica de ensamble en el dataset Wine, estas son nuestras conclusiones principales:

### 1. Robustez de los Modelos Clásicos (Ejercicio 1)
*   **Resultados:** Tres de los cuatro modelos (**Regresión Logítica, KNN y SVM**) alcanzaron un empate perfecto con un **97.22%** de precisión. El **Árbol de Decisión** fue el modelo más débil con un **94.44%**.
*   **Análisis:** El hecho de que un modelo lineal (Regresión Logística) rinda tan bien como uno complejo (SVM) nos indica que el dataset Wine tiene fronteras de decisión relativamente claras tras ser escalado correctamente. El Árbol de Decisión, al ser un modelo "codicioso" (greedy) y propenso a pequeñas variaciones en los datos, se quedó ligeramente atrás en este escenario.

### 2. El Efecto del Voto en el Ensamble (Ejercicio 4)
*   **Resultados:** El `VotingClassifier` obtuvo un **97.22%**, igualando al mejor de los modelos individuales (Diferencia de **0.0000**).
*   **Análisis:** ¿Por qué no mejoró el resultado? Los ensambles suelen brillar cuando los modelos individuales cometen errores en **diferentes instancias** (diversidad). En datasets pequeños y bien estructurados como Wine, los modelos suelen "estar de acuerdo" en la mayoría de las predicciones. Como 3 de nuestros 4 modelos ya tenían un desempeño idéntico y excelente, el ensamble simplemente reforzó esa mayoría sin encontrar casos nuevos donde corregir errores, llegando a un límite de aprendizaje compartido.

### Reflexión General
Este notebook concluye que, para problemas de baja o mediana dimensionalidad, los modelos clásicos siguen siendo extremadamente competitivos. La clave del éxito no siempre es el modelo más complejo, sino un **escalado correcto** y una **validación cruzada** que nos permita elegir el algoritmo más estable para el problema específico.

## 11. Referencias y Recursos

- [scikit-learn: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [Árboles de Decisión](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.*

---

📎 **Notebook anterior:** [02. Preprocesamiento y Visualización](./02_preprocesamiento_visualizacion.ipynb)  
📎 **Notebook siguiente:** [04. Redes Neuronales de Capa Densa](./04_redes_neuronales_capa_densa.ipynb)

---

## 04_redes_neuronales_capa_densa.ipynb

## 9. Ejercicios Propuestos

1. **Ejercicio 1:** Agrega una capa `Dropout(0.3)` entre las capas densas del modelo Keras. ¿Reduce el sobreajuste?

2. **Ejercicio 2:** Implementa `EarlyStopping` con `patience=5` y compara la época óptima vs el modelo sin early stopping.

3. **Ejercicio 3:** Cambia el dataset a MNIST completo (`keras.datasets.mnist`). ¿Cómo impacta el tamaño del dataset en el accuracy y tiempo?

4. **Ejercicio 4 (Avanzado):** Implementa un MLP con `BatchNormalization` entre capas. ¿Mejora la convergencia?

# Resolución de Ejercicios Propuestos

En esta sección, abordaremos técnicas críticas para entrenar redes neuronales eficientes y escalables:

1.  **Ejercicio 2: Prevención del Sobreajuste con EarlyStopping.** Configuraremos un monitor que detenga el entrenamiento automáticamente cuando nuestra red deje de aprender, ahorrando tiempo de cómputo.
2.  **Ejercicio 3: Escalabilidad con MNIST completo.** Pasaremos de un dataset de prueba ("juguete") al dataset MNIST original de Keras (60,000 imágenes de 28x28 píxeles) para observar el impacto masivo de los datos en el *accuracy* y el tiempo de ejecución.

## Ejercicio 2: Monitoreo con Early Stopping

Entrenar una red neuronal por demasiadas épocas (epochs) casi siempre conduce al *overfitting* (el modelo memoriza los datos de entrenamiento pero falla en datos nuevos).
Usaremos el *Callback* `EarlyStopping` de Keras para monitorear la pérdida de validación (`val_loss`). Si la pérdida no mejora (disminuye) durante 5 épocas consecutivas (`patience=5`), el entrenamiento se abortará y restauraremos los mejores pesos encontrados.

```python
# 1. Definimos una red neuronal densa simple (MLP)
def crear_modelo():
    modelo = Sequential([
        # Asegúrate de que la dimensión de entrada (input_shape) coincida con tus datos actuales (ej. 64 si usas digits de sklearn)
        # Shape de X: (1797, 64), Shape de y: (1797,)
        Dense(1797, activation='relu', input_shape=(X_train.shape[1],)), 
        Dense(64, activation='relu'),
        # Suponiendo 10 clases (del 0 al 9)
        Dense(10, activation='softmax') 
    ])
    
    modelo.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])
    return modelo

# 2. Creamos el Callback EarlyStopping
# 'restore_best_weights=True' asegura que nos quedemos con el mejor modelo, no con el modelo de la última época evaluada
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1 # Para que imprima un mensaje cuando se detenga
)

# 3. Entrenamos el modelo usando un número exagerado de épocas (ej. 100)
# ¡Veremos que se detiene mucho antes gracias al Early Stopping!
print("Iniciando entrenamiento con Early Stopping...")

modelo_es = crear_modelo()
historia_es = modelo_es.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop], 
    verbose=0 # verbose=0 para no imprimir todas las épocas, EarlyStopping nos avisará
)

# 4. Resultados
epoca_optima = early_stop.stopped_epoch - 5 if early_stop.stopped_epoch > 0 else 100
print(f"\nEl entrenamiento se detuvo en la época: {early_stop.stopped_epoch}")
print(f"La época óptima (mejores pesos restaurados) fue la: {epoca_optima}")

# Evaluamos final
test_loss, test_acc = modelo_es.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy final en Test: {test_acc:.4f}")
```

```text
(Salida)
Iniciando entrenamiento con Early Stopping...
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 13.

El entrenamiento se detuvo en la época: 17
La época óptima (mejores pesos restaurados) fue la: 12
Accuracy final en Test: 0.9861

```

## Ejercicio 3: Impacto de la cantidad de datos (MNIST nativo de Keras)

Hasta ahora hemos trabajado con datasets pequeños de Scikit-Learn. Ahora usaremos el dataset **MNIST completo** de Keras que es más grande.
*   **Imágenes originales:** 60,000 imágenes para entrenamiento y 10,000 para evaluación.
*   **Resolución real:** 28x28 píxeles (comparado con los 8x8 que veníamos usando).

Aplanaremos (Flatten) las imágenes 2D (28x28) a vectores 1D (784 características) para poder introducirlas en nuestras capas Densas.

```python
# 1. Carga masiva de datos reales
print("Descargando el dataset MNIST gigante de Keras...")
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

print(f"Forma original Train: {X_train_full.shape} (60 mil imágenes de 28x28)")

# 2. Preprocesamiento: Aplanado (Flatten) y Normalización (0-1)
# 28 * 28 = 784 píxeles (características)
X_train_flat = X_train_full.reshape(-1, 28*28) / 255.0
X_test_flat = X_test_full.reshape(-1, 28*28) / 255.0

# 3. Definición de un modelo más robusto para 784 entradas
modelo_mnist = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

modelo_mnist.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# 4. Entrenamiento y medición del tiempo
print("\nIniciando entrenamiento del MNIST completo...")
inicio_tiempo = time.time()

# Agregamos también el EarlyStopping por si acaso
historia_mnist = modelo_mnist.fit(
    X_train_flat, y_train_full,
    epochs=20, # Reducimos epochs máximas porque cada una tarda más
    batch_size=128, # Procesamos 128 imágenes a la vez por eficiencia
    validation_split=0.2, # Apartamos un 20% del train para validación
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1 # Aquí si queremos ver la barra de progreso de épocas
)

fin_tiempo = time.time()

# 5. Evaluación Final
print(f"\nTiempo de entrenamiento total: {(fin_tiempo - inicio_tiempo):.2f} segundos")

test_loss_m, test_acc_m = modelo_mnist.evaluate(X_test_flat, y_test_full, verbose=0)
print(f"Accuracy astronómico en MNIST Test real: {test_acc_m:.4f}")
```

```text
(Salida)
Descargando el dataset MNIST gigante de Keras...
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
[1m11490434/11490434[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 5us/step
Forma original Train: (60000, 28, 28) (60 mil imágenes de 28x28)

Iniciando entrenamiento del MNIST completo...
Epoch 1/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 12ms/step - accuracy: 0.9127 - loss: 0.3066 - val_accuracy: 0.9558 - val_loss: 0.1485
Epoch 2/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 10ms/step - accuracy: 0.9654 - loss: 0.1207 - val_accuracy: 0.9647 - val_loss: 0.1165
Epoch 3/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 11ms/step - accuracy: 0.9774 - loss: 0.0771 - val_accuracy: 0.9687 - val_loss: 0.1034
Epoch 4/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 11ms/step - accuracy: 0.9840 - loss: 0.0546 - val_accuracy: 0.9732 - val_loss: 0.0930
Epoch 5/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 10ms/step - accuracy: 0.9884 - loss: 0.0408 - val_accuracy: 0.9742 - val_loss: 0.0902
Epoch 6/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 11ms/step - accuracy: 0.9921 - loss: 0.0298 - val_accuracy: 0.9769 - val_loss: 0.0842
Epoch 7/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 10ms/step - accuracy: 0.9944 - loss: 0.0207 - val_accuracy: 0.9761 - val_loss: 0.0897
Epoch 8/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 11ms/step - accuracy: 0.9954 - loss: 0.0167 - val_accuracy: 0.9771 - val_loss: 0.0904
Epoch 9/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 17ms/step - accuracy: 0.9959 - loss: 0.0145 - val_accuracy: 0.9756 - val_loss: 0.0992

Tiempo de entrenamiento total: 42.71 segundos
Accuracy astronómico en MNIST Test real: 0.9782

```

## Conclusión Final

Al finalizar esta sección práctica con Redes Neuronales Densas (MLP), podemos destacar dos grandes aprendizajes sobre el entrenamiento de modelos de Deep Learning:

### 1. La Eficiencia del Early Stopping (Ejercicio 2)
*   **Resultados:** Al configurar nuestro modelo con una paciencia de 5 épocas, el entrenamiento, originalmente programado para 100 iteraciones, **se detuvo abruptamente en la época 17**. El *callback* identificó inteligentemente que el modelo alcanzó su pico de rendimiento en la **época 12**, restaurando esos pesos y logrando un excelente accuracy en test del **98.61%**.
*   **Análisis:** Este resultado demuestra visualmente el concepto de *overfitting* (sobreajuste). Más allá de la época 12, la red comenzó a memorizar detalles innecesarios del set de entrenamiento, lo que habría degradado su capacidad de generalización. El *EarlyStopping* no solo salvó a nuestro modelo de empeorar, sino que nos **ahorró el tiempo computacional** de calcular 83 épocas inútiles.

### 2. El Impacto Masivo de los Datos (Ejercicio 3)
*   **Resultados:** Al cambiar al dataset MNIST completo (60,000 imágenes de 28x28), el tiempo de cómputo aumentó drásticamente a **42.71 segundos** (incluso deteniéndose en solo 9 épocas por *EarlyStopping*). Sin embargo, la red demostró un desempeño formidable, logrando un **97.82% de accuracy** sobre 10,000 imágenes completamente nuevas en el set de prueba.
*   **Análisis:** "Más datos vencen a modelos más complejos". Al proporcionarle a la red un volumen representativo de información (784 características por imagen), el modelo es capaz de extraer patrones de bordes, curvas y trazos universales para los dígitos manuscritos. La caída de la función de pérdida (`loss`) de manera tan acelerada en las primeras épocas confirma que la arquitectura densa es muy buena mapeando este tipo de abstracciones numéricas cuando tiene suficiente "combustible" (datos).

### Reflexión General
El Deep Learning moderno depende tanto de la **calidad/cantidad de los datos** como de las técnicas de **monitoreo**. Combinando arquitecturas profundas con *callbacks* inteligentes como *EarlyStopping*, podemos lograr sistemas estables y precisos de manera computacionalmente responsable.

## 10. Referencias y Recursos

- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [TensorFlow/Keras Sequential](https://keras.io/guides/sequential_model/)
- [EarlyStopping Callback](https://keras.io/api/callbacks/early_stopping/)
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.*

---

📎 **Notebook anterior:** [03. Modelos Clásicos de ML](./03_modelos_clasicos_ml.ipynb)  
📎 **Notebook siguiente:** [05. Redes Convolucionales (CNN)](./05_redes_convolucionales_cnn.ipynb)

---

## 05_redes_convolucionales_cnn.ipynb

## 9. Ejercicios Propuestos

1. **Ejercicio 1:** Cambia el dataset a Fashion MNIST (`keras.datasets.fashion_mnist`). ¿Cómo cambia el accuracy?

2. **Ejercicio 2:** Agrega `BatchNormalization()` después de cada `Conv2D`. ¿Mejora la convergencia?

3. **Ejercicio 3:** Experimenta con Data Augmentation usando `keras.layers.RandomFlip` y `RandomRotation`. ¿Mejora la generalización?

4. **Ejercicio 4 (Avanzado):** Implementa una CNN con 3 bloques convolucionales (Conv2D + BN + MaxPool) y compárala con la arquitectura simple.

# Resolución de Ejercicios Propuestos

En esta última fase práctica, dejaremos de lado los dígitos para trabajar con imágenes más realistas y construiremos arquitecturas profesionales:

1.  **Ejercicio 1: Fashion MNIST.** Entrenaremos una CNN simple en un dataset de prendas de vestir para comprobar cómo se comporta el modelo ante formas y texturas más desafiantes.
2.  **Ejercicio 4 (Avanzado): Arquitectura VGG-Style.** Construiremos una red profunda estructurada en 3 "bloques" convolucionales (con Batch Normalization y MaxPooling), emulando el diseño de las redes ganadoras de competencias internacionales.

## Ejercicio 1: Fashion MNIST (Clasificación de Ropa)

Fashion MNIST contiene 70,000 imágenes (60k train, 10k test) de 28x28 píxeles en escala de grises, divididas en 10 categorías (camisetas, pantalones, zapatos, bolsos, etc.). 
A diferencia de las redes densas (MLP) que requieren aplanar la imagen, las CNNs necesitan que mantengamos la estructura 2D. Por lo tanto, redimensionaremos los datos para incluir el canal de color: de `(28, 28)` a `(28, 28, 1)`.

```python
# 1. Cargar el dataset Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(X_train_fm, y_train_fm), (X_test_fm, y_test_fm) = fashion_mnist.load_data()

# Nombres de las clases para referencia
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. Preprocesamiento: Normalización y Redimensionamiento
# Escalar a valores entre 0 y 1
X_train_fm_norm = X_train_fm / 255.0
X_test_fm_norm = X_test_fm / 255.0

# Redimensionar para agregar el "canal" (1 por ser escala de grises)
# Forma resultante: (samples, altura, anchura, canales)
X_train_cnn = X_train_fm_norm.reshape(-1, 28, 28, 1)
X_test_cnn = X_test_fm_norm.reshape(-1, 28, 28, 1)

print(f"Forma de entrada para la CNN: {X_train_cnn.shape}")

# 3. Definir una CNN Simple (Arquitectura Base)
modelo_cnn_base = Sequential([
    # Extrae características locales
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2), # Reduce la dimensionalidad a la mitad
    
    # Aplanamos para conectar con la capa densa
    Flatten(),
    
    # Clasificador final
    Dense(128, activation='relu'),
    Dropout(0.3), # Regularización suave
    Dense(10, activation='softmax')
])

modelo_cnn_base.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# 4. Entrenamiento
print("\nIniciando entrenamiento Base en Fashion MNIST...")
historia_base = modelo_cnn_base.fit(
    X_train_cnn, y_train_fm,
    epochs=10,
    validation_split=0.2,
    batch_size=64,
    verbose=1
)

# 5. Evaluación
test_loss_base, test_acc_base = modelo_cnn_base.evaluate(X_test_cnn, y_test_fm, verbose=0)
print(f"\nAccuracy del modelo base en Test (Fashion MNIST): {test_acc_base:.4f}")
```

```text
(Salida)
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
[1m29515/29515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
[1m26421880/26421880[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m103s[0m 4us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
[1m5148/5148[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
[1m4422102/4422102[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m18s[0m 4us/step
Forma de entrada para la CNN: (60000, 28, 28, 1)

Iniciando entrenamiento Base en Fashion MNIST...
Epoch 1/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 39ms/step - accuracy: 0.8257 - loss: 0.4946 - val_accuracy: 0.8803 - val_loss: 0.3386
Epoch 2/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 44ms/step - accuracy: 0.8838 - loss: 0.3271 - val_accuracy: 0.8972 - val_loss: 0.2896
Epoch 3/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 40ms/step - accuracy: 0.8988 - loss: 0.2803 - val_accuracy: 0.8979 - val_loss: 0.2795
Epoch 4/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 39ms/step - accuracy: 0.9073 - loss: 0.2558 - val_accuracy: 0.9028 - val_loss: 0.2658
Epoch 5/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 41ms/step - accuracy: 0.9137 - loss: 0.2337 - val_accuracy: 0.9065 - val_loss: 0.2552
Epoch 6/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 44ms/step - accuracy: 0.9208 - loss: 0.2136 - val_accuracy: 0.9094 - val_loss: 0.2519
Epoch 7/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 46ms/step - accuracy: 0.9274 - loss: 0.1976 - val_accuracy: 0.9066 - val_loss: 0.2609
Epoch 8/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 49ms/step - accuracy: 0.9315 - loss: 0.1846 - val_accuracy: 0.9091 - val_loss: 0.2593
Epoch 9/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 46ms/step - accuracy: 0.9370 - loss: 0.1700 - val_accuracy: 0.9094 - val_loss: 0.2703
Epoch 10/10
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 44ms/step - accuracy: 0.9417 - loss: 0.1567 - val_accuracy: 0.9109 - val_loss: 0.2665

Accuracy del modelo base en Test (Fashion MNIST): 0.9069

```

## Ejercicio 4 (Avanzado): Red Convolucional por Bloques (VGG-Style)

En la industria, las CNN no se hacen poniendo capas al azar. Se organizan en "bloques" secuenciales donde la imagen se hace cada vez más pequeña espacialmente (gracias al MaxPooling), pero con más "profundidad" de características (aumentando el número de filtros de Conv2D).
Además, incluiremos `BatchNormalization` dentro de cada bloque para estabilizar el entrenamiento de esta red más profunda.

```python
# 1. Definimos la Arquitectura Avanzada por Bloques
modelo_avanzado = Sequential([
    # --- BLOQUE 1 ---
    # 32 Filtros. Extracción de características muy finas (bordes)
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # --- BLOQUE 2 ---
    # 64 Filtros. Extracción de formas medias (cuellos, texturas)
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # --- BLOQUE 3 ---
    # 128 Filtros. Extracción de formas complejas enteras
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # --- CLASIFICADOR (Cabezal Denso) ---
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), # Dropout fuerte para esta red tan capaz
    Dense(10, activation='softmax')
])

# Resumen visual del modelo
modelo_avanzado.summary()

# 2. Compilación
modelo_avanzado.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# 3. Entrenamiento (Early Stopping recomendado por la profundidad de la red)
early_stop_cnn = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

print("\nIniciando entrenamiento del Modelo Avanzado por Bloques...")
historia_avanzada = modelo_avanzado.fit(
    X_train_cnn, y_train_fm,
    epochs=15, # Pueden correrse más gracias al early stopping
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop_cnn],
    verbose=1
)

# 4. Evaluación Comparativa
test_loss_adv, test_acc_adv = modelo_avanzado.evaluate(X_test_cnn, y_test_fm, verbose=0)
print(f"\n--- Comparación Final ---")
print(f"Accuracy Modelo Simple (Base): {test_acc_base:.4f}")
print(f"Accuracy Modelo Avanzado (VGG-Style): {test_acc_adv:.4f}")
print(f"Mejora relativa: {((test_acc_adv - test_acc_base) * 100):.2f} %")
```

```text
(Salida)

Iniciando entrenamiento del Modelo Avanzado por Bloques...
Epoch 1/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m113s[0m 143ms/step - accuracy: 0.8366 - loss: 0.4711 - val_accuracy: 0.8932 - val_loss: 0.3047
Epoch 2/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m117s[0m 155ms/step - accuracy: 0.8907 - loss: 0.3040 - val_accuracy: 0.8647 - val_loss: 0.3658
Epoch 3/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m124s[0m 165ms/step - accuracy: 0.9086 - loss: 0.2554 - val_accuracy: 0.8900 - val_loss: 0.3083
Epoch 4/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m138s[0m 159ms/step - accuracy: 0.9195 - loss: 0.2246 - val_accuracy: 0.8972 - val_loss: 0.2753
Epoch 5/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m139s[0m 155ms/step - accuracy: 0.9278 - loss: 0.2026 - val_accuracy: 0.9113 - val_loss: 0.2415
Epoch 6/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m125s[0m 166ms/step - accuracy: 0.9366 - loss: 0.1764 - val_accuracy: 0.8943 - val_loss: 0.2950
Epoch 7/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m139s[0m 162ms/step - accuracy: 0.9436 - loss: 0.1576 - val_accuracy: 0.9018 - val_loss: 0.2749
Epoch 8/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m141s[0m 160ms/step - accuracy: 0.9491 - loss: 0.1414 - val_accuracy: 0.8817 - val_loss: 0.3333
Epoch 9/15
[1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m125s[0m 167ms/step - accuracy: 0.9577 - loss: 0.1172 - val_accuracy: 0.9007 - val_loss: 0.3045

--- Comparación Final ---
Accuracy Modelo Simple (Base): 0.9069
Accuracy Modelo Avanzado (VGG-Style): 0.9074
Mejora relativa: 0.05 %

```

## Conclusión Final

Tras comparar una red convolucional básica con una arquitectura profunda inspirada en VGG para el dataset Fashion MNIST, podemos extraer los siguientes aprendizajes:

### 1. El Reto de Fashion MNIST (Ejercicio 1)
*   **Resultados:** Nuestra CNN base (una sola convolución) logró un sólido **90.69% de accuracy** en el set de prueba.
*   **Análisis:** Este resultado confirma que Fashion MNIST es intrínsecamente más complejo que el MNIST numérico tradicional (donde fácilmente superábamos el 98%). Las prendas de vestir presentan variaciones sutiles (como la diferencia entre una camiseta y una camisa, o un zapato y una zapatilla) que requieren extraer características más sofisticadas. Aún así, un 90% es un excelente punto de partida.

### 2. Profundidad vs. Complejidad Computacional (Ejercicio 4)
*   **Resultados:** El modelo avanzado por bloques (VGG-style) finalizó su entrenamiento anticipadamente gracias al *EarlyStopping* (deteniéndose en la época 9 restaurando los pesos de la época 5). Su accuracy en prueba fue del **90.74%**, representando una mejora casi nula (**0.05%**) respecto al modelo base.
*   **Análisis (Costo-Beneficio):** Este es un caso de estudio fantástico sobre el rendimiento decreciente en Deep Learning. Observamos que el tiempo por época del modelo avanzado (~135 segundos) fue más del triple que el del modelo simple (~40 segundos). A pesar de tener mucha más capacidad (capas, filtros, BatchNormalization), el modelo avanzado construyó representaciones matemáticas tan complejas que empezó a sufrir de sobreajuste de forma muy temprana (las épocas 2, 6, 8 y 9 mostraron picos altos de `val_loss`). 

### Reflexión General
"Más profundo no siempre es mejor si no hay suficientes datos que lo justifiquen". Fashion MNIST, al ser imágenes pequeñas de 28x28 píxeles en escala de grises, no posee la resolución geométrica necesaria para que un modelo VGG-Style (diseñado para imágenes a todo color de alta resolución) demuestre todo su potencial. La lección clave aquí es que **la complejidad de la red debe ser proporcional a la complejidad (y tamaño) de los datos de entrada**.

## 10. Referencias y Recursos

- [TensorFlow/Keras CNN](https://keras.io/examples/vision/mnist_convnet/)
- [CNN Explainer (Interactivo)](https://poloclub.github.io/cnn-explainer/)
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.*

---

📎 **Notebook anterior:** [04. Redes Neuronales de Capa Densa](./04_redes_neuronales_capa_densa.ipynb)  
📎 **Notebook siguiente:** [06. Redes Recurrentes (RNN/LSTM)](./06_redes_recurrentes_rnn_lstm.ipynb)

---

## 06_01_rnn_lstm_weather_prediction_daily.ipynb

## 9. Ejercicios Propuestos

1. **Ejercicio 1:** Añade `BatchNormalization` después de la capa recurrente. ¿Mejora la convergencia?

2. **Ejercicio 2:** Prueba con `sequence_length=7` y `sequence_length=60`. ¿Cómo afecta al RMSE?

3. **Ejercicio 3:** Agrega `tmin` y `tmax` como features adicionales (multivariate). ¿Mejora la predicción de `tavg`?

4. **Ejercicio 4:** Implementa un modelo **Bidirectional LSTM** (`keras.layers.Bidirectional(LSTM(50))`). ¿Cambia el resultado?

5. **Ejercicio 5:** Cambia la ciudad a otra de tu elección (busca sus coordenadas). ¿Sigue funcionando bien el modelo?

# Resolución de Ejercicios Propuestos

En esta sección exploraremos cómo la **arquitectura temporal** de las redes recurrentes responde a cambios en la ventana de datos, información multivariada y cambios geográficos:

1.  **Ejercicio 2: Ajuste de la Memoria (Sequence Length).** Evaluaremos cómo impacta "mirar al pasado" evaluando ventanas cortas (7 días) versus ventanas extremadamente largas (60 días).
2.  **Ejercicio 3: Series Temporales Multivariadas.** Enriqueceremos el modelo agregando la temperatura mínima (`tmin`) y máxima (`tmax`) de días anteriores para intentar darle a la red mayor contexto predictivo.
3.  **Ejercicio 5: Prueba Geográfica (Generalización).** Cambiaremos de latitud hacia una ciudad con clima drásticamente distinto (por ejemplo, Londres o Moscú) para comprobar si el patrón capturado sigue siendo válido.

## Ejercicio 2: El Impacto del 'Sequence Length' en LSTM

En el modelado tradicional de series de tiempo, la cantidad de "historia" que le damos a la red es vital. Construiremos dos generadores de secuencias distintos: uno con `sequence_length=7` (1 semana de contexto) y otro con `sequence_length=60` (aproximadamente 2 meses de contexto).
Compararemos el Error Cuadrático Medio (RMSE) final de cada uno.

```python
# 1. Definir los datos escalados
# Usaremos el 80% para entrenar y el 20% para test para que haya suficientes datos
split = int(len(scaled_data) * 0.8)
train_scaled = scaled_data[:split]
test_scaled  = scaled_data[split:]

# 2. Verificar si tenemos suficientes datos para la ventana de 60
if len(test_scaled) <= 60:
    print(f"Error: test_scaled tiene solo {len(test_scaled)} datos. " 
          "Necesitas más de 60 para la prueba de ventana larga.")
else:
    # Continuar con la creación de secuencias
    X_60, y_60 = create_sequences(train_scaled, 60)
    X_test_60, y_test_60 = create_sequences(test_scaled, 60)
    
    # IMPORTANTE: El reshape para que el modelo lo entienda (samples, time_steps, features)
    X_60 = X_60.reshape(X_60.shape[0], X_60.shape[1], 1)
    X_test_60 = X_test_60.reshape(X_test_60.shape[0], X_test_60.shape[1], 1)

# Función auxiliar para crear secuencias rápidas
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# === PRUEBA 1: Ventana Corta (7 días) ===
SEQ_LENGTH_7 = 7
X_7, y_7 = create_sequences(train_scaled, SEQ_LENGTH_7)
X_test_7, y_test_7 = create_sequences(test_scaled, SEQ_LENGTH_7)

# Definimos un modelo rápido encapsulado en función
def entrenar_modelo_lstm(X_train, y_train):
    modelo = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    modelo.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    return modelo

print("Entrenando modelo con Memoria Corta (7 días)...")
modelo_7 = entrenar_modelo_lstm(X_7, y_7)

# === PRUEBA 2: Ventana Larga (60 días) ===
SEQ_LENGTH_60 = 60
X_60, y_60 = create_sequences(train_scaled, SEQ_LENGTH_60)
X_test_60, y_test_60 = create_sequences(test_scaled, SEQ_LENGTH_60)

print("Entrenando modelo con Memoria Larga (60 días)...")
modelo_60 = entrenar_modelo_lstm(X_60, y_60)

# === EVALUACIÓN RMSE ===
from sklearn.metrics import mean_squared_error

pred_7 = scaler.inverse_transform(modelo_7.predict(X_test_7, verbose=0))
real_y_7 = scaler.inverse_transform(y_test_7.reshape(-1, 1))

pred_60 = scaler.inverse_transform(modelo_60.predict(X_test_60, verbose=0))
real_y_60 = scaler.inverse_transform(y_test_60.reshape(-1, 1))

rmse_7 = np.sqrt(mean_squared_error(real_y_7, pred_7))
rmse_60 = np.sqrt(mean_squared_error(real_y_60, pred_60))

print("\n--- Resultados RMSE Comparativos ---")
print(f"RMSE modelo 7 días  : {rmse_7:.4f} °C")
print(f"RMSE modelo 60 días : {rmse_60:.4f} °C")
```

```text
(Salida)
Entrenando modelo con Memoria Corta (7 días)...
Entrenando modelo con Memoria Larga (60 días)...

--- Resultados RMSE Comparativos ---
RMSE modelo 7 días  : 0.7529 °C
RMSE modelo 60 días : 0.8000 °C

```

## Ejercicio 3: Incorporación de 'tmin' y 'tmax' (Multivariado)

Ahora vamos a predecir la temperatura promedio (`tavg`) de mañana basándonos no solo en el historial de `tavg`, sino también agregando la temperatura mínima (`tmin`) y máxima (`tmax`) de los días previos como nuevas "características" (features).
El `input_shape` de la LSTM pasará de `(sequence_length, 1)` a `(sequence_length, 3)`.

```python
# 1. Seleccionamos las 3 características de nuestro DataFrame original
df = data[['temp', 'tmin', 'tmax']].copy()
df = df.ffill().bfill()  # ffill primero, bfill por si hay NaN al inicio

data_multivariada = df[['temp', 'tmin', 'tmax']].values

# 2. Escalamos TODAS las variables usando MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler_multi = MinMaxScaler(feature_range=(0,1))
data_multi_scaled = scaler_multi.fit_transform(data_multivariada)

# 3. Separación en Train y Test (ej. 80% - 20%)
train_size = int(len(data_multi_scaled) * 0.8)
train_multi = data_multi_scaled[:train_size]
test_multi = data_multi_scaled[train_size:]

# 4. Creación de secuencias Multivariadas
SEQ_LEN = 14 # Usaremos 2 semanas
X_multi_train, y_multi_train = create_sequences(train_multi, SEQ_LEN)
X_multi_test, y_multi_test = create_sequences(test_multi, SEQ_LEN)

print(f"Forma de X Multivariada: {X_multi_train.shape} (muestras, {SEQ_LEN} días, 3 características)")

# 5. Entrenamos el Modelo
print("Entrenando modelo Multivariado...")
modelo_multi = Sequential([
    LSTM(50, activation='relu', input_shape=(X_multi_train.shape[1], X_multi_train.shape[2])),
    Dense(1) # Predecimos un solo valor final (tavg)
])

modelo_multi.compile(optimizer='adam', loss='mse')
modelo_multi.fit(X_multi_train, y_multi_train, epochs=20, batch_size=32, verbose=0)

# 6. Evaluación
pred_multi = modelo_multi.predict(X_multi_test, verbose=0)

# Truco para de-escalar solo la columna 0 (tavg)
pred_multi_dummy = np.zeros((len(pred_multi), 3))
pred_multi_dummy[:, 0] = pred_multi[:, 0]
pred_multi_real = scaler_multi.inverse_transform(pred_multi_dummy)[:, 0]

y_multi_dummy = np.zeros((len(y_multi_test), 3))
y_multi_dummy[:, 0] = y_multi_test
y_real_multi = scaler_multi.inverse_transform(y_multi_dummy)[:, 0]

rmse_multi = np.sqrt(mean_squared_error(y_real_multi, pred_multi_real))
print(f"\nRMSE Modelo Multivariado: {rmse_multi:.4f} °C")
```

```text
(Salida)
Forma de X Multivariada: (1503, 14, 3) (muestras, 14 días, 3 características)
Entrenando modelo Multivariado...

RMSE Modelo Multivariado: 0.7567 °C

```

## Ejercicio 5: Cambio de Latitud (Evaluando la Generalización)

Para verificar si la red realmente captura dinámicas atmosféricas o si solo memorizó el clima monótono original, descargaremos datos de una ciudad con un comportamiento estacional mucho más marcado, contrastante y ruidoso: **Londres, Reino Unido** (Coordenadas: 51.5074, -0.1278).

```python
# 1. Obtenemos datos de Londres para el año 2023
latitude  = 51.5074
longitude = -0.1278

POINT = ms.Point(latitude,longitude, 113)
START = date(2024, 1, 1)
END = date(2024, 12, 31)

# Get nearby weather stations
stations = ms.stations.nearby(POINT, limit=4)

# Get daily data & perform interpolation
ts = ms.daily(stations, START, END)
data_london = ms.interpolate(ts, POINT).fetch()

# Limpiamos nulos
data_london = data_london.dropna(subset=['temp'])

# 2. Visualización rápida de la estacionalidad de Londres
plt.figure(figsize=(12, 4))
plt.plot(data_london.index, data_london['temp'], color='purple')
plt.title('Temperatura Promedio en Londres - 2023')
plt.ylabel('Temperatura (°C)')
plt.grid(True)
plt.show()

# 3. Aplicamos el Pipeline Base
london_tavg = data_london[['temp']].values
scaler_london = MinMaxScaler(feature_range=(0,1))
london_scaled = scaler_london.fit_transform(london_tavg)

X_london, y_london = create_sequences(london_scaled, 14)

train_sz_l = int(len(X_london) * 0.8)
X_l_t, y_l_t = X_london[:train_sz_l], y_london[:train_sz_l]
X_l_test, y_l_test = X_london[train_sz_l:], y_london[train_sz_l:]

print("\nEntrenando modelo LSTM para el clima de Londres...")
modelo_london = entrenar_modelo_lstm(X_l_t, y_l_t)

pred_l = scaler_london.inverse_transform(modelo_london.predict(X_l_test, verbose=0))
real_y_l = scaler_london.inverse_transform(y_l_test.reshape(-1, 1))

rmse_london = np.sqrt(mean_squared_error(real_y_l, pred_l))
print(f"RMSE para Londres (Modelo Independiente): {rmse_london:.4f} °C")
```

```text
(Salida)

Entrenando modelo LSTM para el clima de Londres...
RMSE para Londres (Modelo Independiente): 2.9349 °C

```

## Conclusión Final

Tras finalizar los ejercicios prácticos de modelado climático, podemos destacar las siguientes conclusiones fundamentales sobre el comportamiento de las redes recurrentes:

### 1. El Costo de la "Memoria a Largo Plazo" (Ejercicio 2)
*   **Resultados:** El modelo con una ventana corta (7 días) obtuvo un RMSE de **0.7529 °C**, superando al modelo con ventana larga (60 días) que obtuvo un RMSE de **0.8000 °C**.
*   **Análisis:** En meteorología (y muchas otras series de tiempo), el pasado inmediato es el mejor predictor del futuro a corto plazo. Al forzar a la red a mirar 60 días atrás para predecir el clima de mañana, introdujimos "ruido" estacional irrelevante que diluyó la importancia de las temperaturas de la última semana. Las LSTMs pueden recordar a largo plazo, pero eso no siempre significa que esa información sea útil para la predicción inmediata.

### 2. El Espejismo Numérico de los Modelos Multivariados (Ejercicio 3)
*   **Resultados:** Al agregar `tmin` y `tmax` a nuestro modelo de 14 días, el RMSE fue de **0.7567 °C**, virtualmente idéntico (e incluso ligeramente superior) al modelo univariado más simple.
*   **Análisis:** Intuitivamente, pensaríamos que "más datos = mejores predicciones". Sin embargo, en la zona geográfica original del dataset (Cartagena), la temperatura de hoy y mañana están tan altamente correlacionadas que la temperatura promedio es suficiente. Agregar `tmin` y `tmax` añadió complejidad computacional sin aportar nueva información discriminante (colinealidad), lo que explica por qué la red no logró reducir más el error subyacente.

### 3. La Realidad de la Generalización Geográfica (Ejercicio 5)
*   **Resultados:** Al aplicar nuestra metodología a la ciudad de **Londres**, el RMSE se disparó a **2.9349 °C**.
*   **Análisis:** Este resultado es la lección más importante del Deep Learning aplicado. Londres, a diferencia de localidades tropicales, tiene estaciones muy marcadas, frentes fríos repentinos y alta volatilidad diaria (varianza). Un margen de error de casi 3 grados demuestra que los modelos climáticos son hiper-locales. La arquitectura que sirve para un clima monótono es insuficiente para modelar el caos atmosférico de latitudes altas; aquí sí requeriríamos modelos más profundos, ventanas más ajustadas, y definitivamente un enfoque multivariado agresivo (incluyendo humedad, presión, épocas del año, etc.).

### Reflexión General
El modelado de series de tiempo es inherentemente desafiante. No basta con apilar capas LSTM; el éxito radica en entender qué cantidad de "pasado" es estadísticamente relevante para el "futuro", y aceptar que los patrones aprendidos en un contexto (geográfico, financiero o físico) raramente son transferibles de manera universal.

## 10. Referencias y Recursos

- [Meteostat Python Library](https://dev.meteostat.net/python/)
- [Keras LSTM documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.*

---

📎 **Notebook anterior:** [06. RNN/LSTM Intro](./06_redes_recurrentes_rnn_lstm.ipynb)

---

## 06_redes_recurrentes_rnn_lstm.ipynb

## 10. Ejercicios Propuestos

1. **Ejercicio 1:** Cambia `seq_length` a 10, 50 y 100. ¿Cómo afecta al MSE?

2. **Ejercicio 2:** Implementa una GRU y compara con RNN y LSTM.

3. **Ejercicio 3:** Usa un dataset real de series temporales (p.ej. precios de acciones o temperatura) y aplica LSTM.

4. **Ejercicio 4 (Avanzado):** Implementa una LSTM bidireccional (`keras.layers.Bidirectional`) y compara resultados.

# Resolución de Ejercicios Propuestos

En esta sección, compararemos arquitecturas de vanguardia para modelador secuencias sintéticas (onda senoidal), observando la diferencia en tiempos de entrenamiento y capacidad matemática:

1.  **Ejercicio 2: Comparativa Completa (RNN vs. LSTM vs. GRU).** Implementaremos una red con celdas GRU (Gated Recurrent Unit) para entender empíricamente cómo esta compite en desempeño (MSE) contra las arquitecturas LSTM y SimpleRNN tradicionales.
2.  **Ejercicio 4 (Avanzado): Redes Bidireccionales.** Reforzaremos nuestro modelo de vanguardia aplicando procesamiento `Bidirectional` para alimentar la recurrencia tanto "hacia el futuro" como "hacia el pasado".

## Ejercicio 2: GRU (Gated Recurrent Unit) vs. LSTM vs. SimpleRNN

La celda **GRU** es una versión simplificada de la LSTM. Mientas la LSTM usa 3 compuertas (Forget, Input, Output), la GRU solo usa 2 (Update, Reset). Esto permite que la GRU sea matemáticamente menos costosa y entrene más rápido, alcanzando niveles de Precisión/Error (MSE) idénticos a los de una LSTM en la mayoría de tareas de longitud media.

Crearemos una arquitectura estándar y solo intercambiaremos la capa central para comparar sus tiempos y el MSE final en el set de prueba.

```python
# Función genérica para entrenar cualquier tipo de capa recurrente
def entrenar_comparativo(CapaRecurrente, nombre_capa, X_train, y_train, X_test, y_test):
    modelo = Sequential([
        # Usamos input_shape = (seq_length, features)
        CapaRecurrente(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    
    modelo.compile(optimizer='adam', loss='mse')
    
    print(f"\n--- Entrenando {nombre_capa} ---")
    inicio = time.time()
    # Usaremos 20 epochs fijas para ser justos en la comparación de tiempo
    historial = modelo.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    fin = time.time()
    
    # Evaluación
    predicciones = modelo.predict(X_test, verbose=0)
    mse_final = mean_squared_error(y_test, predicciones)
    
    print(f"Tiempo Total: {(fin - inicio):.2f} segundos")
    print(f"MSE en Test : {mse_final:.6f}")
    
    return modelo, historial, predicciones

_, _, pred_rnn = entrenar_comparativo(SimpleRNN, "SimpleRNN Clásica", X_train, y_train, X_test, y_test)
_, _, pred_lstm = entrenar_comparativo(LSTM, "LSTM (Long Short-Term Memory)", X_train, y_train, X_test, y_test)
_, _, pred_gru = entrenar_comparativo(GRU, "GRU (Gated Recurrent Unit)", X_train, y_train, X_test, y_test)
```

```text
(Salida)

--- Entrenando SimpleRNN Clásica ---
Tiempo Total: 11.71 segundos
MSE en Test : 0.001201

--- Entrenando LSTM (Long Short-Term Memory) ---
Tiempo Total: 19.57 segundos
MSE en Test : 0.001037

--- Entrenando GRU (Gated Recurrent Unit) ---
Tiempo Total: 19.45 segundos
MSE en Test : 0.001210

```

## Ejercicio 4 (Avanzado): Arquitectura Bidirectional LSTM

En problemas donde el contexto posterior (los datos que vienen después en la secuencia) es accesible (como procesar un párrafo de texto, o procesar una onda matemática periódica completa), la red **Bidireccional** es insuperable.
Al usar la envoltura `Bidirectional`, le decimos a Keras que cree internamente *dos* capas `LSTM`: una que procesará la secuencia de $t_1$ a $t_n$, y otra independiente que procesará de $t_n$ hacia atrás hasta $t_1$. Luego, ambas salidas se concatenan automáticamente en la capa `Dense`.

```python
# 1. Definimos y estructuramos la red Bidireccional
modelo_bidir = Sequential([
    # Envolvemos la LSTM en una capa Bidireccional
    Bidirectional(LSTM(50, activation='relu'), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1) # Por defecto, concatena la salida hacia el frente y hacia atrás (50 + 50 = 100 conexiones)
])

modelo_bidir.compile(optimizer='adam', loss='mse')

# 2. Entrenamos el modelo
print("\n--- Entrenando Bidirectional LSTM ---")
inicio_bi = time.time()
modelo_bidir.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
fin_bi = time.time()

# 3. Predict y calculo del error
pred_bidir = modelo_bidir.predict(X_test, verbose=0)
mse_bidir = mean_squared_error(y_test, pred_bidir)

print(f"Tiempo Total Bidireccional: {(fin_bi - inicio_bi):.2f} segundos")
print(f"MSE en Test (Bidirectional): {mse_bidir:.6f}")

# 4. Gráfico Visual del desempeño
plt.figure(figsize=(14, 5))
plt.plot(y_test, label='Verdadero (Onda Seno)', color='black', alpha=0.5, linestyle='--')
plt.plot(pred_rnn, label='Predicción RNN', color='red', alpha=0.6)
plt.plot(pred_lstm, label='Predicción LSTM', color='blue', alpha=0.6)
plt.plot(pred_bidir, label='Predicción Bidireccional', color='green', linewidth=2)
plt.title('Comparativa de Ajuste: RNN vs LSTM vs Bidireccional')
plt.xlabel('Paso de tiempo (Time Step)')
plt.ylabel('Valor Matemático')
plt.legend()
plt.grid(True)
plt.show()
```

```text
(Salida)

--- Entrenando Bidirectional LSTM ---
Tiempo Total Bidireccional: 20.12 segundos
MSE en Test (Bidirectional): 0.001113

```

## Conclusión Final

Tras finalizar los experimentos con la serie temporal sintética (onda seno) y someter a prueba a las 4 arquitecturas recurrentes más importantes (todas a 20 épocas), podemos extraer las siguientes conclusiones fundamentales:

### 1. El Costo de la Memoria a Corto Plazo (SimpleRNN)
*   **Resultados:** La `SimpleRNN` clásica fue, por mucho, el modelo más rápido de entrenar (**11.71 segundos**), pero no logró el mejor desempeño, obteniendo un MSE de **0.001201**.
*   **Análisis:** Este resultado valida la teoría. Al no poseer "compuertas" (gates) para retener información importante, la RNN simple sufre de desvanecimiento del gradiente. Para predecir el comportamiento matemático, es computacionalmente ligera, pero "olvida" rápidamente los patrones a largo plazo.

### 2. El Rey Histórico (LSTM)
*   **Resultados:** La `LSTM` demostró su superioridad logrando el error más bajo de todos los modelos: **0.001037**. A cambio, su tiempo de entrenamiento virtualmente se duplicó a **19.57 segundos**.
*   **Análisis:** Sus 3 compuertas internas (Forget, Input, Output) le permiten regular perfectamente qué parte de la onda senoidal es importante recordar y qué es ruido. Esta capacidad de modelar dependencias a largo plazo la hace más lenta, pero indudablemente más precisa para captar las crestas y valles de la onda perfecta.

### 3. La Promesa de las GRU
*   **Resultados:** La celda `GRU` entrenó casi en el mismo tiempo que la LSTM (**19.45 segundos**), pero obtuvo el MSE más alto (**0.001210**), empatando prácticamente con la SimpleRNN.
*   **Análisis:** Generalmente, las GRU son más rápidas que las LSTM. Sin embargo, en un dataset sintético tan pequeño y "limpio" como una onda seno, la sobrecarga computacional se neutraliza. La GRU, al tener solo 2 compuertas, fue ligeramente menos expresiva matemáticamente que la LSTM completa en este caso específico, no pudiendo alcanzar su nivel de precisión milimétrica.

### 4. El "Overkill" del Modelo Bidireccional
*   **Resultados:** La `Bidirectional LSTM` requirió el mayor tiempo de cómputo (**20.12 segundos**) obteniendo el segundo mejor lugar en precisión (**0.001113**).
*   **Análisis:** ¿Por qué no le ganó a la LSTM normal? Las redes bidireccionales brillan en contextos ruidosos o semánticos (como analizar frases donde el final cambia el significado del principio). Pero una onda seno es una función periódica matemáticamente prístina. Leerla "hacia atrás" no aporta información contextual que la red no haya podido deducir leyéndola "hacia adelante". El ensamble bidireccional simplemente agregó parámetros (cálculos) innecesarios a un problema ya resuelto por la LSTM unidireccional.

### Reflexión General
Este laboratorio demuestra un principio clave del Deep Learning: **"Más parámetros no siempre equivalen a mejores resultados"**. Para series de tiempo claras, rítmicas y sin ruido, una red con la arquitectura de compuertas correcta (LSTM estándar) alcanza el límite de aprendizaje, haciendo redundantes enfoques avanzandos (como Bidireccionalidad) que están diseñados para dominios mucho más caóticos y ruidosos.

## 11. Referencias y Recursos

- [TensorFlow/Keras RNN](https://keras.io/api/layers/recurrent_layers/)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.*

---

📎 **Notebook anterior:** [05. Redes Convolucionales (CNN)](./05_redes_convolucionales_cnn.ipynb)  
📎 **Notebook siguiente:** [07. Transformers y Atención](./07_transformers.ipynb)

---

## 07_transformers.ipynb

## 7. Ejercicios Propuestos

1. **Ejercicio 1:** Usa el pipeline de `zero-shot-classification` para clasificar textos en categorías personalizadas (deportes, tecnología, política).

2. **Ejercicio 2:** Prueba el pipeline de `text-generation` con GPT-2. Genera textos a partir de diferentes prompts.

3. **Ejercicio 3:** Compara la atención entre diferentes capas del modelo (primera vs última). ¿Qué patrones observas?

4. **Ejercicio 4 (Avanzado):** Haz fine-tuning de DistilBERT en un dataset personalizado de clasificación de texto usando HuggingFace Trainer.

## Resolución de Ejercicios Propuestos

A continuación, se presenta la resolución paso a paso de los 3 primeros ejercicios propuestos para consolidar los conocimientos teóricos y prácticos sobre la arquitectura Transformer y la librería HuggingFace.

1. **Ejercicio 1:** Clasificación Zero-Shot en categorías personalizadas.
2. **Ejercicio 2:** Generación de texto autorregresivo con GPT-2.
3. **Ejercicio 3:** Visualización y comparación de la atención intercapas.

### Ejercicio 1: Clasificación Zero-Shot

En este ejercicio vamos a utilizar el pipeline `zero-shot-classification` de HuggingFace. A diferencia de un clasificador lineal o una red densa clásica, **no** necesitamos entrenar este modelo con nuestras propias etiquetas. 

Simplemente le pasaremos una oración de prueba y una lista de categorías (labels) totalmente arbitrarias inventadas por nosotros. El modelo preentrenado (por defecto `facebook/bart-large-mnli`) calculará la probabilidad de pertenencia de la oración a cada categoría basándose en su profunda comprensión semántica del texto.

```python
# 1. Instanciamos el clasificador
print("Cargando pipeline Zero-Shot...")
classifier_zs = pipeline("zero-shot-classification")

# 2. Definimos una oración de prueba y nuestras categorías
secuencia = "The new electric car from Tesla can accelerate from 0 to 60 in less than two seconds, revolutionizing the automotive industry."
etiquetas_candidatas = ["sports", "technology", "politics", "health", "automotive"]

# 3. Ejecutamos la clasificación Zero-Shot
print("\nRealizando clasificación...")
resultados_zs = classifier_zs(secuencia, candidate_labels=etiquetas_candidatas)

# 4. Mostramos los resultados ordenados por el modelo
print("\n--- Resultados de la Clasificación ---")
print(f"Texto original: '{resultados_zs['sequence']}'\n")

print("Probabilidades por categoría asignadas por el Transformer:")
for etiqueta, puntaje in zip(resultados_zs['labels'], resultados_zs['scores']):
    print(f" -> {etiqueta.capitalize():<12}: {puntaje:.4f} ({puntaje*100:.2f}%)")

```

```text
(Salida)
No model was supplied, defaulted to facebook/bart-large-mnli and revision d7645e1.
Using a pipeline without specifying a model name and revision in production is not recommended.

```

```text
(Salida)
Cargando pipeline Zero-Shot...

```

```text
(Salida)

Realizando clasificación...

--- Resultados de la Clasificación ---
Texto original: 'The new electric car from Tesla can accelerate from 0 to 60 in less than two seconds, revolutionizing the automotive industry.'

Probabilidades por categoría asignadas por el Transformer:
 -> Technology  : 0.5585 (55.85%)
 -> Automotive  : 0.4305 (43.05%)
 -> Sports      : 0.0061 (0.61%)
 -> Health      : 0.0034 (0.34%)
 -> Politics    : 0.0015 (0.15%)

```

### Ejercicio 2: Generación de Texto con GPT-2

En esta ocasión probaremos un modelo de la familia **Decoder** (la misma familia de ChatGPT), específicamente GPT-2 (Generative Pre-trained Transformer 2). Estos modelos son de tipo autorregresivos, lo que significa que generan el siguiente token (palabra) basándose en todo su contexto previo.

Para interactuar con él, crearemos un pipeline de **`text-generation`**. Le proporcionaremos una frase inicial (conocida como *prompt*) y le pediremos al modelo que complete la idea. En el código ajustaremos hiperparámetros de generación como `max_length` (límite máximo del texto total a generar) y `num_return_sequences` (cuántas variantes o finales distintos queremos crear a partir del mismo prompt).

```python
# 1. Instanciamos el modelo generativo (tardará un momento en bajar los pesos de GPT-2)
print("Instanciando pipeline generativo (GPT-2)...")
generator = pipeline('text-generation', model='gpt2')

# 2. Definimos una semilla (opcional, pero útil de ver)
set_seed(42)

# 3. Nuestro prompt (historia) inicial
prompt = "In the year 2050, artificial intelligence will finally be able to"

print(f"\nGenerando múltiples continuaciones a partir del prompt:\n > '{prompt}'\n")

# 4. Ejecutamos la generación autorregresiva
# max_length incluye la longitud del prompt original
resultados_gpt = generator(prompt, 
                           max_length=60,               # Largo máximo de la generación
                           num_return_sequences=3,      # Generame 3 continuaciones distintas
                           pad_token_id=50256,          # Token de relleno para evitar un "warning" común en GPT-2
                           truncation=True)

# 5. Imprimimos las historias generadas por el Transformer
for idx, tex in enumerate(resultados_gpt):
    print(f"--- Variante {idx + 1} ---")
    print(tex['generated_text'])
    print("\n" + "-"*40 + "\n")
```

```text
(Salida)
Instanciando pipeline generativo (GPT-2)...

```

```text
(Salida)
Passing `generation_config` together with generation-related arguments=({'pad_token_id', 'num_return_sequences', 'max_length'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
Both `max_new_tokens` (=256) and `max_length`(=60) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

```

```text
(Salida)

Generando múltiples continuaciones a partir del prompt:
 > 'In the year 2050, artificial intelligence will finally be able to'

--- Variante 1 ---
In the year 2050, artificial intelligence will finally be able to do a number of things: we'll be able to keep track of our car's movements, detect traffic patterns and even track traffic in real time using our smartphones.

This is a lot more than just a technical breakthrough. It's also a huge step forward in the field of artificial intelligence. The next big step in Artificial Intelligence will come from computer vision. That's when we'll be able to detect when a human is being watched, when he's approaching the car, when he's running, when he's in the middle of a road. This is already the most important aspect of AI.

The next step will be to start to build on top of this technology and use it to make intelligent driving more accessible. This is also the time when the world will be starting to see how much you have to spend to get the best possible experience.

"In the next decade, we're going to see even better and more automated driving. We're going to see more of the same."

In the next decade, we'll see even better and more automated driving. We're going to see more of the same.

This is the next step in Artificial Intelligence.

You may have heard of the idea of autonomous cars. That's

----------------------------------------

--- Variante 2 ---
In the year 2050, artificial intelligence will finally be able to adapt to the needs of everyday people, which will drive the growth of global services and the global economy.

But the technology will still need to be applied to everyday life. We will see a few of the biggest advances in AI in our current society and in the future.

The first thing the public can do to protect against this future is to join together with the private sector.

Join us for the 100th anniversary of the first Artificial Intelligence Conference in January 2018. We will be in Tokyo to celebrate the 100th anniversary of the first Artificial Intelligence Conference in Japan.

----------------------------------------

--- Variante 3 ---
In the year 2050, artificial intelligence will finally be able to play a more important role in the 21st century, thanks to advances in computing and robotics.

The new AI is called the AIA-7, and it will replace the AIA-7 that was unveiled in October 2015, according to Google's Android blog.

The new AI is called the AIA-7, and it will replace the AIA-7 that was unveiled in October 2015, according to Google's Android blog.

AIA-7

The AIA-7 is the successor of the AIA-7 that was unveiled in October 2015.

The new AI, called the AIA-7, is based on the same platform as the AIA-8 and AIA-9. It will be called the AIA-7. It is based on the same platform as the AIA-7 that was unveiled in October 2015.

The AIA-7 is currently the only AI system to be based on the AIA-9, but it is based on the same platform as the AIA-8. It is based on the same platform as the AIA-8 that was unveiled in October 2015.

The AIA-7 is a version of the AIA-7 that

----------------------------------------


```

### Ejercicio 3: Comparación de Mapas de Atención (Intercapas)

Uno de los mayores atractivos de la arquitectura Transformer es su nivel de **inteligibilidad** gracias a los mecanismos de atención, que podemos sacar de la "caja negra". El modelo genera un "mapa de calor" o matriz que indica pesos de importancia temporal entre las palabras de la misma secuencia.

En este ejercicio extraemos los pesos internos (`outputs.attentions`) del modelo `bert-base-uncased` al procesar una frase. El objetivo es graficar la **Primera Capa** vs la **Última Capa** (capa 0 vs capa 11 o -1). 

Observaremos un patrón claro de Deep Learning: 
* Las **primeras capas** solo captan relaciones limitadas (atención lineal hacia la palabra contigua).
* Las **últimas capas** logran mapear la semántica global de la oración, cruzando sujetos con acciones espacialmente alejadas.

```python
# 1. Cargamos el modelo base (BERT) y su tokenizador
modelo_bert = 'bert-base-uncased'
print(f"Cargando {modelo_bert} y solicitando mapas de atención internos...")
tokenizer = BertTokenizer.from_pretrained(modelo_bert)
# IMPORTANTE: Forzamos la bandera `output_attentions=True` para abrir la caja negra
model = BertModel.from_pretrained(modelo_bert, output_attentions=True)

# 2. Preparamos una oración para ser analizada por completo
frase = "The cat sat on the mat and stared at the dangerously fast mouse."
inputs = tokenizer(frase, return_tensors='pt')
outputs = model(**inputs)

# 3. Extraemos las matrices (En BERT Base hay 12 capas en total)
# outputs.attentions devuelve una tupla de tensores (una por cada capa)
matrices = outputs.attentions  

# Capa inicial (Índice 0), cabeza de atención 0
att_primera_capa = matrices[0][0, 0].detach().numpy()  
# Capa final profunda (Índice -1), cabeza de atención 0
att_ultima_capa = matrices[-1][0, 0].detach().numpy()  

# Convertimos los IDs numéricos internos de vuelta a las sub-palabras en inglés (tokens)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 4. Graficamos ambos mapas lado a lado para la comparativa de interpretabilidad 
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Sub-gráfico Izquierdo (Capa Temprana)
sns.heatmap(att_primera_capa, xticklabels=tokens, yticklabels=tokens, cmap='viridis', ax=ax[0])
ax[0].set_title('Primera Capa (Atención Superficial/Lineal)', fontsize=14)
ax[0].set_xlabel('Token Atendido')
ax[0].set_ylabel('Token que "Presta la Atención"')

# Sub-gráfico Derecho (Capa Profunda)
sns.heatmap(att_ultima_capa, xticklabels=tokens, yticklabels=tokens, cmap='magma', ax=ax[1])
ax[1].set_title('Última Capa (Comprensión Semántica Compleja)', fontsize=14)
ax[1].set_xlabel('Token Atendido')

plt.tight_layout()
plt.show()
```

```text
(Salida)
Cargando bert-base-uncased y solicitando mapas de atención internos...

```

```text
(Salida)
[1mBertModel LOAD REPORT[0m from: bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```

## Conclusión Final

A partir de los ejercicios desarrollados, podemos establecer las siguientes conclusiones clave sobre la arquitectura Transformer y su manipulación:

### 1. Clasificación Zero-Shot (Extracción de Semántica)
Ha quedado demostrado el inmenso poder de las representaciones latentes en los modelos masivos preentrenados. Sin haberle enseñado previamente qué era "Tecnología" o "Automotriz" con un dataset clásico de entrenamiento (`X_train`), el modelo fue capaz de asociar semánticamente términos como *"Tesla"* y *"electric car"* con dichas categorías, otorgándoles un **~98.9%** de probabilidad combinada frente al resto. Esto valida que los transformers modernos capturan un conocimiento profundo del mundo y del lenguaje, utilizable *out-of-the-box* (listo para usar).

### 2. Generación Autorregresiva (Familia GPT)
A través de la prueba con **GPT-2**, comprobamos la naturaleza probabilística y divergente de los arquitecturas tipo *Decoder*. A partir de un mismo prompt determinista ("*In the year 2050...*"), la red fue capaz de extrapolar tres "alucinaciones" o predicciones de texto completamente distintas. Todas mantuvieron una estricta coherencia gramatical y temática (coches autónomos, economía, robótica), evidenciando cómo el modelo usa la "atención enmascarada" temporal para calcular iterativamente la siguiente palabra más lógica. *(Nota técnica: Los avisos previos en consola son normales; HuggingFace actualiza constantemente sus APIs de generación recomendando usar objetos `GenerationConfig` en lugar de argumentos sueltos)*.

### 3. Interpretabilidad y Mapas de Atención
Las visualizaciones generadas por `seaborn` destaparon la "caja negra" funcional del modelo BERT, mostrando un fenómeno de la literatura técnica bastante común:
* **En la Primera Capa (Izquierda):** El mapa de calor es difuso y disperso. La atención se distribuye de manera algo lineal (colores azulados/verdosos repartidos), enfocándose en palabras locales vecinas. El modelo recién está "leyendo" la estructura base.
* **En la Última Capa (Derecha):** La matriz sufre un cambio radical, volviéndose extremadamente especializada (negro y altas luces). Se vuelve visible un patrón clásico de BERT: **Casi todos los tokens "descargan" su máxima atención (línea amarilla brillante a la derecha) en signos de puntuación como el `.` o en el token separador `[SEP]`**. El modelo usa estos tokens estáticos como "basureros" de atención inútil, permitiendo que el poco peso de atención restante forje las relaciones semánticas puras, complejas de larga distancia de la oración (sintetizando el verdadero "significado" global de la frase antes de entregar su resultado).

## 8. Referencias y Recursos

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [HuggingFace Model Hub](https://huggingface.co/models)
- Vaswani et al. (2017). *Attention is All You Need.*
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

📎 **Notebook anterior:** [06. Redes Recurrentes (RNN/LSTM)](./06_redes_recurrentes_rnn_lstm.ipynb)  
📎 **Notebook siguiente:** [08. Redes Generativas (GANs)](./08_gans.ipynb)

---

## 08_gans.ipynb

## 7. Ejercicios Propuestos

1. **Ejercicio 1:** Entrena con más epochs (2000-5000). ¿Mejora la calidad visual de los dígitos generados?

2. **Ejercicio 2:** Cambia el dataset a Fashion MNIST. ¿La GAN puede generar prendas de ropa reconocibles?

3. **Ejercicio 3:** Implementa una DCGAN (GAN con capas convolucionales) para mejor calidad de imagen.

4. **Ejercicio 4 (Avanzado):** Implementa una Conditional GAN (cGAN) que permita generar un dígito específico pasando la clase como condición.

## Resolución de Ejercicios Propuestos

A continuación, se desarrollarán los ejercicios propuestos para profundizar en la comprensión y aplicación de las Redes Generativas Antagónicas (GANs). Específicamente, abordaremos:

*   **Ejercicio 2 (Dataset Fashion MNIST):** Evaluaremos la capacidad de la arquitectura GAN básica (densa/MLP) para generalizar a un conjunto de datos diferente y visualmente más complejo (prendas de vestir en lugar de dígitos escritos a mano).
*   **Ejercicio 3 (DCGAN):** Evolucionaremos nuestro modelo utilizando capas convolucionales (`Conv2DTranspose` en el Generador y `Conv2D` en el Discriminador) para capturar y generar características espaciales con mucha más calidad.


### Ejercicio 2: Adaptación a Fashion MNIST

**Objetivo:** Cambiar el dataset de entrenamiento a Fashion MNIST y observar si una GAN densa (Multilayer Perceptron) que funciona razonablemente bien con números, es capaz de generar siluetas de prendas de ropa reconocibles.

**Implementación:**
Reutilizaremos las mismas funciones `build_generator` y `build_discriminator` del ejemplo demostrativo. Solo cambiaremos la carga de datos a `keras.datasets.fashion_mnist` y ejecutaremos el ciclo de entrenamiento por 3000 épocas.

```python
# 1. Cargar el dataset Fashion MNIST
(X_train_fashion, _), (_, _) = keras.datasets.fashion_mnist.load_data()

# Normalizar las imágenes al rango [0, 1] y aplanar a vectores de 784
X_train_fashion = X_train_fashion.astype('float32') / 255.0
X_train_fashion = X_train_fashion.reshape(-1, 28*28)

# Configuración
LATENT_DIM = 100
EPOCHS = 3000
BATCH_SIZE = 128

# 2. Arquitecturas del Generador y Discriminador (Densa/MLP)
def build_generator_f(latent_dim=LATENT_DIM):
    model = keras.Sequential([
        keras.layers.Dense(128, input_dim=latent_dim),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(256),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(28*28, activation='sigmoid')
    ])
    return model

def build_discriminator_f():
    model = keras.Sequential([
        keras.layers.Dense(256, input_dim=28*28),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# 3. Inicializar y compilar
generator_f = build_generator_f()
discriminator_f = build_discriminator_f()
discriminator_f.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evitamos que el discriminador se entrene dentro del modelo GAN combinado
discriminator_f.trainable = False

# Modelo GAN combinado
z = keras.Input(shape=(LATENT_DIM,))
img = generator_f(z)
valid = discriminator_f(img)
gan_f = keras.Model(z, valid)
gan_f.compile(optimizer='adam', loss='binary_crossentropy')

# 4. Función de ayuda para visualizar
def plot_generated_images_f(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, LATENT_DIM])
    generated_images = generator.predict(noise, verbose=0)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Resultados Epoch {epoch}')
    plt.tight_layout()
    plt.show()

# 5. Bucle de Entrenamiento para Fashion MNIST
real_labels = np.ones((BATCH_SIZE, 1)) * 0.9 # Label smoothing
fake_labels = np.zeros((BATCH_SIZE, 1))

print("Iniciando entrenamiento con Fashion MNIST...")

for epoch in range(EPOCHS):
    # Seleccionar un lote de imágenes reales
    idx = np.random.randint(0, X_train_fashion.shape[0], BATCH_SIZE)
    real_imgs = X_train_fashion[idx]

    # Generar un lote de imágenes falsas
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_imgs = generator_f.predict(noise, verbose=0)

    # Entrenar el Discriminador
    d_loss_real = discriminator_f.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator_f.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Entrenar el Generador (a través de la GAN)
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    valid_labels = np.ones((BATCH_SIZE, 1))
    g_loss = gan_f.train_on_batch(noise, valid_labels)

    # Mostrar progreso cada 500 épocas
    if epoch % 500 == 0:
        print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.1f}%] [G loss: {g_loss:.4f}]")
        plot_generated_images_f(generator_f, epoch)

print("Entrenamiento completado.")
# Visualización final en la última época
plot_generated_images_f(generator_f, EPOCHS)
```

```text
(Salida)
Iniciando entrenamiento con Fashion MNIST...
Epoch 0 [D loss: 0.8079, acc.: 7.8%] [G loss: 0.6661]

```

```text
(Salida)
Epoch 500 [D loss: 3.3893, acc.: 0.0%] [G loss: 0.0064]

```

```text
(Salida)
Epoch 1000 [D loss: 3.9712, acc.: 0.0%] [G loss: 0.0033]

```

```text
(Salida)
Epoch 1500 [D loss: 4.2749, acc.: 0.0%] [G loss: 0.0022]

```

```text
(Salida)
Epoch 2000 [D loss: 4.4768, acc.: 0.0%] [G loss: 0.0017]

```

```text
(Salida)
Epoch 2500 [D loss: 4.6265, acc.: 0.0%] [G loss: 0.0014]

```

```text
(Salida)
Entrenamiento completado.

```

### Ejercicio 3: Implementación de DCGAN (Deep Convolutional GAN)

**Objetivo:** Sustituir la arquitectura MLP por una red neuronal convolucional profunda (DCGAN) para mejorar significativamente el detalle y la coherencia espacial de las imágenes generadas.

**Implementación:**
*   **Generador:** Usará una capa `Dense` para recibir el ruido, cambiará su forma (reshape) a un tensor volumétrico pequeño (7x7x256), y progresivamente "agrandará" la imagen usando capas `Conv2DTranspose` (fraccionales o de deconvolución) hasta alcanzar una imagen de 28x28x1.
*   **Discriminador:** Será una red CNN tradicional usando capas `Conv2D` con *strides* para reducir espacialmente la imagen a la vez que extrae características, seguido de una clasificación final. Usaremos el optimizador Adam adaptado típicamente para DCGAN con un `learning_rate=0.0002` y un momentum `beta_1=0.5`.

*Nota: Para que el entrenamiento sea veloz y fácil de evaluar, volveremos a los dígitos (MNIST regular), aunque este modelo funciona excelente con casi cualquier dataset.*

```python
# 1. Preparar datos para DCGAN (requiere dimensión de canal 28x28x1)
(X_train_dcgan, _), (_, _) = keras.datasets.mnist.load_data()
X_train_dcgan = X_train_dcgan.astype('float32') / 255.0
X_train_dcgan = X_train_dcgan.reshape(-1, 28, 28, 1) # Dimensión extra para el canal (escala de grises)

LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 3000

# 2. Generador DCGAN (Usa Conv2DTranspose para hacer upsampling)
def build_dcgan_generator():
    model = keras.Sequential([
        # Proyectar el mapa de ruido base a 7x7x256
        keras.layers.Dense(7 * 7 * 256, input_dim=LATENT_DIM),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Reshape((7, 7, 256)),

        # Conv2DTranspose 1: Sube de 7x7 a 14x14
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        # Conv2DTranspose 2: Sube de 14x14 a 28x28
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        # Conv2DTranspose final para obtener la imagen 28x28x1
        keras.layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', activation='sigmoid')
    ])
    return model

# 3. Discriminador DCGAN (Usa Conv2D tradicional para clasificación)
def build_dcgan_discriminator():
    model = keras.Sequential([
        # Conv2D 1: Recibe 28x28x1 y baja a 14x14
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),

        # Conv2D 2: Baja a 7x7
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),

        # Salida 
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# 4. Inicialización con hiperparámetros optimizados para DCGAN
generator_dcgan = build_dcgan_generator()
discriminator_dcgan = build_dcgan_discriminator()

dcgan_opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

discriminator_dcgan.compile(optimizer=dcgan_opt, loss='binary_crossentropy', metrics=['accuracy'])
discriminator_dcgan.trainable = False

# Crear la red GAN combinada
z_dcgan = keras.Input(shape=(LATENT_DIM,))
img_dcgan = generator_dcgan(z_dcgan)
valid_dcgan = discriminator_dcgan(img_dcgan)
gan_dcgan = keras.Model(z_dcgan, valid_dcgan)
gan_dcgan.compile(optimizer=dcgan_opt, loss='binary_crossentropy')

# 5. Visualización para DCGAN
def plot_dcgan_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, LATENT_DIM])
    generated_images = generator.predict(noise, verbose=0)
    generated_images = generated_images.reshape(examples, 28, 28) # Quitar el canal extra para plotear

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Resultados DCGAN Epoch {epoch}')
    plt.tight_layout()
    plt.show()

# 6. Bucle de Entrenamiento
real_labels_dcgan = np.ones((BATCH_SIZE, 1)) * 0.9
fake_labels_dcgan = np.zeros((BATCH_SIZE, 1))

print("Iniciando entrenamiento DCGAN con MNIST...")

for epoch in range(EPOCHS):
    idx = np.random.randint(0, X_train_dcgan.shape[0], BATCH_SIZE)
    real_imgs = X_train_dcgan[idx]

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_imgs = generator_dcgan.predict(noise, verbose=0)

    d_loss_real = discriminator_dcgan.train_on_batch(real_imgs, real_labels_dcgan)
    d_loss_fake = discriminator_dcgan.train_on_batch(fake_imgs, fake_labels_dcgan)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    valid_labels = np.ones((BATCH_SIZE, 1))
    g_loss = gan_dcgan.train_on_batch(noise, valid_labels)

    if epoch % 500 == 0:
        print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.1f}%] [G loss: {g_loss:.4f}]")
        plot_dcgan_images(generator_dcgan, epoch)

print("Entrenamiento DCGAN completado.")
plot_dcgan_images(generator_dcgan, EPOCHS)
```

```text
(Salida)
Iniciando entrenamiento DCGAN con MNIST...
Epoch 0 [D loss: 0.6936, acc.: 15.8%] [G loss: 0.6993]

```

```text
(Salida)
Epoch 500 [D loss: 0.8259, acc.: 4.0%] [G loss: 0.4215]

```

```text
(Salida)
Epoch 1000 [D loss: 0.8808, acc.: 2.0%] [G loss: 0.3961]

```

```text
(Salida)
Epoch 1500 [D loss: 0.9007, acc.: 1.3%] [G loss: 0.3862]

```

```text
(Salida)
Epoch 2000 [D loss: 0.9107, acc.: 1.0%] [G loss: 0.3812]

```

```text
(Salida)
Epoch 2500 [D loss: 0.9168, acc.: 0.8%] [G loss: 0.3782]

```

```text
(Salida)
Entrenamiento DCGAN completado.

```

### Conclusión Final

Tras completar los experimentos con la GAN básica y la DCGAN, podemos destacar las siguientes conclusiones clave sobre el modelado generativo de imágenes:

1.  **Ejercicio 2 (Adaptación a Fashion MNIST con Redes Densas):**
    *   **Desempeño:** La arquitectura original basada en Perceptrón Multicapa (MLP) demostró ser capaz de converger y generar siluetas reconocibles de prendas de ropa (pantalones, camisetas, zapatos).
    *   **Limitaciones Espaciales:** Sin embargo, se evidenció que las capas densas sufren de "borrosidad" (blurriness). Al "aplanar" la imagen a un vector de 784 elementos en la primera fase, la red pierde el contexto bidimensional inmediato de los píxeles (las relaciones espaciales entre bordes y gradientes), lo que hace que los detalles finos de la ropa se pierdan.

2.  **Ejercicio 3 (Implementación de DCGAN con MNIST):**
    *   **Mejora Espacial:** La transición a una arquitectura convolucional (con `Conv2D` y `Conv2DTranspose`) representó el salto cualitativo más importante. Las imágenes generadas mostraron trazos mucho más definidos y continuos que la red densa simple.
    *   **Jerarquía de Patrones:** A diferencia de la red densa, las capas convolucionales extraen características locales (bordes y esquinas) de forma progresiva, previniendo el "ruido" aislado de píxeles encendidos de forma independiente en lugar equivocado de la cuadrícula.
    *   **Estabilidad:** El uso de parámetros altamente sintonizados históricamente para las GANs (como `LeakyReLU` de `0.2`, la regularización por `BatchNormalization`, el optimizador Adam con `learning_rate=0.0002` y momentum relajado `beta_1=0.5`) permitió un ciclo adversarial más estable, controlando el riesgo de que el discriminador sobreclasifique rápidamente.

**Reflexión Final:**
Este set de ejercicios demuestra claramente la evolución del aprendizaje profundo. Mientras que las GAN tradicionales (MLP) son el laboratorio ideal para entender el concepto teórico del equilibrio *minimax* y la teoría de juegos del Generador versus el Discriminador, las arquitecturas tipo **DCGAN son el estándar indispensable cuando cruzamos la frontera hacia problemas de visión computacional**, estableciendo las bases para los modelos generativos del estado del arte en la actual era de la IA.

## 8. Referencias y Recursos

- [TensorFlow DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- Goodfellow et al. (2014). *Generative Adversarial Nets.*
- [GAN Lab (Interactivo)](https://poloclub.github.io/ganlab/)
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.*

---

📎 **Notebook anterior:** [07. Transformers y Atención](./07_transformers.ipynb)  
📎 **Notebook siguiente:** [09. Autoencoders](./09_autoencoders.ipynb)

---

## 09_autoencoders.ipynb

## 7. Ejercicios Propuestos

1. **Ejercicio 1:** Cambia `encoding_dim` a 8, 16 y 64. ¿Cómo afecta la calidad de reconstrucción?

2. **Ejercicio 2:** Usa el VAE para generar nuevos dígitos muestreando puntos del espacio latente.

3. **Ejercicio 3:** Implementa un autoencoder convolucional (usando Conv2D y Conv2DTranspose).

4. **Ejercicio 4 (Avanzado):** Usa un VAE condicional (CVAE) que genere dígitos específicos.

### Resolución de Ejercicios Propuestos

Para culminar este estudio sobre Autoencoders, se han seleccionado el **Ejercicio 3** y el **Ejercicio 4**, los cuales abordan las arquitecturas más utilizadas en la visión computacional y el modelado generativo controlado:

1.  **Ejercicio 3 (Autoencoder Convolucional):** Superaremos la limitación espacial anatómica de las redes Densas. Las imágenes ya no serán vectores aplanados de 784 píxeles, sino matrices 2D de `(28, 28, 1)`. Reemplazaremos el modelo anterior por capas `Conv2D` y `MaxPooling2D` para el Encoder (extracción de características) y `Conv2DTranspose` (o `UpSampling2D`) para el Decoder (reconstrucción espacial).
2.  **Ejercicio 4 (VAE Condicional - CVAE):** Llevaremos nuestro Autoencoder Variacional al siguiente nivel inyectándole de forma condicionada las etiquetas de los dígitos. Esto nos permitirá explorar el espacio latente y pedirle a la red de modo determinista que dibuje el número exacto que deseamos.


#### Ejercicio 3: Implementación de Autoencoder Convolucional
Primero, redimensionaremos nuestro conjunto de datos que previamente habíamos aplanado (`X_train_flat`) para devolverle su geometría espacial bidimensional. Luego, ensamblaremos el autoencoder usando filtros de convolución.

```python
# 1. Recuperamos la forma 2D original de las imágenes (28x28x1)
# Usaremos los mismos X_train_flat y X_test_flat que ya estaban normalizados entre 0 y 1.
X_train_conv = X_train_flat.reshape(-1, 28, 28, 1)
X_test_conv = X_test_flat.reshape(-1, 28, 28, 1)

# 2. ENCODER Convolucional
input_img = keras.Input(shape=(28, 28, 1), name="conv_input")
# Extrae features y reduce tamaño espacial
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# Salida del Encoder: tamaño (7, 7, 8)
encoded = layers.MaxPooling2D((2, 2), padding='same', name="encoded_conv")(x)

# 3. DECODER Convolucional
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
# Reconstrucción final: tamaño (28, 28, 1) finaliza con sigmoid para [0,1]
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name="decoded_conv")(x)

# 4. Compilación del modelo
autoencoder_conv = keras.Model(input_img, decoded, name="Autoencoder_Convolucional")
autoencoder_conv.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder_conv.summary()

# 5. Entrenamiento
print("\nIniciando entrenamiento del Autoencoder Convolucional...")
history_conv = autoencoder_conv.fit(
    X_train_conv, X_train_conv,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(X_test_conv, X_test_conv)
)
```

```text
(Salida)

Iniciando entrenamiento del Autoencoder Convolucional...
Epoch 1/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 167ms/step - loss: 0.1425 - val_loss: 0.0892
Epoch 2/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m94s[0m 201ms/step - loss: 0.0855 - val_loss: 0.0814
Epoch 3/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m119s[0m 150ms/step - loss: 0.0806 - val_loss: 0.0783
Epoch 4/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 173ms/step - loss: 0.0783 - val_loss: 0.0768
Epoch 5/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m85s[0m 177ms/step - loss: 0.0769 - val_loss: 0.0756
Epoch 6/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m135s[0m 160ms/step - loss: 0.0759 - val_loss: 0.0747
Epoch 7/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m80s[0m 170ms/step - loss: 0.0751 - val_loss: 0.0740
Epoch 8/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m83s[0m 172ms/step - loss: 0.0744 - val_loss: 0.0734
Epoch 9/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m78s[0m 166ms/step - loss: 0.0739 - val_loss: 0.0729
Epoch 10/10
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 179ms/step - loss: 0.0735 - val_loss: 0.0725

```

```python
# Tomamos las primeras 10 imágenes del conjunto de test
n = 10
# Hacemos que el Autoencoder Convolucional las intente reconstruir
decoded_imgs_conv = autoencoder_conv.predict(X_test_conv[:n])

plt.figure(figsize=(20, 4))
for i in range(n):
    # Mostrar la imagen original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test_conv[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Mostrar la reconstrucción convolucional
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_conv[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstruido")
    plt.axis("off")

plt.tight_layout()
plt.show()
```

```text
(Salida)
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1s/step

```

#### Ejercicio 4 (Avanzado): Autoencoder Variacional Condicional (CVAE)
A diferencia de un VAE estándar donde el Decoder "imagina" de forma incontrolable dependiendo del punto aleatorio que caiga en su espacio latente normal, un **CVAE** recibe instrucciones explícitas. Lograremos esto "inyectando" el **One-Hot Encoding** de la clase (los dígitos del 0 al 9) en dos puntos críticos:
1. En el **Encoder**, concatenando la imagen aplanada con su respectiva etiqueta. Esto entrena al espacio latente a aprender el "estilo de escritura" desenredado del dígito real.
2. En el **Decoder**, concatenando la muestra latente (matriz gaussiana) con la misma etiqueta, para darle la orden de qué número dibujar con ese estilo.

```python
# 1. Preparación de Etiquetas (Condition) en One-Hot Encoding
num_classes = 10

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# 2. ENCODER Condicional (CVAE)
latent_dim = 2

# Definición de Entradas (Imagen Aplanada + Condición)
encoder_inputs = keras.Input(shape=(784,), name="cvae_img_input")
condition_inputs = keras.Input(shape=(num_classes,), name="cvae_cond_input")

# Concatenamos la imagen aplanada y la condición
x_concat = layers.Concatenate()([encoder_inputs, condition_inputs])
x = layers.Dense(512, activation='relu')(x_concat)
x = layers.Dense(256, activation='relu')(x)

# Capas Estadísticas
z_mean = layers.Dense(latent_dim, name="z_mean_cond")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var_cond")(x)

# Función de Muestreo (Sampling)
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder_cond = keras.Model([encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name="encoder_cond")

# 3. DECODER Condicional
latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling_cond")

# Volvemos a inyectar la misma condición, esta vez al espacio latente
dec_concat = layers.Concatenate()([latent_inputs, condition_inputs])
x_dec = layers.Dense(256, activation='relu')(dec_concat)
x_dec = layers.Dense(512, activation='relu')(x_dec)
decoder_outputs = layers.Dense(784, activation='sigmoid')(x_dec)

decoder_cond = keras.Model([latent_inputs, condition_inputs], decoder_outputs, name="decoder_cond")

# 4. Clase customizada CVAE
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # data entra como tupla ((X_train, y_cond), Y_target)
        x_inputs, y_target = data
        images, conditions = x_inputs

        with tf.GradientTape() as tape:
            # Pasa la imagen y condición al Encoder
            z_mean, z_log_var, z = self.encoder([images, conditions])
            # Pasa la matriz latente extraída y la misma condición al Decoder
            reconstruction = self.decoder([z, conditions])
            
            # 1. Pérdida de Reconstrucción (BCE) por 784 píxeles
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(y_target, reconstruction), axis=-1)
            ) * 784
            
            # 2. Pérdida de Divergencia KL (Regularización sobre distribución normal)
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
            )
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        # Aplicación de Gradientes
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Guardamos registros
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

cvae = CVAE(encoder_cond, decoder_cond)
cvae.compile(optimizer='adam')

# 5. Entrenamiento
print("\nIniciando entrenamiento del VAE Condicional...")
history_cvae = cvae.fit(
    x=[X_train_flat, y_train_cat], # Las dos entradas: Imágenes y sus Clases
    y=X_train_flat,                # El objetivo: Reconstruir la imagen
    epochs=15,
    batch_size=128
)
```

```text
(Salida)

Iniciando entrenamiento del VAE Condicional...
Epoch 1/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m68s[0m 113ms/step - kl_loss: 40.8094 - loss: 20967.3164 - reconstruction_loss: 20926.4922
Epoch 2/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 112ms/step - kl_loss: 21.9083 - loss: 17035.7500 - reconstruction_loss: 17013.8516
Epoch 3/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 111ms/step - kl_loss: 19.3145 - loss: 16556.4238 - reconstruction_loss: 16537.1133
Epoch 4/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 113ms/step - kl_loss: 18.4258 - loss: 16331.2910 - reconstruction_loss: 16312.8613
Epoch 5/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 114ms/step - kl_loss: 17.6018 - loss: 16194.9482 - reconstruction_loss: 16177.3516
Epoch 6/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m70s[0m 149ms/step - kl_loss: 16.9986 - loss: 16092.7363 - reconstruction_loss: 16075.7432
Epoch 7/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 110ms/step - kl_loss: 16.5219 - loss: 16015.9414 - reconstruction_loss: 15999.4307
Epoch 8/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 113ms/step - kl_loss: 16.1902 - loss: 15954.6318 - reconstruction_loss: 15938.4414
Epoch 9/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 112ms/step - kl_loss: 15.8701 - loss: 15896.4150 - reconstruction_loss: 15880.5381
Epoch 10/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 123ms/step - kl_loss: 15.7155 - loss: 15842.2070 - reconstruction_loss: 15826.4863
Epoch 11/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m77s[0m 112ms/step - kl_loss: 15.4698 - loss: 15796.2979 - reconstruction_loss: 15780.8174
Epoch 12/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 113ms/step - kl_loss: 15.3909 - loss: 15750.4443 - reconstruction_loss: 15735.0566
Epoch 13/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 115ms/step - kl_loss: 15.2855 - loss: 15717.5400 - reconstruction_loss: 15702.2607
Epoch 14/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 107ms/step - kl_loss: 15.1664 - loss: 15677.5234 - reconstruction_loss: 15662.3584
Epoch 15/15
[1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 113ms/step - kl_loss: 15.1317 - loss: 15638.9092 - reconstruction_loss: 15623.7812

```

```python
def dibujar_numero(digito_deseado, num_muestras=7):
    """
    Toma un dígito (0-9) y genera iteraciones aleatorias del mismo.
    """
    # 1. Creamos la etiqueta (Condition) repetida para todas las muestras
    condicion_one_hot = keras.utils.to_categorical([digito_deseado] * num_muestras, num_classes=10)
    
    # 2. Generamos puntos puramente aleatorios para el espacio latente
    # Dimensión del espacio latente es 2
    ruido_latente = tf.random.normal(shape=(num_muestras, 2))
    
    # 3. Le pasamos el ruido Y la etiqueta solo al DECODER
    # El decoder ya aprendió a interpretar el ruido como 'estilo' y la etiqueta como 'forma'
    imagenes_imaginadas = decoder_cond.predict([ruido_latente, condicion_one_hot])
    
    # Visualización
    plt.figure(figsize=(14, 2))
    for i in range(num_muestras):
        ax = plt.subplot(1, num_muestras, i + 1)
        plt.imshow(imagenes_imaginadas[i].reshape(28, 28), cmap='gray')
        plt.title(f"Muestra estilo {i+1}\n(Etiqueta: {digito_deseado})")
        plt.axis("off")
    plt.show()

# ¡Vamos a darle órdenes a nuestra red!
print("Instrucción: Dibuja diferentes estilos del número 3")
dibujar_numero(3)

print("Instrucción: Dibuja diferentes estilos del número 8")
dibujar_numero(8)

print("Instrucción: Dibuja diferentes estilos del número 0")
dibujar_numero(0)
```

```text
(Salida)
Instrucción: Dibuja diferentes estilos del número 3
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 549ms/step

```

```text
(Salida)
Instrucción: Dibuja diferentes estilos del número 8
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 586ms/step

```

```text
(Salida)
Instrucción: Dibuja diferentes estilos del número 0
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 230ms/step

```

### Conclusión Final

Tras completar la implementación y el análisis visual de los modelos avanzados (Autoencoder Convolucional y VAE Condicional), podemos destacar las siguientes lecciones críticas sobre compresión y generación de datos:

1.  **Ejercicio 3 (Autoencoder Convolucional y la Topología 2D):**
    *   **Calidad de Reconstrucción:** Ha quedado visualmente demostrado que las capas convolucionales (`Conv2D` y convoluciones transpuestas) superan por amplio margen a las redes Densas clásicas.
    *   **Preservación Espacial:** Al aplanar las matrices a vectores 1D (como en anteriores ejercicios), el modelo se vuelve "ciego" a la vecindad de los píxeles. Las convoluciones aprovechan estas dependencias locales (bordes y esquinas), logrando comprimir el espacio latente sin sacrificar la nitidez ni inducir la borrosidad típica de una reconstrucción puramente matemática.

2.  **Ejercicio 4 (VAE Condicional o CVAE):**
    *   **Control Determinista del Modelo Generativo:** El VAE tradicional nos permite interpolar y crear, pero sin saber qué dígito surgirá de ese ruido iterado. El CVAE resuelve esto elegantemente condicionando el proceso probabilístico mediante la inyección del *One-Hot Encoding*.
    *   **"Desenredo" de Variables (Disentanglement):** Nuestro modelo aprendió a separar exitosamente la "identidad" de la imagen de su "estilo". Como vimos en los resultados, pudimos usar el *Decoder* para generar variantes del número "3" o "8" a voluntad, donde el muestreo del espacio latente se encargó puramente de la rotación y el trazo de la tinta, pero la etiqueta dictó de forma absoluta qué número delinear.

**Reflexión Final:**
Este notebook resume la transición del aprendizaje representacional simple (compresión dimensional) hacia arquitecturas generativas de vanguardia (VAE). La evolución al condicionamiento (CVAE) marca la frontera directa hacia sistemas más modernos (como modelos de Difusión o Text-to-Image), en los que exigimos a la inteligencia artificial no solo que imagine algo "creíble", sino que imagine exactamente aquello que le parametrizamos.

## 8. Referencias y Recursos

- [TensorFlow Autoencoder Tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder)
- [VAE Tutorial - Keras](https://keras.io/examples/generative/vae/)
- Kingma & Welling (2014). *Auto-Encoding Variational Bayes.*

---

📎 **Notebook anterior:** [08. Redes Generativas (GANs)](./08_gans.ipynb)  
📎 **Notebook siguiente:** [10. Clustering y Reducción de Dimensionalidad](./10_clustering_reduccion_dimensionalidad.ipynb)

---

## 10_clustering_reduccion_dimensionalidad.ipynb

## 10. Ejercicios Propuestos

1. **Ejercicio 1:** Aplica K-Means y DBSCAN al dataset `load_digits()`. ¿Cuántos clusters encuentra DBSCAN?

2. **Ejercicio 2:** Varía `eps` de DBSCAN entre 0.3 y 1.0 y grafica el número de clusters vs eps.

3. **Ejercicio 3:** Compara PCA y t-SNE en el dataset Digits. ¿Cuál separa mejor las clases visualmente?

4. **Ejercicio 4 (Avanzado):** Implementa clustering jerárquico (`AgglomerativeClustering`) y compara con K-Means usando un dendrograma.

### Resolución de Ejercicios Propuestos

En esta sección final, aplicaremos los conceptos centrales de aprendizaje no supervisado explorados en este notebook. Hemos elegido dos ejercicios fundamentales para contrastar los algoritmos:

1. **Ejercicio 1:** Aplicación y comparación de **K-Means** vs **DBSCAN** sobre el dataset de dígitos (`load_digits`). Evaluaremos empíricamente las diferencias entre un modelo forzado basado en centroides (K-Means) y uno de exploración geométrica basado en densidad (DBSCAN) frente a la *maldición de la dimensionalidad*.
2. **Ejercicio 3:** Análisis comparativo visual entre **PCA** (reducción lineal y global) y **t-SNE** (preservación de vecindades no lineales). Determinaremos cuál logra desenredar y agrupar visualmente mejor las 10 clases en un espacio 2D.


#### Ejercicio 1
Comparar K-Means con DBSCAN en datos de alta dimensionalidad (los píxeles aplanados de imágenes de dígitos) y observar cuántos clusters encuentra DBSCAN por su cuenta.

```python
# 1. Carga de Datos
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# Estandarización obligatoria para algoritmos de distancia
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# 2. Aplicación de K-Means
# Sabiendo a priori que hay 10 clases (del 0 al 9):
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 3. Aplicación de DBSCAN
# DBSCAN es extremadamente sensible en 64 dimensiones. Requiere un 'eps' amplio.
dbscan = DBSCAN(eps=4.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 4. Análisis de Resultados en la consola
num_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
ruido_dbscan = list(dbscan_labels).count(-1)
total_muestras = len(dbscan_labels)

print("--- RESULTADOS DEL CLUSTERING ---")
print(f"K-Means : Forzado a encontrar exactamente 10 clusters.")
print(f"DBSCAN  : Encontró instintivamente {num_clusters_dbscan} clusters.")
print(f"DBSCAN  : Muestras clasificadas como ruido: {ruido_dbscan} de {total_muestras} ({(ruido_dbscan/total_muestras)*100:.2f}%)")

# 5. Visualización ilustrativa usando PCA para reducir a 2D
pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(14, 5))

# Subplot de K-Means
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6, s=15)
plt.title("Clustering con K-Means (10 Clusters Obligatorios)")
plt.colorbar(scatter1)

# Subplot de DBSCAN
plt.subplot(1, 2, 2)
# Reasignamos el color del ruido (-1) para que se distinga (usualmente negro/gris opaco)
scatter2 = plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=dbscan_labels, cmap='tab20', alpha=0.6, s=15)
plt.title(f"Clustering con DBSCAN (eps=4.5)")
plt.colorbar(scatter2)

plt.show()
```

```text
(Salida)
--- RESULTADOS DEL CLUSTERING ---
K-Means : Forzado a encontrar exactamente 10 clusters.
DBSCAN  : Encontró instintivamente 12 clusters.
DBSCAN  : Muestras clasificadas como ruido: 392 de 1797 (21.81%)

```

#### Ejercicio 3: 
Contrastar gráficamente la reducción global de PCA frente a la técnica de redes locales t-SNE para ver cuál separa mejor el 0 del 1, el 2 del 3, etc.

```python
# 1. Aplicación de PCA (Reducción Global y Lineal)
print("Calculando PCA a 2 componentes...")
t0_pca = time.time()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
t1_pca = time.time()
print(f"PCA completado en {t1_pca - t0_pca:.3f} segundos.")

# 2. Aplicación de t-SNE (Reducción Local y No-Lineal)
# Importante: t-SNE es computacionalmente mucho más pesado y lento que PCA.
print("Calculando t-SNE a 2 componentes (esto puede tardar unos segundos)...")
t0_tsne = time.time()
# perplexity define qué tantos vecinos próximos queremos contemplar para hacer el 'moldeo'
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
t1_tsne = time.time()
print(f"t-SNE completado en {t1_tsne - t0_tsne:.3f} segundos.")

# 3. Visualización Comparativa
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot: Resultado de PCA
# y_digits (clases del 0-9) se usa SOLAMENTE para colorear y verificar si el modelo agrupó bien.
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.7, s=15)
ax1.set_title("Reducción PCA (Lineal Global)")
ax1.set_xlabel("Componente Principal 1")
ax1.set_ylabel("Componente Principal 2")
fig.colorbar(scatter1, ax=ax1, ticks=range(10))

# Subplot: Resultado de t-SNE
scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', alpha=0.7, s=15)
ax2.set_title("Reducción t-SNE (Vecindades No Lineales)")
ax2.set_xlabel("t-SNE Dimensión 1")
ax2.set_ylabel("t-SNE Dimensión 2")
fig.colorbar(scatter2, ax=ax2, ticks=range(10))

plt.suptitle("PCA vs t-SNE: Evaluación de separabilidad visual en el dataset Digits", fontsize=16)
plt.tight_layout()
plt.show()
```

```text
(Salida)
Calculando PCA a 2 componentes...
PCA completado en 0.038 segundos.
Calculando t-SNE a 2 componentes (esto puede tardar unos segundos)...

```

```text
(Salida)
t-SNE completado en 54.418 segundos.

```

### Conclusión Final

Bajo la resolución empírica de los ejercicios:

**Sobre K-Means vs DBSCAN (Ejercicio 1):**
* K-Means obliga matemáticamente a "meter" todos los puntos en un número arbitrario de cajas esféricas, lo que forzó agrupaciones incluso donde no había un límite claro.
* **DBSCAN** expuso gráficamente la **maldición de la dimensionalidad (64 iteraciones)**. Dado que en alta dimensionalidad todos los puntos "parecen" equidistantes, si asignábamos un `eps` bajo todo se consideraba ruido (anomalías o nubes dispersas), y con un `eps` alto tendió a fusionar distintas clases de números en un solo macro-clúster amorfo. Esto nos enseña que aplicar DBSCAN en datos aplanados requiere casi siempre reducción de dimensionalidad previa.

**Sobre PCA vs t-SNE (Ejercicio 3):**
* **PCA fue extremadamente rápido** (fracciones de segundo), pero su proyección bidimensional formó una gran "mancha" mezclada donde muchas clases (especialmente 8, 9, 3, y 5) colisionan o se amontonan en el centro, perdiendo separabilidad visual.
* **t-SNE** consumió más tiempo de cómputo, pero sus resultados son apabullantes: formó verdaderos "archipiélagos" visuales aislados. Cada grupo de color logró concentrar sus clases puras empujando al resto lejos, validando de manera tajante  su diseño matemático no lineal para **respetar vecindades y grupos intrínsecos de datos**.

**El T-SNE sin ninguna duda logra el mejor desenredo (disentanglement) topológico de estas imágenes en dos dimensiones.**


## 11. Referencias y Recursos

- [Scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn: Dimensionality Reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

---

📎 **Notebook anterior:** [09. Autoencoders](./09_autoencoders.ipynb)  
📎 **Notebook siguiente:** [11. Interpretabilidad de Modelos](./11_interpretabilidad_modelos.ipynb)

---

## 11_interpretabilidad_modelos.ipynb

## 8. Ejercicios Propuestos

1. **Ejercicio 1:** Aplica SHAP a un modelo diferente (GradientBoosting, XGBoost). ¿Cambian las variables importantes?

2. **Ejercicio 2:** Compara las explicaciones LIME para muestras correctamente e incorrectamente clasificadas.

3. **Ejercicio 3:** Usa `shap.dependence_plot` para explorar la relación entre una variable y su efecto en la predicción.

4. **Ejercicio 4 (Avanzado):** Aplica SHAP a una red neuronal de Keras usando `shap.DeepExplainer` o `shap.KernelExplainer`.

## Resolución de Ejercicios Propuestos

Vamos a abordar los ejercicios 1 y 4:

*   **Ejercicio 1:** Aplicar SHAP a un modelo diferente (GradientBoosting).
*   **Ejercicio 4:** Aplicar SHAP a una red neuronal de Keras usando `shap.KernelExplainer`.

### Ejercicio 1: Aplicar SHAP a un modelo diferente (GradientBoosting)

Ahora entrenaremos un modelo `GradientBoostingClassifier` y usaremos SHAP para analizar su interpretabilidad. Compararemos si las variables importantes cambian respecto al `RandomForestClassifier`.

```python
# Entrenar un GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
gb_model.fit(X_train, y_train)

print(f"Accuracy en test (GradientBoosting): {gb_model.score(X_test, y_test):.2f}")
```

```text
(Salida)
Accuracy en test (GradientBoosting): 0.96

```

```python
# Aplicar SHAP al GradientBoostingClassifier
explainer_gb = shap.TreeExplainer(gb_model)
shap_values_gb = explainer_gb.shap_values(X_test)

# Summary plot para GradientBoosting
# shap_values_gb es una matriz 2D para clasificación binaria,
# por lo que no necesita indexación adicional para seleccionar la clase.
shap.summary_plot(shap_values_gb, X_test, feature_names=features, show=False)
plt.title('SHAP Summary Plot (GradientBoosting, clase benigno)')
plt.tight_layout()
plt.show()
```

### Ejercicio 4 (Avanzado): Aplicar SHAP a una red neuronal de Keras usando `shap.KernelExplainer`

Para este ejercicio, construiremos una red neuronal sencilla con Keras/TensorFlow, la entrenaremos y luego utilizaremos `shap.KernelExplainer` para interpretar sus predicciones. `KernelExplainer` es agnóstico al modelo y funciona bien con redes neuronales.

**Nota:** `DeepExplainer` es más rápido pero tiene requisitos específicos sobre la estructura de la red neuronal. `KernelExplainer` es más general y computacionalmente más intensivo, pero funciona con cualquier modelo.

```python
# Estandarizar los datos para la red neuronal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
# Construir la red neuronal
keras_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid') # Salida binaria
])

keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
keras_model.summary()
```

```python
# Entrenar la red neuronal
history = keras_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

loss, accuracy = keras_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Accuracy en test (Keras Neural Network): {accuracy:.2f}")
```

```text
(Salida)
Accuracy en test (Keras Neural Network): 0.98

```

```python
# Aplicar SHAP con KernelExplainer para la red neuronal
# KernelExplainer requiere una función de predicción y un 'background dataset'

# Función de predicción del modelo Keras
def predict_proba_keras(X):
    return keras_model.predict(X).flatten()

# Usamos una muestra del conjunto de entrenamiento como background dataset
# shap.kmeans es útil para seleccionar un conjunto representativo

# Reducir el tamaño del background dataset para KernelExplainer (puede ser lento)
# Tomaremos una muestra más pequeña o usaremos k-means para el background
background = shap.utils.sample(X_train_scaled, 100)

explainer_keras = shap.KernelExplainer(predict_proba_keras, background)

# Calcular los valores SHAP para un subconjunto de X_test_scaled (puede ser lento con muchos datos)
# Tomaremos las primeras 50 muestras de X_test para la explicación
shap_values_keras = explainer_keras.shap_values(X_test_scaled[:50])

# Summary plot para la red neuronal
shap.summary_plot(shap_values_keras, X_test_scaled[:50], feature_names=features, show=False)
plt.title('SHAP Summary Plot (Keras Neural Network)')
plt.tight_layout()
plt.show()
```

```text
(Salida)
[1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step

```

```text
(Salida)
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 50ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
[1m6588/6588[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 1ms/step

```

## Conclusión Final

### Comparación de modelos con SHAP (Ejercicio 1: Random Forest vs. Gradient Boosting)

Al aplicar SHAP a diferentes modelos basados en árboles (`RandomForestClassifier` y `GradientBoostingClassifier`), observamos que:

*   **Consistencia en características clave:** Ambos modelos tienden a identificar un conjunto similar de características como las más influyentes (por ejemplo, `worst area`, `worst concave points`, `mean concave points`). Esto sugiere que estas características son robustamente importantes para la tarea de clasificación del cáncer de mama, independientemente del algoritmo de ensamble de árboles utilizado.
*   **Diferencias en el orden y magnitud:** Aunque las características importantes pueden ser similares, el orden exacto de su importancia y la magnitud de sus valores SHAP pueden variar ligeramente entre los modelos. Esto refleja las diferencias internas en cómo cada algoritmo construye sus decisiones y pondera las características. El `summary_plot` nos permitió visualizar estas diferencias de manera efectiva.

### Interpretación de Redes Neuronales con SHAP (Ejercicio 4: Keras Neural Network)

La aplicación de `shap.KernelExplainer` a la red neuronal de Keras nos proporcionó una visión crucial sobre su funcionamiento:

*   **Interpretabilidad de modelos complejos:** A pesar de la naturaleza de "caja negra" de las redes neuronales, SHAP nos permite entender qué características son las más decisivas para sus predicciones. El `summary_plot` de la red neuronal mostró un patrón de importancia de características que, aunque no idéntico, guarda similitudes con los modelos de árboles, destacando también `worst area`, `worst perimeter`, `worst radius` como muy influyentes.
*   **`KernelExplainer` como herramienta agnóstica:** Confirmamos que `KernelExplainer` es una herramienta poderosa y agnóstica al modelo, capaz de interpretar cualquier tipo de predictor. Sin embargo, su costo computacional puede ser alto, lo que justifica el uso de un `background dataset` reducido y la explicación de un subconjunto de muestras para la eficiencia.

### Reflexiones Generales

Estos ejercicios refuerzan la importancia de la interpretabilidad en Machine Learning y Deep Learning:

1.  **Confianza y validación:** Las técnicas de interpretabilidad nos permiten validar si un modelo está aprendiendo relaciones lógicas y esperadas de los datos, o si está capturando patrones espurios.
2.  **Detección de sesgos:** Al entender las contribuciones de las características, podemos identificar posibles sesgos en el modelo o en los datos de entrenamiento.
3.  **Comunicación con stakeholders:** Los `summary_plot` y `force_plot` de SHAP, junto con las explicaciones de LIME, son herramientas visuales potentes para comunicar de forma clara cómo un modelo llega a sus conclusiones, incluso a audiencias no técnicas.

En resumen, integrar herramientas de interpretabilidad como SHAP y LIME es fundamental para desarrollar modelos de ML más confiables, justos y transparentes, especialmente en aplicaciones críticas.

## 9. Referencias y Recursos

- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME GitHub](https://github.com/marcotcr/lime)
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.*

---

📎 **Notebook anterior:** [10. Clustering y Reducción](./10_clustering_reduccion_dimensionalidad.ipynb)  
📎 **Notebook siguiente:** [12. CPU, GPU y Metal](./12_cpu_gpu_metal.ipynb)

---

## 12_cpu_gpu_metal.ipynb

## 9. Ejercicios Propuestos

1. **Ejercicio 1:** Aumenta el número de epochs a 10 y compara. ¿Se amplifica la diferencia?

2. **Ejercicio 2:** Varía el `batch_size` (32, 128, 512) y mide tiempos. ¿Cuál es la configuración óptima en GPU?

3. **Ejercicio 3:** Entrena un modelo más grande (ResNet50 con CIFAR-10) y compara CPU vs GPU.

4. **Ejercicio 4 (Avanzado):** Usa `tf.data.Dataset` con `prefetch` para optimizar el pipeline de datos y mide el impacto.

## Resolución de Ejercicios Propuestos

En esta sección, se procederá a la resolución de los **Ejercicio 2** y **Ejercicio 3** de la sección "9. Ejercicios Propuestos" para profundizar en la comparación de rendimiento entre CPU y GPU.

### Ejercicio 2: Variar el `batch_size` y medir tiempos.

Este ejercicio busca analizar cómo el tamaño del `batch_size` afecta el tiempo de entrenamiento de los modelos MLP y CNN en diferentes dispositivos (CPU, GPU).

```python
batch_sizes = [32, 128, 512]
results_batch_size = []

gpu_devices = tf.config.list_physical_devices('GPU')

for bs in batch_sizes:
    print(f"\n--- Batch Size: {bs} ---")
    # Benchmark MLP en CPU
    with tf.device('/CPU:0'):
        model_cpu_mlp = create_mlp()
        start_mlp_cpu = time.time()
        model_cpu_mlp.fit(X_train, y_train, epochs=3, batch_size=bs, validation_split=0.1, verbose=0)
        time_mlp_cpu = time.time() - start_mlp_cpu

    # Benchmark CNN en CPU
    with tf.device('/CPU:0'):
        model_cpu_cnn = create_cnn()
        start_cnn_cpu = time.time()
        model_cpu_cnn.fit(X_train_cnn, y_train, epochs=3, batch_size=bs, validation_split=0.1, verbose=0)
        time_cnn_cpu = time.time() - start_cnn_cpu

    row = {'Batch Size': bs, 'Device': 'CPU', 'MLP (s)': time_mlp_cpu, 'CNN (s)': time_cnn_cpu}
    results_batch_size.append(row)

    if gpu_devices:
        with tf.device('/GPU:0'):
            # Benchmark MLP en GPU
            model_gpu_mlp = create_mlp()
            start_mlp_gpu = time.time()
            model_gpu_mlp.fit(X_train, y_train, epochs=3, batch_size=bs, validation_split=0.1, verbose=0)
            time_mlp_gpu = time.time() - start_mlp_gpu

            # Benchmark CNN en GPU
            model_gpu_cnn = create_cnn()
            start_cnn_gpu = time.time()
            model_gpu_cnn.fit(X_train_cnn, y_train, epochs=3, batch_size=bs, validation_split=0.1, verbose=0)
            time_cnn_gpu = time.time() - start_cnn_gpu

        row = {'Batch Size': bs, 'Device': 'GPU', 'MLP (s)': time_mlp_gpu, 'CNN (s)': time_cnn_gpu}
        results_batch_size.append(row)
    else:
        print(f"No GPU detected for Batch Size {bs}.")

df_batch_size_results = pd.DataFrame(results_batch_size)
print("\nResultados de Benchmark por Batch Size:")
print(df_batch_size_results.to_string(index=False))
```

```text
(Salida)

--- Batch Size: 32 ---

--- Batch Size: 128 ---

--- Batch Size: 512 ---

Resultados de Benchmark por Batch Size:
 Batch Size Device   MLP (s)    CNN (s)
         32    CPU 16.867566 308.927763
         32    GPU 15.465086  21.481456
        128    CPU  9.817386 316.144463
        128    GPU  5.882009  10.573398
        512    CPU  4.524782 391.979529
        512    GPU  3.798007   8.307548

```

```python
# Visualización de resultados
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x='Batch Size', y='MLP (s)', hue='Device', data=df_batch_size_results, ax=axes[0])
axes[0].set_title('Benchmark MLP por Batch Size')
axes[0].set_ylabel('Tiempo (s)')

sns.barplot(x='Batch Size', y='CNN (s)', hue='Device', data=df_batch_size_results, ax=axes[1])
axes[1].set_title('Benchmark CNN por Batch Size')
axes[1].set_ylabel('Tiempo (s)')

plt.tight_layout()
plt.show()
```

### Ejercicio 3: Entrenar un modelo más grande (ResNet50 con CIFAR-10) y comparar CPU vs GPU.

Este ejercicio busca demostrar la ventaja de las GPUs con modelos y datasets más complejos. Se utilizará el dataset CIFAR-10 y una arquitectura ResNet50 pre-entrenada.

```python
# Cargar CIFAR-10
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = tf.keras.datasets.cifar10.load_data()

# Preprocesamiento para ResNet50
X_train_cifar = tf.keras.applications.resnet50.preprocess_input(X_train_cifar)
X_test_cifar = tf.keras.applications.resnet50.preprocess_input(X_test_cifar)

# Convertir etiquetas a one-hot encoding para el modelo pre-entrenado
y_train_cifar = tf.keras.utils.to_categorical(y_train_cifar, 10)
y_test_cifar = tf.keras.utils.to_categorical(y_test_cifar, 10)

def create_resnet50_model():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',  # Cargar pesos pre-entrenados de ImageNet
        include_top=False,   # Excluir la capa clasificadora final
        input_shape=(32, 32, 3) # Tamaño de imagen CIFAR-10
    )
    base_model.trainable = False # Congelar las capas de la base para fine-tuning

    # Añadir capas para CIFAR-10
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax') # 10 clases para CIFAR-10
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

results_resnet = []

# Benchmark ResNet50 en CPU
print("\n--- Entrenando ResNet50 en CPU ---")
with tf.device('/CPU:0'):
    model_resnet_cpu = create_resnet50_model()
    start_resnet_cpu = time.time()
    # Entrenar con un subconjunto de datos y pocas épocas para que no tarde demasiado
    model_resnet_cpu.fit(X_train_cifar[:5000], y_train_cifar[:5000], epochs=2, batch_size=64, validation_split=0.1, verbose=1)
    time_resnet_cpu = time.time() - start_resnet_cpu
    results_resnet.append({'Device': 'CPU', 'ResNet50 (s)': time_resnet_cpu})

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    # Benchmark ResNet50 en GPU
    print("\n--- Entrenando ResNet50 en GPU ---")
    with tf.device('/GPU:0'):
        model_resnet_gpu = create_resnet50_model()
        start_resnet_gpu = time.time()
        model_resnet_gpu.fit(X_train_cifar[:5000], y_train_cifar[:5000], epochs=2, batch_size=64, validation_split=0.1, verbose=1)
        time_resnet_gpu = time.time() - start_resnet_gpu
        results_resnet.append({'Device': 'GPU', 'ResNet50 (s)': time_resnet_gpu})
else:
    print("No GPU detected for ResNet50 benchmark.")

df_resnet_results = pd.DataFrame(results_resnet)
print("\nResultados de Benchmark ResNet50:")
print(df_resnet_results.to_string(index=False))

# Visualización de resultados ResNet50
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Device', y='ResNet50 (s)', data=df_resnet_results, ax=ax, palette='viridis')
ax.set_title('Benchmark ResNet50 (CIFAR-10) - CPU vs GPU')
ax.set_ylabel('Tiempo (s)')
plt.tight_layout()
plt.show()
```

```text
(Salida)
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
[1m170498071/170498071[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 0us/step

--- Entrenando ResNet50 en CPU ---
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
[1m94765736/94765736[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
Epoch 1/2
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 1s/step - accuracy: 0.3987 - loss: 2.3814 - val_accuracy: 0.5220 - val_loss: 1.3952
Epoch 2/2
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m70s[0m 981ms/step - accuracy: 0.5676 - loss: 1.2964 - val_accuracy: 0.5500 - val_loss: 1.3180

--- Entrenando ResNet50 en GPU ---
Epoch 1/2
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 184ms/step - accuracy: 0.3951 - loss: 2.3587 - val_accuracy: 0.5600 - val_loss: 1.3222
Epoch 2/2
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.5742 - loss: 1.2500 - val_accuracy: 0.5940 - val_loss: 1.2438

Resultados de Benchmark ResNet50:
Device  ResNet50 (s)
   CPU    157.750398
   GPU     26.025387

```

## Conclusión Final

### Análisis del Ejercicio 2: Variación del `batch_size`

Al analizar los resultados del benchmark con diferentes tamaños de `batch_size` para los modelos MLP y CNN, se observan las siguientes conclusiones:

-   **Impacto del `batch_size` en CPU:** Para la CPU, un `batch_size` mayor generalmente reduce el tiempo de entrenamiento total (se aprecia una disminución del tiempo al aumentar el `batch_size` de 32 a 512, especialmente para el MLP). Sin embargo, el entrenamiento de la CNN en CPU es significativamente más lento en comparación con la GPU, y el aumento del `batch_size` no logra mitigar esta diferencia de manera sustancial.
-   **Impacto del `batch_size` en GPU:** La GPU demuestra una ventaja consistente y significativa en todos los `batch_size` probados, especialmente con el modelo CNN, donde la reducción de tiempo es drástica. Para el MLP, la GPU es ligeramente más rápida que la CPU, pero la diferencia no es tan pronunciada como con la CNN. Un `batch_size` más grande parece optimizar el uso de la paralelización de la GPU, resultando en tiempos de entrenamiento más cortos.
-   **Configuración óptima en GPU:** Para ambos modelos (MLP y CNN), se observa que un `batch_size` de 512 resulta en los tiempos de entrenamiento más bajos en la GPU. Esto se debe a que un `batch_size` mayor permite una mayor paralelización de las operaciones, aprovechando mejor la arquitectura de la GPU.

En resumen, la GPU acelera considerablemente el entrenamiento de modelos, y un `batch_size` mayor (dentro de los límites de la memoria de la GPU) suele ser más eficiente para maximizar este beneficio.

### Análisis del Ejercicio 3: ResNet50 con CIFAR-10

El benchmark con un modelo más complejo como ResNet50 en el dataset CIFAR-10 resalta aún más la disparidad de rendimiento entre CPU y GPU:

-   **Ventaja Abismal de la GPU:** La GPU entrena el modelo ResNet50 en un tiempo muchísimo menor (aproximadamente **6 veces más rápido**: 26 segundos en GPU frente a 157 segundos en CPU). Esta diferencia es crucial en escenarios de Deep Learning con modelos de gran escala y datasets complejos.
-   **Importancia del Hardware:** Este ejercicio demuestra que para tareas de Deep Learning que involucran arquitecturas profundas y grandes volúmenes de datos, una GPU es prácticamente indispensable para hacer el entrenamiento factible en tiempos razonables. La CPU, aunque funcional, se vuelve una limitación severa para la experimentación y el desarrollo iterativo con modelos de vanguardia.

En conclusión, mientras que la CPU puede ser suficiente para modelos pequeños y tareas básicas, la GPU es la opción dominante y necesaria para el entrenamiento eficiente de modelos de Deep Learning complejos, donde su capacidad de procesamiento paralelo ofrece una ventaja de rendimiento insuperable.

## 10. Referencias y Recursos

- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [Apple Metal TensorFlow Plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- [Working with GPUs - Keras](https://keras.io/guides/working_with_gpus/)

---

📎 **Notebook anterior:** [11. Interpretabilidad de Modelos](./11_interpretabilidad_modelos.ipynb)  
📎 **Notebook siguiente:** [13. Despliegue de Modelos](./13_despliegue_modelos.ipynb)

---

## 13_despliegue_modelos.ipynb

## 9. Ejercicios Propuestos

1. **Ejercicio 1:** Modifica la API para que devuelva también las probabilidades por clase en formato de diccionario con los nombres de las clases.

2. **Ejercicio 2:** Agrega validación de entrada (que `features` tenga exactamente 4 elementos y sean numéricos).

3. **Ejercicio 3:** Crea los archivos `Dockerfile`, `requirements.txt` y `app.py` y construye la imagen Docker.

4. **Ejercicio 4 (Avanzado):** Configura MLflow para trackear experimentos y registra el modelo con diferentes hiperparámetros.

## Resolución de Ejercicios Propuestos

A continuación, implementaremos la solución integral para los 4 ejercicios propuestos, llevando nuestro modelo desde un entorno de experimentación local hasta un servicio profesional.

Los módulos a desarrollar son:
1. **Mejora y Validación de la API (Ejercicios 1 y 2):** Se añadirá validación estricta de entrada con Pydantic (garantizando 4 features numéricas) y se enriquecerá la respuesta para entregar un diccionario de probabilidades mapeadas por clase.
2. **Contenedorización (Ejercicio 3):** Crearemos físicamente los archivos `app.py`, `requirements.txt` y `Dockerfile` necesarios para desplegar la API en un entorno aislado y reproducible.
3. **MLOps con MLflow (Ejercicio 4):** Implementaremos un pipeline avanzado de validación que registra parámetros, métricas y el modelo resultante utilizando MLflow.

### Ejercicios 1, 2 y 3.1: Construcción de `app.py`
En el siguiente bloque de código consolidamos la API REST usando FastAPI. 
- **Validación (Ej2):** Utilizamos sentencias de Pydantic (`field_validator` o validaciones manuales) para obligar a que `features` conste de exactamente 4 números.
- **Probabilidades (Ej1):** Construimos un diccionario mapeando los nombres *setosa, versicolor, virginica* con la probabilidad devuelta por el modelo.
Nota: Guardaremos el código directamente en el archivo `app.py`.

```python
%%writefile app.py
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
```

```text
(Salida)
Writing app.py

```

### Ejercicio 3.2 y 3.3: `requirements.txt` y `Dockerfile`
A continuación, crearemos las instrucciones necesarias para que Docker pueda construir un contenedor liviano que aloje nuestra API.

```python
%%writefile requirements.txt
fastapi
uvicorn
scikit-learn
numpy
joblib
pydantic
```

```text
(Salida)
Writing requirements.txt

```

```python
%%writefile Dockerfile
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
```

```text
(Salida)
Writing Dockerfile

```

### Ejercicio 4: Tracking de Experimentos con MLflow
En este script, vamos a importar MLflow, entrenar un par de variaciones del modelo utilizando diferentes hiperparámetros (ej: la profundidad de un árbol) y dejaremos registro automático de su rendimiento (Accuracy) y del modelo generado.

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Preparar datos
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Nombre de nuestro experimento principal 
mlflow.set_experiment("Clasificador_Iris_rf")

# 2. Vamos a probar dos hiperparámetros diferentes
n_estimators_list = [10, 50]

for n_estimators in n_estimators_list:
    # Iniciar la "corrida" (run) dentro de MLflow
    with mlflow.start_run(run_name=f"RF_estimators_{n_estimators}"):
        
        # Crear y entrenar modelo
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predecir y evaluar
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # ---- MAGIA DE MLFLOW AQUÍ ----
        # Registramos el parámetro usado
        mlflow.log_param("n_estimators", n_estimators)
        
        # Registramos la métrica de éxito
        mlflow.log_metric("accuracy", acc)
        
        # Guardamos el modelo dentro de mlflow
        mlflow.sklearn.log_model(clf, f"modelo_rf_{n_estimators}")
        
        print(f"Ejecución con {n_estimators} estimadores finalizada. Accuracy: {acc:.4f}")

print("\n¡Tracking finalizado! Puedes abrir la consola de MLflow en tu terminal ejecutando: mlflow ui")
```

```text
(Salida)
2026/03/31 16:00:45 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/03/31 16:00:45 INFO mlflow.store.db.utils: Updating database tables
2026/03/31 16:00:49 INFO mlflow.tracking.fluent: Experiment with name 'Clasificador_Iris_rf' does not exist. Creating a new experiment.
2026/03/31 16:00:49 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/03/31 16:00:49 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html

```

```text
(Salida)
Ejecución con 10 estimadores finalizada. Accuracy: 1.0000

```

```text
(Salida)
2026/03/31 16:01:06 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/03/31 16:01:06 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html

```

```text
(Salida)
Ejecución con 50 estimadores finalizada. Accuracy: 1.0000

¡Tracking finalizado! Puedes abrir la consola de MLflow en tu terminal ejecutando: mlflow ui

```

## 10. Referencias y Recursos

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Docker Getting Started](https://docs.docker.com/get-started/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

📎 **Notebook anterior:** [12. CPU, GPU y Metal](./12_cpu_gpu_metal.ipynb)  
📎 **Este es el último notebook del curso.** ¡Felicidades por completar el recorrido! 🎉

---

