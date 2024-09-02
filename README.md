# Regresión Lineal con Keras: Predicción de Peso basado en la Altura

Este proyecto implementa un modelo de regresión lineal utilizando Keras para predecir el peso de una persona en función de su altura. Los datos utilizados provienen de un archivo CSV llamado `altura_peso.csv`. A continuación se describe el proceso de entrenamiento, evaluación y visualización del modelo.

## Requisitos

Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar el código:

```bash
pip install pandas numpy matplotlib scikit-learn keras
``` 

Descripción del Código

    Carga de Datos
        Los datos se cargan desde el archivo altura_peso.csv utilizando pandas.
        Se muestra una vista previa de los primeros registros con print(df.head()).

    Procesamiento de Datos
        Se selecciona la columna Altura como variable independiente (X) y la columna Peso como variable dependiente (y).
        Los datos se dividen en conjuntos de entrenamiento y prueba utilizando train_test_split con un 80% para entrenamiento y 20% para prueba.
        Los datos se normalizan utilizando StandardScaler.

    Construcción del Modelo
        Se construye un modelo secuencial con una sola capa densa (Dense) que tiene una unidad de salida y usa una función de activación lineal.

    Compilación del Modelo
        El modelo se compila utilizando el optimizador Adam con una tasa de aprendizaje de 0.01 y la función de pérdida mean_squared_error.

    Entrenamiento del Modelo
        El modelo se entrena durante 100 épocas con una división del 20% de los datos de entrenamiento para validación.

    Predicciones y Evaluación
        Se realizan predicciones sobre los datos de prueba y se desnormalizan los valores predichos.
        Se calcula el error cuadrático medio (Mean Squared Error) para evaluar el rendimiento del modelo.

    Visualización de Resultados
        Se generan gráficos utilizando matplotlib para visualizar:
            Los datos de entrenamiento.
            Los datos de prueba y las predicciones realizadas por el modelo.
        El gráfico final se guarda como regresion_lineal.png.