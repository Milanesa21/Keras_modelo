import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam

# Cargar los datos
df = pd.read_csv('altura_peso.csv')

# Verificar los datos cargados
print(df.head())

# Procesar los datos
X = df[['Altura']].values
y = df['Peso'].values

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Construir el modelo
model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, verbose=1, validation_split=0.2)

# Realizar predicciones
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

# Evaluar el modelo
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')

# Visualizar los resultados
plt.figure(figsize=(10, 6))

# Datos de entrenamiento
plt.scatter(X_train, y_train, color='blue', label='Datos de entrenamiento')

# Datos de prueba y predicciones
plt.scatter(X_test, y_test, color='green', label='Datos de prueba')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicciones')

plt.title('Regresión Lineal: Altura vs Peso')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.savefig('regresion_lineal.png') 