import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Навчальні дані
X_train = np.array([
    [54.4, 165.1],  # Валерія
    [65.44, 183],   # Євген
    [62.2, 178],    # Антон
    [49, 152]      # Діана
])
y_train = np.array([0, 1, 1, 0])  # 0 - Жінка, 1 - Чоловік

# Масштабування вхідних даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Створення моделі нейронної мережі
model = keras.Sequential([
    keras.layers.Dense(4, activation='sigmoid', input_shape=(2,)),  # Прихований шар
    keras.layers.Dense(1, activation='sigmoid')  # Вихідний шар
])

# Компіляція моделі
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Навчання моделі
model.fit(X_train_scaled, y_train, epochs=500, verbose=0)

# Отримання вагових коефіцієнтів
weights = model.get_weights()
print("Вагові коефіцієнти:", weights)

# Тестування моделі
X_test = np.array([
    [60, 170],  # Нові параметри для тесту
    [70, 185]
])
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("Результати тестування:", predictions)

# Обчислення середньоквадратичної функції втрат
loss = np.mean((predictions - np.array([0, 1]))**2)
print("Середньоквадратична функція втрат:", loss)

# Збереження моделі
model.save("gender_classification_model.h5")