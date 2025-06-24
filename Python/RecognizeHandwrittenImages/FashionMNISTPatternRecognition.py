import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Завантаження датасету Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Нормалізація вхідних даних
x_train = x_train / 255.0
x_test = x_test / 255.0

# Перетворення міток у категоріальний формат
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Побудова моделі
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),   # Один прихований шар (64 нейрони)
    tf.keras.layers.Dense(10, activation='softmax')
])

# Виведення структури моделі
model.summary()

# Компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
history = model.fit(x_train, y_train_cat, batch_size=25, epochs=5, validation_split=0.12)

# Оцінка моделі
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
print(f"Точність на тестовій вибірці: {test_accuracy}")

# Збереження результатів
model.save('fashion_mnist_model_variant2.h5')
print("Модель збережена у файл 'fashion_mnist_model_variant2.h5'")