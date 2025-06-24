import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Завантаження датасету MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормалізація вхідних даних (0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Перетворення міток у категоріальний формат
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Побудова моделі нейронної мережі
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Вхідний шар (28x28 -> 784)
    tf.keras.layers.Dense(64, activation='relu'),   # Один прихований шар (64 нейрони)
    tf.keras.layers.Dense(10, activation='softmax') # Вихідний шар (10 нейронів)
])

# Виведення структури моделі
model.summary()

# Компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
history = model.fit(x_train, y_train_cat, batch_size=25, epochs=5, validation_split=0.12)

# Оцінка моделі на тестовій вибірці
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
print(f"Точність на тестовій вибірці: {test_accuracy}")

# Розпізнавання одного тестового зображення
n = 1
x = np.expand_dims(x_test[n], axis=0)
prediction = model.predict(x)
predicted_digit = np.argmax(prediction)
print(f"Передбачена цифра: {predicted_digit}")
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Визначення неправильно розпізнаних зразків
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
mask = predicted_labels != y_test
x_false = x_test[mask]
y_false = y_test[mask]
print(f"Кількість неправильно розпізнаних зразків: {len(x_false)}")

# Виведення 5 неправильно розпізнаних зразків
for i in range(min(5, len(x_false))):
    print(f"Реальне значення: {y_false[i]}, Передбачене значення: {predicted_labels[mask][i]}")
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()

# Виведення перших 25 зразків з датасету
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()






