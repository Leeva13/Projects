import numpy as np

# Сигмоїдна функція активації
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна сигмоїди для навчання
def sigmoid_derivative(x):
    return x * (1 - x)

# Ініціалізація даних (приклад для сегментації клієнтів "Єва")
# [Сума покупок, Частота покупок], Клас (0 - звичайний, 1 - VIP)
training_data = np.array([
    [5000, 5],  # Клієнт 1: сума 5000 грн, 5 покупок - VIP
    [1500, 2],  # Клієнт 2: сума 1500 грн, 2 покупки - Звичайний
    [7000, 7],  # Клієнт 3: сума 7000 грн, 7 покупок - VIP
    [2000, 3]   # Клієнт 4: сума 2000 грн, 3 покупки - Звичайний
])

# Вихідні дані (класи)
training_outputs = np.array([[1], [0], [1], [0]])

# Ініціалізація ваг і зміщень (bias) випадковими значеннями
np.random.seed(42)
input_layer_neurons = 2  # 2 вхідні параметри
hidden_layer_neurons = 4  # 4 нейрони у прихованому шарі
output_layer_neurons = 1  # 1 вихід (0 або 1)

# Вагові коефіцієнти
w1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
w2 = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
b2 = np.random.uniform(size=(1, output_layer_neurons))

# Гіперпараметри
learning_rate = 0.01
epochs = 10000

# Навчання нейронної мережі
for epoch in range(epochs):
    # Пряме поширення (feedforward)
    hidden_layer_input = np.dot(training_data, w1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, w2) + b2
    predicted_output = sigmoid(output_layer_input)
    
    # Обчислення помилки
    error = training_outputs - predicted_output
    
    # Зворотне поширення (backpropagation)
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = np.dot(d_predicted_output, w2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Оновлення ваг і зміщень
    w2 += learning_rate * np.dot(hidden_layer_output.T, d_predicted_output)
    b2 += learning_rate * np.sum(d_predicted_output, axis=0, keepdims=True)
    w1 += learning_rate * np.dot(training_data.T, d_hidden_layer)
    b1 += learning_rate * np.sum(d_hidden_layer, axis=0, keepdims=True)
    
    # Виведення помилки кожні 1000 епох
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss}")

# Виведення вагових коефіцієнтів після навчання
print("\nВагові коефіцієнти між вхідним і прихованим шаром (w1):\n", w1)
print("Зміщення прихованого шару (b1):\n", b1)
print("Вагові коефіцієнти між прихованим і вихідним шаром (w2):\n", w2)
print("Зміщення вихідного шару (b2):\n", b2)

# Тестування нейронної мережі
test_data = np.array([
    [6000, 6],  # Тест 1: сума 6000 грн, 6 покупок
    [1000, 1]   # Тест 2: сума 1000 грн, 1 покупка
])

hidden_layer_input = np.dot(test_data, w1) + b1
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, w2) + b2
test_output = sigmoid(output_layer_input)

print("\nРезультати тестування:")
for i, result in enumerate(test_output):
    segment = "VIP" if result > 0.5 else "Звичайний"
    print(f"Тест {i+1}: {test_data[i]} -> {result[0]:.4f} -> Сегмент: {segment}")

# Обчислення середньоквадратичної функції втрат для тестових даних
test_targets = np.array([[1], [0]])  # Очікувані значення для тестових даних
loss = np.mean(np.square(test_targets - test_output))
print(f"\nСередньоквадратична функція втрат на тестових даних: {loss:.4f}")

# Збереження результатів у файл
with open("neural_network_results.txt", "w") as f:
    f.write("Вагові коефіцієнти між вхідним і прихованим шаром (w1):\n")
    f.write(str(w1) + "\n")
    f.write("Зміщення прихованого шару (b1):\n")
    f.write(str(b1) + "\n")
    f.write("Вагові коефіцієнти між прихованим і вихідним шаром (w2):\n")
    f.write(str(w2) + "\n")
    f.write("Зміщення вихідного шару (b2):\n")
    f.write(str(b2) + "\n")
    f.write("\nРезультати тестування:\n")
    for i, result in enumerate(test_output):
        segment = "VIP" if result > 0.5 else "Звичайний"
        f.write(f"Тест {i+1}: {test_data[i]} -> {result[0]:.4f} -> Сегмент: {segment}\n")
    f.write(f"\nСередньоквадратична функція втрат: {loss:.4f}\n")

print("\nРезультати збережено у файл 'neural_network_results.txt'")