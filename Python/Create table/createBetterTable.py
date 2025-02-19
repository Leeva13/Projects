import pandas as pd
import numpy as np

# Вхідні дані
data = {
    "X": [6, 12, 18, 24, 30],
    "Y": [4, 6, 8, 10],
    "n_ij": [
        [0, 2, 0, 0],  # X=6
        [1, 10, 3, 0],  # X=12
        [2, 4, 4, 0],  # X=18
        [0, 1, 5, 2],  # X=24
        [0, 0, 0, 1],  # X=30
    ],
}

# Преобразуем данные в DataFrame для удобства расчетов
df = pd.DataFrame(data["n_ij"], index=data["X"], columns=data["Y"])

# Преобразуйте індекси стовпців до числового типу
df.columns = df.columns.astype(float)

# Обчислюємо n_xi (сума частот за рядками)
df["n_xi"] = df.sum(axis=1)

# Обчислюємо n_yj (сума частот за стовпцями)
n_yj = df.sum(axis=0)

# Обчислення умовного середнього Y_x
Y_x_mean = (df.mul(df.columns, axis=1)).sum(axis=1) / df["n_xi"]

# Обчислення умовного середнього X_y
X_y_mean = (df.T * df.index).sum(axis=1) / n_yj

# Розрахунок сум для рівняння регресії
sum_X = df.index @ df["n_xi"]
sum_Y = df.columns @ n_yj
sum_X2 = (df.index**2) @ df["n_xi"]
sum_XY = sum((i * j * df.loc[i, j] for i in df.index for j in df.columns))

# Кількість спостережень
n = df["n_xi"].sum()

# Складання системи рівнянь для регресії Y_x = aX + b
A = np.array([
    [sum_X2, sum_X],
    [sum_X, n]
])
B = np.array([sum_XY, sum_Y])

# Розв'язання системи
a, b = np.linalg.solve(A, B)

# Збереження результатів
results = {
    "Умовні середні Y_x": Y_x_mean,
    "Умовні середні X_y": X_y_mean,
    "Сума X": sum_X,
    "Сума Y": sum_Y,
    "Сума X^2": sum_X2,
    "Сума XY": sum_XY,
    "Коефіцієнт a (Y_x)": a,
    "Коефіцієнт b (Y_x)": b
}

# Розрахунок середніх квадратів та тісноти зв'язку
X_mean = sum_X / n
Y_mean = sum_Y / n

# Відхилення
delta_Y = ((Y_x_mean - Y_mean)**2 * df["n_xi"]).sum() / n
delta_X = ((X_y_mean - X_mean)**2 * n_yj).sum() / n

# Стандартні відхилення
S_Y = np.sqrt(((df.columns - Y_mean)**2 * n_yj).sum() / n)
S_X = np.sqrt(((df.index - X_mean)**2 * df["n_xi"]).sum() / n)

# Кореляційні відношення
eta_Y_X = delta_Y / S_Y
eta_X_Y = delta_X / S_X

# Результати тісноти зв'язку
correlation_results = {
    "delta_Y": delta_Y,
    "delta_X": delta_X,
    "S_Y": S_Y,
    "S_X": S_X,
    "eta_Y_X": eta_Y_X,
    "eta_X_Y": eta_X_Y
}

# Підготовка до експорту в Excel
excel_data = {
    "X": df.index,
    "n_xi": df["n_xi"],
    "Y_x_mean": Y_x_mean
}
excel_df = pd.DataFrame(excel_data)

# Збереження до файлу
output_path = "C:\My PC\Projects\Python\Create table\egression_analysis.xlsx"
with pd.ExcelWriter(output_path) as writer:
    excel_df.to_excel(writer, sheet_name="Умовні середні", index=False)
    df.to_excel(writer, sheet_name="Таблиця частот")
    pd.DataFrame([results, correlation_results]).to_excel(writer, sheet_name="Результати")

output_path
