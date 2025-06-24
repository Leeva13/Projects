import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Генерація синтетичного датасету
np.random.seed(42)
n_samples = 500  # Кількість клієнтів

data = {
    'age': np.random.randint(18, 66, n_samples),
    'monthly_spending': np.random.uniform(100, 5000, n_samples),
    'purchase_frequency': np.random.randint(1, 21, n_samples),
    'purchase_type': np.random.choice([0, 1, 2], n_samples),
    'loyalty_program': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)
df.head()  # Переглянемо перші рядки



# Вибираємо ознаки для кластеризації
X_cluster = df[['age', 'monthly_spending', 'purchase_frequency']]

# Масштабування даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Застосовуємо KMeans (3 кластери як приклад)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Візуалізація кластерів
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='monthly_spending', hue='cluster', size='purchase_frequency', data=df, palette='viridis')
plt.title('Сегментація клієнтів "Єва" за віком і сумою покупок')
plt.xlabel('Вік')
plt.ylabel('Сума покупок (грн)')
plt.show()



# Ознаки та цільова змінна
X = df[['age', 'monthly_spending', 'purchase_frequency', 'purchase_type']]
y = df['loyalty_program']

# Розбиваємо на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Побудова моделі логістичної регресії
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Прогноз і оцінка точності
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі класифікації: {accuracy:.2f}")

# Візуалізація точності на тестовій вибірці
plt.figure(figsize=(8, 6))
sns.histplot(y_test, color='blue', alpha=0.5, label='Реальні значення')
sns.histplot(y_pred, color='orange', alpha=0.5, label='Прогнозовані значення')
plt.title('Порівняння реальних і прогнозованих значень участі у програмі лояльності')
plt.legend()
plt.show()


# Середні значення для кожного кластера
cluster_summary = df.groupby('cluster').mean()
print(cluster_summary)

# Візуалізація середніх значень
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_summary.T, annot=True, cmap='coolwarm', fmt='.1f')
plt.title('Середні характеристики кластерів клієнтів "Єва"')
plt.show()


df.to_csv('eva_customer_segmentation.csv', index=False)
print("Результати збережено у файл 'eva_customer_segmentation.csv'")