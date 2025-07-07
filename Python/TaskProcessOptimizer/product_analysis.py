import csv
import os

# ЗАВДАННЯ 1
# Функція для читання даних з CSV файлу і повертання списка словників
def read_product(file_path):
    # Перевіряємо чи існує файл
    if not os.path.exists(file_path):
        print(f"Помилка: файл {file_path} не знайдено!")
        return []
    
    products = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                try:
                    # Конвертуємо числові поля
                    row['price'] = float(row['price'])
                    row['stock'] = int(row['stock'])
                    row['sales_last_30_days'] = int(row['sales_last_30_days'])
                    row['rating'] = float(row['rating'])
                    row['returns_last_30_days'] = int(row['returns_last_30_days']) 
                    products.append(row)
                except (ValueError, KeyError) as e:
                    print(f"Помилка при обробці рядка: {row}. Причина: {str(e)}")

    except Exception as e:
        print(f"Помилка при читанні файлу: {str(e)}")
        return []

    return products

# ЗАВДАННЯ 2
# Знаходить товари для оптимізації
#     - продажі = 0
#     - залишок на складі > 0
#     - немає повернень
#     - рейтинг < 4.2
def find_unoptimized_products(products):
    unoptimized_products = []

    for product in products:
        # Перевіряємо, чи товар підходить під критерії оптимізації
        if (product['sales_last_30_days'] == 0 
            and product['stock'] > 0
            and product['returns_last_30_days'] == 0
            and product['rating'] < 4.2):
            unoptimized_products.append(product)
    return unoptimized_products
            
# ЗАВДАННЯ 3            
# Бонус: функція для оцінки ефективності товару
# (продажі - повернення) / залишок
def calculate_product_effectiveness(product):
    if product['stock'] == 0:
        return 0.0
    return (product['sales_last_30_days'] - product['returns_last_30_days']) / product['stock']

if __name__ == "__main__":
    products = read_product("products.csv")

    # Перевірка на наявність даних перед обчисленнями
    if not products:
        print("Немає даних для аналізу. Програма завершує роботу.")
        exit()

    # 1. Середні продажі
    avg_sales = sum(p["sales_last_30_days"] for p in products) / len(products)
    print(f"Середні продажі: {avg_sales:.2f}")
    
    # 2. Товари з продажами 0 та наявним залишком
    zero_sales = [p for p in products if p["sales_last_30_days"] == 0 and p["stock"] > 0]
    print("\nТовари без продажів (наявний залишок):")
    for p in zero_sales:
        print(f"- {p['title']} (залишок: {p['stock']})")
    
    # 3. Товари з низьким рейтингом без повернень
    low_rating = [p for p in products if p["rating"] < 4.0 and p["returns_last_30_days"] == 0]
    print("\nТовари для доопрацювання (рейтинг < 4.0, без повернень):")
    for p in low_rating:
        print(f"- {p['title']} (рейтинг: {p['rating']})")
    
    # Оптимізаційні товари
    unoptimized = find_unoptimized_products(products)
    print("\nТовари для оптимізації:")
    for p in unoptimized:
        print(f"- {p['title']} (рейтинг: {p['rating']}, залишок: {p['stock']})")
    
    # Бонус: виводимо ефективність
    print("\nЕфективність товарів:")
    for p in products:
        eff = calculate_product_effectiveness(p)
        print(f"- {p['title']}: {eff:.2f}")