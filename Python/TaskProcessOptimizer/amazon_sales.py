# ЗАВДАННЯ 1
# Обчислення загальої виручки для кожного замовлення 
def calculate_total_revenue(orders):
    if not orders:
        return 0.0
    
    total_revenue = 0.0
    for order in orders:
        # Сумуємо виручку по кожному товару
        total_revenue += order["price"] * order["quantity"]
    return round(total_revenue, 2) # Округлення для грошових значень

# ЗАВДАННЯ 2
# Повертає словник з категоріями та їх виручкою
def get_category_revenue(orders):
    if not orders:
        return {}
    
    category_revenue = {}
    for order in orders:
        category = order["category"]
        revenue = order["price"] * order["quantity"]
        
        # Додаємо виручку до категорії / або створюємо новий запис
        if category in category_revenue:
            category_revenue[category] += revenue
        else:
            category_revenue[category] = revenue
    

    # Округлюємо значення
    for category in category_revenue:
        category_revenue[category] = round(category_revenue[category], 2)
    return category_revenue

# ЗАВДАННЯ 3
# Повертає товар з найбільшою виручкою та її значення
def find_top_product(orders):
    if not orders: 
        return (None, 0.0)

    top_product = ""
    max_revenue = 0.0

    for order in orders:
        revenue = order["price"] * order["quantity"]
        if revenue > max_revenue:
            max_revenue = revenue
            top_product = order["product_name"]

    return (top_product, round(max_revenue, 2))

# ЗАВДАННЯ 4
# Повертає список замовлень, відсортований за датою
def filter_by_date(orders, target_date):
    if not orders or not target_date:
        return []
    
    filtered_orders = []

    for order in orders:
        if order["date"] == target_date:
            filtered_orders.append(order)
    return filtered_orders

if __name__ == "__main__":
    # Вхідні дані для тестування
    orders = [
        {"order_id": "A001", "product_name": "Wireless Headphones", "category": "Electronics", "price": 59.99, "quantity": 2, "date": "2025-05-01"},
        {"order_id": "A002", "product_name": "Yoga Mat", "category": "Fitness", "price": 29.99, "quantity": 1, "date": "2025-05-02"},
        {"order_id": "A003", "product_name": "Smartphone Charger", "category": "Electronics", "price": 19.99, "quantity": 3, "date": "2025-05-03"},
        {"order_id": "A004", "product_name": "Running Shoes", "category": "Fitness", "price": 79.99, "quantity": 1, "date": "2025-05-03"},
        {"order_id": "A005", "product_name": "Bluetooth Speaker", "category": "Electronics", "price": 39.99, "quantity": 2, "date": "2025-05-04"},
    ]

    print("Загальна виручка:", calculate_total_revenue(orders))
    print("Виручка по категоріям:", get_category_revenue(orders)) 
    # Фактичний результат у категорії Electronics: 259.93
    # У прикладі завдання вказано 199.95 - не враховано Smartphone Charger 
    # у цій програмі обчислення йшлося по всіх товарах Electronics

    print("Топовий товар:", find_top_product(orders))
    print("Замовлення за 2025-05-03:", filter_by_date(orders, "2025-05-03"))
