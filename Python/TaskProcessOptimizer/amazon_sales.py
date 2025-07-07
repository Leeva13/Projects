# TASK 1
# Calculate total revenue for each order 
def calculate_total_revenue(orders):
    if not orders:
        return 0.0
    
    total_revenue = 0.0
    for order in orders:
        # Summing up the revenue for each product
        total_revenue += order["price"] * order["quantity"]
    return round(total_revenue, 2) # Rounding for monetary values

# TASK 2
# Returns a dictionary with categories and their revenue
def get_category_revenue(orders):
    if not orders:
        return {}
    
    category_revenue = {}
    for order in orders:
        category = order["category"]
        revenue = order["price"] * order["quantity"]
        
        # Add revenue to the category / or create a new record
        if category in category_revenue:
            category_revenue[category] += revenue
        else:
            category_revenue[category] = revenue
    

    # Round up the values
    for category in category_revenue:
        category_revenue[category] = round(category_revenue[category], 2)
    return category_revenue

# TASK 3
# Returns the product with the highest revenue and its value
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

# TASK 4
# Returns a list of orders sorted by date
def filter_by_date(orders, target_date):
    if not orders or not target_date:
        return []
    
    filtered_orders = []

    for order in orders:
        if order["date"] == target_date:
            filtered_orders.append(order)
    return filtered_orders

if __name__ == "__main__":
    # Input data for testing
    orders = [
        {"order_id": "A001", "product_name": "Wireless Headphones", "category": "Electronics", "price": 59.99, "quantity": 2, "date": "2025-05-01"},
        {"order_id": "A002", "product_name": "Yoga Mat", "category": "Fitness", "price": 29.99, "quantity": 1, "date": "2025-05-02"},
        {"order_id": "A003", "product_name": "Smartphone Charger", "category": "Electronics", "price": 19.99, "quantity": 3, "date": "2025-05-03"},
        {"order_id": "A004", "product_name": "Running Shoes", "category": "Fitness", "price": 79.99, "quantity": 1, "date": "2025-05-03"},
        {"order_id": "A005", "product_name": "Bluetooth Speaker", "category": "Electronics", "price": 39.99, "quantity": 2, "date": "2025-05-04"},
    ]

    print("Total revenue:", calculate_total_revenue(orders))
    print("Revenue by category:", get_category_revenue(orders)) 
    # Actual result in the Electronics category: 259.93
    # The example task shows 199.95 - Smartphone Charger is not included
    # in this programme, the calculation was performed for all Electronics products

    print("Top product:", find_top_product(orders))
    print("Order by 2025-05-03:", filter_by_date(orders, "2025-05-03"))
