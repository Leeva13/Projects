import csv
import os

# TASK 1
# Function to read data from a CSV file and return a list of dictionaries
def read_product(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: file {file_path} not found!")
        return []
    
    products = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                try:
                    # Convert numeric fields
                    row['price'] = float(row['price'])
                    row['stock'] = int(row['stock'])
                    row['sales_last_30_days'] = int(row['sales_last_30_days'])
                    row['rating'] = float(row['rating'])
                    row['returns_last_30_days'] = int(row['returns_last_30_days']) 
                    products.append(row)
                except (ValueError, KeyError) as e:
                    print(f"Error processing a string: {row}. Reason: {str(e)}")

    except Exception as e:
        print(f"Error reading a file: {str(e)}")
        return []

    return products

# TASK 2
# Finds products to optimise
# - sales = 0
# - stock balance > 0
# - no returns
# - rating < 4.2
def find_unoptimized_products(products):
    unoptimized_products = []

    for product in products:
        # We check whether the product meets the optimisation criteria
        if (product['sales_last_30_days'] == 0 
            and product['stock'] > 0
            and product['returns_last_30_days'] == 0
            and product['rating'] < 4.2):
            unoptimized_products.append(product)
    return unoptimized_products
            
# TASK 3
# Bonus: a function for evaluating the effectiveness of a product
# (sales - returns) / balance
def calculate_product_effectiveness(product):
    if product['stock'] == 0:
        return 0.0
    return (product['sales_last_30_days'] - product['returns_last_30_days']) / product['stock']

if __name__ == "__main__":
    products = read_product("products.csv")

    # Check for data availability before calculations
    if not products:
        print("No data for analysis. The application exits.")
        exit()

    # 1. Average sales
    avg_sales = sum(p["sales_last_30_days"] for p in products) / len(products)
    print(f"Average sales: {avg_sales:.2f}")
    
    # 2. Goods with sales of 0 and available balance
    zero_sales = [p for p in products if p["sales_last_30_days"] == 0 and p["stock"] > 0]
    print("\nGoods without sales (available balance):")
    for p in zero_sales:
        print(f"- {p['title']} (stock: {p['stock']})")
    
    # 3. Low-rated products with no returns
    low_rating = [p for p in products if p["rating"] < 4.0 and p["returns_last_30_days"] == 0]
    print("\nProducts for improvement (rating < 4.0, no returns):")
    for p in low_rating:
        print(f"- {p['title']} (rating: {p['rating']})")
    
    # Optimisation products
    unoptimized = find_unoptimized_products(products)
    print("\nProducts for optimisation:")
    for p in unoptimized:
        print(f"- {p['title']} (rating: {p['rating']}, stock: {p['stock']})")
    
    # Bonus: we derive efficiency
    print("\nProduct efficiency:")
    for p in products:
        eff = calculate_product_effectiveness(p)
        print(f"- {p['title']}: {eff:.2f}")