import numpy as np
import matplotlib.pyplot as plt

# Membership functions
def demand_low(x):
    return np.maximum(1 - x / 3000, 0)

def demand_medium(x):
    return np.maximum(1 - np.abs(x - 3500) / 1500, 0)

def demand_high(x):
    return np.maximum((x - 4000) / 3000, 0)

def stock_low(x):
    return np.maximum(1 - x / 400, 0)

def stock_medium(x):
    return np.maximum(1 - np.abs(x - 500) / 200, 0)

def stock_high(x):
    return np.maximum((x - 600) / 400, 0)

def production_decrease(x):
    return np.maximum(1 - x / 4000, 0)

def production_increase(x):
    return np.maximum((x - 4000) / 4000, 0)

# Fuzzification
def fuzzify_demand(x):
    return demand_low(x), demand_medium(x), demand_high(x)

def fuzzify_stock(x):
    return stock_low(x), stock_medium(x), stock_high(x)

# Apply rules
def infer_rules(demand, stock):
    rules = {
        "increase": [
            min(demand[0], stock[0]),  # Rule 1
            min(demand[1], stock[0]),  # Rule 4
            min(demand[2], stock[0]),  # Rule 7
            min(demand[2], stock[1]),  # Rule 8
        ],
        "decrease": [
            min(demand[0], stock[1]),  # Rule 2
            min(demand[0], stock[2]),  # Rule 3
            min(demand[1], stock[1]),  # Rule 5
            min(demand[1], stock[2]),  # Rule 6
            min(demand[2], stock[2]),  # Rule 9
        ],
    }
    return max(rules["increase"]), max(rules["decrease"])

# Defuzzification
def defuzzify(increase, decrease):
    x = np.linspace(0, 8000, 1000)
    agg_increase = np.fmin(increase, production_increase(x))
    agg_decrease = np.fmin(decrease, production_decrease(x))
    aggregated = np.fmax(agg_increase, agg_decrease)

    centroid = np.sum(x * aggregated) / np.sum(aggregated)
    return centroid

# Example input
demand_value = 3000  # Permintaan
stock_value = 500    # Persediaan

# Fuzzification
demand = fuzzify_demand(demand_value)
stock = fuzzify_stock(stock_value)

# Inference
increase, decrease = infer_rules(demand, stock)

# Defuzzification
result = defuzzify(increase, decrease)

print(f"Recommended Production: {result:.2f} units")

# Plot membership functions
x_demand = np.linspace(0, 7000, 1000)
x_stock = np.linspace(0, 1000, 1000)
x_production = np.linspace(0, 8000, 1000)

plt.figure(figsize=(10, 8))

# Demand
plt.subplot(3, 1, 1)
plt.plot(x_demand, demand_low(x_demand), label="turun", linestyle="dashed")
plt.plot(x_demand, demand_medium(x_demand), label="tetap", linestyle="dashed")
plt.plot(x_demand, demand_high(x_demand), label="naik", linestyle="dashed")
plt.title("Permintaan")
plt.legend()

# Stock
plt.subplot(3, 1, 2)
plt.plot(x_stock, stock_low(x_stock), label="sedikit", linestyle="dashed")
plt.plot(x_stock, stock_medium(x_stock), label="sedang", linestyle="dashed")
plt.plot(x_stock, stock_high(x_stock), label="banyak", linestyle="dashed")
plt.title("Persediaan")
plt.legend()

# Production
plt.subplot(3, 1, 3)
plt.plot(x_production, production_decrease(x_production), label="berkurang", linestyle="dashed")
plt.plot(x_production, production_increase(x_production), label="bertambah", linestyle="dashed")
plt.title("Produksi")
plt.legend()

plt.tight_layout()
plt.show()
