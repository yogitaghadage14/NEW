# Write a program to solve a fractional Knapsack problem using a greedy method.

# Define a class to represent an item with weight and value
class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

# Function to perform fractional knapsack
def fractional_knapsack(items, capacity):
    # Sort items by descending order of value-to-weight ratio
    items.sort(key=lambda x: x.value / x.weight, reverse=True)

    total_value = 0  # Variable to store the total value of items in knapsack
    for item in items:
        if capacity > 0 and item.weight <= capacity:
            # If item can be fully taken, add its value
            capacity -= item.weight
            total_value += item.value
        else:
            # If only part of the item can be taken
            fraction = capacity / item.weight
            total_value += item.value * fraction
            capacity = 0  # Knapsack is now full
            break

    return total_value

# List of items (value, weight)
items = [
    Item(60, 10),
    Item(100, 20),
    Item(120, 30)
]

# Capacity of the knapsack
knapsack_capacity = 50

# Solve the fractional knapsack problem
max_value = fractional_knapsack(items, knapsack_capacity)

# Display the result
print("Maximum value in Knapsack:", max_value)
