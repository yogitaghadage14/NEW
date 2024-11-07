#Write a program to solve a 0-1 Knapsack problem using dynamic programming or branch and bound strategy.

# Function to solve 0-1 Knapsack problem using dynamic programming
def knapsack(values, weights, capacity):
    n = len(values)
    # Initialize a DP table with dimensions (n+1) x (capacity+1)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Build the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                # Include the item or exclude it, whichever gives a higher value
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                # Cannot include the item
                dp[i][w] = dp[i - 1][w]

    # The maximum value will be at dp[n][capacity]
    return dp[n][capacity]

# Example items with values and weights
values = [60, 100, 120]
weights = [10, 20, 30]
knapsack_capacity = 50

# Solve the knapsack problem
max_value = knapsack(values, weights, knapsack_capacity)

# Display the result
print("Maximum value in Knapsack:", max_value)
