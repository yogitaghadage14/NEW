# Implement Gradient Descent Algorithm to find the local minima of a function.
# For example, find the local minima of the function y=(x+3)² starting from the point x=2.

import numpy as np
import matplotlib.pyplot as plt

# Define the function y = (x + 3)^2
def func(x):
    return (x + 3)**2

# Define the derivative of the function (gradient)
def grad_func(x):
    return 2 * (x + 3)

# Gradient Descent Algorithm
def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    history = []  # To store the history of x values during iterations
    
    for _ in range(num_iterations):
        # Calculate the gradient at the current point
        gradient = grad_func(x)
        # Update x using the gradient descent rule
        x = x - learning_rate * gradient
        # Store the history of x values
        history.append(x)
    
    return x, history

# Parameters
starting_point = 2    # Starting point x=2
learning_rate = 0.1   # Learning rate
num_iterations = 50   # Number of iterations

# Run gradient descent to find the local minima
local_min, history = gradient_descent(starting_point, learning_rate, num_iterations)

# Print the local minimum point found by gradient descent
print(f"The local minima is approximately at x = {local_min}")

# Plot the function and the gradient descent steps
x_values = np.linspace(-10, 5, 400)
y_values = func(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="y = (x + 3)^2", color='b')
plt.scatter(history, [func(x) for x in history], color='r', label="Gradient Descent Steps", zorder=5)
plt.title("Gradient Descent to Find Local Minima of y = (x + 3)^2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
