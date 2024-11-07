# Write a program non-recursive and recursive program to calculate Fibonacci numbers and
# analyze their time and space complexity.


import time

# Non-recursive (Iterative) Fibonacci
def fibonacci_iterative(n):   # Time O(n)  Space O(1)  Better
    if n <= 1:
        return n
    fib_0, fib_1 = 0, 1
    for _ in range(2, n + 1):
        fib_0, fib_1 = fib_1, fib_0 + fib_1
    return fib_1

# Recursive Fibonacci
def fibonacci_recursive(n):   # Time O(2^n)  Space O(n)  Not Better
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2) 

# Time analysis
def time_analysis(func, n):
    start_time = time.time()
    result = func(n)
    end_time = time.time()
    time_taken = end_time - start_time
    return result, time_taken

n = int(input("Enter the number: "))

# Calculating with Iterative
iter_result, iter_time = time_analysis(fibonacci_iterative, n)
print(f"Iterative Fibonacci({n}) = {iter_result}, Time Taken = {iter_time:.6f} seconds")

# Calculating with Recursive
rec_result, rec_time = time_analysis(fibonacci_recursive, n)
print(f"Recursive Fibonacci({n}) = {rec_result}, Time Taken = {rec_time:.6f} seconds")
