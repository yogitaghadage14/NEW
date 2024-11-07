# Write a program to calculate Fibonacci numbers and find its step count.
def fibonacci_iter(n):
    if n < 0:
        return -1, 1
    if n == 0 or n == 1:
        return n, 1
    steps = 0
    a, b = 0, 1
    for i in range(2, n + 1):
        c = a + b
        a = b
        b = c
        steps += 1
    return b, steps

def fibonacci_recur(n):
    if n < 0:
        return -1, 1
    if n == 0 or n == 1:
        return n, 1
    fib1, steps1 = fibonacci_recur(n - 1)
    fib2, steps2 = fibonacci_recur(n - 2)
    return fib1 + fib2, steps1 + steps2 + 1

if __name__ == '__main__':
    n = int(input("Enter a number: "))
    iter_result, iter_steps = fibonacci_iter(n)
    print("Iterative:", iter_result)
    print("Steps:", iter_steps)
    
    recur_result, recur_steps = fibonacci_recur(n)
    print("Recursive:", recur_result)
    print("Steps:", recur_steps)
