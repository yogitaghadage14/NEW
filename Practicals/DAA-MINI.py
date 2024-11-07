# Implement merge sort and multithreaded merge sort. Compare time required by both the
# algorithms. Also analyze the performance of each algorithm for the best case and the worst
# case.

import random
import time
from threading import Thread

# Merge function used by both single-threaded and multithreaded merge sort
def merge(arr, left, mid, right):
    left_subarray = arr[left:mid+1]
    right_subarray = arr[mid+1:right+1]
    i = j = 0
    k = left

    while i < len(left_subarray) and j < len(right_subarray):
        if left_subarray[i] <= right_subarray[j]:
            arr[k] = left_subarray[i]
            i += 1
        else:
            arr[k] = right_subarray[j]
            j += 1
        k += 1

    # Copy remaining elements of left and right subarrays, if any
    while i < len(left_subarray):
        arr[k] = left_subarray[i]
        i += 1
        k += 1
    while j < len(right_subarray):
        arr[k] = right_subarray[j]
        j += 1
        k += 1

# Regular single-threaded merge sort
def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)

# Multithreaded merge sort
def threaded_merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        # Create two threads for sorting the two halves
        left_thread = Thread(target=threaded_merge_sort, args=(arr, left, mid))
        right_thread = Thread(target=threaded_merge_sort, args=(arr, mid + 1, right))

        left_thread.start()
        right_thread.start()

        # Wait for both threads to finish
        left_thread.join()
        right_thread.join()

        # Merge the two halves
        merge(arr, left, mid, right)

# Function to analyze and compare the performance of both merge sorts
def compare_sorting_algorithms():
    # Generate a large random array
    size = 10000
    arr = [random.randint(1, 10000) for _ in range(size)]

    # Best case: Array is already sorted
    best_case_arr = sorted(arr)

    # Worst case: Array is sorted in reverse order
    worst_case_arr = sorted(arr, reverse=True)

    # Run single-threaded merge sort on best case
    arr_copy = best_case_arr[:]
    start = time.time()
    merge_sort(arr_copy, 0, len(arr_copy) - 1)
    single_thread_best_time = time.time() - start

    # Run multithreaded merge sort on best case
    arr_copy = best_case_arr[:]
    start = time.time()
    threaded_merge_sort(arr_copy, 0, len(arr_copy) - 1)
    multi_thread_best_time = time.time() - start

    # Run single-threaded merge sort on worst case
    arr_copy = worst_case_arr[:]
    start = time.time()
    merge_sort(arr_copy, 0, len(arr_copy) - 1)
    single_thread_worst_time = time.time() - start

    # Run multithreaded merge sort on worst case
    arr_copy = worst_case_arr[:]
    start = time.time()
    threaded_merge_sort(arr_copy, 0, len(arr_copy) - 1)
    multi_thread_worst_time = time.time() - start

    print("Performance Comparison:")
    print(f"Single-threaded Merge Sort (Best Case): {single_thread_best_time:.6f} seconds")
    print(f"Multithreaded Merge Sort (Best Case): {multi_thread_best_time:.6f} seconds")
    print(f"Single-threaded Merge Sort (Worst Case): {single_thread_worst_time:.6f} seconds")
    print(f"Multithreaded Merge Sort (Worst Case): {multi_thread_worst_time:.6f} seconds")

# Run the comparison
compare_sorting_algorithms()
