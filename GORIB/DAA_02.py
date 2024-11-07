# Write a program to implement Huffman Encoding using a greedy strategy.
class Job:
    def __init__(self, id, deadline, profit):
        self.id = id
        self.deadline = deadline
        self.profit = profit

def job_sequencing(jobs, n):
    jobs.sort(key=lambda x: x.profit, reverse=True)
    
    time_slots = [-1] * n
    
    job_sequence = []
    total_profit = 0
    
    for job in jobs:
        for slot in range(min(n, job.deadline) - 1, -1, -1):
            if time_slots[slot] == -1:  
                time_slots[slot] = job.id  
                job_sequence.append(job.id)
                total_profit += job.profit
                break

    print("Jobs selected:", job_sequence)
    print("Total Profit:", total_profit)

jobs = [
    Job('a', 4, 20), 
    Job('b', 1, 10),
    Job('c', 1, 40),
    Job('d', 1, 30)
]

n = len(jobs)
job_sequencing(jobs, n)
