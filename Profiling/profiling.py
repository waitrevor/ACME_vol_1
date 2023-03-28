# profiling.py
"""Python Essentials: Profiling.
<Name> Trevor
<Class> Section 1
<Date> 2/27/23
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from math import sqrt
from numba import jit
from time import perf_counter
import matplotlib.pyplot as plt


# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()] for line in infile.readlines()]
        n = len(data)
        #Loops through the rows
        for i in range(len(data)-1):
            #Loops through the entries of each row
            for j in range(n-1-i):
                data[n-2-i][j] = max(data[n-1-i][j], data[n-1-i][j+1]) + data[n-2-i][j]

    return data[0][0]


# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]
    current = 3
    while len(primes_list) < N:
        isprime = True
        root = int(sqrt(current))
        for i in primes_list:     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
                break
            #p <= sqrt(n)
            if i > root:
                break
        if isprime:
            primes_list.append(current)
        #Except for 2, primes are always odd
        current += 2
    return primes_list


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    #Find the index of the column of A
    return np.argmin(np.linalg.norm(A - x.reshape((A.shape[0], 1)), axis=0))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
        #Use a dictionary to give each letter a number
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        dictionary = dict(zip(alphabet, range(1,27)))

        #create the list of alphanumeric scores
        values = [sum([dictionary[char] for char in name]) for name in names]
        return sum(np.multiply(range(1, len(names) + 1), values))



# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    #Base Case
    f1 = 1
    f2 = 1
    yield f1
    yield f2
    #Calculate the Fib sequence
    while True:
        f = f1 + f2
        yield f
        f1 = f2
        f2 = f

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    #Find the index of the first term in the Fibonacci sequence with N digits
    for i in enumerate(fibonacci()):
        if len(str(i[1])) >= N:
            return i[0] + 1


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    l = np.arange(2,N+1)
    #While the list ins't empty
    while len(l) > 0:
        yield l[0]
        #Get rid of the entires divisible by the first entry
        l = l[l % l[0] != 0]


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temp_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temp_array[j] = total
            product[i] = temp_array
    return product
    
def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    call = matrix_power_numba(np.random.random((2,2)), 2)
    ms = 2 ** np.arange(2, 8)
    nptimes = []
    nbtimes = []
    bad_times = []

    for m in ms:
        A = np.random.random((m,m))

        #Time matix_power()
        start = perf_counter()
        comp1 = matrix_power(A, n)
        end = perf_counter() - start
        bad_times.append(end)

        #Time matrix_power_numba()
        start = perf_counter()
        comp2 = matrix_power_numba(A, n)
        end = perf_counter() - start
        nbtimes.append(end)

        #Time np.linalg.matrix_power()
        start = perf_counter()
        comp3 = np.linalg.matrix_power(A, n)
        end = perf_counter() - start
        nptimes.append(end)


    #Plots
    plt.loglog(ms, bad_times, label='naive')
    plt.loglog(ms, nbtimes, label='numba')
    plt.loglog(ms, nptimes, label='numpy')
    plt.xlabel('m')
    plt.ylabel('Time')
    plt.legend()
    plt.title('Matrix Power Methods')
    plt.tight_layout()
    plt.show()