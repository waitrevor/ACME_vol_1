import calculator
import box
from itertools import combinations
import random
import time

# standard_library.py
"""Python Essentials: The Standard Library.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/6/22
"""


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    #returns min max and average of elements from list L
    return min(L), max(L), sum(L)/len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """
    
    a = 1
    b = a
    a += 1
    print(a == b)
    
    a = 'Hello World!'
    b = a
    a += ' Goodbye World!'
    print(a == b)

    a = [1,2,3]
    b = a
    a.append(4)
    print(a == b)

    a = (4,5,6)
    b = a
    a += (1, )
    print(a == b)
    
    a = {1,2,3}
    b = a
    a.add(4)
    print(a == b)

    print("Mutable: lists, sets Immutable: int, string, tuple")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """

    hyp = calculator.sqrt(calculator.sum(calculator.prod(a,a), calculator.prod(b,b)))

    return hyp


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    L = []

    for i in range(len(A) + 1):
        L += [set(j) for j in list(combinations(A,i))]

    return L

# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    
    numbers_left = [1,2,3,4,5,6,7,8,9]
    time_left = timelimit
    while time_left > 0:
        sum = random.randint(2,12)


        if numbers_left == []:
            print("Win") 
            break  

