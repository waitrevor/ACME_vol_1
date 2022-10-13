# linear_systems.py
"""Volume 1: Linear Systems.
<Name> Trevor Wai
<Class> Section 2
<Date> 10/11/22
"""
import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
from time import time
from matplotlib import pyplot as plt

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    k = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if j > k:
                A[j] = A[j] - A[j,i] * (A[k]/A[k,i])
            else:
                continue
        k += 1
    return A



# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    m,n = np.shape(A)
    U = np.copy(A)
    L = np.eye(m)
    for j in range(0, n):
        for i in range(j + 1, m):
                L[i][j] = U[i][j] / U[j][j]
                U[i][j:] = U[i][j:] - L[i][j] * U[j][j:]


    return L,U


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    m,n = np.shape(A)
    L,U = lu(A)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - (L[i,:i] @ y[:i]) 

    for j in reversed(range(n)):
        x[j] = (y[j] - (U[j,:] @ x)) / U[j,j]

    return x
    


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    L1 = []
    L2 = []
    L3 = []
    L4 = []

    domain = 2**np.arange(1, 13)
    
    for n in domain:
        A = np.random.random((n,n))
        b = np.random.random(n)
        start = time()
        ans1 = np.matmul(la.inv(A),b)
        end = time() - start
        L1.append(end)

        start = time()
        ans2 = la.solve(A, b)
        end = time() - start
        L2.append(end)

        start = time()
        L, P = la.lu_factor(A)
        x = la.lu_solve((L,P), b)
        end = time() - start
        L3.append(end)

        L , P = la.lu_factor(A)
        start = time()
        x = la.lu_solve((L,P), b)
        end = time() - start
        L4.append(end)

    plt.plot(domain, L1)
    plt.plot(domain, L2)
    plt.plot(domain, L3)
    plt.plot(domain, L4)
    plt.title("Different Methods to solve Ax = b")
    plt.legend(["la.inv()", "la.solve()", "la.lu_factor() and la.lu_solve()", "la.lu_solve()"])
    plt.xlabel("n")
    plt.ylabel("time")
    plt.tight_layout()
    plt.show()

    


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    B = sparse.diags([1,-4,1], [-1, 0, 1], shape=(n,n))
    A = sparse.block_diag([B] * n)

    A.setdiag(1,-n)
    A.setdiag(1,n)

    return A
    


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """

    

#Testing