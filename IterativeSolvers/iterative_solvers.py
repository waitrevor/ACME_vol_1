# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name> Trevor Wai
<Class> Section 1
<Date> 4/17/23
"""

import numpy as np
from scipy import sparse, linalg as la
from matplotlib import pyplot as plt


# Helper function
def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.
    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.
        as_sparse: If True, an equivalent sparse CSR matrix is returned.
    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = sparse.dok_matrix((n,n))
    rows = np.random.choice(n, size=num_entries)
    cols = np.random.choice(n, size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    B = A.tocsr() # convert to row format for the next step
    for i in range(n):
        A[i,i] = abs(B[i]).sum() + 1
    return A.tocsr() if as_sparse else A.toarray()

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize
    D_inv = np.diag(1/np.diag(A))
    n = A.shape[0]
    x_0 = np.zeros(n)
    errors = []

    for i in range(maxiter):
        #Calculates the error if plot is True
        if plot == True:
            errors.append(la.norm(A@x_0 - b, ord=np.inf))

        #Jacobi Method
        x = x_0 + (D_inv @ (b - (A @ x_0)))

        #Tests the convergence
        if la.norm(x - x_0, ord=np.inf) < tol:
            break
        x_0 = x

    #Plots
    if plot == True:
        domain = range(i + 1)
        plt.semilogy(domain, errors)
        plt.title('Convergence of Jacobi Method')
        plt.ylabel('Absolute Error of Approximation')
        plt.xlabel('Iteration')
        plt.tight_layout()
        plt.show()

    return x


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize
    D_inv = 1/np.diag(A)
    n = np.shape(A)[0]
    x_0 = np.zeros(n)
    x = np.zeros(n)
    errors = []

    for k in range(maxiter):
        #Calculates the Errors
        if plot == True:
            errors.append(la.norm(A@x_0 - b, ord=np.inf))

        #Gauss-Seidel Method
        for i in range(n):
            x[i] = x_0[i] + D_inv[i] * (b[i] - A[i] @ x_0)

        #Tests for convergence
        if la.norm(x - x_0, ord=np.inf) < tol:
            break

        #Store past iterations as copy
        x_0 = np.copy(x)

    #Plots
    if plot == True:
        domain = np.arange(len(errors))
        plt.semilogy(domain, errors)
        plt.title('Convergence of Gauss-Seidel Method')
        plt.ylabel('Absolute Error of Approximation')
        plt.xlabel('Iteration')
        plt.tight_layout()
        plt.show()

    return x


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #Initials
    n = len(b)
    x_0 = np.zeros(n)

    for k in range(maxiter):

        x = np.copy(x_0)

        #Start and end rows of A
        for i in range(n):
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]

            #Gauss Seidel Sparse Method
            x[i] = x[i] + (b[i] - A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]) / A[i,i]      

        #Test of convergence
        if la.norm(x - x_0, ord=np.inf) < tol:
            break
        x_0 = np.copy(x)

    return x


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #Initialize
    n = len(b)
    x_0 = np.zeros(n)

    for k in range(maxiter):

        x = np.copy(x_0)

        for i in range(n):
            #Update components
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]

            #Sor Method
            x[i] = x[i] + omega * (b[i] - A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]) / A[i,i]

        #Test for convergence
        if la.norm(x - x_0, ord=np.inf) < tol:
            break

        x_0 = np.copy(x)

    return x, i+1 < maxiter, i + 1


#Came from linear systems lab to generate A
def prob5(n):
# Create the block B.
    B = sparse.diags([1, -4, 1], [-1, 0, 1], shape=(n, n))

    # Create the sparse matrix A.
    A = sparse.block_diag((B for i in range(n)))

    A.setdiag([1 for i in range(n**2 - n)], n)
    A.setdiag([1 for i in range(n**2 - n)], -n)
    return A


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    A = prob5(n)

    #Generate b
    tile = np.array([-100])
    tile = np.append(tile, [0 for i in range(n - 2)])
    tile = np.append(tile, -100)
    b = np.tile(tile, n)

    #Solution Au = b
    u, conv, iters = sor(A.tocsr(), b, omega, tol=tol, maxiter=maxiter)

    #Plots
    if plot == True:
        u_reshape = u.reshape((n,n))
        plt.title('Heat Map')
        plt.pcolormesh(u_reshape, cmap='coolwarm')
        plt.tight_layout()
        plt.show()

    return u, conv, iters


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    omega_list = np.arange(1, 2, 0.05)
    n = 20
    iters = []

    #Iterate over the omegas
    for omega in omega_list:
        iters.append(hot_plate(n, omega, tol=1e-2, maxiter=1000)[2])

    #Plot Itersations
    plt.plot(omega_list, iters)
    plt.tight_layout()
    plt.show()

    return omega_list[np.argmin(iters)]
