# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name> Trevor Wai
<Class> Section 1
<Date> 2/13/23
"""

import numpy as np
import sympy as sy
import scipy.linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    #Find the Conditional Numbers
    k = la.svdvals(A)[0] / la.svdvals(A)[-1]
    if k == 0:
        return np.inf
    else:
        return k


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    #Conditional Numbers
    k = []
    k_hat = []

    
    for i in range(100):
        #Perturbed values
        r = np.random.normal(1,1e-10,21)
        new_roots = np.sort(np.roots(np.poly1d(w_coeffs * r)))
        #Absolute Conditional Number
        k.append(la.norm(new_roots - w_roots, np.inf) / la.norm(r, np.inf))
        #Relative Condition Number
        k_hat.append(k[-1] * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf))
        plt.scatter(new_roots.real, new_roots.imag, marker=',', c='black', s=0.7)

    plt.scatter(new_roots.real, new_roots.imag, marker=',', label='perturbed', c='black', s=0.7)
    plt.scatter(w_roots.real, w_roots.imag, label='original', c='blue')
    plt.axis('equal')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title('Wilkinson Polynomial Roots')
    plt.legend()
    plt.show()

    return np.mean(k), np.mean(k_hat)



# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    lam = la.eig(A)[0]
    #Eigenvalues of perturbed matrix
    lam_hat = la.eig(A + H)[0]
    #Reorder the perturbed matrix
    lam_hat = reorder_eigvals(lam, lam_hat)

    #Calculate Relative and absolute conditional number
    k_hat = la.norm(lam - lam_hat) / la.norm(H,2)
    k = k_hat * la.norm(A,2) / la.norm(lam)

    return k_hat, k




# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    x = np.linspace(domain[0], domain[1], res)
    y = np.linspace(domain[2], domain[3], res)
    con_nums = np.zeros((res,res))

    for i in range(res):
        for j in range(res):
            #Set x and y entries
            A = np.array([[1,x[i]], [y[j],1]])
            #Calculate Relative condtion 
            con_nums[i,j] = eig_cond(A)[1]

    plt.pcolormesh(x, y, con_nums, cmap='gray_r')
    plt.colorbar()
    plt.show()


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    xk, yk = np.load('stability_data.npy').T
    A = np.vander(xk, n+1)

    #Solve using la.inv
    x_solve = la.inv(A.T @ A) @ A.T @ yk

    #Solve using la.qr
    Q,R = la.qr(A, mode='economic')
    x = la.solve_triangular(R, Q.T @ yk)

    domain = np.linspace(0, 1, 1000)

    plt.scatter(xk, yk, label='data')
    plt.plot(domain, np.polyval(x_solve, domain), c='g', label='la.inv')
    plt.plot(domain, np.polyval(x, domain), c='r', label='la.qr')
    plt.ylim([0,4])
    plt.legend()
    plt.title('Ways to solve the Least Squares')
    plt.tight_layout()
    plt.show()

    return la.norm(A @ x_solve - yk), la.norm(A @ x - yk)

# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    x = sy.symbols('x')
    ns = np.arange(5,51,5)
    exact = []
    facts =[]

    for i in ns:
        n = int(i)
        #Using sy.integrate
        exact.append(float(sy.integrate(x**n * sy.exp(x-1), (x,0,1))))
        #Using sy.subfactorial
        facts.append((-1)**n*(sy.subfactorial(n) - sy.factorial(n)/np.e))
    #Relative Foward Error
    errors = (np.abs(np.array(exact) - np.array(facts))) / np.abs(exact)
    plt.semilogy(ns,errors)
    plt.title('Relative Forward Error')
    plt.xlabel('n')
    plt.ylabel('Relative Error')
    plt.tight_layout()
    plt.show()
