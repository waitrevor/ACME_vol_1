# newtons_method.py
"""Volume 1: Newton's Method.
<Name> Trevor Wai
<Class> Section 1
<Date> 2/1/23
"""

import numpy as np
import sympy as sy
from math import exp
import scipy.linalg as la
from matplotlib import pyplot as plt

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    x = x0

    is_scalar = np.isscalar(f(x0))

    if is_scalar:
        for i in range(1, maxiter + 1):

            x -= alpha * (f(x) / Df(x))

            if la.norm(x0 - x) < tol:
                return x, True, i
            
            x0 = x
    else:
        for i in range(1, maxiter + 1):

            x = x - alpha * la.solve(Df(x), f(x))

            if la.norm(x0 - x) < tol:
                return x, True, i
            
            x0 = x

    return x0, False, maxiter


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    r = sy.symbols('r')
    f = P1 * ((1 + r)**N1 - 1) - P2 * (1 - (1 + r)**(-N2))
    Df = sy.diff(f, r)
    f = sy.lambdify(r, f, 'numpy')
    Df = sy.lambdify(r, Df, 'numpy')

    return newton(f, 0.1, Df)[0]


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    alpha = np.linspace(0,1, 1000)
    L = []

    for a in alpha[1:]:
        L.append(newton(f, x0, Df, alpha = a)[2])
    plt.plot(alpha[1:], L)
    plt.show()



# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    f = lambda x: np.array([5 * x[0] * x[1] - x[0] * (1 + x[1]),
                            -x[0] * x[1] + (1 - x[1]) * (1 + x[1])])
    Df = lambda x: np.array([4 * x[1] - 1, 4 * x[0], -x[1], -x[0] - 2 * x[1]]).reshape((2,2))

    x1 = newton(f, np.array([0,1]), Df)[0]
    x2 = newton(f, np.array([0,-1]), Df)[0]
    x3 = newton(f, np.array([3.75, 0.25]), Df, alpha=0.55)[0]

    return np.array([x1, x2, x3])


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    rmin, rmax, imin, imax = domain
    r = np.linspace(rmin, rmax, res)
    i = np.linspace(imin, imax, res)
    grid = np.array(np.meshgrid(r, i))
    grid = grid[0] + 1j * grid[1]

    for x in range(iters):
        grid = grid - la.solve(Df(grid), f(grid))

    index = np.argmin(abs(zeros[:, None, None] - grid), axis=0)

    plt.pcolormesh(index, extent=domain, origin='lower', cmap='brg')
    plt.show()
