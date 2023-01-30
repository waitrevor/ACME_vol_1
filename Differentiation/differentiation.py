# differentiation.py
"""Volume 1: Differentiation.
<Name> Trevor Wai
<Class> Section 1
<Date> 1/23/23
"""

import sympy as sy
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax import grad
import time as time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')
    f_prime = sy.diff((sy.sin(x) + 1)**(sy.sin(sy.cos(x))), x)
    f_prime = sy.lambdify(x, f_prime, 'numpy')
    return f_prime


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x)) / h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x+h) - f(x+2*h)) / (2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x-h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x-h) + f(x-2*h)) / (2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x+h) - f(x-h)) / (2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h)) / (12*h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    x = sy.symbols('x')
    f = sy.lambdify(x, (sy.sin(x) + 1)**(sy.sin(sy.cos(x))), 'numpy')
    h = np.logspace(-8,0,9)
    plt.loglog(h, abs(prob1()(x0) - fdq1(f, x0, h)), label='Order 1 Forward')
    plt.loglog(h, abs(prob1()(x0) - fdq2(f, x0, h)), label='Order 2 Forward')
    plt.loglog(h, abs(prob1()(x0) - bdq1(f, x0, h)), label='Order 1 Backward')
    plt.loglog(h, abs(prob1()(x0) - bdq2(f, x0, h)), label='Order 2 Backward')
    plt.loglog(h, abs(prob1()(x0) - cdq2(f, x0, h)), label='Order 2 Centered')
    plt.loglog(h, abs(prob1()(x0) - cdq4(f, x0, h)), label='Order 4 Centered')
    plt.ylabel('Absolte Error')
    plt.xlabel('h')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    data = np.deg2rad(np.load('plane.npy'))
    
    x = lambda t: 500 * np.tan(data[t][2]) / (np.tan(data[t][2]) - np.tan(data[t][1]))
    y = lambda t: (500 * np.tan(data[t][2]) * np.tan(data[t][1])) / (np.tan(data[t][2]) - np.tan(data[t][1]))

    x_7 = fdq1(x, 0, 1)
    y_7 = fdq1(y, 0, 1)
    t_7 = np.sqrt(x_7**2 + y_7**2)

    x_14 = bdq1(x, 7, 1)
    y_14 = bdq1(y, 7, 1)
    t_14 = np.sqrt(x_14**2 + y_14**2)

    t_i = []
    for i in range(1,7):
          x_i = cdq2(x, i, 1)
          y_i = cdq2(y, i, 1)
          t_i.append(np.sqrt(x_i**2 + y_i**2))

    return t_7, t_i[0], t_i[1], t_i[2], t_i[3], t_i[4], t_i[5], t_14


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    n = len(x)
    I = np.eye(n)
    J = np.array([])
    df = [(f(x + h * I[:,i]) - f(x - h * I[:,i])) / (2 * h) for i in range(n)]
    J = np.column_stack(df)
    return J


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    if n == 1:
        return x
    elif n == 0:
        return jnp.ones_like(x)
    return 2 * x * cheb_poly(x, n-1) - cheb_poly(x, n-2)

def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = jnp.linspace(-1,1, 1000)
    for n in range(0,5):
        poly = lambda x: cheb_poly(x, n)
        df = jnp.vectorize(grad(poly))

        plt.plot(domain, df(domain), label=f'n = {n}')


    plt.title(f'Derivative of Cheby Poly')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Problem 7
def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """
    f = lambda x: (jnp.sin(x) + 1)**(jnp.sin(jnp.cos(x)))
    
    prob1_time = []
    cdq4_time = []
    cdq4_err = []
    jax_time = []
    jax_err = []

    for n in range(N):
        x0 = np.random.random()
        #time using sympy
        start1 = time.perf_counter()
        df = prob1()(x0)
        end1 = time.perf_counter()
        prob1_time.append(end1 - start1)

        #time using cdq4
        start2 = time.perf_counter()
        df_cdq4 = cdq4(f, x0)
        end2 = time.perf_counter()
        cdq4_err.append(abs(df - df_cdq4))
        cdq4_time.append(end2 - start2)

        #time using jax
        start3 = time.perf_counter()
        df_jax = grad(f)(x0)
        end3 = time.perf_counter()
        jax_err.append(abs(df - df_jax))
        jax_time.append(end3 - start3)


    domain = np.ones(N)
    plt.scatter(prob1_time, domain*(1e-18), label='sympy', alpha=.25)
    plt.scatter(cdq4_time, cdq4_err, label='cdq4', alpha=.25)
    plt.scatter(jax_time, jax_err, label='jax', alpha=.25)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Computation Time')
    plt.ylabel('Absolute Error')
    plt.title('Computation Time v. Error')
    plt.legend()
    plt.tight_layout()
    plt.show()
