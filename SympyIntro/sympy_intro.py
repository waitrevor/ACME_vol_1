# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Name> Trevor Wai
<Class> Section 2
<Date> 1/16/23
"""

import sympy as sy
from matplotlib import pyplot as plt
import numpy as np

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    #Returns teh expression (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1) with symbolic fractions
    x, y = sy.symbols('x, y')
    return sy.Rational(2,5)*sy.E**(x**2 - y)*sy.cosh(x+y) + sy.Rational(3,7)*sy.log(x*y + 1)


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    #Computes product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    x, j, i = sy.symbols('x, j, i')
    return sy.simplify(sy.product(sy.summation(j * (sy.sin(x) + sy.cos(x)),(j, i, 5)), (i, 1, 5)))


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-2,2]. Plot e^(-y^2) over the same domain for comparison.
    """
    #Defines expression
    y, n = sy.symbols('y,n')
    domain = np.linspace(-2,2, 1000)
    f = sy.lambdify(y, sy.summation((-y**2)**n / sy.factorial(n), (n, 0, N)), 'numpy')
    #Plot sthe series
    plt.plot(domain, f(domain), label='Summation')
    #Plots the origional function e^x
    plt.plot(domain, sy.E**(-domain**2), label='e^(-y^2)')
    plt.legend()
    plt.title('Graph of McLaurin series of e^x')
    plt.tight_layout()
    plt.show()



# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    x, y, r, theta = sy.symbols('x,y, r, theta')
    domain = np.linspace(0, 2*np.pi, 1000)
    #Non-zero side of expression
    expr = 1 - ((x**2 + y**2)**sy.Rational(7,2) + 18 * x**5 * y - 60 * x**3 * y**3 + 18 * x * y**5) / (x**2 + y**2)**3
    #Simplifies and solves for r to get r(theta)
    trig = sy.simplify(expr.subs({x: r * sy.cos(theta), y:r*sy.sin(theta)}))
    sol = sy.solve(trig, r)[0]
    f = sy.lambdify(theta, sol, 'numpy')
    #Plots x(theta) against y(theta)
    plt.plot(f(domain) * np.cos(domain), f(domain) * np.sin(domain))
    plt.title('Rose Curve')
    plt.tight_layout()
    plt.show()




# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    #Initialize Matrices
    x, y, l = sy.symbols('x, y, l')
    A = sy.Matrix([[x-y, x, 0], [x, x-y, x], [0, x, x-y]])
    I = sy.Matrix([[1,0,0], [0,1,0], [0,0,1]])
    #Finds the eigen values
    eigs = sy.det(A - l*I)
    vals = sy.solve(eigs, l)
    dict = {}
    #Maps eigen vectors to eigen values
    for eigen in vals:
        dict[eigen] = (A - eigen*I).nullspace()
    return dict


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points over [-5,5]. Determine which
    points are maxima and which are minima. Plot the maxima in one color and the
    minima in another color. Return the minima and maxima (x values) as two
    separate sets.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    #Defines the polynomial
    domain = np.linspace(-5, 5, 1000)
    x = sy.symbols('x')
    p = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    #Differentiates the polynimials to find critical points
    p_prime = sy.diff(p, x)
    p_prime_prime = sy.diff(p_prime, x)
    crits = np.array(sy.solve(p_prime, x))
    f = sy.lambdify(x, p, 'numpy')
    f_prime_prime = sy.lambdify(x, p_prime_prime, 'numpy')

    #Determine if the critical points are mins or maxs
    max_p = np.array([])
    min_p = np.array([])
    for point in crits:
        if f_prime_prime(point) < 0:
            max_p = np.append(max_p, point)
        else:
            min_p = np.append(min_p, point)
    #Plots the polynomial with min and max points labeled
    plt.scatter(max_p, f(max_p), label='max')
    plt.scatter(min_p, f(min_p), label='min')
    plt.plot(domain, f(domain), label='p(x)')
    plt.legend()
    plt.title('Graph of p(x) with min and max points')
    plt.show()
    return set(min_p), set(max_p)
    




# Problem 7
def prob7():
    """Calculate the volume integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    #Creates the parts of the Jacobian
    x, y, z, rho, theta, psi, r = sy.symbols('x, y, z, rho, theta, psi, r')
    domain = np.linspace(0, 3, 1000)
    h_1 = rho * sy.sin(psi) * sy.cos(theta)
    h_2 = rho * sy.sin(psi) * sy.sin(theta)
    h_3 = rho * sy.cos(psi)
    #Volume to integrate
    f = sy.lambdify((x, y, z), (x**2 + y**2 + z**2)**2, 'numpy')
    #Compute the integral
    h = sy.Matrix([h_1, h_2, h_3])
    J = h.jacobian([rho, theta, psi])
    vol = sy.integrate(sy.simplify(f(h_1, h_2, h_3) * -J.det()), (rho, 0, r), (theta, 0, 2*sy.pi), (psi, 0, sy.pi))
    v = sy.lambdify(r, vol, 'numpy')
    #Plots the volume as the radius grows larger
    plt.plot(domain, v(domain))
    plt.title('Volume as radius increases')
    plt.show()
    return v(2)

