# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name> Trevor Wai
<Class> Section 2
<Date> 10/24/22
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import cmath
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    #Finds the QR decomposition of A
    Q,R = la.qr(A, mode='economic')
    #Using the upper traingular matrix R solves for x_hat
    prod = Q.T @ b
    x_hat = la.solve_triangular(R, prod)
    return x_hat

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    #Loads the data from the housing.npy file
    infile = np.load('housing.npy')
    m,n = infile.shape

    #Initializes the years from the infile and prices from the infile
    year = infile[:,0]
    price = infile[:,1]
    
    #Creates a Matrix using the years and vector using the prices
    A = np.column_stack((year, np.ones(m)))
    b = price

    #Returns the least square of A and b
    x = least_squares(A,b)

    #Plots the data
    plt.plot(np.linspace(0,16), np.polyval(x, np.linspace(0,16)), label='Least Square Fit')
    plt.plot(year, price, 'o', label='Data Point')
    plt.legend()
    plt.xlabel("year")
    plt.ylabel("Price Index")
    plt.title("Best Fit Line")
    plt.show()

    


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    #Loads the data and initializes variables
    infile = np.load('housing.npy')
    plot = 1
    year = infile[:,0]
    price = infile[:,1]
        
    #Loops through the degrees of polynimials
    for i in [4,7,10,13]:
        #Creates matrix A and vector b
        A = np.vander(year, i)
        b = price
        #Returns the least square of A and b
        x = least_squares(A,b)
        #Plots the data
        plt.subplot(2,2,plot)
        plt.plot(np.linspace(0,16), np.polyval(x, np.linspace(0,16)), label='polynomial best fit')
        plt.plot(year, price, 'o', label='Data Point')
        plot +=1 
        plt.legend()
        plt.xlabel("year")
        plt.ylabel("Price Index")
        plt.title(f"{i-1} degree polynomial best fit")
    plt.tight_layout()
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    #Loads the data from ellipse.npy and puts it into a matrix
    x, y = np.load('ellipse.npy').T
    A = np.column_stack((x**2, x, x*y, y, y**2))
    m,n = A.shape
    #Gives the coeffiecents of the equation for an ellipse
    a, b, c, d, e= la.lstsq(A, np.ones(m))[0]
    #Plots the Ellipse
    plot_ellipse(a,b,c,d,e)
    plt.plot(x, y, 'o', label='data points')
    plt.legend()
    plt.xlabel("x axis")
    plt.ylabel('y label')
    plt.title('Title')
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    #Gets the shape of the square nxn matrix A
    m,n = A.shape
    #Gets a random vector of length n
    x = np.random.random(n)
    #Normalize x
    x = x / la.norm(x)
    #Computes the dominant eigenvalue of A and a corresponding eigenvector via the power method
    for k in range(N):
        copy_of_x = np.copy(x)
        x = A @ x
        x = x / la.norm(x)
        #Checks to make sure that the norm is still within the tolerance
        if la.norm(x - copy_of_x) < tol:
            break

    return np.dot(x, A @ x), x


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    #Finds the shape of the matrix A
    m,n = A.shape
    #Puts A into upper Hessenberg Form
    S = la.hessenberg(A)
    for k in range(N):
        #Gets the QR decomposition of A
        Q,R = la.qr(S)
        #Recombine R and Q into S
        S = R @ Q
        #Initialize an empty list of eigenvalues
    eigs = []
    i = 0
    
    while i < n:
        #Base Case
        if i == n-1:
            eigs.append(S[i][i])
            break
        #Ensures still within the tolerance
        if abs(S[i+1,i]) < tol:
            eigs.append(S[i,i])
        #Calculates the eigen values andappends them to eigs
        else:
            B = - S[i][i] - S[i+1][i+1]
            C = (S[i][i]*S[i+1][i+1] - S[i][i+1]*S[i+1][i])
            thing_plus = (-B + cmath.sqrt(B**2 - (4*C))) / 2
            thing_minus = (-B - cmath.sqrt(B**2 - (4*C))) / 2
            eigs.append(thing_plus)
            eigs.append(thing_minus)
            i += 1
        #Move on to the next S
        i += 1

    return eigs
