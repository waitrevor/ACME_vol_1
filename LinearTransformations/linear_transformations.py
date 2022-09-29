# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/27/22
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
from time import time


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """

    #Stretch
    stretch = np.array([A[0] * a,A[1] * b])
    return stretch
    

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    #Shear
    shear = np.array([A[0] + A[1] * a, A[1] + A[0]* b])
    return shear
    

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    #Reflection
    reflected = np.array([((1/(a**2 + b**2))*A[0]*(a**2 - b**2) + (1/(a**2 + b**2))*A[1]*(2*a*b)),((1/(a**2 + b**2))*A[1]*(b**2 - a**2) + (1/(a**2 + b**2))*A[0]*(2*a*b))])
    return reflected

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    #Elementary Matrix
    elem_rotate = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #Rotates
    rotated = np.dot(elem_rotate,A)
    return rotated


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    #Breaks up the time into intervals
    time = np.linspace(0,T,200)
    #Position of the Earth
    earth_array = [[x_e],[0]]
    #Position of the Moon
    moon_array = [[x_m],[0]]
    #Loops through the time interval
    for t in time[1:]:
        #Rotates the earth and appends the new location to the position list
        new_earth = rotate([x_e,0],t*omega_e)
        earth_array[0].append(new_earth[0])
        earth_array[1].append(new_earth[1])

        #Rotates the moon and appends the new location to the position list
        new_moon = rotate([x_m-x_e,0],t*omega_m)
        moon_array[0].append(new_moon[0] + new_earth[0])
        moon_array[1].append(new_moon[1] + new_earth[1])


    #Plots the earth and moon
    plt.plot(earth_array[0],earth_array[1])
    plt.plot(moon_array[0], moon_array[1])
    #Gives titles and labels
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("This is the title")
    #Makes the legend
    plt.legend(["Earth","Moon"])
    plt.gca().set_aspect("equal")
    plt.show()
        
        


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    mvTime = []
    mmTime = []
    domain = 2**np.arange(1,9)
    for n in domain:
        matrixOne = random_matrix(n)
        matrixTwo = random_matrix(n)
        vector = random_vector(n)
        start = time()
        matrix_matrix_product(matrixOne, matrixTwo)
        mmTime.append(time() - start)
        start = time()
        matrix_vector_product(matrixOne, vector)
        mvTime.append(time() - start)

    plt.subplot(122)
    plt.subplot(121)
    plt.plot(domain, mvTime)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Matrix-Vector Multiplication')

    plt.subplot(122)
    plt.plot(domain, mmTime)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Matrix-Matrix Multiplication')

    plt.tight_layout()
    plt.show()


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    mvTime = []
    mvDotTime = []
    mmDotTime = []
    mmTime = []
    domain = 2**np.arange(1,9)
    for n in domain:
        matrixOne = random_matrix(n)
        matrixTwo = random_matrix(n)
        vector = random_vector(n)
        start = time()
        matrix_matrix_product(matrixOne, matrixTwo)
        mmTime.append(time() - start)
        start = time()
        np.dot(matrixOne, matrixTwo)
        mmDotTime.append(time() - start)
        start = time()
        matrix_vector_product(matrixOne, vector)
        mvTime.append(time() - start)
        start = time()
        np.dot(matrixOne, vector)
        mvDotTime.append(time() - start)

    plt.subplot(122)
    plt.subplot(121)
    plt.plot(domain,mvTime)
    plt.plot(domain, mmTime)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Multiplication Graph')
    plt.legend(['Matrix-Vector', 'Matrix-Matrix'])
    

    plt.subplot(122)
    plt.plot(domain,mvDotTime)
    plt.plot(domain, mmDotTime)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Dot Graph')

    
    plt.tight_layout()
    plt.show()


#testing

# horse = np.load('horse.npy')

# def showplot(A):
# #   # This function displays the image produce by the collection of coordinates given in H
#     cougarplot=plt.plot(A[0,:],A[1,:],'k.',markersize=3.5)
#     plt.axis([-1.5,1.5,-1.5,1.5])
#     plt.gca().set_aspect("equal")
#     plt.show()
#     return None


# # # Let's test the function above by plotting the data in our NumPy array cougar  
# showplot(rotate(horse,np.pi/2))

prob4()
