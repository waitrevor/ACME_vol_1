# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name> Trevor Wai
<Class> Section 2
<Date> 11/1/2022
"""

import numpy as np
import scipy.linalg as la
from imageio.v2 import imread
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import laplacian as lap
import scipy.sparse.linalg as spla


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #Computes the Laplacian Matrix
    return np.diag(np.sum(A, axis=0)) - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    #Initialize Variables
    L = laplacian(A)
    eigs = sorted(np.real(la.eigvals(L)))
    zeros = 0

    #Loops through all the eigen values and sets eigen values less than tolerance to zero
    for i in eigs:
        if i < tol:
            zeros+=1
        else:
            break
    
    return zeros, eigs[1]


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        #Scales the image to be between 0 and 1
        self.image = imread(filename) / 255

        #Checks to see if the image is color or gray scaled
        self.colored = bool(len(self.image.shape) == 3)

        #Flattens brightness
        if self.colored:
            self.brightness = np.ravel(self.image.mean(axis=2))
        else:
            self.brightness = np.ravel(self.image)


    # Problem 3
    def show_original(self):
        """Display the original image."""
        #Checks if the image is color or grayscaled
        if not self.colored:
            plt.imshow(self.image, cmap="gray")
            
        else:
            plt.imshow(self.image)

        #Shows image
        plt.axis('off')
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        #Initialize Variables
        m,n = self.image.shape[:2]
        A = sparse.lil_matrix((m*n,m*n))
        D = np.zeros(m*n)

        #Loops through the size of A and D
        for i in range(m*n):
            #Uses get_neighbors to find the weight and updates them to A and D
            J, dist = get_neighbors(i, r, m, n)
            w = np.exp(-abs(self.brightness[i] - self.brightness[J]) / sigma_B2 - dist / sigma_X2)
            A[i, J] = w
            D[i] = sum(w)

        return A.tocsc(), D
        

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        #Initialize Variables making matrix L and D
        m,n = self.image.shape[:2]
        L = lap(A)
        new_D = sparse.diags(D ** -(1/2))
        #Computes the second smalles eigen value and turns it into a matrix
        prod = new_D@L@new_D
        small_eig = spla.eigsh(prod, which='SM', k=2)[1][:,1].reshape((m,n))
        #Creates a greater than zero mask of the smallest eigen matrix
        mask = small_eig > 0

        return mask


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        #Gets matrices from adjacency and cut
        A, D = self.adjacency()
        mask = self.cut(A, D)

        #Checks to see if the image is colored
        if self.colored:
            mask = np.dstack([mask, mask, mask])

        #Updates the positive and negative images
        pos = self.image * mask
        neg = self.image * ~mask

        #Plots original image
        plt.subplot(131)
        if not self.colored:
            plt.imshow(self.image, cmap="gray")
        else:
            plt.imshow(self.image)
        plt.axis('off')

        #PLots the negative image
        plt.subplot(132)
        if not self.colored:
            plt.imshow(neg, cmap='gray')
        else:
            plt.imshow(neg)
        plt.axis('off')

        #PLots the positive image
        plt.subplot(133)
        if not self.colored:
            plt.imshow(pos, cmap='gray')
        else:
            plt.imshow(pos)
        plt.axis('off')

        plt.show()



# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
