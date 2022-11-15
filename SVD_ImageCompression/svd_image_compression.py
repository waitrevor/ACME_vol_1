"""Volume 1: The SVD and Image Compression."""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #Calculates the eigen values and eigen vectors, and singular values
    lam, V = la.eig(A.conj().T @ A)
    sig = np.sqrt(np.array(lam))
    #Sorts the singlular values and eigen vectors greates to least
    V_sort = np.argsort(sig)[::-1]
    sig = np.array(sorted(sig)[::-1])
    r = sum(sig > tol)
    #Keeps the positive singular values and corresponding eigenvectors
    sig = sig[:r]
    V = V[:,V_sort[:r]]
    U = (A @ V) / sig

    return np.real(U), np.real(sig), V.conj().T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #Initialize Variables
    domain = np.linspace(0, 2 * np.pi, 200)
    S = np.row_stack((np.cos(domain), np.sin(domain)))
    E = np.array([[1,0,0], [0,0,1]])
    U, s, Vh = la.svd(A)
    
    #Graph S and E
    plt.subplot(221)
    plt.plot(S[0], S[1])
    plt.plot(E[0], E[1])
    plt.axis('equal')
    plt.title('S and E')

    #Graph VhS and VhE
    VhS = Vh @ S
    VhE = Vh @ E
    plt.subplot(222)
    plt.plot(VhS[0], VhS[1])
    plt.plot(VhE[0], VhE[1])    
    plt.axis('equal')
    plt.title('VhS and VhE')

    #Graph ΣVhS and ΣVhE
    ΣVhS = np.diag(s) @ VhS
    ΣVhE = np.diag(s) @ VhE
    plt.subplot(223)
    plt.plot(ΣVhS[0], ΣVhS[1])
    plt.plot(ΣVhE[0], ΣVhE[1])
    plt.axis('equal')
    plt.title('ΣVhS and ΣVhE')

    #Graph UΣVhS and UΣVhE
    UΣVhS = U @ ΣVhS
    UΣVhE = U @ ΣVhE
    plt.subplot(224)
    plt.plot(UΣVhS[0], UΣVhS[1])
    plt.plot(UΣVhE[0], UΣVhE[1])
    plt.axis('equal')
    plt.title('UΣVhS and UΣVhE')

    plt.suptitle('Transformations')
    plt.tight_layout()
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Initialize variables
    U, sig, Vh = la.svd(A)

    #Raises an error if Rank A is less than s
    if s > len(sig[sig > 0]):
        raise ValueError('Rank of A is less than s')

    #Keep only the first s columns of U, s rows of Vh and    
    U = U[:,:s]
    Vh = Vh[:s,:]
    sig = sig[:s]

    # Calculate the best rank s approximation of A and number of entries to store the truncated SVD
    UΣVh = U @ np.diag(sig) @ Vh
    amount = U.size + Vh.size + len(sig)

    return UΣVh, amount 


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Initialize Variables and check tolerance
    s = compact_svd(A)[1]
    if s[-1] > err:
        raise ValueError('Err is less than or equal to ')

    #Finds the largest singular value less than or equal to err
    sig = np.argmax(s < err)

    return svd_approx(A, sig)


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    #Accepts the image and color
    image = imread(filename) / 255
    colored = False
    if len(image.shape) == 3:
        colored = True

    if colored:
        #Calculate the low rank approximations of red, green, and blue
        R_s = svd_approx(image[:,:,0], s)
        G_s = svd_approx(image[:,:,1], s)
        B_s = svd_approx(image[:,:,2], s)

        #Plots Original
        plt.subplot(121)
        plt.title('Original')
        plt.imshow(image)
        plt.axis('off')

        #Plots Compressed
        plt.subplot(122)
        plt.title('Compressed')
        plt.imshow(np.dstack((np.clip(R_s[0], 0, 1), np.clip(G_s[0], 0, 1), np.clip(B_s[0], 0, 1))))

        plt.suptitle(f'Amount of Entries Saved: {np.size(image) - R_s[1] - G_s[1] - B_s[1]}')

    else:

        #Calculates the low rank approximation of gray
        M_s = svd_approx(image, s)

        #Plots Original
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        #Plots Compressed
        plt.subplot(122)
        plt.title('Compressed')
        plt.imshow(np.clip(M_s[0], 0, 1), cmap='gray')
        
        plt.suptitle(f'Amount of Entries Saved: {np.size(image) - M_s[1]}')

    plt.axis('off')
    plt.show()
