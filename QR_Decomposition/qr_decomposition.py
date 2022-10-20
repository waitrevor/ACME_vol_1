# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name> Trevor Wai
<Class> Section
<Date> 10/18/22
"""
import numpy as np
import scipy.linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    #Initializes variables
    m, n = A.shape
    Q = A.copy().astype('float64')
    R = np.zeros((n,n))

    
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        #Normalize the ith column of Q
        Q[:,i] = Q[:,i] / R[i,i]

        #Loops through 
        for j in range(i+1, n):
            R[i,j] = np.dot((Q[:,j].T), Q[:,i])
            #Orthoganlize the ith column of Q
            Q[:,j] = Q[:,j] - (R[i,j] * Q[:,i])

    return Q, R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    #Calculates the determinate using qr decomp
    return abs(np.prod(np.diag(qr_gram_schmidt(A)[1])))


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    #Initialize Variables
    m,n = np.shape(A)
    Q,R = qr_gram_schmidt(A)
    x = np.zeros(n)

    #Solves for y
    y = np.transpose(Q) @ b

    #Uses y to solve for x
    for j in reversed(range(n)):
        x[j] = (y[j] - (R[j,:] @ x)) / R[j,j]

    return x


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    #Initializes variables
    m,n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        u = np.copy(R[k:,k])
        u[0] = u[0] + np.sign(u[0]) * la.norm(u)
        #Normalize u
        u = u / la.norm(u)
        #Apply the reflection to R
        R[k:,k:] = R[k:,k:] - 2 * np.outer(u, (np.dot(np.transpose(u), R[k:,k:])))
        #Apply the reflection to Q
        Q[k:,:] = Q[k:,:] - 2 * np.outer(u, (np.dot(np.transpose(u), Q[k:,:])))

    return np.transpose(Q), R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    sign = lambda x: 1 if x >=0 else -1
    #Initialize variavles
    m,n = A.shape
    H = A.copy()
    Q = np.eye(m)
    for k in range(n - 2):
        u = np.copy(H[k+1:,k])
        u[0] = u[0] + sign(u[0]) * la.norm(u)
        u = u / la.norm(u)
        #Apply Qk to H
        H[k+1:,k:] = H[k+1:, k:] - 2 * np.outer(u, np.dot(np.transpose(u), H[k+1:,k:]))
        #Apply Qk transpose to H
        H[:,k+1:] = H[:,k+1:] - 2 * np.outer(np.dot(H[:,k+1:], u), np.transpose(u))
        #Apply Qk to Q
        Q[k+1:,:] = Q[k+1:,:] - 2 * np.outer(u, np.dot(np.transpose(u), Q[k+1:,:]))

    return H, np.transpose(Q)

#testing
A = np.random.random((3,3))
H, Q = hessenberg(A)
print(np.allclose(np.triu(H, -1), H))
print(np.allclose(Q @ H @ Q.T, A))
print(A)
print(Q @ H @ Q.T)