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
    m, n = A.shape
    Q = A.copy().astype('float64')
    R = np.zeros((n,n))
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i+1, n):
            R[i,j] = np.dot((Q[:,j].T), Q[:,i])
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
    m,n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for k in range(n - 1):
        u = np.copy(R[k:,k])
        u[0] = u[0] + np.sign(u[0]) * la.norm(u)
        u = u / la.norm(u)
        R[k:,k:] = R[k:,k:] - 2 * np.outer(u, (np.dot(np.transpose(u), R[k:,k:])))
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
    m,n = A.shape
    H = A.copy()
    Q = np.eye(m)
    for k in range(n - 3):
        u = np.copy(H[k+1:,k])
        u[0] = u[0] + np.sign(u[0]) * la.norm(u)
        u = u / la.norm(u)
        H[k+1:,k:] = H[k+1:, k:] - 2 * np.outer(u, np.dot(np.transpose(u), H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2 * np.outer(np.dot(H[:,k+1:], u), np.transpose(u))
        Q[k+1:,:] = Q[k+1:,:] - 2 * np.outer(u, np.dot(np.transpose(u), Q[k+1:,:]))

    return H, np.transpose(Q)


#Testing
A = np.random.random((4,4))
b = np.random.random(4)
Q = hessenberg(A)[1]
H = hessenberg(A)[0]
print(np.allclose(Q @ H @ Q.T, A))