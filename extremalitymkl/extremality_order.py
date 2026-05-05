"""
extremality_order.py
---------------------
Core of the Extremality MKL ordering step.

The algorithm rotates the kernel-metric space so that the direction *u*
(encoding which metrics are "better") aligns with the first canonical axis,
then counts Pareto-dominance to rank kernels.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gram-Schmidt
# ---------------------------------------------------------------------------

def gram_schmidt(A: np.ndarray):
    """
    Gram-Schmidt QR factorisation of matrix *A*.

    Parameters
    ----------
    A : ndarray of shape (m, n)

    Returns
    -------
    Q : ndarray of shape (m, n)  – orthonormal columns
    R : ndarray of shape (n, n)  – upper-triangular
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def matrix_rotation(u: np.ndarray) -> np.ndarray:
    """
    Build the rotation matrix R_u that maps *u/||u||* to *e_1/||e_1||*.

    The construction follows the thesis: place *u* (normalised) as the first
    column of M_u, apply Gram-Schmidt to both M_u and the standard basis
    completion M_e, then set R_u = Q_e · Q_u^T.

    Parameters
    ----------
    u : ndarray of shape (n,)

    Returns
    -------
    R_u : ndarray of shape (n, n)  – orthogonal rotation matrix
    """
    n = len(u)
    I = np.eye(n)

    # M_u: first column is u/||u||, remaining columns are sign(u_i)*e_i
    M_u = np.sign(u)[:, None] * I          # broadcast signs across rows
    M_u[:, 0] = u / np.linalg.norm(u)

    # M_e: standard completion with first column = 1/sqrt(n)
    M_e = I.copy()
    M_e[:, 0] = np.ones(n) / np.sqrt(n)

    Q_u, _ = gram_schmidt(M_u)
    Q_e, _ = gram_schmidt(M_e)

    return Q_e @ Q_u.T


# ---------------------------------------------------------------------------
# Extremality order
# ---------------------------------------------------------------------------

def order_compar(kernels_metrics: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Assign an extremality rank to each kernel.

    After rotating the metric space so that *u* becomes the positive axis,
    kernel *i* receives a score equal to the number of kernels that
    **dominate** it (i.e. are at least as good in every rotated metric).
    A lower score → better kernel.

    Parameters
    ----------
    kernels_metrics : ndarray of shape (k, m)
        Metric values for *k* kernels across *m* metrics.
    u : ndarray of shape (m,)
        Direction vector encoding metric preferences
        (+1 = higher is better, -1 = lower is better).

    Returns
    -------
    extremality_order : ndarray of shape (k,)
        Dominance count for each kernel.
    """
    R_u = matrix_rotation(u)
    rotated = (R_u @ kernels_metrics.T).T    # shape (k, m)

    # Count how many kernels dominate each kernel i (vectorised)
    # rotated[j] >= rotated[i] for all metrics  ⟺  kernel j dominates i
    # shape: (k, k)  →  True where j dominates i
    dominance = np.all(rotated[:, None, :] >= rotated[None, :, :], axis=2)
    extremality_order = dominance.sum(axis=0).astype(float)

    return extremality_order
