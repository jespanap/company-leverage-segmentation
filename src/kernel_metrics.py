"""
kernel_metrics.py
-----------------
Kernel quality metrics used by the Extremality MKL algorithm.

Available metrics
-----------------
- kernel_alignment   : Normalized alignment between a kernel and the ideal kernel.
- kernel_polarization: Vectorised pairwise margin contribution.
- FSM                : Feature Space Measure (intra/inter-class similarity ratio).
- complex_ratio      : Trace of the kernel matrix (complexity proxy).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ideal_kernel(y: np.ndarray) -> np.ndarray:
    """
    Build the ideal (+1 / -1) kernel matrix from a label vector.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary labels in {-1, +1}.

    Returns
    -------
    ndarray of shape (n, n)  –  outer(y, y)
    """
    return np.outer(y, y)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def complex_ratio(K: np.ndarray, y: np.ndarray = None) -> float:
    """
    Complexity proxy: trace of the kernel matrix.

    Parameters
    ----------
    K : ndarray of shape (n, n)
    y : ignored (kept for a uniform API signature across all metrics)

    Returns
    -------
    float
    """
    return float(np.trace(K))


def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Uncentered kernel alignment with the ideal kernel.

        alignment = <K, K_ideal>_F  /  (||K||_F * n)

    Parameters
    ----------
    K : ndarray of shape (n, n)
    y : ndarray of shape (n,)  – binary labels in {-1, +1}

    Returns
    -------
    float
    """
    K_ideal = ideal_kernel(y)
    numerator   = np.sum(K * K_ideal)          # Frobenius inner product
    denominator = np.linalg.norm(K, "fro") * len(y)
    return float(numerator / denominator)


# Backwards-compatible alias that fixes the original typo
kernel_aligment = kernel_alignment


def kernel_polarization(K: np.ndarray, y: np.ndarray) -> float:
    """
    Vectorised kernel polarization.

        polarization = Σ_{i<j}  -y_i * y_j * (K_ii + K_jj - 2·K_ij)

    The original O(n²) double Python loop is replaced by NumPy broadcasting,
    giving a ~100× speed-up on typical dataset sizes.

    Parameters
    ----------
    K : ndarray of shape (n, n)
    y : ndarray of shape (n,)  – binary labels in {-1, +1}

    Returns
    -------
    float
    """
    diag_K = np.diag(K)                                    # (n,)
    D = diag_K[:, None] + diag_K[None, :] - 2.0 * K       # (n, n)  margin matrix
    A = -np.outer(y, y) * D                                # (n, n)
    # Sum only the strictly upper triangle to count each pair once
    return float(np.sum(np.triu(A, k=1)))


def FSM(K: np.ndarray, y: np.ndarray) -> float:
    """
    Feature Space Measure: ratio of intra/inter-class similarity spread.

    Parameters
    ----------
    K : ndarray of shape (n, n)
    y : ndarray of shape (n,)  – binary labels in {-1, +1}

    Returns
    -------
    float
    """
    mask_neg = y == -1
    mask_pos = y == 1
    n_neg = int(mask_neg.sum())
    n_pos = int(mask_pos.sum())

    # Per-sample mean similarities (row averages within each block)
    d_i = K[np.ix_(mask_neg, mask_neg)].sum(axis=1) / n_neg   # neg intra
    a_i = K[np.ix_(mask_pos, mask_pos)].sum(axis=1) / n_pos   # pos intra
    c_i = K[np.ix_(mask_neg, mask_pos)].sum(axis=1) / n_pos   # neg→pos inter
    b_i = K[np.ix_(mask_pos, mask_neg)].sum(axis=1) / n_neg   # pos→neg inter

    A = float(a_i.mean())
    B = float(b_i.mean())
    C = float(c_i.mean())
    D = float(d_i.mean())

    phi_sq = A + D - B - C   # normalization factor (must be > 0)

    aux_1 = np.sum((b_i - a_i + A - B) ** 2) / (phi_sq * (n_pos - 1))
    aux_2 = np.sum((c_i - d_i + D - C) ** 2) / (phi_sq * (n_neg - 1))

    return float((np.sqrt(aux_1) + np.sqrt(aux_2)) / np.sqrt(phi_sq))
