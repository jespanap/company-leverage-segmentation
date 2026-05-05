"""
weight_linear_combination.py
-----------------------------
Utility to convert a raw score vector into normalised convex weights.
"""

import numpy as np


def weight(b: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Convert a score vector *b* into normalised convex weights.

    Steps
    -----
    1. Normalise b  →  x = b / sum(b)
    2. Raise to power n  →  x = x^n   (sharpens / flattens the distribution)
    3. Re-normalise  →  x = x / sum(x)

    Parameters
    ----------
    b : ndarray of shape (k,)
        Raw (non-negative) scores, e.g. extremality ranks.
    n : int, optional
        Sharpening exponent.  n=1 → proportional weights;
        larger n concentrates weight on the highest-scored kernels.

    Returns
    -------
    ndarray of shape (k,)
        Convex weights that sum to 1.
    """
    x = b / b.sum()
    x = np.power(x, n)
    x = x / x.sum()
    return x
