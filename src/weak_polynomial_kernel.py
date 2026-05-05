"""
weak_polynomial_kernel.py
--------------------------
Generates families of weak polynomial kernels by randomly sampling feature
subsets and polynomial degrees.
"""

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel


def create_weak_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    num_kernels: int = 3,
    t: int = 5,
    max_degree: int = 3,
):
    """
    Build *num_kernels* weak polynomial kernels from random feature subsets.

    For each kernel
    ---------------
    * A random subset of at most *t* feature indices is drawn **with replacement**
      (allowing feature repetition, which increases kernel diversity).
    * A random polynomial degree in [1, max_degree] is chosen.
    * ``sklearn.metrics.pairwise.polynomial_kernel`` is applied with
      ``coef0=0`` and ``gamma=1``.

    Parameters
    ----------
    X_train     : ndarray of shape (n_train, d)
    X_test      : ndarray of shape (n_test, d), optional
    num_kernels : int
        Number of weak kernels to generate.
    t           : int
        Maximum number of (possibly repeated) features per kernel.
    max_degree  : int
        Maximum polynomial degree.

    Returns
    -------
    KL_train : ndarray of shape (num_kernels, n_train, n_train)
    KL_test  : ndarray of shape (num_kernels, n_test,  n_train)  or None
    """
    n_features = X_train.shape[1]
    KL_train = []
    KL_test  = [] if X_test is not None else None

    for _ in range(num_kernels):
        # Random feature subset (with repetition, size in [1, t])
        k = np.random.randint(1, t + 1)
        idx = np.random.randint(0, n_features, size=k)

        degree = np.random.randint(1, max_degree + 1)

        X1 = X_train[:, idx]
        KL_train.append(polynomial_kernel(X1, degree=degree, coef0=0, gamma=1))

        if X_test is not None:
            X2 = X_test[:, idx]
            KL_test.append(polynomial_kernel(X2, X1, degree=degree, coef0=0, gamma=1))

    KL_train = np.array(KL_train)
    if X_test is not None:
        KL_test = np.array(KL_test)

    return (KL_train, KL_test) if X_test is not None else KL_train
