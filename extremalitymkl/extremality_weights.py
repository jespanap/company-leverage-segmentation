"""
extremality_weights.py
-----------------------
Computes kernel combination weights using extremality-based ordering.

Two complementary weight vectors are produced:
  w_1  ("natural")     – concentrates mass on kernels with the best metrics.
  w_2  ("anti-natural")– concentrates mass on kernels with the worst metrics
                         (useful as a contrastive / regularisation baseline).
"""

import numpy as np

from src.kernel_metrics import (
    kernel_alignment,
    kernel_polarization,
    FSM,
    complex_ratio,
)
from src.weight_linear_combination import weight
from .extremality_order import order_compar


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# All available metrics  (must accept (K, y) with y possibly None)
METRICS = {
    "alignment"   : kernel_alignment,
    "polarization": kernel_polarization,
    "FSM"         : FSM,
    "complex_ratio": complex_ratio,
}

# Direction convention: +1 → higher value is better; -1 → lower is better
DIRECTIONS = {
    "alignment"   :  1,
    "polarization":  1,
    "FSM"         :  1,
    "complex_ratio": -1,
}

# Subset used by default in kernel_extremaly_weights
DEFAULT_METRICS = {
    "alignment": kernel_alignment,
    "FSM"      : FSM,
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def metrics_kernels(
    KL_train: np.ndarray,
    y_train: np.ndarray,
    metrics: dict = None,
):
    """
    Evaluate all selected metrics on every kernel in *KL_train*.

    Parameters
    ----------
    KL_train : ndarray of shape (k, n, n)
    y_train  : ndarray of shape (n,)  – binary labels in {-1, +1}
    metrics  : dict {name: callable}, optional
               Defaults to all metrics in METRICS.

    Returns
    -------
    measures   : ndarray of shape (k, m)  – metric values
    directions : ndarray of shape (m,)    – +1 / -1 per metric
    """
    if metrics is None:
        metrics = METRICS

    k = KL_train.shape[0]
    m = len(metrics)
    measures   = np.zeros((k, m))
    directions = np.array([DIRECTIONS[name] for name in metrics])

    for i, (name, fn) in enumerate(metrics.items()):
        for j in range(k):
            measures[j, i] = fn(KL_train[j], y_train)

    return measures, directions


# ---------------------------------------------------------------------------
# Weight container
# ---------------------------------------------------------------------------

class KernelWeights:
    """
    Stores the two complementary weight vectors produced by
    :func:`kernel_extremaly_weights`.

    Attributes
    ----------
    w_1 : ndarray – natural weights   (best kernels get more weight)
    w_2 : ndarray – anti-natural weights (worst kernels get more weight)
    """

    def __init__(self, w_1: np.ndarray, w_2: np.ndarray):
        self.w_1 = w_1
        self.w_2 = w_2

    def __repr__(self):
        return (
            f"KernelWeights(\n"
            f"  w_1={np.round(self.w_1, 4)},\n"
            f"  w_2={np.round(self.w_2, 4)}\n)"
        )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def kernel_extremaly_weights(
    KL_train: np.ndarray,
    y_train: np.ndarray,
    metrics: dict = None,
    n: int = 1,
) -> KernelWeights:
    """
    Compute natural and anti-natural kernel combination weights.

    Parameters
    ----------
    KL_train : ndarray of shape (k, n_train, n_train)
        Stack of kernel matrices on the training set.
    y_train  : ndarray of shape (n_train,)
        Binary labels in {-1, +1}.
    metrics  : dict, optional
        Metrics used for ordering. Defaults to {alignment, FSM}.
    n        : int, optional
        Sharpening exponent for :func:`weight`. Default 1.

    Returns
    -------
    KernelWeights
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    measures, directions = metrics_kernels(KL_train, y_train, metrics)

    # Natural order: kernels better in the direction of 'directions' rank first
    order_1 = order_compar(measures, directions)
    w_1 = weight(len(order_1) - order_1, n)   # lower dominance count → higher weight

    # Anti-natural order: flip directions
    order_2 = order_compar(measures, -directions)
    w_2 = weight(order_2, n)

    return KernelWeights(w_1, w_2)
