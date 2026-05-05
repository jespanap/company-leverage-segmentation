"""
pipeline/evaluation.py
-----------------------
Evaluación de clustering: corrección de etiquetas y métricas de clasificación.
"""

import numpy as np
from scipy.stats import mode as scipy_mode
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    silhouette_score,
)

from config import K_FINAL


def leverage_ground_truth(deuda_activos: np.ndarray) -> np.ndarray:
    """
    Etiqueta heurística basada en deuda/activos.
      < 0.4  → 0 (Bajo)
      0.4–0.7→ 1 (Medio)
      > 0.7  → 2 (Alto)
    """
    labels = np.ones(len(deuda_activos), dtype=int)
    labels[deuda_activos < 0.4] = 0
    labels[deuda_activos > 0.7] = 2
    return labels


def correct_labels(y_true: np.ndarray, y_pred: np.ndarray,
                   n_clusters: int = K_FINAL) -> np.ndarray:
    """
    Reasigna cada cluster al label verdadero más frecuente (votación mayoritaria).
    Devuelve y_pred corregido.
    """
    mapping = {}
    for c in range(n_clusters):
        subset = y_true[y_pred == c]
        mapping[c] = int(scipy_mode(subset, keepdims=True).mode[0]) if len(subset) else c
    return np.array([mapping[c] for c in y_pred]), mapping


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Devuelve accuracy, f1, precision y recall (weighted)."""
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "f1"       : f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall"   : recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def evaluate_clustering(X, y_true: np.ndarray, y_pred_raw: np.ndarray,
                         n_clusters: int = K_FINAL, label: str = "") -> dict:
    """
    Wrapper completo: calcula silhouette, corrige etiquetas y devuelve métricas.

    Returns
    -------
    dict con claves: silhouette, y_corr, mapping, accuracy, f1, precision, recall
    """
    sil = silhouette_score(X, y_pred_raw, sample_size=5000, random_state=42)
    y_corr, mapping = correct_labels(y_true, y_pred_raw, n_clusters)
    metrics = compute_metrics(y_true, y_corr)

    if label:
        print(f"\n  [{label}]")
        print(f"    Silhouette : {sil:.4f}")
        print(f"    Mapeo      : {mapping}")
        print(f"    Accuracy   : {metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  "
              f"Precisión={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}")

    return {"silhouette": sil, "y_corr": y_corr, "mapping": mapping, **metrics}
