"""
pipeline/clustering.py
----------------------
Wrappers de clustering sklearn-compatibles.

Clases
------
- KMeansClusterer    : wrapper de KMeans con selección automática de K
- KernelKMeans       : Kernel K-Means sobre una matriz de kernel precomputada
- EMKLClusterer      : construye el kernel EMKL y aplica Kernel K-Means
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import K_FINAL, K_RANGE, EMKL_NUM_KERNELS, EMKL_T, EMKL_MAX_DEGREE, EMKL_N


# ─────────────────────────────────────────────────────────────
class KMeansClusterer(BaseEstimator, ClusterMixin):
    """KMeans con n_clusters fijo."""

    def __init__(self, n_clusters: int = K_FINAL, random_state: int = 42):
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self._model       = None

    def fit(self, X, y=None):
        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20, max_iter=500,
        )
        self._model.fit(X)
        self.labels_   = self._model.labels_
        self.inertia_  = self._model.inertia_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    @staticmethod
    def best_k(X, k_range=K_RANGE, sample_size: int = 5000, random_state: int = 42):
        """Devuelve (inertias, sil_scores, best_k) para el rango dado."""
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil_scores.append(
                silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)
            )
        best = list(k_range)[int(np.argmax(sil_scores))]
        return inertias, sil_scores, best


# ─────────────────────────────────────────────────────────────
class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-Means sobre una matriz de kernel precomputada.

    Minimiza la inercia en el espacio de características inducido por K.
    """

    def __init__(self, n_clusters: int = K_FINAL,
                 n_init: int = 10, max_iter: int = 300, random_state: int = 42):
        self.n_clusters   = n_clusters
        self.n_init       = n_init
        self.max_iter     = max_iter
        self.random_state = random_state

    def fit(self, K: np.ndarray, y=None):
        self.labels_ = self._fit_kernel(K)
        return self

    def fit_predict(self, K: np.ndarray, y=None):
        return self.fit(K).labels_

    def _fit_kernel(self, K: np.ndarray) -> np.ndarray:
        n      = K.shape[0]
        rng    = np.random.default_rng(self.random_state)
        diag_K = np.diag(K)
        best_labels, best_score = None, np.inf

        for _ in range(self.n_init):
            labels = rng.integers(0, self.n_clusters, size=n)

            for _ in range(self.max_iter):
                cross = np.zeros((n, self.n_clusters))
                means = np.zeros(self.n_clusters)

                for c in range(self.n_clusters):
                    idx_c = np.where(labels == c)[0]
                    if len(idx_c) == 0:
                        continue
                    cross[:, c] = K[:, idx_c].mean(axis=1)
                    means[c]    = K[np.ix_(idx_c, idx_c)].mean()

                dist       = diag_K[:, None] - 2.0 * cross + means[None, :]
                new_labels = dist.argmin(axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels

            score = sum(
                diag_K[labels == c].sum()
                - K[np.ix_(labels == c, labels == c)].sum() / max(1, (labels == c).sum())
                for c in range(self.n_clusters)
            )
            if score < best_score:
                best_score, best_labels = score, labels.copy()

        return best_labels


# ─────────────────────────────────────────────────────────────
class EMKLClusterer(BaseEstimator, ClusterMixin):
    """
    Pipeline completo EMKL:
      1. Genera kernels débiles polinomiales
      2. Calcula pesos por extremalidad
      3. Construye el kernel combinado normalizado
      4. Aplica Kernel K-Means

    Requiere que `fit` reciba (X_scaled, y_weak) donde y_weak son
    etiquetas {-1, +1} para la evaluación de métricas de kernel.

    Para datasets grandes se trabaja sobre una muestra (`sample_size`):
    los pesos se aprenden en la muestra y luego se propagan al dataset
    completo mediante asignación por vecino más cercano en espacio original.
    """

    def __init__(self, n_clusters: int = K_FINAL,
                 num_kernels: int = EMKL_NUM_KERNELS,
                 t: int = EMKL_T,
                 max_degree: int = EMKL_MAX_DEGREE,
                 n: int = EMKL_N,
                 sample_size: int = 5_000,
                 random_state: int = 42):
        self.n_clusters   = n_clusters
        self.num_kernels  = num_kernels
        self.t            = t
        self.max_degree   = max_degree
        self.n            = n
        self.sample_size  = sample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from src.weak_polynomial_kernel import create_weak_kernels
        from extremalitymkl.extremality_weights import kernel_extremaly_weights

        rng = np.random.default_rng(self.random_state)

        # ── Muestra para construir los kernels ────────────────
        n_total = X.shape[0]
        if n_total > self.sample_size:
            idx = rng.choice(n_total, self.sample_size, replace=False)
            print(f"  [EMKL] Dataset grande ({n_total:,} filas). "
                  f"Usando muestra de {self.sample_size:,} para construir kernels.")
        else:
            idx = np.arange(n_total)

        X_sample = X[idx]
        y_sample = y[idx]

        # ── Kernels débiles sobre la muestra ──────────────────
        np.random.seed(self.random_state)
        KL_sample = create_weak_kernels(X_sample, num_kernels=self.num_kernels,
                                        t=self.t, max_degree=self.max_degree)

        # ── Pesos por extremalidad (aprendidos en la muestra) ─
        self.kernel_weights_ = kernel_extremaly_weights(KL_sample, y_sample, n=self.n)
        self.sample_idx_      = idx

        # ── Kernels combinados (solo en la muestra) ───────────
        K_combined      = np.einsum("ijk,i->jk", KL_sample, self.kernel_weights_.w_1)
        self.K_natural_ = self._normalize(K_combined)
        K_anti          = np.einsum("ijk,i->jk", KL_sample, self.kernel_weights_.w_2)
        self.K_anti_    = self._normalize(K_anti)

        # ── Kernel K-Means sobre la muestra ───────────────────
        sample_labels = KernelKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        ).fit_predict(self.K_natural_)

        # ── Propagar etiquetas al dataset completo ────────────
        if n_total > self.sample_size:
            print(f"  [EMKL] Propagando etiquetas al dataset completo ({n_total:,} filas) ...")
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            knn.fit(X_sample, sample_labels)
            self.labels_ = knn.predict(X)
        else:
            self.labels_ = sample_labels

        return self

    def fit_predict(self, X: np.ndarray, y: np.ndarray):
        return self.fit(X, y).labels_

    @staticmethod
    def _normalize(K: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(K))
        d[d == 0] = 1.0
        return K / np.outer(d, d)