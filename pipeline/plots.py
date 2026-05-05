"""
pipeline/plots.py
-----------------
Todas las funciones de visualización del proyecto, agrupadas por sección.

Cada función recibe los datos que necesita, guarda la figura en OUTPUT_DIR
y no devuelve nada (side-effect puro).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import silhouette_samples

from config import OUTPUT_DIR, MONETARY_COLS, LEVERAGE_FEATURES, K_FINAL

COLORS       = sns.color_palette("Set2", K_FINAL)
LABEL_NAMES  = ["Bajo (0)", "Medio (1)", "Alto (2)"]
_save = lambda name: plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")


# ═════════════════════════════════════════════════════════════
# EDA
# ═════════════════════════════════════════════════════════════

def plot_eda_distribucion(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["Año de Corte"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set(title="Empresas por Año de Corte", xlabel="Año", ylabel="Cantidad")
    axes[0].tick_params(axis="x", rotation=0)
    df["MACROSECTOR"].value_counts().plot(
        kind="barh", ax=axes[1], color="coral", edgecolor="white")
    axes[1].set(title="Empresas por Macrosector", xlabel="Cantidad")
    plt.tight_layout()
    _save("eda_01_distribucion_empresas.png")
    plt.close()
    print("  [OK] eda_01_distribucion_empresas.png")


def plot_eda_financieras(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flatten(), MONETARY_COLS):
        data = df[col][df[col] > 0]
        ax.hist(np.log1p(data), bins=60, color="teal", edgecolor="white", alpha=0.8)
        ax.set(title=f"log(1+{col})", xlabel="log(1+valor)", ylabel="Frecuencia")
    axes.flatten()[-1].axis("off")
    plt.suptitle("Distribución de Variables Financieras (escala log)", fontsize=13)
    plt.tight_layout()
    _save("eda_02_distribuciones_financieras.png")
    plt.close()
    print("  [OK] eda_02_distribuciones_financieras.png")


def plot_eda_boxplot_leverage(df: pd.DataFrame):
    sector_order = (df.groupby("MACROSECTOR")["TOTAL PASIVOS"]
                    .median().sort_values(ascending=False).index)
    df_pos = df[df["TOTAL ACTIVOS"] > 0].copy()
    df_pos["Pasivos/Activos"] = df_pos["TOTAL PASIVOS"] / df_pos["TOTAL ACTIVOS"]
    df_clip = df_pos[df_pos["Pasivos/Activos"].between(0, 5)]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_clip, x="MACROSECTOR", y="Pasivos/Activos",
                order=sector_order, ax=ax, palette="Set2", linewidth=0.8)
    ax.set(title="Ratio Pasivos/Activos por Macrosector", xlabel="", ylabel="Pasivos/Activos")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save("eda_03_boxplot_leverage_macrosector.png")
    plt.close()
    print("  [OK] eda_03_boxplot_leverage_macrosector.png")


def plot_eda_correlacion(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[MONETARY_COLS].corr()
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlación entre Variables Financieras")
    plt.tight_layout()
    _save("eda_04_correlacion.png")
    plt.close()
    print("  [OK] eda_04_correlacion.png")


def plot_eda_evolucion_temporal(df: pd.DataFrame):
    evol = df.groupby("Año de Corte")[
        ["TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO"]].median().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, color in zip(evol.columns[1:], ["steelblue", "coral", "seagreen"]):
        ax.plot(evol["Año de Corte"], evol[col], marker="o", label=col, color=color)
    ax.set(title="Mediana Activos/Pasivos/Patrimonio por Año",
           xlabel="Año", ylabel="Billones COP (mediana)")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    _save("eda_05_evolucion_temporal.png")
    plt.close()
    print("  [OK] eda_05_evolucion_temporal.png")


# ═════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ═════════════════════════════════════════════════════════════

def plot_prepro_distribuciones(X_scaled_df: pd.DataFrame):
    transformed_cols = [c + "_lm" for c in LEVERAGE_FEATURES]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flatten(), transformed_cols):
        ax.hist(X_scaled_df[col], bins=60, color="mediumpurple", edgecolor="white", alpha=0.85)
        ax.set(title=col.replace("_lm","").replace("_"," ").title(),
               xlabel="Valor escalado", ylabel="Frecuencia")
    axes.flatten()[-1].axis("off")
    plt.suptitle("Distribuciones Post-Procesamiento (log-modulus + RobustScaler)", fontsize=13)
    plt.tight_layout()
    _save("prepro_01_distribuciones_finales.png")
    plt.close()
    print("  [OK] prepro_01_distribuciones_finales.png")


def plot_prepro_pairplot(X_scaled_df: pd.DataFrame):
    transformed_cols = [c + "_lm" for c in LEVERAGE_FEATURES]
    sample = X_scaled_df[transformed_cols[:4]].sample(min(2000, len(X_scaled_df)), random_state=42)
    g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10})
    g.figure.suptitle("Pairplot – Features de Apalancamiento (muestra 2,000)", y=1.02)
    g.figure.savefig(os.path.join(OUTPUT_DIR, "prepro_02_pairplot_features.png"),
                     dpi=130, bbox_inches="tight")
    plt.close()
    print("  [OK] prepro_02_pairplot_features.png")


def plot_prepro_correlacion(X_scaled_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = X_scaled_df.corr()
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlación – Features Procesadas")
    plt.tight_layout()
    _save("prepro_03_correlacion_post.png")
    plt.close()
    print("  [OK] prepro_03_correlacion_post.png")


# ═════════════════════════════════════════════════════════════
# CLUSTERING
# ═════════════════════════════════════════════════════════════

def plot_codo_silhouette(k_range, inertias: list, sil_scores: list, best_k: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
    axes[0].axvline(x=3, color="red", linestyle="--", alpha=0.6, label="K elegido")
    axes[0].set(title="Método del Codo", xlabel="K", ylabel="Inercia")
    axes[0].legend()
    axes[1].plot(list(k_range), sil_scores, marker="s", color="darkorange")
    axes[1].axvline(x=best_k, color="red", linestyle="--", alpha=0.6,
                    label=f"K={best_k} (mejor)")
    axes[1].set(title="Silhouette Score", xlabel="K", ylabel="Score")
    axes[1].legend()
    plt.suptitle("Selección de K Óptimo", fontsize=13)
    plt.tight_layout()
    _save("cluster_01_codo_silhouette.png")
    plt.close()
    print("  [OK] cluster_01_codo_silhouette.png")


def plot_silhouette_analysis(X, labels: np.ndarray, sil_mean: float, title_suffix: str = ""):
    sil_vals = silhouette_samples(X, labels)
    fig, ax  = plt.subplots(figsize=(10, 6))
    y_lower  = 10
    for i in range(K_FINAL):
        ith = np.sort(sil_vals[labels == i])
        ax.fill_betweenx(np.arange(y_lower, y_lower + len(ith)), 0, ith,
                         facecolor=COLORS[i], edgecolor=COLORS[i], alpha=0.7,
                         label=f"Cluster {i} (n={len(ith):,})")
        y_lower += len(ith) + 10
    ax.axvline(x=sil_mean, color="red", linestyle="--", label=f"Media={sil_mean:.3f}")
    ax.set(title=f"Análisis Silhouette {title_suffix}",
           xlabel="Coeficiente Silhouette", ylabel="Muestra")
    ax.legend(loc="upper right"); ax.set_yticks([])
    plt.tight_layout()
    _save("cluster_02_silhouette_analisis.png")
    plt.close()
    print("  [OK] cluster_02_silhouette_analisis.png")


def plot_pca_comparacion(X_pca, var_exp, labels_km: np.ndarray, labels_emkl: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, labels, title in zip(axes,
                                  [labels_km, labels_emkl],
                                  ["K-Means estándar", "Kernel K-Means (EMKL)"]):
        for i in range(K_FINAL):
            mask = labels == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=8, alpha=0.4, color=COLORS[i], label=f"Cluster {i}")
        ax.set(title=f"PCA – {title}",
               xlabel=f"PC1 ({var_exp[0]*100:.1f}%)",
               ylabel=f"PC2 ({var_exp[1]*100:.1f}%)")
        ax.legend(markerscale=3)
    plt.suptitle("Comparación de Segmentación: K-Means vs EMKL", fontsize=13)
    plt.tight_layout()
    _save("cluster_03_pca_comparacion.png")
    plt.close()
    print("  [OK] cluster_03_pca_comparacion.png")


def plot_perfil_heatmap(profile: pd.DataFrame, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    norm = (profile - profile.mean()) / (profile.std() + 1e-9)
    sns.heatmap(norm.T, annot=profile.T.round(3), fmt=".3f",
                cmap="RdYlGn_r", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set(title=title, xlabel="Cluster", ylabel="Feature")
    plt.tight_layout()
    _save(filename)
    plt.close()
    print(f"  [OK] {filename}")


def plot_distribucion_macrosector(df: pd.DataFrame, cluster_col: str, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    cross = pd.crosstab(df["MACROSECTOR"], df[cluster_col], normalize="index") * 100
    cross.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set(title=title, xlabel="", ylabel="% dentro del macrosector")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Cluster")
    plt.tight_layout()
    _save(filename)
    plt.close()
    print(f"  [OK] {filename}")


# ═════════════════════════════════════════════════════════════
# EMKL
# ═════════════════════════════════════════════════════════════

def plot_emkl_pesos(w_1: np.ndarray, w_2: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, w, color, title in zip(
            axes, [w_1, w_2],
            ["steelblue", "coral"],
            ["Pesos Naturales (ordenados)", "Pesos Anti-Naturales (ordenados)"]):
        ax.bar(np.arange(len(w)), np.sort(w)[::-1], color=color, edgecolor="white")
        ax.set(title=title, xlabel="Kernel (rank)", ylabel="Peso")
    plt.suptitle("Distribución de Pesos EMKL", fontsize=13)
    plt.tight_layout()
    _save("emkl_01_distribucion_pesos.png")
    plt.close()
    print("  [OK] emkl_01_distribucion_pesos.png")


def plot_emkl_kernel_heatmap(K_natural: np.ndarray, K_anti: np.ndarray,
                              deuda_activos_sample: np.ndarray):
    sort_idx = np.argsort(deuda_activos_sample)
    n = len(sort_idx)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, K, title in zip(axes,
                              [K_natural[np.ix_(sort_idx, sort_idx)],
                               K_anti[np.ix_(sort_idx, sort_idx)]],
                              ["Kernel Natural EMKL", "Kernel Anti-Natural EMKL"]):
        im = ax.imshow(K, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set(title=f"{title}\n({n:,} empresas, ord. por deuda/activos)",
               xlabel="Empresa", ylabel="Empresa")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    _save("emkl_02_kernel_heatmap.png")
    plt.close()
    print("  [OK] emkl_02_kernel_heatmap.png")


# ═════════════════════════════════════════════════════════════
# EVALUACIÓN
# ═════════════════════════════════════════════════════════════

def plot_metricas_comparacion(metrics_km: dict, metrics_emkl: dict):
    names  = ["Accuracy", "F1", "Precisión", "Recall"]
    keys   = ["accuracy", "f1", "precision", "recall"]
    vals_km   = [metrics_km[k]   for k in keys]
    vals_emkl = [metrics_emkl[k] for k in keys]
    x, w  = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, vals_km,   w, label="K-Means",        color="steelblue",  alpha=0.85)
    b2 = ax.bar(x + w/2, vals_emkl, w, label="EMKL (natural)", color="darkorange", alpha=0.85)
    ax.set(ylabel="Score",
           title="Comparación de Métricas – K-Means vs EMKL (con corrección de etiquetas)",
           ylim=(0, 1.15))
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend()
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _save("cluster_07_metricas_comparacion.png")
    plt.close()
    print("  [OK] cluster_07_metricas_comparacion.png")


def plot_confusion_matrices(y_true, y_km_corr, y_emkl_corr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_pred, title in zip(axes,
                                  [y_km_corr, y_emkl_corr],
                                  ["K-Means", "EMKL Natural"]):
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                               display_labels=LABEL_NAMES).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Matriz de Confusión – {title} (corregido)")
    plt.tight_layout()
    _save("cluster_08_confusion_matrix.png")
    plt.close()
    print("  [OK] cluster_08_confusion_matrix.png")


def plot_evolucion_temporal(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, title in zip(axes,
                               ["cluster_kmeans", "cluster_emkl"],
                               ["K-Means", "EMKL"]):
        evol = (df.groupby(["Año de Corte", col]).size()
                .reset_index(name="count")
                .pivot(index="Año de Corte", columns=col, values="count")
                .fillna(0))
        pct = evol.div(evol.sum(axis=1), axis=0) * 100
        for c in pct.columns:
            ax.plot(pct.index, pct[c], marker="o", label=f"Cluster {c}", color=COLORS[c])
        ax.set(title=f"Evolución Temporal – {title} (% por año)",
               xlabel="Año", ylabel="% de empresas")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    _save("cluster_09_evolucion_temporal.png")
    plt.close()
    print("  [OK] cluster_09_evolucion_temporal.png")