"""
main.py
-------
Orquestador del pipeline de segmentación por apalancamiento.

Flujo
-----
  1. Carga
  2. EDA
  3. Preprocesamiento  (MonetaryCleaner → LeverageFeatureEngineer
                        → OutlierIQRRemover → LogModulusScaler)
  4. Clustering K-Means (línea de base)
  5. Clustering EMKL    (Kernel K-Means sobre kernel combinado)
  6. Evaluación y comparación
  7. Exportación
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Añade la raíz del proyecto al path para los módulos EMKL ─
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

from config import (
    FILE_PATH, OUTPUT_DIR, MONETARY_COLS, LEVERAGE_FEATURES, ID_COLS,
    K_FINAL, K_RANGE, CLUSTER_NAMES,
)
from pipeline.transformers import (
    MonetaryCleaner, LeverageFeatureEngineer,
    OutlierIQRRemover, LogModulusScaler,
)
from pipeline.clustering  import KMeansClusterer, EMKLClusterer
from pipeline.evaluation  import evaluate_clustering, leverage_ground_truth
from pipeline import plots

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════
# 1. CARGA
# ═════════════════════════════════════════════════════════════
print("=" * 60)
print("1. CARGA DE DATOS")
print("=" * 60)

df_raw = pd.read_csv(FILE_PATH)
print(f"  Filas: {df_raw.shape[0]:,}  |  Columnas: {df_raw.shape[1]}")

# ═════════════════════════════════════════════════════════════
# 2. EDA  (sobre datos crudos)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. EDA")
print("=" * 60)

# Limpieza mínima para el EDA (solo tipos, sin filtros)
df_eda = MonetaryCleaner().fit_transform(df_raw)
print(df_eda[MONETARY_COLS + ["Año de Corte"]].describe().T.to_string())

plots.plot_eda_distribucion(df_eda)
plots.plot_eda_financieras(df_eda)
plots.plot_eda_boxplot_leverage(df_eda)
plots.plot_eda_correlacion(df_eda)
plots.plot_eda_evolucion_temporal(df_eda)

# ═════════════════════════════════════════════════════════════
# 3. PREPROCESAMIENTO
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. PREPROCESAMIENTO")
print("=" * 60)

preprocess_pipeline = Pipeline([
    ("clean",      MonetaryCleaner()),
    ("features",   LeverageFeatureEngineer()),
    ("outliers",   OutlierIQRRemover()),
    ("scale",      LogModulusScaler()),
])

df_processed = preprocess_pipeline.fit_transform(df_raw)
scaler_step  = preprocess_pipeline.named_steps["scale"]
transformed_cols = scaler_step.transformed_cols   # [feat_lm, ...]

X_scaled_df = df_processed[transformed_cols]
print(f"  Shape final: {X_scaled_df.shape}")
print(f"  NaN residuales: {X_scaled_df.isnull().sum().sum()}")

plots.plot_prepro_distribuciones(X_scaled_df)
plots.plot_prepro_pairplot(X_scaled_df)
plots.plot_prepro_correlacion(X_scaled_df)

# Exportar dataset procesado
output_cols = ID_COLS + LEVERAGE_FEATURES + transformed_cols
df_processed[output_cols].to_csv(
    os.path.join(OUTPUT_DIR, "dataset_procesado_apalancamiento.csv"), index=False)
print("  [OK] dataset_procesado_apalancamiento.csv")

# ═════════════════════════════════════════════════════════════
# 4. CLUSTERING K-MEANS (línea de base)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. CLUSTERING K-MEANS")
print("=" * 60)

# Selección de K
print("  Buscando K óptimo ...")
inertias, sil_scores, best_k = KMeansClusterer.best_k(X_scaled_df, K_RANGE)
plots.plot_codo_silhouette(K_RANGE, inertias, sil_scores, best_k)
print(f"  K con mejor silhouette: {best_k}  |  K usado: {K_FINAL}")

# Entrenamiento
km_clusterer = KMeansClusterer(n_clusters=K_FINAL)
labels_km    = km_clusterer.fit_predict(X_scaled_df)
df_processed["cluster_kmeans"] = labels_km

plots.plot_silhouette_analysis(
    X_scaled_df, labels_km,
    sil_mean=sil_scores[K_FINAL - list(K_RANGE)[0]],
    title_suffix="– K-Means"
)

# PCA (se reutiliza más adelante con EMKL)
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_df)
var_exp = pca.explained_variance_ratio_

# ═════════════════════════════════════════════════════════════
# 5. EMKL – KERNEL K-MEANS
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. EXTREMALITY MKL")
print("=" * 60)

# Etiqueta de supervisión débil {-1, +1} para evaluación de kernels
y_weak = np.where(df_processed["deuda_activos"].values < 0.55, 1.0, -1.0)

emkl = EMKLClusterer(n_clusters=K_FINAL)
labels_emkl = emkl.fit_predict(X_scaled_df.values, y_weak)
df_processed["cluster_emkl"] = labels_emkl

plots.plot_emkl_pesos(emkl.kernel_weights_.w_1, emkl.kernel_weights_.w_2)

# Heatmap: K_natural_ y K_anti_ son de la muestra EMKL
deuda_sample = df_processed["deuda_activos"].values[emkl.sample_idx_]
plots.plot_emkl_kernel_heatmap(emkl.K_natural_, emkl.K_anti_, deuda_sample)

# ═════════════════════════════════════════════════════════════
# 6. EVALUACIÓN Y COMPARACIÓN
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. EVALUACIÓN")
print("=" * 60)

y_true = leverage_ground_truth(df_processed["deuda_activos"].values)

results_km   = evaluate_clustering(X_scaled_df, y_true, labels_km,   K_FINAL, "K-Means")
results_emkl = evaluate_clustering(X_scaled_df, y_true, labels_emkl, K_FINAL, "EMKL")

# Perfiles de cluster
profile_km   = df_processed.groupby("cluster_kmeans")[LEVERAGE_FEATURES].median()
profile_emkl = df_processed.groupby("cluster_emkl")[LEVERAGE_FEATURES].median()

# Gráficas de comparación
plots.plot_pca_comparacion(X_pca, var_exp, labels_km, labels_emkl)
plots.plot_perfil_heatmap(profile_km,   "Perfil Clusters K-Means",  "cluster_05_perfil_heatmap_kmeans.png")
plots.plot_perfil_heatmap(profile_emkl, "Perfil Clusters EMKL",     "emkl_04_perfil_heatmap_emkl.png")
plots.plot_distribucion_macrosector(df_processed, "cluster_kmeans",
    "cluster_06_macrosector_kmeans.png", "Distribución Clusters K-Means por Macrosector (%)")
plots.plot_distribucion_macrosector(df_processed, "cluster_emkl",
    "emkl_05_macrosector_emkl.png",     "Distribución Clusters EMKL por Macrosector (%)")
plots.plot_metricas_comparacion(results_km, results_emkl)
plots.plot_confusion_matrices(y_true, results_km["y_corr"], results_emkl["y_corr"])
plots.plot_evolucion_temporal(df_processed)

# ═════════════════════════════════════════════════════════════
# 7. EXPORTACIÓN
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. EXPORTACIÓN")
print("=" * 60)

df_processed["cluster_kmeans_corr"]  = results_km["y_corr"]
df_processed["cluster_emkl_corr"]    = results_emkl["y_corr"]
df_processed["perfil_kmeans"]        = df_processed["cluster_kmeans_corr"].map(CLUSTER_NAMES)
df_processed["perfil_emkl"]          = df_processed["cluster_emkl_corr"].map(CLUSTER_NAMES)

final_cols = (ID_COLS + LEVERAGE_FEATURES
              + ["cluster_kmeans", "cluster_kmeans_corr", "perfil_kmeans",
                 "cluster_emkl",   "cluster_emkl_corr",   "perfil_emkl"])

df_final = df_processed[final_cols]
df_final.to_csv(os.path.join(OUTPUT_DIR, "dataset_final_clusters.csv"), index=False)
print(f"  [OK] dataset_final_clusters.csv  ({df_final.shape[0]:,} filas)")
print(f"\n  Distribución K-Means:\n{df_final['perfil_kmeans'].value_counts().to_string()}")
print(f"\n  Distribución EMKL:\n{df_final['perfil_emkl'].value_counts().to_string()}")

# ═════════════════════════════════════════════════════════════
# RESUMEN
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESUMEN COMPARATIVO")
print("=" * 60)
print(f"\n  {'Método':<22} {'Silhouette':>12} {'Accuracy':>10} {'F1':>8}")
print(f"  {'-'*54}")
print(f"  {'K-Means':<22} {results_km['silhouette']:>12.4f} "
      f"{results_km['accuracy']:>10.4f} {results_km['f1']:>8.4f}")
print(f"  {'EMKL (natural)':<22} {results_emkl['silhouette']:>12.4f} "
      f"{results_emkl['accuracy']:>10.4f} {results_emkl['f1']:>8.4f}")
print("\n  Pipeline completo FINALIZADO.")