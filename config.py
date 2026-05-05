"""
config.py
---------
Centraliza todas las constantes del proyecto.
Edita este archivo para cambiar rutas, parámetros o columnas.
"""

import os

# ── Rutas ─────────────────────────────────────────────────────
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "10.000_Empresas_mas_Grandes_del_País_20260210.csv")
OUTPUT_DIR = "out"

# ── Columnas monetarias en el CSV original ────────────────────
MONETARY_COLS = [
    "INGRESOS OPERACIONALES",
    "GANANCIA (PÉRDIDA)",
    "TOTAL ACTIVOS",
    "TOTAL PASIVOS",
    "TOTAL PATRIMONIO",
]

# ── Features de apalancamiento a construir ────────────────────
LEVERAGE_FEATURES = [
    "deuda_activos",
    "deuda_patrimonio",
    "multiplicador_cap",
    "cobertura_ingresos",
    "margen_neto",
    "roa",
    "roe",
]

# ── Columnas de identidad a conservar en el output ────────────
ID_COLS = ["NIT", "RAZÓN SOCIAL", "MACROSECTOR", "REGIÓN", "Año de Corte"]

# ── Clustering ────────────────────────────────────────────────
K_FINAL    = 3
K_RANGE    = range(2, 11)

CLUSTER_NAMES = {
    0: "Bajo Apalancamiento",
    1: "Apalancamiento Medio",
    2: "Alto Apalancamiento",
}

# ── EMKL ──────────────────────────────────────────────────────
EMKL_NUM_KERNELS = 30
EMKL_T           = 4
EMKL_MAX_DEGREE  = 3
EMKL_N           = 2      # exponente de sharpening de pesos

# ── IQR outlier removal ───────────────────────────────────────
IQR_FACTOR = 5.0
