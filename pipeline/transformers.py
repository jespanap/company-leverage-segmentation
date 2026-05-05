"""
pipeline/transformers.py
------------------------
Transformadores sklearn-compatibles para el pipeline de apalancamiento.

Clases
------
- MonetaryCleaner      : limpieza de columnas monetarias con formato "$1,234"
- LeverageFeatureEngineer : construye los 7 ratios de apalancamiento
- OutlierIQRRemover    : elimina outliers extremos por IQR (como TransformerMixin)
- LogModulusScaler     : aplica log-modulus + RobustScaler
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

from config import MONETARY_COLS, LEVERAGE_FEATURES, IQR_FACTOR


# ─────────────────────────────────────────────────────────────
class MonetaryCleaner(BaseEstimator, TransformerMixin):
    """
    Limpia columnas monetarias con formato '$1,234.56' y convierte a float.
    También normaliza 'Año de Corte' y 'NIT'.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in MONETARY_COLS:
            df[col] = (
                df[col].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
                .astype(float)
            )
        df["Año de Corte"] = df["Año de Corte"].astype(str).str.replace(",", "").astype(int)
        df["NIT"]          = df["NIT"].astype(str).str.replace(",", "").str.strip()
        return df


# ─────────────────────────────────────────────────────────────
class LeverageFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Construye los 7 ratios de apalancamiento y elimina filas con activos <= 0.
    """

    EPS = 1e-9

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        e  = self.EPS

        df = df[df["TOTAL ACTIVOS"] > 0].copy()

        df["deuda_activos"]      = df["TOTAL PASIVOS"]  / (df["TOTAL ACTIVOS"]            + e)
        df["deuda_patrimonio"]   = df["TOTAL PASIVOS"]  / (df["TOTAL PATRIMONIO"].abs()   + e)
        df["multiplicador_cap"]  = df["TOTAL ACTIVOS"]  / (df["TOTAL PATRIMONIO"].abs()   + e)
        df["cobertura_ingresos"] = df["INGRESOS OPERACIONALES"] / (df["TOTAL PASIVOS"]    + e)
        df["margen_neto"]        = df["GANANCIA (PÉRDIDA)"] / (df["INGRESOS OPERACIONALES"].abs() + e)
        df["roa"]                = df["GANANCIA (PÉRDIDA)"] / (df["TOTAL ACTIVOS"]        + e)
        df["roe"]                = df["GANANCIA (PÉRDIDA)"] / (df["TOTAL PATRIMONIO"].abs() + e)

        return df


# ─────────────────────────────────────────────────────────────
class OutlierIQRRemover(BaseEstimator, TransformerMixin):
    """
    Elimina filas cuyos valores en `cols` estén fuera de [Q1 - f·IQR, Q3 + f·IQR].
    También imputa Inf/-Inf con la mediana y rellena NaN.
    """

    def __init__(self, cols: list = None, factor: float = IQR_FACTOR):
        self.cols   = cols or LEVERAGE_FEATURES
        self.factor = factor
        self._lower = {}
        self._upper = {}
        self._medians = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.cols:
            q1, q3 = X[col].quantile(0.01), X[col].quantile(0.99)
            iqr = q3 - q1
            self._lower[col]   = q1 - self.factor * iqr
            self._upper[col]   = q3 + self.factor * iqr
            clean = X[col].replace([np.inf, -np.inf], np.nan)
            self._medians[col] = clean.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df   = X.copy()
        mask = pd.Series(True, index=df.index)
        for col in self.cols:
            mask &= df[col].between(self._lower[col], self._upper[col])
        df = df[mask].copy()
        for col in self.cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(self._medians[col])
        return df


# ─────────────────────────────────────────────────────────────
class LogModulusScaler(BaseEstimator, TransformerMixin):
    """
    Aplica la transformación log-modulus  sign(x)·log(1+|x|)
    seguida de RobustScaler sobre las columnas de apalancamiento.

    Devuelve el DataFrame completo con las columnas `*_lm` añadidas
    y las features originales sustituidas por sus versiones escaladas.
    """

    def __init__(self, cols: list = None):
        self.cols = cols or LEVERAGE_FEATURES
        self._scaler = RobustScaler()

    @staticmethod
    def _log_modulus(x: pd.Series) -> pd.Series:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X: pd.DataFrame, y=None):
        lm = X[self.cols].apply(self._log_modulus)
        self._scaler.fit(lm)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        lm_cols = [c + "_lm" for c in self.cols]
        lm = X[self.cols].apply(self._log_modulus)
        scaled = self._scaler.transform(lm)
        df[lm_cols] = scaled
        return df

    @property
    def transformed_cols(self):
        return [c + "_lm" for c in self.cols]
