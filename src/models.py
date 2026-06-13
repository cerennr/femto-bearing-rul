"""
models.py
---------
Track A ML modelleri: SVR, Random Forest, XGBoost.

Her model aynı arayüzü paylaşır:
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    model.save(path)
    model = ModelClass.load(path)

Tüm modeller negatif RUL tahmini döndürmez (clip ile 0'a sabitlenir).
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# SVR
# ─────────────────────────────────────────────────────────────────────────────

# class SVRModel:
#     """
#     Support Vector Regressor (RBF kernel).

#     Küçük veri setlerinde (FEMTO gibi) genellikle güçlü sonuçlar verir.
#     Kernel trick sayesinde doğrusal olmayan RUL-feature ilişkilerini yakalar.

#     Hiperparametreler:
#         C       : regularization (büyük C → train'e daha sıkı fit)
#         epsilon : hata tüpü genişliği (küçük epsilon → daha hassas)
#         gamma   : RBF kernel genişliği ('scale' = 1/n_features/var)
#     """

#     name = "SVR (RBF)"

#     def __init__(
#         self,
#         C:       float = 10.0,
#         epsilon: float = 0.1,
#         gamma:   str   = "scale",
#     ):
#         self.model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
#         self.params = {"C": C, "epsilon": epsilon, "gamma": gamma}

#     def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRModel":
#         self.model.fit(X, y)
#         return self

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         return np.clip(self.model.predict(X), 0, None)

#     def save(self, path: str | Path):
#         joblib.dump(self, path)

#     @classmethod
#     def load(cls, path: str | Path) -> "SVRModel":
#         return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel:
    """
    Random Forest Regressor.

    Avantajları:
    - Doğal olarak öznitelik önem skoru verir (SHAP ile zenginleştirilebilir)
    - Outlier'lara dayanıklı
    - Az hiperparametre ayarı gerektir

    Hiperparametreler:
        n_estimators    : ağaç sayısı (fazla = daha kararlı ama yavaş)
        max_depth       : None = tam büyüme (overfitting riski var)
        min_samples_leaf: yaprak düğüm minimum örnek (regularization)
    """

    name = "Random Forest"

    def __init__(
        self,
        n_estimators:     int  = 200,
        max_depth:        int  = None,
        min_samples_leaf: int  = 10,
        n_jobs:           int  = -1,
        random_state:     int  = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X), 0, None)

    def feature_importances(self, feature_names: list = None) -> dict:
        """Gini importance skorlarını döndürür."""
        imp = self.model.feature_importances_
        if feature_names:
            return dict(
                sorted(zip(feature_names, imp), key=lambda x: -x[1])
            )
        return imp

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestModel":
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    name = "XGBoost"

    def __init__(
        self,
        n_estimators:         int   = 500,
        learning_rate:        float = 0.05,
        max_depth:            int   = 5,
        subsample:            float = 0.8,
        colsample_bytree:     float = 0.8,
        early_stopping_rounds: int  = None,  # None = early stopping yok (sabit tur)
        n_jobs:               int   = -1,
        random_state:         int   = 42,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        # early_stopping_rounds XGBRegressor'a VERILMIYOR:
        # eval_set olmadan fit() cagrilirsa XGBoost hata verir.
        # Bunun yerine fit() icinde eval_set varsa dinamik olarak gecilir.
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
        }
        self._best_iteration = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray = None,
        y_val:   np.ndarray = None,
    ) -> "XGBoostModel":
        if X_val is not None and y_val is not None and self.early_stopping_rounds:
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self._best_iteration = self.model.best_iteration
            print(f"    XGBoost best iteration: {self._best_iteration}")
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X), 0, None)

    def feature_importances(self, feature_names: list = None) -> dict:
        imp = self.model.feature_importances_
        if feature_names:
            return dict(sorted(zip(feature_names, imp), key=lambda x: -x[1]))
        return imp

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        return joblib.load(path)

# ─────────────────────────────────────────────────────────────────────────────
# Model listesi — train scripti için
# ─────────────────────────────────────────────────────────────────────────────

def get_all_models() -> list:
    """Karşılaştırılacak tüm modelleri döndürür."""
    return [
        # SVRModel(C=10.0, epsilon=0.1),
        # SVRModel(C=100.0, epsilon=0.05),
        RandomForestModel(n_estimators=200, min_samples_leaf=2),
        XGBoostModel(n_estimators=500, learning_rate=0.05, max_depth=5),
    ]
