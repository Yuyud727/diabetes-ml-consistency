"""
src/models.py
=============
Factory untuk keempat model: Random Forest, XGBoost, KNN, dan Soft Voting Ensemble.

Soft Voting Ensemble mengikuti definisi pada BAB 3:
    p_hat(c|x) = (1/M) * sum_m p_hat_m(c|x),   y_hat = argmax_c p_hat(c|x)
Diimplementasikan dengan sklearn VotingClassifier(voting="soft") atas ketiga
base model (RF + XGB + KNN). Karena seed dan fold latih identik, base model di
dalam ensemble identik dengan base model yang dievaluasi terpisah.

Seluruh hyperparameter dibaca dari config.HYPERPARAMS sehingga terdokumentasi
dan tetap lintas dataset.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

import config


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**config.HYPERPARAMS["RandomForest"])


def make_xgboost():
    if not _HAS_XGB:
        raise ImportError(
            "Paket 'xgboost' belum terpasang. Jalankan: pip install xgboost"
        )
    return XGBClassifier(**config.HYPERPARAMS["XGBoost"])


def make_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(**config.HYPERPARAMS["KNN"])


def make_base_estimators() -> list[tuple[str, object]]:
    """Daftar (nama, estimator) base model untuk Soft Voting Ensemble."""
    return [
        ("rf", make_random_forest()),
        ("xgb", make_xgboost()),
        ("knn", make_knn()),
    ]


def make_soft_voting() -> VotingClassifier:
    """Soft Voting Ensemble (RF + XGB + KNN) dengan bobot seragam."""
    return VotingClassifier(
        estimators=make_base_estimators(),
        voting="soft",
        weights=None,      # bobot seragam => rata-rata probabilitas
        n_jobs=None,       # hindari nested parallel berlebihan
        flatten_transform=True,
    )


# Registry: nama model -> fungsi pembuat
MODEL_FACTORY = {
    "RandomForest": make_random_forest,
    "XGBoost": make_xgboost,
    "KNN": make_knn,
    "SoftVoting": make_soft_voting,
}


def make_model(name: str):
    """Membuat instance model baru berdasarkan nama."""
    if name not in MODEL_FACTORY:
        raise KeyError(f"Model tidak dikenal: {name}. Pilihan: {list(MODEL_FACTORY)}")
    return MODEL_FACTORY[name]()


def is_tree_model(name: str) -> bool:
    """True untuk model berbasis pohon (RF/XGB) yang memakai TreeSHAP."""
    return name in ("RandomForest", "XGBoost")
