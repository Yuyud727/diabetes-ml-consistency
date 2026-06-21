"""
src/evaluation.py
=================
Evaluasi performa melalui Stratified 10-Fold Cross-Validation.

Untuk setiap sel (model, dataset) dihasilkan:
    - distribusi 10 skor per metrik (F1, AUC, Recall, Accuracy, Precision)
    - rata-rata dan simpangan baku tiap metrik
    - prediksi out-of-fold (untuk confusion matrix & kurva ROC)

Seluruh perhitungan memakai cross_validate / cross_val_predict dengan pipeline
anti-leakage, sehingga preprocessing + SMOTE selalu di-fit pada fold latih saja.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

import config
from preprocessing import build_preprocessor, build_pipeline
from models import make_model

logger = logging.getLogger(__name__)


@dataclass
class CellResult:
    """Hasil evaluasi satu sel (model, dataset)."""
    dataset: str
    model: str
    per_fold: dict[str, np.ndarray] = field(default_factory=dict)   # metrik -> array(10,)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    y_true: np.ndarray | None = None      # label asli (urutan asli)
    y_pred: np.ndarray | None = None      # prediksi OOF
    y_proba: np.ndarray | None = None     # probabilitas OOF kelas positif


def get_cv() -> StratifiedKFold:
    """StratifiedKFold dengan seed tetap (skema validasi terkontrol)."""
    return StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=config.CV_SHUFFLE,
        random_state=config.RANDOM_STATE,
    )


def evaluate_cell(model_name: str, X: pd.DataFrame, y: pd.Series) -> CellResult:
    """
    Mengevaluasi satu model pada satu dataset melalui 10-fold CV.
    """
    cv = get_cv()
    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor, make_model(model_name), use_smote=True)

    # --- skor per-fold untuk seluruh metrik ---
    scoring = {m: config.SCORERS[m] for m in config.METRICS}
    cv_out = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=None,            # paralelisme ada di level estimator (n_jobs=-1)
        return_train_score=False,
        error_score="raise",
    )

    res = CellResult(dataset="", model=model_name)
    for m in config.METRICS:
        scores = np.asarray(cv_out[f"test_{m}"], dtype=float)
        res.per_fold[m] = scores
        res.mean[m] = float(np.nanmean(scores))
        res.std[m] = float(np.nanstd(scores))

    # --- prediksi out-of-fold untuk confusion matrix & ROC ---
    # (pipeline yang sama, di-fit ulang per fold di balik layar)
    pipeline_pred = build_pipeline(build_preprocessor(X), make_model(model_name),
                                   use_smote=True)
    y_pred = cross_val_predict(pipeline_pred, X, y, cv=cv, method="predict",
                               n_jobs=None)
    try:
        proba = cross_val_predict(pipeline_pred, X, y, cv=cv,
                                  method="predict_proba", n_jobs=None)
        y_proba = proba[:, 1]
    except Exception as exc:  # pragma: no cover
        logger.warning("predict_proba gagal untuk %s: %s", model_name, exc)
        y_proba = None

    res.y_true = np.asarray(y)
    res.y_pred = np.asarray(y_pred)
    res.y_proba = y_proba

    logger.info(
        "    %-12s | F1=%.4f±%.4f  AUC=%.4f  Recall=%.4f",
        model_name, res.mean["f1"], res.std["f1"],
        res.mean.get("roc_auc", float("nan")), res.mean.get("recall", float("nan")),
    )
    return res


def evaluate_dataset(dataset_key: str, X: pd.DataFrame, y: pd.Series
                     ) -> dict[str, CellResult]:
    """Mengevaluasi seluruh model pada satu dataset."""
    short = config.DATASETS[dataset_key]["short"]
    logger.info("  Evaluasi dataset %s (n=%d, positif=%.1f%%)",
                short, len(y), 100.0 * float(np.mean(y)))
    results: dict[str, CellResult] = {}
    for model_name in config.MODEL_ORDER:
        res = evaluate_cell(model_name, X, y)
        res.dataset = dataset_key
        results[model_name] = res
    return results


# --------------------------------------------------------------------------- #
# Agregasi ke tabel performa (Tabel 4.1)
# --------------------------------------------------------------------------- #
def results_to_performance_table(all_results: dict[str, dict[str, CellResult]]
                                 ) -> pd.DataFrame:
    """
    Membentuk tabel performa lebar:
        kolom: Dataset, Model, lalu mean & std setiap metrik.
    """
    rows = []
    for dkey in config.DATASET_ORDER:
        if dkey not in all_results:
            continue
        short = config.DATASETS[dkey]["short"]
        for mname in config.MODEL_ORDER:
            if mname not in all_results[dkey]:
                continue
            r = all_results[dkey][mname]
            row = {"Dataset": short, "Model": config.MODEL_LABELS[mname]}
            for m in config.METRICS:
                row[f"{config.METRIC_LABELS[m]}_mean"] = round(r.mean[m], 4)
                row[f"{config.METRIC_LABELS[m]}_std"] = round(r.std[m], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_f1_folds(all_results: dict[str, dict[str, CellResult]]
                     ) -> dict[str, dict[str, np.ndarray]]:
    """
    Mengumpulkan distribusi F1 per-fold:
        {model: {dataset_key: array(n_splits,)}}
    Dipakai untuk uji Wilcoxon (berpasangan antar dataset).
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for mname in config.MODEL_ORDER:
        out[mname] = {}
        for dkey in config.DATASET_ORDER:
            if dkey in all_results and mname in all_results[dkey]:
                out[mname][dkey] = all_results[dkey][mname].per_fold["f1"]
    return out


def collect_mean_metric(all_results: dict[str, dict[str, CellResult]],
                        metric: str = "f1") -> pd.DataFrame:
    """
    Matriks rata-rata metrik: indeks = model, kolom = dataset (short).
    Dipakai untuk ranking, Friedman, dan korelasi Spearman.
    """
    data = {}
    for dkey in config.DATASET_ORDER:
        if dkey not in all_results:
            continue
        short = config.DATASETS[dkey]["short"]
        col = {}
        for mname in config.MODEL_ORDER:
            if mname in all_results[dkey]:
                col[config.MODEL_LABELS[mname]] = all_results[dkey][mname].mean[metric]
        data[short] = col
    return pd.DataFrame(data)
