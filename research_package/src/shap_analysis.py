"""
src/shap_analysis.py
====================
Analisis explainability berbasis SHAP (SHapley Additive exPlanations).

Strategi (sesuai BAB 3, Subbab 3.4):
    - RF & XGBoost  -> TreeSHAP (eksak, efisien)
    - KNN           -> KernelSHAP (model-agnostik)
    - Soft Voting   -> KernelSHAP atas predict_proba ansambel

Operasionalisasi (didokumentasikan agar reproducible):
    Untuk setiap (model, dataset) dilakukan satu pembagian terstratifikasi
    (seed tetap). Preprocessor di-fit pada data latih; SMOTE diterapkan pada
    data latih terproses; classifier dilatih pada data latih ter-resample.
    Nilai SHAP dihitung pada SAMPEL data uji terproses dengan background yang
    berasal dari data latih terproses (diringkas via shap.kmeans untuk KernelSHAP).
    Kepentingan global = rata-rata |SHAP| per fitur; top-3 dan peringkat fitur
    diturunkan darinya. Pendekatan ini dapat diperluas ke rata-rata lintas fold
    bila sumber daya komputasi memadai.

Catatan tractability:
    KernelSHAP sangat mahal. Ukuran background dan jumlah sampel uji dibatasi
    melalui config (SHAP_BACKGROUND_SIZE, SHAP_NSAMPLES_EXPLAIN). Untuk dataset
    sangat besar, data uji dibatasi (SHAP_LARGE_DATASET_CAP) khusus untuk SHAP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - SHAP opsional bila dijalankan --no-shap
    shap = None
    SHAP_AVAILABLE = False

import config
from preprocessing import build_preprocessor, get_feature_names
from models import make_model, is_tree_model
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


@dataclass
class ShapResult:
    """Hasil SHAP satu sel (model, dataset)."""
    dataset: str
    model: str
    feature_names: list[str] = field(default_factory=list)
    mean_abs_shap: np.ndarray | None = None       # (n_features,)
    importance: pd.Series | None = None           # fitur -> mean|SHAP| terurut desc
    top_k: list[str] = field(default_factory=list)
    ranking: pd.Series | None = None              # fitur -> peringkat (1 = terpenting)
    ok: bool = True
    error: str | None = None
    # objek untuk force plot (disimpan agar dipakai modul visualisasi)
    explanation: object | None = None
    sample_index: int | None = None


def _prepare_fit_explain(model_name: str, X: pd.DataFrame, y: pd.Series):
    """
    Membagi data, fit preprocessor + classifier, mengembalikan komponen yang
    diperlukan untuk perhitungan SHAP pada ruang fitur ter-transform.
    """
    # Pembatasan ukuran khusus SHAP untuk dataset besar
    if len(X) > config.SHAP_LARGE_DATASET_CAP:
        X, _, y, _ = train_test_split(
            X, y, train_size=config.SHAP_LARGE_DATASET_CAP,
            stratify=y, random_state=config.RANDOM_STATE,
        )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=config.RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_tr)
    Xt_tr = preprocessor.fit_transform(X_tr)
    Xt_te = preprocessor.transform(X_te)
    feature_names = get_feature_names(preprocessor)

    # SMOTE hanya pada data latih ter-transform
    Xs_tr, ys_tr = SMOTE(random_state=config.RANDOM_STATE).fit_resample(Xt_tr, y_tr)

    clf = make_model(model_name)
    clf.fit(Xs_tr, ys_tr)

    return clf, np.asarray(Xt_tr), np.asarray(Xt_te), feature_names


def _positive_class_shap(shap_values) -> np.ndarray:
    """
    Menormalkan keluaran SHAP menjadi matriks (n_samples, n_features) untuk
    kelas positif, menangani beragam format yang dikembalikan pustaka shap.
    """
    # shap.Explanation
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values)
    elif isinstance(shap_values, list):
        # list per kelas -> ambil kelas positif (indeks 1) bila ada
        vals = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        vals = np.asarray(shap_values)

    # Bentuk (n, f, n_classes) -> ambil kelas positif
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    return vals


def compute_shap_for_cell(model_name: str, dataset_key: str,
                          X: pd.DataFrame, y: pd.Series) -> ShapResult:
    """Menghitung SHAP untuk satu sel (model, dataset)."""
    res = ShapResult(dataset=dataset_key, model=model_name)
    try:
        clf, Xt_tr, Xt_te, feature_names = _prepare_fit_explain(model_name, X, y)
        res.feature_names = feature_names

        # Batasi jumlah sampel yang dijelaskan
        n_explain = min(config.SHAP_NSAMPLES_EXPLAIN, Xt_te.shape[0])
        rng = np.random.RandomState(config.RANDOM_STATE)
        idx = rng.choice(Xt_te.shape[0], size=n_explain, replace=False)
        X_explain = Xt_te[idx]

        if is_tree_model(model_name):
            # ---- TreeSHAP ----
            n_tree = min(config.SHAP_MAX_TREE_EXPLAIN, X_explain.shape[0])
            X_tree = X_explain[:n_tree]
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_tree, check_additivity=False)
            vals = _positive_class_shap(shap_values)
            try:
                res.explanation = explainer(X_tree[: min(50, len(X_tree))])
            except Exception:
                res.explanation = None
        else:
            # ---- KernelSHAP (KNN & Soft Voting) ----
            bg_size = min(config.SHAP_BACKGROUND_SIZE, Xt_tr.shape[0])
            background = shap.kmeans(Xt_tr, bg_size)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_explain, nsamples="auto",
                                                silent=True)
            vals = _positive_class_shap(shap_values)

        # Kepentingan global = rata-rata |SHAP| per fitur
        mean_abs = np.abs(vals).mean(axis=0)
        mean_abs = np.asarray(mean_abs, dtype=float).ravel()

        # Selaraskan panjang dengan nama fitur (defensif)
        if len(mean_abs) != len(feature_names):
            k = min(len(mean_abs), len(feature_names))
            mean_abs = mean_abs[:k]
            feature_names = feature_names[:k]
            res.feature_names = feature_names

        importance = pd.Series(mean_abs, index=feature_names).sort_values(
            ascending=False)
        res.mean_abs_shap = mean_abs
        res.importance = importance
        res.top_k = list(importance.head(config.SHAP_TOP_K).index)
        # Peringkat: 1 = terpenting
        res.ranking = importance.rank(ascending=False, method="min").astype(int)
        res.sample_index = int(idx[config.SHAP_FORCE_SAMPLE_INDEX % len(idx)])

        logger.info("    SHAP %-12s @ %-5s -> top3: %s",
                    model_name, config.DATASETS[dataset_key]["short"],
                    ", ".join(res.top_k))
    except Exception as exc:  # pragma: no cover
        logger.warning("    SHAP gagal untuk %s @ %s: %s",
                       model_name, dataset_key, exc)
        res.ok = False
        res.error = str(exc)
    return res


def compute_all_shap(datasets: dict[str, tuple[pd.DataFrame, pd.Series]]
                     ) -> dict[str, dict[str, ShapResult]]:
    """
    Menghitung SHAP untuk seluruh sel (model x dataset).
    Mengembalikan: {dataset_key: {model_name: ShapResult}}
    """
    out: dict[str, dict[str, ShapResult]] = {}
    if not SHAP_AVAILABLE:
        logger.warning("Pustaka 'shap' tidak terpasang; analisis SHAP dilewati. "
                       "Pasang via `pip install shap` untuk mengaktifkannya.")
        return out
    for dkey in config.DATASET_ORDER:
        if dkey not in datasets:
            continue
        X, y = datasets[dkey]
        out[dkey] = {}
        for mname in config.MODEL_ORDER:
            out[dkey][mname] = compute_shap_for_cell(mname, dkey, X, y)
    return out


# --------------------------------------------------------------------------- #
# Tabel top-3 (Tabel 4.6)
# --------------------------------------------------------------------------- #
def shap_to_top3_table(all_shap: dict[str, dict[str, ShapResult]]) -> pd.DataFrame:
    rows = []
    for dkey in config.DATASET_ORDER:
        if dkey not in all_shap:
            continue
        short = config.DATASETS[dkey]["short"]
        for mname in config.MODEL_ORDER:
            r = all_shap[dkey].get(mname)
            if r is None:
                continue
            top = (r.top_k + ["-", "-", "-"])[:3] if r.ok else ["-", "-", "-"]
            rows.append({
                "Dataset": short,
                "Model": config.MODEL_LABELS[mname],
                "Top-1": top[0], "Top-2": top[1], "Top-3": top[2],
            })
    return pd.DataFrame(rows)
