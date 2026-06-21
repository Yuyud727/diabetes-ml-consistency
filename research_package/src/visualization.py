"""
src/visualization.py
====================
Pembuatan dan penyimpanan seluruh figur yang dirujuk pada BAB 4.

    - Confusion matrix per (model, dataset)        -> Gambar 4.1
    - ROC curve seluruh model per dataset          -> Gambar 4.2
    - SHAP summary plot per (model, dataset)        -> Gambar 4.4
    - SHAP force plot (satu sampel)                 -> Gambar 4.5
    - Ranking plot lintas dataset                   -> Gambar 4.3
    - Feature comparison plot (mean|SHAP|)          -> Gambar 4.6

Semua figur disimpan ke outputs/figures/<subfolder>/.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif (aman tanpa display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - SHAP opsional
    shap = None
    SHAP_AVAILABLE = False

import config

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": config.FIG_DPI,
    "savefig.bbox": "tight",
    "font.size": 10,
})


def _save(fig, path):
    fig.savefig(path, dpi=config.FIG_DPI, format=config.FIG_FORMAT)
    plt.close(fig)
    logger.info("    figur disimpan: %s", path.name)


# --------------------------------------------------------------------------- #
# Gambar 4.1 - Confusion Matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cell_result, dataset_key: str):
    short = config.DATASETS[dataset_key]["short"]
    label = config.MODEL_LABELS[cell_result.model]
    cm = confusion_matrix(cell_result.y_true, cell_result.y_pred)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Diabetes", "Diabetes"])
    ax.set_yticklabels(["Non-Diabetes", "Diabetes"])
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix - {label} @ {short}")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fname = f"cm_{cell_result.model}_{short}.{config.FIG_FORMAT}"
    _save(fig, config.FIG_CONFUSION / fname)


# --------------------------------------------------------------------------- #
# Gambar 4.2 - ROC Curve (seluruh model per dataset)
# --------------------------------------------------------------------------- #
def plot_roc_curves(cell_results: dict, dataset_key: str):
    short = config.DATASETS[dataset_key]["short"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for mname in config.MODEL_ORDER:
        r = cell_results.get(mname)
        if r is None or r.y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(r.y_true, r.y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8,
                label=f"{config.MODEL_LABELS[mname]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Kurva ROC seluruh model @ {short}")
    ax.legend(loc="lower right", fontsize=8)
    fname = f"roc_{short}.{config.FIG_FORMAT}"
    _save(fig, config.FIG_ROC / fname)


# --------------------------------------------------------------------------- #
# Gambar 4.4 - SHAP Summary Plot
# --------------------------------------------------------------------------- #
def plot_shap_summary(shap_result, dataset_key: str):
    if not shap_result.ok or shap_result.importance is None:
        return
    short = config.DATASETS[dataset_key]["short"]
    label = config.MODEL_LABELS[shap_result.model]
    imp = shap_result.importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(5.6, max(3.2, 0.3 * len(imp) + 1.2)))
    ax.barh(range(len(imp)), imp.values, color="#1f77b4")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Summary (mean|SHAP|) - {label} @ {short}")
    fname = f"shap_summary_{shap_result.model}_{short}.{config.FIG_FORMAT}"
    _save(fig, config.FIG_SHAP_SUMMARY / fname)


# --------------------------------------------------------------------------- #
# Gambar 4.5 - SHAP Force Plot (satu sampel)
# --------------------------------------------------------------------------- #
def plot_shap_force(shap_result, dataset_key: str):
    """
    Membuat force plot untuk satu sampel. Bila objek Explanation tersedia
    (TreeSHAP) digunakan waterfall plot; jika tidak, dibuat bar kontribusi
    fitur teratas sebagai pendekatan force plot.
    """
    if not shap_result.ok:
        return
    short = config.DATASETS[dataset_key]["short"]
    label = config.MODEL_LABELS[shap_result.model]
    fname = f"shap_force_{shap_result.model}_{short}.{config.FIG_FORMAT}"

    try:
        if shap_result.explanation is not None:
            expl = shap_result.explanation
            single = expl[0]
            # Untuk classifier pohon, Explanation dapat berdimensi
            # (n_sampel, n_fitur, n_kelas). Ambil kontribusi KELAS POSITIF (indeks 1)
            # agar waterfall menampilkan satu penjelasan tunggal.
            try:
                if getattr(single, "values", None) is not None and \
                        np.ndim(single.values) > 1:
                    pos = min(1, single.values.shape[-1] - 1)
                    single = single[..., pos]
            except Exception:
                pass
            fig = plt.figure(figsize=(7, 4))
            shap.plots.waterfall(single, max_display=10, show=False)
            fig = plt.gcf()
            fig.suptitle(f"SHAP Force/Waterfall - {label} @ {short}", fontsize=10)
            _save(fig, config.FIG_SHAP_FORCE / fname)
            return
    except Exception as exc:
        logger.warning("    waterfall gagal (%s); memakai bar kontribusi.", exc)

    # Fallback: bar kontribusi fitur teratas berdasarkan importance global
    if shap_result.importance is not None:
        imp = shap_result.importance.head(8)[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(range(len(imp)), imp.values, color="#d62728")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=8)
        ax.set_xlabel("Kontribusi |SHAP| (global, proksi force plot)")
        ax.set_title(f"Kontribusi fitur - {label} @ {short}")
        _save(fig, config.FIG_SHAP_FORCE / fname)


# --------------------------------------------------------------------------- #
# Gambar 4.3 - Ranking Plot (lintas dataset)
# --------------------------------------------------------------------------- #
def plot_ranking(mean_f1: pd.DataFrame):
    """
    Garis peringkat tiap model lintas dataset (rank 1 di atas).
    mean_f1 : indeks=model label, kolom=dataset short.
    """
    ranks = mean_f1.rank(ascending=False, method="min")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = range(len(ranks.columns))
    for label in ranks.index:
        ax.plot(x, ranks.loc[label].values, marker="o", lw=1.8, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(ranks.columns))
    ax.set_ylabel("Peringkat (1 = terbaik)")
    ax.set_xlabel("Dataset")
    ax.set_title("Perbandingan Ranking Model pada Tiga Dataset")
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(ranks.index) + 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, config.FIG_RANKING / f"ranking_comparison.{config.FIG_FORMAT}")


# --------------------------------------------------------------------------- #
# Gambar 4.6 - Feature Comparison (mean|SHAP| antar dataset)
# --------------------------------------------------------------------------- #
def plot_feature_comparison(all_shap, model_name: str):
    """
    Heatmap mean|SHAP| fitur (baris) x dataset (kolom) untuk satu model,
    menyoroti fitur yang konsisten penting lintas dataset.
    """
    short = {k: config.DATASETS[k]["short"] for k in config.DATASET_ORDER}
    series = {}
    for dkey in config.DATASET_ORDER:
        r = all_shap.get(dkey, {}).get(model_name)
        if r and r.ok and r.importance is not None:
            series[short[dkey]] = r.importance
    if not series:
        return

    df = pd.DataFrame(series)
    # urutkan fitur berdasarkan rata-rata kepentingan lintas dataset
    df["__mean__"] = df.mean(axis=1)
    df = df.sort_values("__mean__", ascending=False).drop(columns="__mean__")
    df = df.head(12).fillna(0.0)

    fig, ax = plt.subplots(figsize=(1.6 * len(df.columns) + 3, 0.45 * len(df) + 1.5))
    im = ax.imshow(df.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title(f"Feature Importance SHAP - {config.MODEL_LABELS[model_name]}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean|SHAP|")
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=7)
    fname = f"feature_comparison_{model_name}.{config.FIG_FORMAT}"
    _save(fig, config.FIG_FEATURE / fname)


# --------------------------------------------------------------------------- #
# Orkestrasi seluruh figur
# --------------------------------------------------------------------------- #
def generate_all_figures(all_results, all_shap, mean_f1):
    logger.info("  Membuat figur ...")
    # Confusion matrix & ROC
    for dkey, cells in all_results.items():
        for mname, r in cells.items():
            plot_confusion_matrix(r, dkey)
        plot_roc_curves(cells, dkey)
    # SHAP summary & force
    for dkey, cells in all_shap.items():
        for mname, r in cells.items():
            plot_shap_summary(r, dkey)
            plot_shap_force(r, dkey)
    # Ranking
    plot_ranking(mean_f1)
    # Feature comparison per model
    for mname in config.MODEL_ORDER:
        plot_feature_comparison(all_shap, mname)
