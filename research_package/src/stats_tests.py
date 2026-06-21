"""
src/stats_tests.py
=================
Uji statistik nonparametrik pendukung kerangka konsistensi.

    - Wilcoxon signed-rank : signifikansi selisih F1 antar pasang dataset (per model)
    - Friedman test        : perbedaan peringkat antar model lintas dataset
    - Nemenyi post-hoc     : pasangan model yang berbeda signifikan (bila Friedman sig.)
    - Spearman correlation : kesamaan urutan (ranking model & peringkat SHAP)

Catatan kehati-hatian (sesuai Subbab 3.10.4):
    Lipatan CV tidak sepenuhnya independen dan jumlah dataset hanya tiga,
    sehingga p-value ditafsirkan berhati-hati dan dilaporkan berdampingan
    dengan ukuran efek (selisih & korelasi).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr

try:
    import scikit_posthocs as sp
    _HAS_POSTHOCS = True
except Exception:  # pragma: no cover
    _HAS_POSTHOCS = False

import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank (Tabel 4.3)
# --------------------------------------------------------------------------- #
def wilcoxon_across_pairs(f1_folds: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Menjalankan Wilcoxon signed-rank pada distribusi F1 per-fold untuk setiap
    pasang dataset dan setiap model.

    f1_folds : {model: {dataset_key: array(n_splits,)}}
    """
    rows = []
    for mname in config.MODEL_ORDER:
        per_ds = f1_folds.get(mname, {})
        for (a, b) in config.DATASET_PAIRS:
            if a not in per_ds or b not in per_ds:
                continue
            xa, xb = np.asarray(per_ds[a]), np.asarray(per_ds[b])
            pair_label = f"{config.DATASETS[a]['short']}-{config.DATASETS[b]['short']}"
            stat, p = _safe_wilcoxon(xa, xb)
            rows.append({
                "Model": config.MODEL_LABELS[mname],
                "Dataset Pair": pair_label,
                "statistic": None if stat is None else round(float(stat), 4),
                "p_value": None if p is None else round(float(p), 4),
                "Signifikan (p<0.05)": "Ya" if (p is not None and p < config.ALPHA) else "Tidak",
            })
    return pd.DataFrame(rows)


def _safe_wilcoxon(xa: np.ndarray, xb: np.ndarray):
    """Wilcoxon yang aman terhadap kasus selisih nol / sampel kecil."""
    diff = xa - xb
    if np.allclose(diff, 0.0):
        # Tidak ada perbedaan -> tidak signifikan
        return 0.0, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = wilcoxon(xa, xb, zero_method="wilcox", correction=False,
                               alternative="two-sided")
        return stat, p
    except Exception as exc:  # pragma: no cover
        logger.warning("Wilcoxon gagal: %s", exc)
        return None, None


# --------------------------------------------------------------------------- #
# Friedman + Nemenyi (Tabel 4.5)
# --------------------------------------------------------------------------- #
def friedman_test(mean_metric: pd.DataFrame) -> dict:
    """
    Friedman test atas matriks rata-rata metrik (indeks=model, kolom=dataset).

    Setiap dataset diperlakukan sebagai blok; setiap model sebagai perlakuan.
    Mengembalikan dict berisi chi-square, p-value, df, k, n, dan rata-rata
    peringkat tiap model.
    """
    # Kolom = dataset (blok). scipy.friedmanchisquare menerima tiap sampel
    # sebagai argumen terpisah: di sini tiap argumen = skor seluruh model pada
    # satu dataset. Namun friedman membutuhkan format: tiap "grup" = pengukuran
    # berulang. Kita susun: baris=model, kolom=dataset, lalu transpose agar
    # tiap dataset menjadi satu pengamatan blok.
    data = mean_metric.dropna(axis=0, how="any")
    models = list(data.index)
    # matriks (n_datasets, n_models)
    mat = data.T.values  # baris=dataset, kolom=model
    n_blocks, k = mat.shape

    result = {"chi_square": None, "p_value": None, "df": k - 1,
              "k_models": k, "n_datasets": n_blocks, "avg_ranks": {}}

    if n_blocks < 2 or k < 3:
        logger.warning("Friedman butuh >=3 model & >=2 dataset; dilewati.")
        return result

    try:
        # tiap kolom model sebagai satu sampel berukuran n_blocks
        samples = [mat[:, j] for j in range(k)]
        chi2, p = friedmanchisquare(*samples)
        result["chi_square"] = float(chi2)
        result["p_value"] = float(p)
    except Exception as exc:  # pragma: no cover
        logger.warning("Friedman gagal: %s", exc)

    # Rata-rata peringkat (1 = terbaik). Peringkat dihitung per dataset.
    ranks = np.zeros_like(mat, dtype=float)
    for i in range(n_blocks):
        # peringkat menurun: skor tertinggi -> rank 1
        order = pd.Series(mat[i, :]).rank(ascending=False, method="average").values
        ranks[i, :] = order
    avg_ranks = ranks.mean(axis=0)
    result["avg_ranks"] = {models[j]: float(avg_ranks[j]) for j in range(k)}
    return result


def nemenyi_posthoc(mean_metric: pd.DataFrame) -> pd.DataFrame | None:
    """
    Nemenyi post-hoc. Mengembalikan matriks p-value antar model, atau None bila
    paket scikit-posthocs tidak tersedia.
    """
    if not _HAS_POSTHOCS:
        logger.warning("scikit-posthocs tidak tersedia; Nemenyi dilewati.")
        return None
    data = mean_metric.dropna(axis=0, how="any")
    # scikit_posthocs.posthoc_nemenyi_friedman: baris=blok, kolom=grup
    mat = data.T  # baris=dataset, kolom=model
    try:
        pvals = sp.posthoc_nemenyi_friedman(mat.values)
        pvals.index = list(data.index)
        pvals.columns = list(data.index)
        return pvals.round(4)
    except Exception as exc:  # pragma: no cover
        logger.warning("Nemenyi gagal: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Spearman: ranking model antar dataset (Tabel pendukung 4.4)
# --------------------------------------------------------------------------- #
def model_ranks_per_dataset(mean_metric: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah matriks rata-rata metrik menjadi peringkat model per dataset.
    Indeks = model, kolom = dataset. Rank 1 = terbaik.
    """
    ranks = mean_metric.rank(ascending=False, method="min").astype("Int64")
    return ranks


def spearman_ranking_across_pairs(mean_metric: pd.DataFrame) -> pd.DataFrame:
    """
    Korelasi Spearman peringkat model antar pasang dataset.
    """
    ranks = model_ranks_per_dataset(mean_metric)
    short_to_key = {config.DATASETS[k]["short"]: k for k in config.DATASET_ORDER}
    rows = []
    for (a, b) in config.DATASET_PAIRS:
        sa, sb = config.DATASETS[a]["short"], config.DATASETS[b]["short"]
        if sa not in ranks.columns or sb not in ranks.columns:
            continue
        ra = ranks[sa].astype(float).values
        rb = ranks[sb].astype(float).values
        rho, p = _safe_spearman(ra, rb)
        rows.append({
            "Pair": f"{sa}-{sb}",
            "spearman_rho": None if rho is None else round(float(rho), 4),
            "p_value": None if p is None else round(float(p), 4),
            f"Konsisten (rho>={config.SPEARMAN_THRESHOLD})":
                "Ya" if (rho is not None and rho >= config.SPEARMAN_THRESHOLD) else "Tidak",
        })
    return pd.DataFrame(rows)


def _safe_spearman(a: np.ndarray, b: np.ndarray):
    """Spearman yang aman terhadap input konstan."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        # urutan identik & konstan -> korelasi sempurna bila sama persis
        return (1.0, 0.0) if np.allclose(a, b) else (0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = spearmanr(a, b)
    if np.isnan(rho):
        return None, None
    return rho, p


# --------------------------------------------------------------------------- #
# Spearman: peringkat SHAP antar dataset (dimensi eksplanasi)
# --------------------------------------------------------------------------- #
def spearman_shap_across_pairs(all_shap, model_name: str) -> pd.DataFrame:
    """
    Korelasi Spearman peringkat fitur SHAP sebuah model antar pasang dataset.
    Hanya fitur yang muncul di kedua dataset yang dibandingkan (irisan kolom).
    """
    rows = []
    for (a, b) in config.DATASET_PAIRS:
        sa, sb = config.DATASETS[a]["short"], config.DATASETS[b]["short"]
        ra = all_shap.get(a, {}).get(model_name)
        rb = all_shap.get(b, {}).get(model_name)
        if not ra or not rb or not ra.ok or not rb.ok:
            rows.append({"Pair": f"{sa}-{sb}", "spearman_rho": None,
                         "n_common_features": 0})
            continue
        common = [f for f in ra.ranking.index if f in rb.ranking.index]
        if len(common) < 3:
            rows.append({"Pair": f"{sa}-{sb}", "spearman_rho": None,
                         "n_common_features": len(common)})
            continue
        va = ra.ranking.loc[common].astype(float).values
        vb = rb.ranking.loc[common].astype(float).values
        rho, _ = _safe_spearman(va, vb)
        rows.append({
            "Pair": f"{sa}-{sb}",
            "spearman_rho": None if rho is None else round(float(rho), 4),
            "n_common_features": len(common),
        })
    return pd.DataFrame(rows)
