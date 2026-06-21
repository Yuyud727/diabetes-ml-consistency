"""
config.py
=========
Konfigurasi terpusat untuk seluruh eksperimen cross-dataset consistency.

Seluruh parameter yang bersifat "variabel kontrol" (sesuai BAB 3) didefinisikan
di sini agar protokol eksperimen sepenuhnya dapat direplikasi dan diaudit.
Tidak ada satu pun parameter eksperimen yang di-hardcode di modul lain.

Prinsip:
- random_state = 42 dipakai di SEMUA proses yang melibatkan keacakan.
- Pipeline preprocessing, skema validasi, definisi metrik, dan ambang konsistensi
  dijaga identik lintas dataset (controlled comparison).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. PATH & DIREKTORI
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = ROOT_DIR / "datasets"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"
TABLE_DIR: Path = OUTPUT_DIR / "tables"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
LOG_DIR: Path = OUTPUT_DIR / "logs"
DOCS_DIR: Path = ROOT_DIR / "docs"

# Subfolder figur
FIG_CONFUSION: Path = FIGURE_DIR / "confusion_matrix"
FIG_ROC: Path = FIGURE_DIR / "roc_curve"
FIG_SHAP_SUMMARY: Path = FIGURE_DIR / "shap_summary"
FIG_SHAP_FORCE: Path = FIGURE_DIR / "shap_force"
FIG_RANKING: Path = FIGURE_DIR / "ranking_plot"
FIG_FEATURE: Path = FIGURE_DIR / "feature_comparison"

ALL_DIRS = [
    OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, DOCS_DIR,
    FIG_CONFUSION, FIG_ROC, FIG_SHAP_SUMMARY, FIG_SHAP_FORCE,
    FIG_RANKING, FIG_FEATURE,
]


def ensure_dirs() -> None:
    """Membuat seluruh direktori output bila belum ada."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 2. REPRODUSIBILITAS
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# 3. SKEMA VALIDASI
# --------------------------------------------------------------------------- #
N_SPLITS: int = 10          # Stratified 10-Fold Cross-Validation
CV_SHUFFLE: bool = True

# --------------------------------------------------------------------------- #
# 4. DATASET
#    Setiap dataset dideskripsikan secara deklaratif. Modul preprocessing
#    membaca spesifikasi ini sehingga penambahan dataset baru cukup dilakukan
#    di sini tanpa mengubah logika.
#
#    field:
#      key                : identifier internal & nama subfolder
#      name               : nama panjang (untuk caption/laporan)
#      short              : singkatan (PIDD / DPD / Early)
#      path               : lokasi file CSV
#      target             : nama kolom label
#      positive_label     : nilai label yang dianggap "positif" (diabetes)
#      zero_as_missing    : daftar kolom yang nilai 0-nya diperlakukan sebagai NaN
#      drop_columns       : kolom yang dibuang (mis. ID) sebelum pemodelan
#      sep                : pemisah CSV
# --------------------------------------------------------------------------- #
DATASETS: dict[str, dict] = {
    "pidd": {
        "key": "pidd",
        "name": "Pima Indians Diabetes Dataset",
        "short": "PIDD",
        "path": str(DATASET_DIR / "pidd" / "pidd.csv"),
        "target": "Outcome",
        "positive_label": 1,
        "zero_as_missing": [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        ],
        "drop_columns": [],
        "sep": ",",
    },
    "dpd": {
        "key": "dpd",
        "name": "Diabetes Prediction Dataset",
        "short": "DPD",
        "path": str(DATASET_DIR / "diabetes_prediction" / "diabetes_prediction.csv"),
        "target": "diabetes",
        "positive_label": 1,
        "zero_as_missing": [],
        "drop_columns": [],
        "sep": ",",
    },
    "early": {
        "key": "early",
        "name": "Early Stage Diabetes Risk Prediction Dataset",
        "short": "Early",
        "path": str(DATASET_DIR / "early_stage" / "early_stage.csv"),
        "target": "class",
        "positive_label": "Positive",   # sebagian sumber memakai "Positive"/"Negative"
        "zero_as_missing": [],
        "drop_columns": [],
        "sep": ",",
    },
}

# Urutan dataset yang konsisten untuk pelaporan & uji statistik
DATASET_ORDER: list[str] = ["pidd", "dpd", "early"]

# Pasangan dataset untuk uji berpasangan (Wilcoxon, ΔF1, overlap SHAP)
DATASET_PAIRS: list[tuple[str, str]] = [
    ("pidd", "dpd"),
    ("pidd", "early"),
    ("dpd", "early"),
]

# --------------------------------------------------------------------------- #
# 5. MODEL & HYPERPARAMETER (terdokumentasi, tetap, lintas dataset)
# --------------------------------------------------------------------------- #
MODEL_ORDER: list[str] = ["RandomForest", "XGBoost", "KNN", "SoftVoting"]

# Label ringkas untuk caption/tabel
MODEL_LABELS: dict[str, str] = {
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "KNN": "KNN",
    "SoftVoting": "SVE",
}

HYPERPARAMS: dict[str, dict] = {
    "RandomForest": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "KNN": {
        "n_neighbors": 11,
        "weights": "distance",
        "n_jobs": -1,
    },
}

# --------------------------------------------------------------------------- #
# 6. METRIK
#    F1 = metrik utama. AUC & Recall = utama pendukung. Acc & Precision = pendukung.
#    Fokus pada kelas positif (penderita diabetes).
# --------------------------------------------------------------------------- #
PRIMARY_METRIC: str = "f1"
METRICS: list[str] = ["f1", "roc_auc", "recall", "accuracy", "precision"]

# Pemetaan nama metrik internal -> scorer scikit-learn
SCORERS: dict[str, str] = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "accuracy": "accuracy",
    "precision": "precision",
}

# Label rapi metrik untuk tabel
METRIC_LABELS: dict[str, str] = {
    "f1": "F1",
    "roc_auc": "AUC",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "precision": "Precision",
}

# --------------------------------------------------------------------------- #
# 7. AMBANG KONSISTENSI (tiga dimensi)
# --------------------------------------------------------------------------- #
DELTA_F1_THRESHOLD: float = 0.05    # Dimensi performa : |ΔF1| <= 0.05
SPEARMAN_THRESHOLD: float = 0.80    # Dimensi ranking  : Spearman rho >= 0.80
SHAP_TOP_K: int = 3                 # Dimensi eksplanasi: top-3 fitur
SHAP_OVERLAP_MIN: int = 2           # overlap >= 2 dari 3 pasang dataset
ALPHA: float = 0.05                 # taraf signifikansi uji statistik

# --------------------------------------------------------------------------- #
# 8. PENGATURAN SHAP (untuk menjaga tractability pada dataset besar)
#    KernelSHAP sangat mahal; kita membatasi ukuran background dan sampel uji.
#    Nilai dapat diturunkan otomatis pada mode --quick.
# --------------------------------------------------------------------------- #
SHAP_BACKGROUND_SIZE: int = 100     # ukuran ringkasan background (kmeans)
SHAP_NSAMPLES_EXPLAIN: int = 200    # jumlah sampel uji yang dijelaskan
SHAP_MAX_TREE_EXPLAIN: int = 2000   # batas sampel untuk TreeSHAP (efisiensi)
SHAP_FORCE_SAMPLE_INDEX: int = 0    # indeks sampel untuk force plot (relatif)

# --------------------------------------------------------------------------- #
# 9. PENGATURAN EKSEKUSI / MODE CEPAT (SMOKE TEST)
#    Mode quick memangkas beban komputasi agar pipeline dapat diverifikasi
#    end-to-end dengan cepat TANPA mengubah logika ilmiah.
# --------------------------------------------------------------------------- #
QUICK_MODE: bool = bool(int(os.environ.get("RP_QUICK", "0")))

if QUICK_MODE:
    N_SPLITS = 3
    SHAP_BACKGROUND_SIZE = 30
    SHAP_NSAMPLES_EXPLAIN = 40
    SHAP_MAX_TREE_EXPLAIN = 400
    # Subsampling dataset besar pada mode quick (lihat preprocessing.load_dataset)
    QUICK_MAX_ROWS = 3000
else:
    QUICK_MAX_ROWS = None

# Subsampling KHUSUS untuk dataset sangat besar pada perhitungan SHAP non-tree.
# (Tidak memengaruhi evaluasi performa; hanya membatasi beban KernelSHAP.)
SHAP_LARGE_DATASET_CAP: int = 5000

# Format penyimpanan figur
FIG_DPI: int = 150
FIG_FORMAT: str = "png"
