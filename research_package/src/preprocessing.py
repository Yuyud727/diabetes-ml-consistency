"""
src/preprocessing.py
====================
Pemuatan dataset dan konstruksi pipeline preprocessing yang ANTI-KEBOCORAN.

Kunci anti-leakage:
    Seluruh estimasi parameter (imputasi, encoding, scaling, SMOTE) dilakukan
    sebagai bagian dari sebuah objek Pipeline (imblearn). Karena Pipeline ini
    selalu di-`fit` di dalam loop cross-validation pada DATA LATIH setiap lipatan,
    tidak ada satu pun statistik yang dipelajari dari data uji.

Urutan pipeline (identik lintas dataset, sesuai Algoritma 3.1):
    1. Penanganan nilai hilang  -> SimpleImputer(mean) untuk numerik,
                                   SimpleImputer(most_frequent) untuk kategorikal
    2. Encoding                 -> OneHotEncoder(drop="if_binary") untuk kategorikal
                                   (biner Ya/Tidak -> satu kolom 0/1; nominal -> one-hot)
    3. Standardisasi            -> StandardScaler untuk fitur numerik
    4. SMOTE                    -> hanya pada data latih (otomatis, karena di dalam CV)

Catatan PIDD:
    Nilai 0 pada {Glucose, BloodPressure, SkinThickness, Insulin, BMI} dikonversi
    menjadi NaN SEBELUM pipeline. Ini adalah pembersihan deterministik per-nilai
    (bukan estimasi berbasis data), sehingga tidak menimbulkan kebocoran.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# imblearn Pipeline memungkinkan SMOTE menjadi langkah yang HANYA aktif saat fit
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pemuatan dataset
# --------------------------------------------------------------------------- #
def load_dataset(key: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Memuat sebuah dataset berdasarkan key pada config.DATASETS.

    Mengembalikan:
        X : pd.DataFrame fitur (kategorikal masih bertipe object/category)
        y : pd.Series label biner {0, 1} (1 = positif/diabetes)

    Penanganan khusus:
        - PIDD: konversi 0 -> NaN pada kolom fisiologis tertentu.
        - Label non-numerik (mis. "Positive"/"Negative", "Yes"/"No") dipetakan
          ke {0, 1} berdasarkan positive_label di config.
        - Kolom drop_columns dibuang (mis. kolom ID).
    """
    spec = config.DATASETS[key]
    path = spec["path"]

    df = pd.read_csv(path, sep=spec.get("sep", ","))
    logger.info("Memuat %s: %d baris, %d kolom", spec["short"], *df.shape)

    # Normalisasi nama kolom (hilangkan spasi di tepi)
    df.columns = [str(c).strip() for c in df.columns]

    # Buang kolom yang tidak relevan untuk pemodelan
    drop_cols = [c for c in spec.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    target = spec["target"]
    if target not in df.columns:
        raise KeyError(
            f"Kolom target '{target}' tidak ditemukan pada dataset '{key}'. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    # Pisahkan fitur dan label
    X = df.drop(columns=[target]).copy()
    y_raw = df[target].copy()

    # --- Penanganan nilai 0 -> NaN (khusus PIDD) ---
    zero_cols = [c for c in spec.get("zero_as_missing", []) if c in X.columns]
    for col in zero_cols:
        n_zero = int((X[col] == 0).sum())
        X.loc[X[col] == 0, col] = np.nan
        logger.info("  PIDD: %d nilai 0 pada '%s' dikonversi -> NaN", n_zero, col)

    # --- Pemetaan label -> {0, 1} ---
    y = _encode_target(y_raw, spec["positive_label"])

    # --- Subsampling opsional pada mode quick (untuk dataset besar) ---
    if config.QUICK_MAX_ROWS is not None and len(X) > config.QUICK_MAX_ROWS:
        X, y = _stratified_subsample(X, y, config.QUICK_MAX_ROWS)
        logger.info("  [QUICK] %s disubsample menjadi %d baris", spec["short"], len(X))

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y


def _encode_target(y_raw: pd.Series, positive_label) -> pd.Series:
    """Memetakan kolom label ke biner {0, 1} dengan 1 = positive_label."""
    # Jika sudah numerik 0/1, normalisasi langsung
    unique_vals = set(pd.unique(y_raw.dropna()))
    if unique_vals.issubset({0, 1, 0.0, 1.0}):
        return y_raw.astype(int)

    # Normalisasi string (case-insensitive, strip)
    def _norm(v):
        return str(v).strip().lower()

    pos = _norm(positive_label)
    y = y_raw.map(lambda v: 1 if _norm(v) == pos else 0).astype(int)

    # Sanity check: pastikan ada dua kelas
    if y.nunique() < 2:
        # Fallback: anggap nilai yang lebih jarang sebagai positif
        logger.warning(
            "positive_label='%s' tidak ditemukan; memakai fallback kelas minoritas.",
            positive_label,
        )
        counts = y_raw.value_counts()
        minority = counts.idxmin()
        y = (y_raw == minority).astype(int)
    return y


def _stratified_subsample(X: pd.DataFrame, y: pd.Series, n: int):
    """Subsampling terstratifikasi (dipakai hanya pada mode quick)."""
    from sklearn.model_selection import train_test_split

    if n >= len(X):
        return X, y
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=config.RANDOM_STATE
    )
    return X_s, y_s


# --------------------------------------------------------------------------- #
# Identifikasi tipe fitur
# --------------------------------------------------------------------------- #
def split_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """
    Mengembalikan (numeric_cols, categorical_cols).

    Heuristik (robust lintas versi pandas, termasuk pandas >= 2.x/3.x yang
    memperkenalkan dtype `string`/`str` khusus):
        - Kolom numerik sejati (int/float) dan BUKAN boolean -> numerik
        - object / string / str / category / bool / lainnya  -> kategorikal

    Menggunakan ``pd.api.types`` alih-alih membandingkan ``dtype == object``
    secara langsung, sebab pada pandas modern kolom teks tidak lagi selalu
    bertipe ``object`` (bisa ``string[python]``/``str``), sehingga pengecekan
    lama akan salah mengklasifikasikan teks sebagai numerik.
    """
    from pandas.api import types as pdt

    categorical_cols: list[str] = []
    numeric_cols: list[str] = []
    for col in X.columns:
        s = X[col]
        if pdt.is_bool_dtype(s):
            categorical_cols.append(col)
        elif pdt.is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            # object, string/str, category, datetime, dll -> perlakukan kategorikal
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


# --------------------------------------------------------------------------- #
# Konstruksi preprocessor & pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Membangun ColumnTransformer:
        numerik     : imputasi mean -> StandardScaler
        kategorikal : imputasi modus -> OneHotEncoder(drop="if_binary")

    Catatan: encoder one-hot dengan drop="if_binary" menghasilkan satu kolom 0/1
    untuk fitur biner (setara label encoding) dan one-hot penuh untuk nominal,
    persis sebagaimana dijelaskan pada BAB 3.
    """
    numeric_cols, categorical_cols = split_feature_types(X)

    numeric_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, classifier,
                   use_smote: bool = True) -> ImbPipeline:
    """
    Merangkai preprocessor + SMOTE + classifier dalam satu imblearn Pipeline.

    SMOTE diletakkan SETELAH preprocessing dan, karena pipeline ini selalu
    di-fit di dalam fold latih, SMOTE secara otomatis hanya menyentuh data latih.
    """
    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Mengambil nama fitur setelah transformasi (preprocessor sudah di-fit)."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - fallback defensif
        return [f"f{i}" for i in range(preprocessor.transform(
            pd.DataFrame()).shape[1])]
